"""CPU inference; the ONNX runtime path intentionally never imports PyTorch."""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_REPO = "ThanhHai25/TomatoCare-UniRepLKNet-A"
CHECKPOINT_NAME = "best_unireplknet_a.pth"
ONNX_NAME = "best_unireplknet_a.onnx"
EXPECTED_CLASSES = ("Healthy", "Early Blight", "Late Blight", "Leaf Mold")
# Confirmed trained-checkpoint contract supplied by the model owner. This is
# preprocessing metadata, never substitute model weights or prediction data.
CONFIRMED_METADATA = {
    "class_names": list(EXPECTED_CLASSES),
    "class_to_idx": {name: index for index, name in enumerate(EXPECTED_CLASSES)},
    "preprocessing": {
        "color_mode": "RGB", "image_size": [224, 224], "resize": "exact", "interpolation": "bilinear",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    },
}


class ModelUnavailable(RuntimeError):
    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_metadata(metadata):
    names = metadata.get("class_names")
    if names != list(EXPECTED_CLASSES):
        raise ValueError(f"Checkpoint class_names must be {EXPECTED_CLASSES}, in this exact order")
    mapping = metadata.get("class_to_idx")
    if mapping is not None and mapping != {name: i for i, name in enumerate(names)}:
        raise ValueError("class_to_idx disagrees with class_names")
    config = metadata.get("preprocessing", {})
    if config.get("color_mode") != "RGB" or config.get("image_size") != [224, 224]:
        raise ValueError("Expected checkpoint RGB 224x224 preprocessing")
    if config.get("resize") != "exact" or config.get("interpolation") != "bilinear":
        raise ValueError("Unsupported checkpoint resize/interpolation; do not guess preprocessing")
    mean, std = np.array(config.get("mean", [])), np.array(config.get("std", []))
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Invalid checkpoint normalization")
    if not np.allclose(mean, [0.485, 0.456, 0.406], rtol=0, atol=1e-7) or not np.allclose(std, [0.229, 0.224, 0.225], rtol=0, atol=1e-7):
        raise ValueError("Checkpoint normalization disagrees with the confirmed ImageNet mean/std")
    return tuple(names), config


def preprocess(image, config):
    """Same RGB, exact bilinear resize and ImageNet normalization as training."""
    image = image.convert("RGB").resize(tuple(config["image_size"])[::-1], Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float32) / np.float32(255.0)
    values = (values - np.asarray(config["mean"], dtype=np.float32)) / np.asarray(config["std"], dtype=np.float32)
    return np.ascontiguousarray(values.transpose(2, 0, 1)[None])


def download_artifact(filename):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=os.getenv("MODEL_REPO_ID", MODEL_REPO), filename=filename,
                               revision=os.getenv("MODEL_REVISION", "main")))


class TorchEngine:
    def __init__(self, checkpoint_path=None):
        import torch
        from .model import create_unireplknet_a
        torch.set_num_threads(int(os.getenv("CPU_THREADS", "1")))
        self.torch = torch
        path = Path(checkpoint_path) if checkpoint_path else download_artifact(CHECKPOINT_NAME)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.class_names, self.config = validate_metadata(checkpoint)
        self.metadata = checkpoint
        self.model = create_unireplknet_a()
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.checkpoint_sha256 = sha256_file(path)
        self.identifier = f"{os.getenv('MODEL_REPO_ID', MODEL_REPO)}:{self.checkpoint_sha256[:12]}:pytorch"
        # Keep metadata, not a second state_dict reference, after model construction.
        self.metadata = {k: v for k, v in checkpoint.items() if k != "state_dict"}

    def logits(self, values):
        with self.torch.inference_mode():
            return self.model(self.torch.from_numpy(values)).cpu().numpy()


class OnnxEngine:
    def __init__(self, model_path=None, metadata_path=None, profile=None):
        import onnxruntime as ort
        if metadata_path and not model_path:
            raise ValueError("A local ONNX_METADATA_PATH requires ONNX_MODEL_PATH")
        path = Path(model_path) if model_path else download_artifact(os.getenv("ONNX_FILENAME", ONNX_NAME))
        model_hash = sha256_file(path)
        sidecar = None
        if metadata_path:
            sidecar = json.loads(Path(metadata_path).read_text())
            if sidecar.get("onnx_sha256") != model_hash:
                raise ValueError("ONNX SHA-256 does not match its verified metadata")
            verification = sidecar.get("verification", {})
            if verification.get("passed") is not True or verification.get("input_count", 0) < 3:
                raise ValueError("ONNX metadata has no successful multi-input parity verification")
        self.profile = profile or os.getenv("ONNX_PROFILE", "default")
        if self.profile not in ("default", "low_memory"):
            raise ValueError("ONNX_PROFILE must be default or low_memory")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if self.profile == "low_memory":
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or inputs[0].type != "tensor(float)" or inputs[0].shape != [1, 3, 224, 224]:
            raise ValueError("Expected one fixed float32 ONNX input [1, 3, 224, 224]")
        if len(outputs) != 1 or outputs[0].type != "tensor(float)" or outputs[0].shape != [1, 4]:
            raise ValueError("Expected one float32 ONNX logits output [1, 4]")
        self.input_name = inputs[0].name
        embedded = self.session.get_modelmeta().custom_metadata_map
        if sidecar is not None:
            metadata = sidecar
            self.metadata_source = "verified_sidecar"
        elif "class_names" in embedded or "preprocessing" in embedded:
            if "class_names" not in embedded or "preprocessing" not in embedded:
                raise ValueError("Incomplete embedded class/preprocessing metadata")
            metadata = {"class_names": json.loads(embedded["class_names"]),
                        "preprocessing": json.loads(embedded["preprocessing"])}
            if "class_to_idx" in embedded:
                metadata["class_to_idx"] = json.loads(embedded["class_to_idx"])
            self.metadata_source = "onnx_embedded"
        else:
            # The owner has already validated this exact four-class contract on
            # Kaggle. Do not require a sidecar that may not exist in the HF repo.
            metadata = CONFIRMED_METADATA
            self.metadata_source = "owner_confirmed_checkpoint_contract"
        self.class_names, self.config = validate_metadata(metadata)
        if "class_names" in embedded and json.loads(embedded["class_names"]) != list(self.class_names):
            raise ValueError("Embedded class_names disagree with the supplied metadata")
        if "preprocessing" in embedded:
            validate_metadata({"class_names": list(self.class_names), "preprocessing": json.loads(embedded["preprocessing"])})
        self.identifier = f"{os.getenv('MODEL_REPO_ID', MODEL_REPO)}:{model_hash[:12]}:onnx"

    def logits(self, values):
        if values.shape != (1, 3, 224, 224) or values.dtype != np.float32:
            raise ValueError("ONNX inference requires batch=1 float32 [1, 3, 224, 224]")
        return self.session.run(None, {self.input_name: values})[0]


class Predictor:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()

    def predict(self, image):
        started = time.perf_counter()
        values = preprocess(image, self.engine.config)
        with self.lock:
            logits = np.asarray(self.engine.logits(values), dtype=np.float32)
        if logits.shape != (1, 4) or not np.isfinite(logits).all():
            raise ValueError("Model returned invalid logits")
        probabilities = np.exp(logits[0] - logits[0].max())
        probabilities /= probabilities.sum()
        winner = int(probabilities.argmax())
        return {
            "predicted_class": self.engine.class_names[winner], "confidence": float(probabilities[winner]),
            "probabilities": {name: float(value) for name, value in zip(self.engine.class_names, probabilities)},
            "model_identifier": self.engine.identifier,
            "inference_ms": round((time.perf_counter() - started) * 1000, 2),
        }


def load_predictor():
    try:
        runtime = os.getenv("INFERENCE_RUNTIME", "onnx")
        if runtime == "pytorch":
            engine = TorchEngine(os.getenv("CHECKPOINT_PATH"))
        elif runtime == "onnx":
            engine = OnnxEngine(os.getenv("ONNX_MODEL_PATH"), os.getenv("ONNX_METADATA_PATH"))
        else:
            raise ValueError("INFERENCE_RUNTIME must be pytorch or onnx")
        return Predictor(engine)
    except Exception as exc:
        raise ModelUnavailable(
            f"Cannot initialize TomatoCare: {exc}. For PyTorch, provide the trusted official trained "
            "best_unireplknet_a.pth using CHECKPOINT_PATH, or allow Hugging Face access. "
            "For ONNX, allow Hugging Face access to best_unireplknet_a.onnx or set ONNX_MODEL_PATH to a trusted downloaded copy. "
            "No mock prediction or random-initialization fallback is used."
        ) from exc
