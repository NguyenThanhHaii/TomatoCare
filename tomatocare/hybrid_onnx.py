"""Manifest-driven two-input hybrid ONNX inference, with no torch/transformers import."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ("Healthy", "Early Blight", "Late Blight", "Leaf Mold")
INPUT_NAMES = ("baseline_images", "dino_images")
SHAPE = [1, 3, 224, 224]
ARCHITECTURE = "unireplknet_a_stage3_dinov3_vits16"
RESAMPLING = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR,
              "bicubic": Image.Resampling.BICUBIC}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalization(config, mean_key, std_key):
    mean, std = np.asarray(config.get(mean_key, []), dtype=np.float32), np.asarray(config.get(std_key, []), dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError(f"Invalid {mean_key}/{std_key} normalization")
    return mean, std


def _size(size, *, shortest=False):
    if not isinstance(size, dict):
        raise ValueError("Resize/crop size must be an explicit object")
    keys = set(size)
    if keys != {"height", "width"} and not (shortest and keys == {"shortest_edge"}):
        raise ValueError("Unsupported size keys; use height/width or shortest_edge")
    if any(type(v) is not int or not 0 < v <= 4096 for v in size.values()):
        raise ValueError("Resize/crop dimensions must be integers in [1,4096]")


def validate_manifest(manifest):
    if manifest.get("architecture") != ARCHITECTURE:
        raise ValueError("Manifest architecture is not the supported hybrid")
    if manifest.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Manifest class_names must preserve Healthy/Early Blight/Late Blight/Leaf Mold order")
    if manifest.get("input_shapes") != {name: SHAPE for name in INPUT_NAMES}:
        raise ValueError("Manifest must declare both fixed [1,3,224,224] inputs")
    filename = manifest.get("model_file")
    if not isinstance(filename, str) or Path(filename).name != filename or filename != "tomatocare_hybrid.onnx":
        raise ValueError("model_file must be tomatocare_hybrid.onnx inside HYBRID_MODEL_DIR")
    digest = manifest.get("model_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Manifest requires a lowercase SHA-256 digest")
    preprocessing = manifest.get("preprocessing", {})
    base, dino = preprocessing.get("unireplknet", {}), preprocessing.get("dinov3", {})
    if base.get("color_mode") != "RGB" or base.get("image_size") != [224, 224] or base.get("resize") != "exact" or base.get("interpolation") not in RESAMPLING:
        raise ValueError("Unsupported baseline RGB/resize/interpolation configuration")
    _normalization(base, "mean", "std")
    if dino.get("image_processor_type") != "DINOv3ViTImageProcessorFast":
        raise ValueError("Only the manifest's DINOv3ViTImageProcessorFast order is supported")
    for flag in ("do_resize", "do_rescale", "do_normalize"):
        if type(dino.get(flag)) is not bool:
            raise ValueError(f"Manifest must explicitly specify {flag}")
    if not dino["do_rescale"]:
        raise ValueError("This Fast processor requires explicit rescale-before-resize; do_rescale=false is unsupported")
    for flag in ("do_center_crop", "do_convert_rgb"):
        if dino.get(flag) is not None and type(dino[flag]) is not bool:
            raise ValueError(f"Invalid {flag}")
    if dino["do_resize"]:
        _size(dino.get("size"), shortest=True)
        if type(dino.get("resample")) is not int or dino["resample"] not in (0, 2, 3):
            raise ValueError("Tensor resize supports only nearest(0), bilinear(2), bicubic(3)")
    if dino.get("do_center_crop"):
        _size(dino.get("crop_size"))
    if dino["do_rescale"]:
        factor = dino.get("rescale_factor")
        if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not np.isfinite(factor) or factor <= 0:
            raise ValueError("Invalid DINO rescale_factor")
    if dino["do_normalize"]:
        _normalization(dino, "image_mean", "image_std")
    if "model_version" in manifest and (not isinstance(manifest["model_version"], str) or not manifest["model_version"].strip()):
        raise ValueError("model_version must be nonempty when supplied")
    return manifest


def load_manifest(directory):
    root = Path(directory).resolve()
    path = root / "inference_manifest.json"
    if not path.is_file():
        raise ValueError("Missing inference_manifest.json in HYBRID_MODEL_DIR")
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Manifest is unexpectedly large")
    return root, validate_manifest(json.loads(path.read_text()))


def _weights(source, target, resample):
    """Separable align_corners=False antialias weights used for float tensor resize.

    Float bilinear/bicubic uses half-pixel centers, widened filters for downsample,
    edge renormalization and the antialiased bicubic coefficient a=-0.5.
    """
    # Match float32 coordinate arithmetic, not a float64 geometric approximation.
    # The difference matters for nonintegral resize ratios after normalization.
    scale = np.float32(source / target)
    support = np.float32((1 if resample == 2 else 2) * max(float(scale), 1.0))
    inverse = np.float32(1.0 / scale) if scale >= 1 else np.float32(1)
    centers = (float(scale) * (np.arange(target, dtype=np.float64) + 0.5)).astype(np.float32)
    source_indices = np.arange(source, dtype=np.float32)[None, :]
    delta = (source_indices - centers[:, None]).astype(np.float64) + 0.5
    distance = np.abs((delta * float(inverse)).astype(np.float32))
    if resample == 2:
        weights = np.maximum(1.0 - distance, 0)
    else:
        x = distance
        weights = np.where(x < 1, ((1.5 * x - 2.5) * x) * x + 1,
                           np.where(x < 2, ((-0.5 * x + 2.5) * x - 4) * x + 2, 0))
    lower = np.maximum(np.trunc((centers - support).astype(np.float64) + 0.5).astype(int), 0)
    upper = np.minimum(np.trunc((centers + support).astype(np.float64) + 0.5).astype(int), source)
    weights *= (source_indices >= lower[:, None]) & (source_indices < upper[:, None])
    weights = weights.astype(np.float32)
    return weights / weights.sum(axis=1, keepdims=True)


def tensor_resize(image, size, resample):
    """HWC NumPy equivalent of antialiased Torchvision float tensor resize."""
    h, w = image.shape[:2]
    if "shortest_edge" in size:
        edge = size["shortest_edge"]
        oh, ow = (edge, int(w * edge / h)) if h <= w else (int(h * edge / w), edge)
    else:
        oh, ow = size["height"], size["width"]
    if max(oh, ow) > 4096:
        raise ValueError("Computed resize dimensions exceed the supported limit")
    if (h, w) == (oh, ow):
        return image
    if resample == 0:
        ys = np.minimum((np.arange(oh) * h / oh).astype(int), h - 1)
        xs = np.minimum((np.arange(ow) * w / ow).astype(int), w - 1)
        return image[ys[:, None], xs[None, :], :]
    def apply_axis(values, weights):
        output = np.empty((len(weights), *values.shape[1:]), dtype=np.float32)
        for i, row in enumerate(weights):
            nonzero = np.flatnonzero(row)
            output[i] = (values[nonzero] * row[nonzero, None, None]).sum(axis=0)
        return output
    # Sparse filter support avoids dense BLAS multiplication and hidden CPU pools.
    resized = apply_axis(image, _weights(h, oh, resample))
    return apply_axis(resized.transpose(1, 0, 2), _weights(w, ow, resample)).transpose(1, 0, 2)


def center_crop(image, size):
    th, tw = size["height"], size["width"]
    h, w = image.shape[:2]
    if th > h or tw > w:
        ph, pw = max(th - h, 0), max(tw - w, 0)
        image = np.pad(image, ((ph // 2, (ph + 1) // 2), (pw // 2, (pw + 1) // 2), (0, 0)))
        h, w = image.shape[:2]
    top, left = int(round((h - th) / 2)), int(round((w - tw) / 2))
    return image[top:top + th, left:left + tw]


def _nchw(array):
    value = np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=np.float32)
    if list(value.shape) != SHAPE or not np.isfinite(value).all():
        raise ValueError("Manifest preprocessing must produce finite float32 [1,3,224,224]")
    return value


def preprocess(image, manifest):
    if max(image.size) > 8192:
        raise ValueError("Image side exceeds experimental 8192-pixel preprocessing limit")
    image = image.convert("RGB")
    configs = manifest["preprocessing"]
    base, dino = configs["unireplknet"], configs["dinov3"]
    # Baseline training: PIL resize (uint8 rounding), then /255, then normalize.
    resized = image.resize((224, 224), RESAMPLING[base["interpolation"]])
    baseline = np.asarray(resized, dtype=np.float32) / np.float32(255)
    mean, std = _normalization(base, "mean", "std")
    baseline = _nchw((baseline - mean) / std)
    # Fast DINO processor: uint8->float rescale BEFORE float antialiased resize.
    pixels = np.asarray(image, dtype=np.float32)
    if dino["do_rescale"]:
        pixels *= np.float32(dino["rescale_factor"])
    if dino["do_resize"]:
        pixels = tensor_resize(pixels, dino["size"], dino["resample"])
    if dino.get("do_center_crop"):
        pixels = center_crop(pixels, dino["crop_size"])
    if dino["do_normalize"]:
        mean, std = _normalization(dino, "image_mean", "image_std")
        pixels = (pixels - mean) / std
    return {"baseline_images": baseline, "dino_images": _nchw(pixels)}


class HybridOnnxPredictor:
    def __init__(self, directory, session_factory=None):
        self.root, self.manifest = load_manifest(directory)
        path = (self.root / self.manifest["model_file"]).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("Missing model file or model path escapes HYBRID_MODEL_DIR")
        # Verify the supplied artifact before ONNX Runtime parses any model bytes.
        if sha256_file(path) != self.manifest["model_sha256"]:
            raise ValueError("Hybrid ONNX SHA-256 mismatch; inference refused")
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # Keep other optimization defaults; no unreported quantization or graph change.
        factory = session_factory or ort.InferenceSession
        self.session = factory(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        if {x.name for x in inputs} != set(INPUT_NAMES) or len(inputs) != 2 or any(x.type != "tensor(float)" or x.shape != SHAPE for x in inputs):
            raise ValueError("Hybrid ONNX requires baseline_images/dino_images float32 [1,3,224,224]")
        outputs = self.session.get_outputs()
        if len(outputs) != 1 or outputs[0].shape != [1, 4] or outputs[0].type != "tensor(float)":
            raise ValueError("Hybrid ONNX must return one float32 [1,4] logits output")
        self.output_name = outputs[0].name
        self.model_version = self.manifest.get("model_version", "hybrid-" + self.manifest["model_sha256"][:12])

    def logits(self, image):
        values = preprocess(image, self.manifest)
        result = np.asarray(self.session.run([self.output_name], values)[0], dtype=np.float32)
        if result.shape != (1, 4) or not np.isfinite(result).all():
            raise ValueError("Hybrid ONNX returned invalid logits")
        return result

    def predict(self, image):
        logits = self.logits(image)[0]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        winner = int(probabilities.argmax())
        return {"predicted_class": CLASS_NAMES[winner], "confidence": float(probabilities[winner]),
                "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probabilities)},
                "model_version": self.model_version}


def load_hybrid_predictor():
    directory = os.getenv("HYBRID_MODEL_DIR")
    if not directory:
        raise ValueError("Set HYBRID_MODEL_DIR to the private directory containing inference_manifest.json and tomatocare_hybrid.onnx")
    return HybridOnnxPredictor(directory)
