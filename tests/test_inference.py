"""Synthetic checkpoint tests validate plumbing, not trained model accuracy."""

import json

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from scripts.export_onnx import export_verified
from tomatocare.data import PREPROCESSING
from tomatocare.inference import EXPECTED_CLASSES, OnnxEngine, Predictor, TorchEngine, preprocess, validate_metadata
from tomatocare.model import create_unireplknet_a


@pytest.fixture
def checkpoint(tmp_path):
    torch.set_num_threads(1)
    model = create_unireplknet_a()
    path = tmp_path / "random_test_checkpoint.pth"
    torch.save({"state_dict": model.state_dict(), "class_names": list(EXPECTED_CLASSES),
                "class_to_idx": {n: i for i, n in enumerate(EXPECTED_CLASSES)},
                "preprocessing": PREPROCESSING}, path)
    return path


def test_preprocessing_matches_training():
    image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (170, 280, 3), dtype=np.uint8))
    training = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
                                   transforms.Normalize(PREPROCESSING["mean"], PREPROCESSING["std"])])(image)
    np.testing.assert_allclose(preprocess(image, PREPROCESSING)[0], training.numpy(), atol=1e-6)


def test_reject_unknown_preprocessing_or_wrong_classes():
    with pytest.raises(ValueError, match="class_names"):
        validate_metadata({"class_names": list(reversed(EXPECTED_CLASSES)), "preprocessing": PREPROCESSING})
    with pytest.raises(ValueError, match="interpolation"):
        validate_metadata({"class_names": list(EXPECTED_CLASSES), "preprocessing": {**PREPROCESSING, "interpolation": "bicubic"}})


def test_torch_checkpoint_inference(checkpoint):
    engine = TorchEngine(checkpoint)
    response = Predictor(engine).predict(Image.new("RGB", (300, 210), "green"))
    assert response["predicted_class"] in EXPECTED_CLASSES
    assert sum(response["probabilities"].values()) == pytest.approx(1.0)
    assert not engine.model.training


def test_export_parity_and_runtime_integrity(checkpoint, tmp_path):
    # Random weights ONLY in this test fixture; real artifacts need a trained checkpoint.
    path, metadata = export_verified(checkpoint, tmp_path / "onnx")
    manifest = json.loads(metadata.read_text())
    assert manifest["verification"]["input_count"] == 7
    assert manifest["verification"]["same_argmax_all"]
    engine = OnnxEngine(path, metadata)
    assert len(Predictor(engine).predict(Image.new("RGB", (250, 300)))["probabilities"]) == 4
    with path.open("ab") as file:
        file.write(b"corruption")
    with pytest.raises(ValueError, match="SHA-256"):
        OnnxEngine(path, metadata)


@pytest.fixture(scope="module")
def onnx_artifacts(tmp_path_factory):
    root = tmp_path_factory.mktemp("onnx_contract_tests")
    torch.set_num_threads(1)
    path = root / "synthetic_fixture.pth"
    torch.save({"state_dict": create_unireplknet_a().state_dict(),
                "class_names": list(EXPECTED_CLASSES), "preprocessing": PREPROCESSING}, path)
    return export_verified(path, root / "export")


@pytest.mark.parametrize("profile", ["default", "low_memory"])
def test_onnx_profiles_and_batch_one(onnx_artifacts, profile, monkeypatch):
    monkeypatch.setenv("CPU_THREADS", "8")  # ONNX production must still use one thread.
    engine = OnnxEngine(onnx_artifacts[0], profile=profile)
    options = engine.session.get_session_options()
    assert options.intra_op_num_threads == options.inter_op_num_threads == 1
    assert engine.session.get_providers() == ["CPUExecutionProvider"]
    assert options.enable_cpu_mem_arena is (profile == "default")
    assert options.enable_mem_pattern is (profile == "default")
    result = Predictor(engine).predict(Image.new("RGB", (250, 270)))
    assert list(result["probabilities"]) == list(EXPECTED_CLASSES)
    with pytest.raises(ValueError, match="batch=1"):
        engine.logits(np.zeros((2, 3, 224, 224), dtype=np.float32))


def test_default_download_uses_real_onnx_filename_no_sidecar(onnx_artifacts, monkeypatch):
    import tomatocare.inference as inference
    monkeypatch.delenv("INFERENCE_RUNTIME", raising=False)
    monkeypatch.delenv("ONNX_MODEL_PATH", raising=False)
    monkeypatch.delenv("ONNX_METADATA_PATH", raising=False)
    monkeypatch.delenv("ONNX_FILENAME", raising=False)
    downloads = []
    def download(filename):
        downloads.append(filename)
        return onnx_artifacts[0]
    monkeypatch.setattr(inference, "download_artifact", download)
    from backend.main import create_app
    from fastapi.testclient import TestClient
    import io
    image = io.BytesIO()
    Image.new("RGB", (32, 48)).save(image, format="PNG")
    with TestClient(create_app()) as client:
        health = client.get("/health").json()
        assert health["runtime"] == "onnx"
        assert health["metadata_source"] == "owner_confirmed_checkpoint_contract"
        for _ in range(2):
            response = client.post("/predict", files={"file": ("leaf.png", image.getvalue(), "image/png")})
            assert response.status_code == 200
            assert sum(response.json()["probabilities"].values()) == pytest.approx(1.0)
    assert downloads == ["best_unireplknet_a.onnx"]


def test_onnx_download_failure_does_not_use_random_or_torch_fallback(monkeypatch):
    import tomatocare.inference as inference
    monkeypatch.setenv("INFERENCE_RUNTIME", "onnx")
    monkeypatch.delenv("ONNX_MODEL_PATH", raising=False)
    def blocked(filename):
        raise RuntimeError("403 Forbidden")
    monkeypatch.setattr(inference, "download_artifact", blocked)
    with pytest.raises(inference.ModelUnavailable, match="403 Forbidden"):
        inference.load_predictor()


def test_embedded_metadata_and_invalid_input_shape(onnx_artifacts, tmp_path):
    import onnx
    model = onnx.load(onnx_artifacts[0])
    onnx.helper.set_model_props(model, {"class_names": json.dumps(list(EXPECTED_CLASSES)),
                                      "preprocessing": json.dumps(PREPROCESSING)})
    path = tmp_path / "embedded.onnx"
    onnx.save(model, path)
    assert OnnxEngine(path).metadata_source == "onnx_embedded"
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_value = 2
    onnx.save(model, path)
    with pytest.raises(ValueError, match="fixed float32 ONNX input"):
        OnnxEngine(path)


def test_imagenet_normalization_is_required():
    with pytest.raises(ValueError, match="ImageNet"):
        validate_metadata({"class_names": list(EXPECTED_CLASSES),
                           "preprocessing": {**PREPROCESSING, "mean": [0.5, 0.5, 0.5]}})
