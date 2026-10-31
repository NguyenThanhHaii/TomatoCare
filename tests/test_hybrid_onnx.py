"""Synthetic manifest/session tests only: no trained model or dataset is bundled."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

from hybrid_backend.main import create_app
from tomatocare.hybrid_onnx import CLASS_NAMES, HybridOnnxPredictor, center_crop, load_hybrid_predictor, preprocess, sha256_file, tensor_resize, validate_manifest


@pytest.fixture
def manifest():
    return json.loads((Path(__file__).parents[1] / "docs/examples/hybrid_inference_manifest.json").read_text())


@pytest.fixture
def artifact(tmp_path, manifest):
    (tmp_path / "tomatocare_hybrid.onnx").write_bytes(b"TEST-ONLY session fixture, not a real ONNX graph")
    manifest["model_sha256"] = sha256_file(tmp_path / "tomatocare_hybrid.onnx")
    (tmp_path / "inference_manifest.json").write_text(json.dumps(manifest))
    return tmp_path


class FakeSession:
    calls = []
    def __init__(self, path, **kwargs):
        self.calls.append(kwargs)
        self.run_count = 0

    def get_inputs(self):
        return [SimpleNamespace(name=n, type="tensor(float)", shape=[1, 3, 224, 224]) for n in ("baseline_images", "dino_images")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits", type="tensor(float)", shape=[1, 4])]

    def run(self, outputs, values):
        self.run_count += 1
        assert outputs == ["logits"]
        assert all(x.shape == (1, 3, 224, 224) and x.dtype == np.float32 for x in values.values())
        return [np.array([[0, 3, 1, -1]], dtype=np.float32)]


def photo(format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (37, 41), "green").save(output, format=format)
    return output.getvalue()


@pytest.mark.parametrize("shape,target,resample", [
    ((37, 41), (224, 224), 2), ((256, 320), (224, 224), 2),
    ((501, 641), (96, 110), 2), ((224, 224), (224, 224), 2),
    ((37, 41), (224, 224), 3), ((256, 320), (224, 224), 3),
    ((501, 641), (96, 110), 3), ((256, 320), (224, 224), 0),
])
def test_float_resize_matches_torchvision(shape, target, resample):
    torch.set_num_threads(2)
    image = np.random.default_rng(42).random((*shape, 3), dtype=np.float32)
    interpolation = {0: transforms.InterpolationMode.NEAREST, 2: transforms.InterpolationMode.BILINEAR, 3: transforms.InterpolationMode.BICUBIC}[resample]
    reference = F.resize(torch.from_numpy(image).permute(2, 0, 1), list(target), interpolation=interpolation, antialias=True).permute(1, 2, 0).numpy()
    actual = tensor_resize(image, {"height": target[0], "width": target[1]}, resample)
    np.testing.assert_allclose(actual, reference, atol=3e-6, rtol=1e-5)


def test_two_real_training_preprocessing_orders(manifest):
    image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (301, 257, 3), dtype=np.uint8))
    values = preprocess(image, validate_manifest(manifest))
    base = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                               transforms.Normalize(manifest["preprocessing"]["unireplknet"]["mean"], manifest["preprocessing"]["unireplknet"]["std"])])(image)
    dino = F.pil_to_tensor(image).float() * manifest["preprocessing"]["dinov3"]["rescale_factor"]
    dino = F.resize(dino, [224, 224], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
    dino = F.normalize(dino, manifest["preprocessing"]["dinov3"]["image_mean"], manifest["preprocessing"]["dinov3"]["image_std"])
    np.testing.assert_allclose(values["baseline_images"][0], base.numpy(), atol=1e-6)
    np.testing.assert_allclose(values["dino_images"][0], dino.numpy(), atol=1e-5, rtol=1e-5)
    assert not np.allclose(values["baseline_images"], values["dino_images"], atol=1e-4)
    assert all(v.dtype == np.float32 and v.flags.c_contiguous for v in values.values())


def test_distinct_normalization_and_center_crop(manifest):
    image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (200, 300, 3), dtype=np.uint8))
    config = manifest["preprocessing"]["dinov3"]
    config.update(size={"shortest_edge": 256}, do_center_crop=True, crop_size={"height": 224, "width": 224}, image_mean=[.1, .2, .3], image_std=[.4, .5, .6])
    values = preprocess(image, validate_manifest(manifest))
    ref = F.resize(F.pil_to_tensor(image).float() / 255, 256, antialias=True)
    ref = F.normalize(F.center_crop(ref, [224, 224]), config["image_mean"], config["image_std"])
    np.testing.assert_allclose(values["dino_images"][0], ref.numpy(), atol=1e-5, rtol=1e-5)


def test_center_crop_padding_matches_torchvision():
    image = np.random.default_rng(2).random((20, 31, 3), dtype=np.float32)
    ref = F.center_crop(torch.from_numpy(image).permute(2, 0, 1), [41, 42]).permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(center_crop(image, {"height": 41, "width": 42}), ref)


@pytest.mark.parametrize("kind", ["classes", "shape", "sha", "path", "processor", "std", "rescale", "resample"])
def test_manifest_rejects_bad_configuration(manifest, kind):
    if kind == "classes": manifest["class_names"] = sorted(CLASS_NAMES)
    elif kind == "shape": manifest["input_shapes"]["dino_images"] = [2, 3, 224, 224]
    elif kind == "sha": manifest["model_sha256"] = "invalid"
    elif kind == "path": manifest["model_file"] = "../tomatocare_hybrid.onnx"
    elif kind == "processor": manifest["preprocessing"]["dinov3"]["image_processor_type"] = "Unknown"
    elif kind == "std": manifest["preprocessing"]["dinov3"]["image_std"] = [0, 1, 1]
    elif kind == "rescale": manifest["preprocessing"]["dinov3"]["do_rescale"] = False
    else: manifest["preprocessing"]["dinov3"]["resample"] = 99
    with pytest.raises(ValueError): validate_manifest(manifest)


def test_sha_checked_before_session_construction(artifact):
    (artifact / "tomatocare_hybrid.onnx").write_bytes(b"corrupt")
    def forbidden(*args, **kwargs):
        pytest.fail("Session must not be constructed before hash verification")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        HybridOnnxPredictor(artifact, session_factory=forbidden)


def test_cached_session_low_memory_and_api(artifact):
    FakeSession.calls = []
    def factory(): return HybridOnnxPredictor(artifact, session_factory=FakeSession)
    with TestClient(create_app(factory)) as client:
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["experimental"]
        for _ in range(2):
            response = client.post("/predict", files={"file": ("leaf.png", photo(), "image/png")})
            assert response.status_code == 200
            result = response.json()
            assert list(result["probabilities"]) == list(CLASS_NAMES)
            assert result["predicted_class"] == "Early Blight"
            assert sum(result["probabilities"].values()) == pytest.approx(1)
            assert result["confidence"] == max(result["probabilities"].values())
            assert result["model_version"] == health.json()["model_version"]
            assert response.headers["cache-control"] == "no-store"
        assert client.app.state.predictor.session.run_count == 2
    assert len(FakeSession.calls) == 1
    options = FakeSession.calls[0]["sess_options"]
    assert options.intra_op_num_threads == options.inter_op_num_threads == 1
    assert not options.enable_cpu_mem_arena and not options.enable_mem_pattern
    assert FakeSession.calls[0]["providers"] == ["CPUExecutionProvider"]


@pytest.mark.parametrize("filename,payload,mime,status", [
    ("a.png", b"broken", "image/png", 422), ("a.png", b"", "image/png", 422),
    ("a.gif", photo(), "image/gif", 415), ("a.png", photo("BMP"), "image/png", 422),
    ("a.png", b"x" * (10 * 1024 * 1024 + 1), "image/png", 413),
])
def test_upload_errors(artifact, filename, payload, mime, status):
    with TestClient(create_app(lambda: HybridOnnxPredictor(artifact, session_factory=FakeSession))) as client:
        assert client.post("/predict", files={"file": (filename, payload, mime)}).status_code == status


def test_missing_artifact_never_produces_mock(monkeypatch, tmp_path):
    monkeypatch.delenv("HYBRID_MODEL_DIR", raising=False)
    with pytest.raises(ValueError, match="HYBRID_MODEL_DIR"): load_hybrid_predictor()
    monkeypatch.setenv("HYBRID_MODEL_DIR", str(tmp_path))
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
        response = client.post("/predict", files={"file": ("leaf.png", photo(), "image/png")})
        assert response.status_code == 503
        assert "Missing inference_manifest" in response.json()["detail"]


def test_invalid_logits_fail_clearly(artifact):
    class BrokenSession(FakeSession):
        def run(self, *args): return [np.full((1, 4), np.nan)]
    with TestClient(create_app(lambda: HybridOnnxPredictor(artifact, session_factory=BrokenSession))) as client:
        assert client.post("/predict", files={"file": ("leaf.png", photo(), "image/png")}).status_code == 503


def test_exif_orientation_and_rgb_are_normalized(artifact):
    # Same shared validator used by the experimental endpoint, not a training change.
    from backend.main import decode_image
    image = Image.new("RGB", (20, 30))
    exif = image.getexif(); exif[274] = 6
    data = io.BytesIO(); image.save(data, format="JPEG", exif=exif)
    result = decode_image(data.getvalue())
    assert result.mode == "RGB" and result.size == (30, 20)


def test_invalid_session_signature_is_rejected(artifact):
    class WrongSession(FakeSession):
        def get_inputs(self):
            return [SimpleNamespace(name="wrong", type="tensor(float)", shape=[1, 3, 224, 224])]
    with pytest.raises(ValueError, match="baseline_images/dino_images"):
        HybridOnnxPredictor(artifact, session_factory=WrongSession)


def test_real_ort_session_with_tiny_synthetic_graph_only(artifact):
    # Exercises actual ORT plumbing, NOT the unavailable trained 90 MiB artifact.
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 3, 224, 224])
              for name in ("baseline_images", "dino_images")]
    nodes = []
    for name in ("baseline", "dino"):
        nodes += [helper.make_node("ReduceMean", [name + "_images"], [name + "_mean"], axes=[1, 2, 3], keepdims=0),
                  helper.make_node("Unsqueeze", [name + "_mean", "axes"], [name + "_feature"])]
    nodes.append(helper.make_node("Concat", ["baseline_feature", "dino_feature", "baseline_feature", "dino_feature"], ["logits"], axis=1))
    graph = helper.make_graph(nodes, "SYNTHETIC-TEST-NOT-HYBRID-WEIGHTS", inputs,
                              [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 4])],
                              [numpy_helper.from_array(np.array([1], dtype=np.int64), "axes")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    path = artifact / "tomatocare_hybrid.onnx"
    onnx.save(model, path)
    manifest = json.loads((artifact / "inference_manifest.json").read_text())
    manifest["model_sha256"] = sha256_file(path)
    (artifact / "inference_manifest.json").write_text(json.dumps(manifest))
    predictor = HybridOnnxPredictor(artifact)
    assert predictor.session.get_providers() == ["CPUExecutionProvider"]
    assert sum(predictor.predict(Image.new("RGB", (310, 241)))["probabilities"].values()) == pytest.approx(1)


def test_multi_image_validation_harness_with_explicit_mock_reference(artifact, tmp_path):
    import csv
    import hashlib
    from scripts.check_hybrid_onnx import compare_loaded_model
    manifest = json.loads((artifact / "inference_manifest.json").read_text())
    root = tmp_path / "synthetic_data"
    root.mkdir()
    (root / "labels.json").write_text(json.dumps({n: i for i, n in enumerate(CLASS_NAMES)}))
    rows = []
    for s, split in enumerate(("train", "val", "test")):
        for c, label in enumerate(CLASS_NAMES):
            relative = Path("images") / split / f"{c}.png"
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (31, 37), (40 * c, 60 * s, 20 + c)).save(path)
            rows.append({"image_path": str(relative), "label": label, "label_id": c,
                         "split": split, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_image_path": str(relative)})
    with (root / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    class MockReference(torch.nn.Module):
        provenance = manifest["provenance"]
        def forward(self, baseline, dino):
            return torch.tensor([[0, 3, 1, -1]], dtype=torch.float32)
    class MockFastProcessor:
        def __call__(self, images, return_tensors):
            pixels = F.pil_to_tensor(images).float() / 255
            pixels = F.resize(pixels, [224, 224], antialias=True)
            return {"pixel_values": F.normalize(pixels, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)}
    predictor = HybridOnnxPredictor(artifact, session_factory=FakeSession)
    report = compare_loaded_model(MockReference(), MockFastProcessor(), artifact, root, tmp_path / "mock_report.json",
                                  predictor=predictor, check_counts=False, per_class=1)
    assert report["passed"] and report["images_checked"] == 4
    assert all("/val/" in row["validation_image"] for row in report["rows"])
    with pytest.raises(ValueError, match="at least one"):
        compare_loaded_model(MockReference(), MockFastProcessor(), artifact, root, tmp_path / "invalid.json", per_class=0)
    class WrongReference(MockReference):
        def forward(self, *args): return torch.zeros(1, 4)
    with pytest.raises(RuntimeError, match="Multi-image parity failed"):
        compare_loaded_model(WrongReference(), MockFastProcessor(), artifact, root, tmp_path / "failure.json",
                             predictor=predictor, check_counts=False, per_class=1)
    assert json.loads((tmp_path / "failure.json").read_text())["passed"] is False


def test_extreme_aspect_ratio_is_rejected_before_resampling(artifact, manifest):
    image = Image.new("RGB", (9000, 2))
    data = io.BytesIO(); image.save(data, format="PNG")
    with pytest.raises(ValueError, match="8192-pixel"):
        preprocess(image, manifest)
    with TestClient(create_app(lambda: HybridOnnxPredictor(artifact, session_factory=FakeSession))) as client:
        response = client.post("/predict", files={"file": ("wide.png", data.getvalue(), "image/png")})
        assert response.status_code == 422
