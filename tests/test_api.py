"""API contract tests use explicit test-only engines, never production mock mode."""

import io
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import MAX_BYTES, create_app, decode_image
from tomatocare.inference import EXPECTED_CLASSES, ModelUnavailable, Predictor


class FixtureEngine:
    class_names = EXPECTED_CLASSES
    identifier = "TEST-ONLY-NOT-TRAINED-WEIGHTS"
    config = {"color_mode": "RGB", "image_size": [224, 224], "mean": [0.485, 0.456, 0.406],
              "std": [0.229, 0.224, 0.225], "resize": "exact", "interpolation": "bilinear"}

    def logits(self, values):
        assert values.shape == (1, 3, 224, 224)
        return np.array([[0.0, 3.0, 1.0, -1.0]], dtype=np.float32)


def photo(format="PNG", mode="RGB"):
    buffer = io.BytesIO()
    Image.new(mode, (32, 48)).save(buffer, format=format)
    return buffer.getvalue()


def test_predict_contract_and_cached_model():
    loads = []
    def factory():
        loads.append(1)
        return Predictor(FixtureEngine())
    with TestClient(create_app(factory)) as client:
        assert client.get("/health").json()["status"] == "ready"
        for _ in range(2):
            response = client.post("/predict", files={"file": ("leaf.png", photo(mode="RGBA"), "image/png")})
            assert response.status_code == 200
            data = response.json()
            assert data["predicted_class"] == "Early Blight"
            assert list(data["probabilities"]) == list(EXPECTED_CLASSES)
            assert sum(data["probabilities"].values()) == pytest.approx(1.0)
            assert data["confidence"] == max(data["probabilities"].values())
            assert response.headers["cache-control"] == "no-store"
        assert len(loads) == 1


@pytest.mark.parametrize("name,payload,mime,status", [
    ("leaf.png", b"broken", "image/png", 422),
    ("leaf.png", b"", "image/png", 422),
    ("leaf.gif", photo(), "image/gif", 415),
    ("leaf.exe", photo(), "image/png", 415),
    ("leaf.png", b"x" * (MAX_BYTES + 1), "image/png", 413),
    ("leaf.png", b"x" * (MAX_BYTES + 100_000), "image/png", 413),
])
def test_upload_validation(name, payload, mime, status):
    with TestClient(create_app(lambda: Predictor(FixtureEngine()))) as client:
        response = client.post("/predict", files={"file": (name, payload, mime)})
        assert response.status_code == status, response.text


def test_exif_orientation_is_applied():
    original = Image.new("RGB", (20, 30))
    exif = original.getexif()
    exif[274] = 6
    output = io.BytesIO()
    original.save(output, format="JPEG", exif=exif)
    result = decode_image(output.getvalue())
    assert result.size == (30, 20)
    assert result.mode == "RGB"


def test_unavailable_model_fails_without_fallback():
    def fail():
        raise ModelUnavailable("TEST checkpoint download blocked")
    with TestClient(create_app(fail)) as client:
        assert client.get("/health").status_code == 503
        response = client.post("/predict", files={"file": ("leaf.png", photo(), "image/png")})
        assert response.status_code == 503
        assert "checkpoint download blocked" in response.json()["detail"]


def test_cors_explicit_origin(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://approved-frontend.example")
    with TestClient(create_app(lambda: Predictor(FixtureEngine()))) as client:
        for origin, allowed in [("https://approved-frontend.example", True), ("https://other.example", False)]:
            response = client.options("/predict", headers={"Origin": origin, "Access-Control-Request-Method": "POST"})
            assert (response.headers.get("access-control-allow-origin") == origin) is allowed


def test_inference_import_does_not_import_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; import tomatocare.inference; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_streamed_oversized_body_is_rejected():
    def chunks():
        for _ in range(12):
            yield b"x" * (1024 * 1024)
    with TestClient(create_app(lambda: Predictor(FixtureEngine()))) as client:
        response = client.post("/predict", content=chunks(), headers={"Content-Type": "multipart/form-data; boundary=test"})
        assert response.status_code == 413


def test_concurrent_prediction_returns_busy():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    entered, release = threading.Event(), threading.Event()
    class SlowEngine(FixtureEngine):
        def logits(self, values):
            entered.set()
            assert release.wait(timeout=10)
            return super().logits(values)
    with TestClient(create_app(lambda: Predictor(SlowEngine()))) as client, ThreadPoolExecutor() as pool:
        first = pool.submit(client.post, "/predict", files={"file": ("leaf.png", photo(), "image/png")})
        try:
            assert entered.wait(timeout=10)
            response = client.post("/predict", files={"file": ("leaf.png", photo(), "image/png")})
            assert response.status_code == 429
        finally:
            release.set()
        assert first.result(timeout=10).status_code == 200
