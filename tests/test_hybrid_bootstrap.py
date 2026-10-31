"""Private bootstrap tests use synthetic files, fake credentials and mocked HF reads."""

import io
import json
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from hybrid_backend.main import create_app
from tomatocare import hybrid_bootstrap as bootstrap
from tomatocare.hybrid_onnx import HybridOnnxPredictor, load_hybrid_predictor, sha256_file

REPO = "test-owner/private-model"
REVISION = "a" * 40
SECRET = "unit-test-secret-do-not-log"


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    model = source / "tomatocare_hybrid.onnx"
    model.write_bytes(b"TEST FIXTURE ONLY: no trained ONNX material")
    manifest = json.loads((Path(__file__).parents[1] / "docs/examples/hybrid_inference_manifest.json").read_text())
    manifest["model_sha256"] = sha256_file(model)
    (source / "inference_manifest.json").write_text(json.dumps(manifest))
    (source / "DINOv3-LICENSE.md").write_text("# DINOv3 License\nTest-only license fixture.")
    return source, manifest["model_sha256"], tmp_path / "cache"


def setup_env(monkeypatch, bundle):
    source, expected, cache = bundle
    for name, value in {"HYBRID_HF_REPO": REPO, "HYBRID_HF_REVISION": REVISION, "HF_TOKEN": SECRET,
                        "HYBRID_EXPECTED_ONNX_SHA256": expected, "HYBRID_MODEL_DIR": str(cache)}.items():
        monkeypatch.setenv(name, value)


def fake_hf(monkeypatch, bundle, failure=None):
    import huggingface_hub
    source, expected, cache = bundle
    calls = []
    def download(**kwargs):
        assert kwargs["repo_id"] == REPO and kwargs["revision"] == REVISION
        assert kwargs["token"] == SECRET and kwargs["repo_type"] == "model"
        assert kwargs["force_download"] is True
        calls.append(kwargs["filename"])
        if failure is not None:
            raise failure
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source / kwargs["filename"]).read_bytes())
        return str(destination)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls


def test_exact_three_same_revision_files_atomically_promoted(monkeypatch, bundle):
    calls = fake_hf(monkeypatch, bundle)
    source, expected, cache = bundle
    verified = []
    original_verify = bootstrap._verify_bundle
    original_rename = Path.rename
    def verify(*args):
        original_verify(*args)
        verified.append(True)
    def rename(path, target):
        if path.name == "candidate":
            assert verified
            assert {p.name for p in path.iterdir()} == set(bootstrap.FILES)
            assert not Path(target).exists()
        return original_rename(path, target)
    monkeypatch.setattr(bootstrap, "_verify_bundle", verify)
    monkeypatch.setattr(Path, "rename", rename)
    result = bootstrap.bootstrap_private_bundle(REPO, REVISION, SECRET, expected, cache)
    assert calls == list(bootstrap.FILES)
    assert result.name.startswith("verified-" + REVISION)
    assert {p.name for p in result.iterdir()} == set(bootstrap.FILES)
    assert not list(cache.glob(".staging-*"))
    assert SECRET not in "".join(p.read_text(errors="ignore") for p in result.iterdir())


@pytest.mark.parametrize("field,value,match", [
    ("revision", "main", "immutable"), ("revision", "abc123", "immutable"),
    ("revision", None, "immutable"), ("token", None, "HF_TOKEN"),
    ("token", "", "HF_TOKEN"), ("expected", None, "SHA-256"),
    ("expected", "123", "SHA-256"), ("repo", "https://host/owner/model", "owner/name"),
])
def test_bad_config_rejected_before_network(monkeypatch, bundle, field, value, match):
    source, expected, cache = bundle
    args = {"repo": REPO, "revision": REVISION, "token": SECRET, "expected": expected}
    args[field] = value
    monkeypatch.setattr(bootstrap, "_download", lambda *args: pytest.fail("Network must not run"))
    with pytest.raises(bootstrap.BootstrapError, match=match):
        bootstrap.bootstrap_private_bundle(args["repo"], args["revision"], args["token"], args["expected"], cache)
    assert not cache.exists()


@pytest.mark.parametrize("failure_kind", ["401", "403", "404", "timeout", "interrupted"])
def test_errors_never_echo_sdk_credentials(monkeypatch, bundle, caplog, failure_kind):
    source, expected, cache = bundle
    failure = TimeoutError(SECRET) if failure_kind == "timeout" else OSError("Authorization: Bearer " + SECRET)
    if failure_kind.isdigit():
        failure.response = SimpleNamespace(status_code=int(failure_kind))
    fake_hf(monkeypatch, bundle, failure)
    with pytest.raises(bootstrap.BootstrapError) as caught:
        bootstrap.bootstrap_private_bundle(REPO, REVISION, SECRET, expected, cache)
    assert SECRET not in str(caught.value)
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert SECRET not in caplog.text
    if failure_kind in ("401", "403"): assert "401/403" in str(caught.value)
    if failure_kind == "timeout": assert "timed out" in str(caught.value)
    assert not list(cache.iterdir())


@pytest.mark.parametrize("kind", ["model", "manifest_hash", "manifest_schema", "license", "license_missing"])
def test_incomplete_or_untrusted_artifact_never_activated(monkeypatch, bundle, kind):
    source, expected, cache = bundle
    if kind == "model": (source / bootstrap.FILES[0]).write_bytes(b"damaged")
    elif kind.startswith("manifest"):
        path = source / bootstrap.FILES[1]
        manifest = json.loads(path.read_text())
        if kind == "manifest_hash": manifest["model_sha256"] = "0" * 64
        else: manifest["class_names"].reverse()
        path.write_text(json.dumps(manifest))
    elif kind == "license": (source / bootstrap.FILES[2]).write_bytes(b"")
    else: (source / bootstrap.FILES[2]).unlink()
    fake_hf(monkeypatch, bundle)
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.bootstrap_private_bundle(REPO, REVISION, SECRET, expected, cache)
    assert not list(cache.iterdir())


def test_no_old_bundle_fallback_after_auth_failure(monkeypatch, bundle):
    source, expected, cache = bundle
    fake_hf(monkeypatch, bundle)
    old = bootstrap.bootstrap_private_bundle(REPO, REVISION, SECRET, expected, cache)
    denied = OSError(SECRET)
    denied.response = SimpleNamespace(status_code=403)
    fake_hf(monkeypatch, bundle, denied)
    with pytest.raises(bootstrap.BootstrapError, match="401/403"):
        bootstrap.bootstrap_private_bundle(REPO, REVISION, SECRET, expected, cache)
    assert old.is_dir() and len(list(cache.iterdir())) == 1


def wait_until(client, predicate):
    for _ in range(300):
        response = client.get("/health")
        if predicate(response): return response
        time.sleep(0.01)
    pytest.fail("Startup state not observed")


def test_remote_startup_pending_then_cached_session(monkeypatch, bundle):
    setup_env(monkeypatch, bundle)
    calls = fake_hf(monkeypatch, bundle)
    entered, release = threading.Event(), threading.Event()
    from tomatocare import hybrid_onnx
    original = hybrid_onnx.HybridOnnxPredictor
    sessions = []
    class FakeSession:
        def __init__(self, path, **kwargs):
            assert len(calls) == 3 and "verified-" in path
            sessions.append(path)
        def get_inputs(self):
            return [SimpleNamespace(name=n, type="tensor(float)", shape=[1,3,224,224]) for n in ("baseline_images", "dino_images")]
        def get_outputs(self): return [SimpleNamespace(name="logits", type="tensor(float)", shape=[1,4])]
        def run(self, *args): return [np.array([[0,3,1,-1]], dtype=np.float32)]
    monkeypatch.setattr(hybrid_onnx, "HybridOnnxPredictor", lambda directory, **kwargs: original(directory, session_factory=FakeSession, **kwargs))
    def factory():
        entered.set()
        assert release.wait(timeout=5)
        return load_hybrid_predictor()
    with TestClient(create_app(factory)) as client:
        try:
            assert entered.wait(timeout=3)
            assert client.get("/health").status_code == 503
        finally:
            release.set()
        health = wait_until(client, lambda r:r.status_code == 200)
        assert health.json()["hf_revision"] == REVISION
        assert health.json()["artifact_source"] == "private-hf"
        image = io.BytesIO(); Image.new("RGB", (31,37)).save(image, format="PNG")
        for _ in range(2):
            assert client.post("/predict", files={"file":("leaf.png",image.getvalue(),"image/png")}).status_code == 200
        assert len(calls) == 3 and len(sessions) == 1


def test_startup_403_health_safe(monkeypatch, bundle, caplog):
    setup_env(monkeypatch, bundle)
    denied = OSError("Authorization: " + SECRET)
    denied.response = SimpleNamespace(status_code=403)
    fake_hf(monkeypatch, bundle, denied)
    with TestClient(create_app()) as client:
        response = wait_until(client, lambda r:"401/403" in r.text)
        assert response.status_code == 503
        assert SECRET not in response.text
    assert SECRET not in caplog.text


def test_unknown_remote_startup_error_is_redacted(monkeypatch, caplog):
    monkeypatch.setenv("HYBRID_HF_REPO", REPO)
    def fail(): raise RuntimeError(SECRET)
    with TestClient(create_app(fail)) as client:
        response = wait_until(client, lambda r:"configuration" in r.text)
        assert response.status_code == 503 and SECRET not in response.text
    assert SECRET not in caplog.text


def test_local_mode_keeps_phase5_contract(monkeypatch, bundle):
    source, expected, cache = bundle
    monkeypatch.delenv("HYBRID_HF_REPO", raising=False)
    monkeypatch.delenv("HYBRID_HF_REVISION", raising=False)
    monkeypatch.setenv("HYBRID_MODEL_DIR", str(source))
    monkeypatch.setattr(bootstrap, "_download", lambda *a:pytest.fail("Local mode must not call HF"))
    assert bootstrap.resolve_hybrid_model_dir() == source
    monkeypatch.setenv("HYBRID_EXPECTED_ONNX_SHA256", expected)
    assert bootstrap.resolve_hybrid_model_dir() == source
    monkeypatch.setenv("HYBRID_EXPECTED_ONNX_SHA256", "0" * 64)
    with pytest.raises(bootstrap.BootstrapError, match="trusted expected"):
        bootstrap.resolve_hybrid_model_dir()


def test_session_rechecks_trusted_expected_digest(bundle):
    source, expected, cache = bundle
    def forbidden(*args, **kwargs): pytest.fail("No session before trusted-hash comparison")
    with pytest.raises(ValueError, match="trusted expected"):
        HybridOnnxPredictor(source, session_factory=forbidden, expected_sha256="0" * 64)


def test_wrapped_sdk_http_error_is_classified_without_echoing_chain(monkeypatch, bundle):
    source, expected, cache = bundle
    denied = OSError(SECRET)
    denied.response = SimpleNamespace(status_code=403)
    wrapper = ValueError("SDK wrapper " + SECRET)
    wrapper.__cause__ = denied
    fake_hf(monkeypatch, bundle, wrapper)
    with pytest.raises(bootstrap.BootstrapError, match="401/403") as caught:
        bootstrap.bootstrap_private_bundle(REPO, REVISION, SECRET, expected, cache)
    assert SECRET not in "".join(traceback.format_exception(caught.value))


def test_sdk_transport_logs_are_suppressed_and_filter_restored(monkeypatch, bundle, caplog):
    import logging
    import huggingface_hub
    source, expected, cache = bundle
    caplog.set_level(logging.WARNING)
    logger = logging.getLogger("huggingface_hub.private_test")
    def denied(**kwargs):
        logger.warning("Authorization Bearer %s; signed-url credential", SECRET)
        logging.getLogger("urllib3.connectionpool").warning("Private URL %s", SECRET)
        error = OSError(SECRET)
        error.response = SimpleNamespace(status_code=403)
        raise error
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", denied)
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.bootstrap_private_bundle(REPO, REVISION, SECRET, expected, cache)
    assert SECRET not in caplog.text
    logger.warning("safe post-download diagnostic")
    assert "safe post-download diagnostic" in caplog.text


def test_prediction_failure_never_echoes_exception_credentials(monkeypatch, caplog):
    monkeypatch.setenv("HYBRID_HF_REPO", REPO)
    class BrokenPredictor:
        model_version = "unit-test"
        def predict(self, image): raise RuntimeError(SECRET)
    with TestClient(create_app(BrokenPredictor)) as client:
        wait_until(client, lambda r:r.status_code == 200)
        image=io.BytesIO(); Image.new("RGB", (20,30)).save(image,format="PNG")
        response=client.post("/predict",files={"file":("leaf.png",image.getvalue(),"image/png")})
        assert response.status_code == 503 and SECRET not in response.text
    assert SECRET not in caplog.text
