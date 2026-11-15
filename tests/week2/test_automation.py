"""T01–T08: in-process TestClient + explicit synthetic engines, no private model."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import MAX_BYTES, create_app as baseline_app
from hybrid_backend.main import create_app as hybrid_app
from tomatocare.inference import Predictor
from tomatocare.hybrid_onnx import CLASS_NAMES, HybridOnnxPredictor
from test_api import FixtureEngine, photo
from test_hybrid_onnx import FakeSession, artifact, manifest  # Reuse original fixtures.


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import huggingface_hub
    from tomatocare import hybrid_bootstrap
    for key in ('HF_TOKEN', 'HF_TOKEN_FILE', 'HYBRID_HF_REPO', 'HYBRID_HF_REVISION'):
        monkeypatch.delenv(key, raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail('T01–T08 must not download a model')
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', forbidden)
    monkeypatch.setattr(hybrid_bootstrap, '_download', forbidden)


@pytest.fixture
def apps(artifact):
    return {
        'Baseline': baseline_app(lambda: Predictor(FixtureEngine())),
        'Hybrid': hybrid_app(lambda: HybridOnnxPredictor(artifact, session_factory=FakeSession)),
    }


def note(record_property, rows):
    record_property('actual', json.dumps(rows, ensure_ascii=False))


def test_T01_health_ready(apps, record_property):
    rows = []
    for name, app in apps.items():
        with TestClient(app) as client:
            r = client.get('/health')
            rows.append(f'{name}: HTTP {r.status_code}, status={r.json().get("status")}')
            note(record_property, rows)
            assert r.status_code == 200 and r.json()['status'] == 'ready'


def test_T02_valid_jpg_png(apps, record_property):
    rows = []
    expected = np.exp(np.array([0., 3., 1., -1.]) - 3)
    expected /= expected.sum()
    for name, app in apps.items():
        with TestClient(app) as client:
            for fmt, suffix, mime in [('JPEG', 'jpg', 'image/jpeg'), ('PNG', 'png', 'image/png')]:
                r = client.post('/predict', files={'file': ('leaf.' + suffix, photo(fmt), mime)})
                rows.append(f'{name}/{suffix}: HTTP {r.status_code}')
                note(record_property, rows)
                assert r.status_code == 200
                data = r.json()
                assert {'predicted_class', 'confidence', 'probabilities'} <= data.keys()
                assert list(data['probabilities']) == list(CLASS_NAMES)
                values = list(data['probabilities'].values())
                assert all(np.isfinite(p) and 0 <= p <= 1 for p in values)
                np.testing.assert_allclose(values, expected, atol=1e-6)
                assert sum(values) == pytest.approx(1)
                assert data['predicted_class'] == 'Early Blight'
                assert data['confidence'] == pytest.approx(max(values))


def test_T03_corrupt_empty(apps, record_property):
    rows = []
    for name, app in apps.items():
        with TestClient(app) as client:
            for label, payload in [('hỏng', b'not an image'), ('rỗng', b'')]:
                r = client.post('/predict', files={'file': ('leaf.png', payload, 'image/png')})
                rows.append(f'{name}/{label}: HTTP {r.status_code}')
                note(record_property, rows)
                assert r.status_code == 422 and 'detail' in r.json()
                assert 'predicted_class' not in r.json()


def test_T04_over_10_mib(apps, record_property):
    rows = []
    for name, app in apps.items():
        with TestClient(app) as client:
            r = client.post('/predict', files={'file': ('leaf.png', b'x' * (MAX_BYTES + 1), 'image/png')})
            rows.append(f'{name}: 10 MiB + 1 byte → HTTP {r.status_code}')
            note(record_property, rows)
            assert r.status_code == 413 and 'predicted_class' not in r.json()


def test_T05_cors_headers(monkeypatch, artifact, record_property):
    origin = 'https://approved-frontend.example'
    monkeypatch.setenv('CORS_ORIGINS', origin)
    monkeypatch.setenv('HYBRID_CORS_ORIGINS', origin)
    rows = []
    for name, app in [('Baseline', baseline_app(lambda: Predictor(FixtureEngine()))),
                      ('Hybrid', hybrid_app(lambda: HybridOnnxPredictor(artifact, session_factory=FakeSession)))]:
        with TestClient(app) as client:
            for incoming, status in [(origin, 200), ('https://other.example', 400)]:
                r = client.options('/predict', headers={'Origin': incoming, 'Access-Control-Request-Method': 'POST'})
                rows.append(f'{name}/CORS {"hợp lệ" if incoming == origin else "khác origin"}: HTTP {r.status_code}')
                note(record_property, rows)
                assert r.status_code == status
                assert r.headers.get('access-control-allow-origin') == (origin if incoming == origin else None)
            for r in (client.get('/health'), client.post('/predict', files={'file': ('leaf.png', photo(), 'image/png')}),
                      client.post('/predict', files={'file': ('leaf.png', b'', 'image/png')})):
                assert r.headers['cache-control'] == 'no-store'
                assert r.headers['x-content-type-options'] == 'nosniff'
    rows.append('no-store và nosniff có trên health, predict thành công và lỗi')
    note(record_property, rows)


def test_T06_model_unavailable(record_property):
    def fail():
        raise RuntimeError('TEST ONLY: model load failure')
    rows = []
    for name, factory in [('Baseline', baseline_app), ('Hybrid', hybrid_app)]:
        with TestClient(factory(fail)) as client:
            for r in (client.get('/health'), client.post('/predict', files={'file': ('leaf.png', photo(), 'image/png')})):
                rows.append(f'{name}: HTTP {r.status_code}')
                note(record_property, rows)
                assert r.status_code == 503 and 'predicted_class' not in r.json()


def test_T07_modified_onnx(artifact, record_property):
    (artifact / 'tomatocare_hybrid.onnx').write_bytes(b'ALTERED TEST FIXTURE ONLY')
    constructed = []
    def forbidden_session(*args, **kwargs):
        constructed.append(True)
        pytest.fail('Must reject bad hash before ONNX session construction')
    def load():
        return HybridOnnxPredictor(artifact, session_factory=forbidden_session)
    with pytest.raises(ValueError, match='SHA-256 mismatch'):
        load()
    with TestClient(hybrid_app(load)) as client:
        health = client.get('/health')
        r = client.post('/predict', files={'file': ('leaf.png', photo(), 'image/png')})
        note(record_property, [f'SHA-256 mismatch; session chưa tạo={not constructed}; health={health.status_code}; predict={r.status_code}'])
        assert not constructed
        assert health.status_code == r.status_code == 503
        assert 'predicted_class' not in r.json()


def test_T08_bad_logits(artifact, record_property):
    rows = []
    for label, logits in [('NaN', np.full((1, 4), np.nan)), ('Inf', np.full((1, 4), np.inf)),
                          ('sai shape', np.zeros((1, 3)))]:
        class BadEngine(FixtureEngine):
            def logits(self, values): return logits
        class BadSession(FakeSession):
            def run(self, *args): return [logits]
        for name, app in [('Baseline', baseline_app(lambda: Predictor(BadEngine()))),
                          ('Hybrid', hybrid_app(lambda: HybridOnnxPredictor(artifact, session_factory=BadSession)))]:
            with TestClient(app) as client:
                r = client.post('/predict', files={'file': ('leaf.png', photo(), 'image/png')})
                rows.append(f'{name}/{label}: HTTP {r.status_code}')
                note(record_property, rows)
                assert r.status_code == 503
                assert not {'predicted_class', 'confidence', 'probabilities'} & r.json().keys()
