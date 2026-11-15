"""Bounded local probes only. Mock engines cannot establish real-model robustness."""
import json
import logging
import struct
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from backend.main import MAX_BYTES, create_app as baseline_app
from hybrid_backend.main import create_app as hybrid_app
from tomatocare.inference import Predictor
from tomatocare.hybrid_onnx import HybridOnnxPredictor, sha256_file
from test_api import FixtureEngine, photo
from test_hybrid_onnx import FakeSession, artifact, manifest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from tomatocare import hybrid_bootstrap
    import huggingface_hub
    for key in ('HF_TOKEN','HF_TOKEN_FILE','HYBRID_HF_REPO','HYBRID_HF_REVISION'):
        monkeypatch.delenv(key,raising=False)
    def forbidden(*args,**kwargs):pytest.fail('Remote model download forbidden in security tests')
    monkeypatch.setattr(hybrid_bootstrap,'_download',forbidden)
    monkeypatch.setattr(huggingface_hub,'hf_hub_download',forbidden)


@pytest.fixture
def apps(artifact):
    return {'Baseline':baseline_app(lambda:Predictor(FixtureEngine())),
            'Hybrid':hybrid_app(lambda:HybridOnnxPredictor(artifact,session_factory=FakeSession))}


def result(record_property, code, attack, actual, assessment, defense):
    record_property('security_result', json.dumps(dict(code=code,attack=attack,actual=actual,
                                                     assessment=assessment,defense=defense),ensure_ascii=False))


def upload(client, payload=None, filename='leaf.png', mime='image/png', **kwargs):
    return client.post('/predict',files={'file':(filename,photo() if payload is None else payload,mime)},**kwargs)


def test_A01_oversize(apps,record_property):
    observations=[]
    for name,app in apps.items():
        with TestClient(app) as client:
            r=upload(client,b'x'*(MAX_BYTES+1));observations.append(f'{name}: {r.status_code}')
            assert r.status_code==413
    result(record_property,'A01','File quá lớn','; '.join(observations)+'; 1 request/API, 10 MiB + 1 byte',
           'ATTACK_BLOCKED','Giữ byte-limit trước multipart và sau upload; thêm giới hạn body tại reverse proxy')


def test_A02_bomb_header_no_allocation(apps,monkeypatch,record_property):
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    # 100M declared pixels, no pixel buffer/large image generation or decompression.
    payload=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',10000,10000,8,2,0,0,0))+chunk(b'IDAT',b'')+chunk(b'IEND',b'')
    assert len(payload)<128
    def forbidden_load(*args,**kwargs):pytest.fail('Must reject declared bomb dimensions before pixel load')
    monkeypatch.setattr(Image.Image,'load',forbidden_load)
    monkeypatch.setattr(PngImagePlugin.PngImageFile,'load',forbidden_load)
    observations=[]
    for name,app in apps.items():
        with TestClient(app) as client:
            r=upload(client,payload);observations.append(f'{name}: {r.status_code}')
            assert r.status_code==422
            assert 'pixels' in r.text.lower() or 'decompression' in r.text.lower()
    result(record_property,'A02','Decompression bomb','; '.join(observations)+'; header <128 bytes, 100M pixels khai báo; không load pixels',
           'ATTACK_BLOCKED','Giữ pixel-limit/Pillow bomb checks; timeout và memory isolation khi decode')


def test_A03_mime_spoof(apps,record_property):
    observations=[]
    cases=[('leaf.exe','image/png',photo(),415),('leaf.png','image/gif',photo(),415),
           ('leaf.png','image/png',photo('BMP'),422)]
    for name,app in apps.items():
        with TestClient(app) as client:
            for filename,mime,payload,expected in cases:
                r=upload(client,payload,filename,mime);observations.append(f'{name}/{filename}/{mime}: {r.status_code}')
                assert r.status_code==expected
    result(record_property,'A03','Giả định dạng/MIME','; '.join(observations),'ATTACK_BLOCKED',
           'Kiểm tra extension + MIME + định dạng decoder; không tin tên file hoặc Content-Type riêng lẻ')


def test_A04_malformed(apps,record_property):
    observations=[]
    for name,app in apps.items():
        with TestClient(app) as client:
            for payload in (b'',b'invalid image',photo()[:30]):
                r=upload(client,payload);observations.append(f'{name}: {r.status_code}')
                assert r.status_code==422
            assert client.get('/health').status_code==200
    result(record_property,'A04','Ảnh hỏng/cắt/dữ liệu bất thường','; '.join(observations)+'; health vẫn 200',
           'ATTACK_BLOCKED','Giữ decode/load bắt buộc, exception boundary, cập nhật Pillow định kỳ')


def test_A05_origin_policy(artifact,monkeypatch,record_property):
    origin='https://approved.example';other='https://unapproved.example'
    monkeypatch.setenv('CORS_ORIGINS',origin);monkeypatch.setenv('HYBRID_CORS_ORIGINS',origin)
    observations=[]
    for name,app in [('Baseline',baseline_app(lambda:Predictor(FixtureEngine()))),
                     ('Hybrid',hybrid_app(lambda:HybridOnnxPredictor(artifact,session_factory=FakeSession)))]:
        with TestClient(app) as client:
            r=client.options('/predict',headers={'Origin':other,'Access-Control-Request-Method':'POST'})
            assert r.status_code==400 and 'access-control-allow-origin' not in r.headers
            direct=upload(client,headers={'Origin':other})
            assert direct.status_code==200 and 'access-control-allow-origin' not in direct.headers
            observations.append(f'{name}: preflight 400, direct POST 200 nhưng không có Allow-Origin')
    result(record_property,'A05','Origin không được phép','; '.join(observations),'ATTACK_BLOCKED',
           'CORS chặn quyền đọc của browser, KHÔNG là xác thực; cần auth/token riêng cho API nếu cần hạn chế người gọi')


def test_A06_small_concurrency(artifact,record_property):
    observations=[]
    for name,base in [('Baseline',lambda:Predictor(FixtureEngine())),
                      ('Hybrid',lambda:HybridOnnxPredictor(artifact,session_factory=FakeSession))]:
        entered,release=threading.Event(),threading.Event()
        predictor=base();original=predictor.predict
        def slow(image):
            entered.set();assert release.wait(timeout=3);return original(image)
        predictor.predict=slow
        app=(baseline_app if name=='Baseline' else hybrid_app)(lambda:predictor)
        with TestClient(app) as client,ThreadPoolExecutor(max_workers=2) as pool:
            first=pool.submit(upload,client)
            try:
                assert entered.wait(timeout=2)
                busy=[upload(client).status_code for _ in range(2)]
                assert busy==[429,429]
            finally:release.set()
            assert first.result(timeout=3).status_code==200
            observations.append(f'{name}: 1 in-flight thành công, 2 request khi bận trả 429')
    result(record_property,'A06','Quá tải đồng thời mức nhỏ','; '.join(observations),'ATTACK_BLOCKED',
           'Admission lock chặn đồng thời; KHÔNG chứng minh rate limiting; thêm quota/rate limit và proxy timeout')


def test_A07_repeated_queries(apps,record_property):
    observations=[]
    for name,app in apps.items():
        with TestClient(app) as client:
            replies=[upload(client) for _ in range(5)]
            assert all(r.status_code==200 for r in replies)
            assert all('probabilities' in r.json() for r in replies)
            observations.append(f'{name}: 5/5 truy vấn tuần tự nhận 200 và đủ 4 xác suất')
    result(record_property,'A07','Truy vấn lặp / nguy cơ sao chép hành vi','; '.join(observations)+
           '; không huấn luyện bản sao, không chứng minh extraction thành công hoặc vắng mọi rate limit',
           'INCONCLUSIVE','Nếu API mở: auth, quota theo tài khoản/IP, giám sát truy vấn; cân nhắc giảm chi tiết xác suất theo nhu cầu')


def test_A08_information_disclosure(apps,caplog,record_property):
    canary='SYNTHETIC_CANARY_ONLY';path='/private-fixture-only/model.onnx'
    observations=[];leaks=[]
    caplog.set_level(logging.DEBUG)
    for name,factory in [('Baseline',baseline_app),('Hybrid',hybrid_app)]:
        caplog.clear()
        def fail():raise RuntimeError(canary+' '+path)
        with TestClient(factory(fail)) as client:
            r=client.get('/health');assert r.status_code==503
            exposed=canary in r.text or path in r.text
            startup_log_leak=canary in caplog.text or path in caplog.text
            leaks.extend([exposed,startup_log_leak]);observations.append(f'{name}/startup: response leak={exposed}, log leak={startup_log_leak}')
        app=apps[name];caplog.clear()
        with TestClient(app) as client:
            client.app.state.predictor.predict=lambda image:fail()
            r=upload(client);assert r.status_code==503
            response_leak=canary in r.text or path in r.text
            log_leak=canary in caplog.text or path in caplog.text
            leaks.extend([response_leak,log_leak])
            observations.append(f'{name}/inference: response leak={response_leak}, log leak={log_leak}')
    result(record_property,'A08','Lộ thông tin khi lỗi','; '.join(observations)+'; chỉ canary giả, không secret thật',
           'WEAKNESS_FOUND' if any(leaks) else 'ATTACK_BLOCKED','Trả lỗi chung + error ID; redaction exception/log; không echo str(exc), giới hạn quyền đọc logs')


def test_A10_fixture_tamper(artifact,record_property):
    config=json.loads((artifact/'inference_manifest.json').read_text())
    trusted=config['model_sha256']
    before=(artifact/'tomatocare_hybrid.onnx').read_bytes()
    (artifact/'tampered-copy.onnx').write_bytes(before+b'changed')
    # Only isolated tmp fixture is modified. No real model file is loaded/touched.
    (artifact/'tomatocare_hybrid.onnx').write_bytes((artifact/'tampered-copy.onnx').read_bytes())
    calls=[]
    def forbidden(*args,**kwargs):calls.append(True);pytest.fail('Hash must fail before session')
    with pytest.raises(ValueError,match='SHA-256 mismatch'):
        HybridOnnxPredictor(artifact,session_factory=forbidden)
    config['model_sha256']=sha256_file(artifact/'tomatocare_hybrid.onnx')
    (artifact/'inference_manifest.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='trusted expected digest'):
        HybridOnnxPredictor(artifact,session_factory=forbidden,expected_sha256=trusted)
    assert not calls
    result(record_property,'A10','Thay đổi model trái phép (bản sao fixture)',
           'Model đổi: SHA-256 mismatch; model+manifest cùng đổi: trusted hash từ chối; session chưa tạo; chỉ tmp fixture giả bị sửa',
           'ATTACK_BLOCKED','Pin trusted expected SHA ngoài bundle, immutable revision, quyền file read-only; hash manifest tự thân không chống thay đồng thời manifest/model')
