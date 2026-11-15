"""The original A08 probe, strengthened boundary checks, and success-contract checks."""
import asyncio
import json
import logging

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import create_app as baseline_app, decode_image
from hybrid_backend.main import create_app as hybrid_app
from tomatocare.hybrid_bootstrap import BootstrapError
from security.test_security_scenarios import apps, offline, test_A08_information_disclosure as original_a08
from test_api import FixtureEngine, photo
from test_hybrid_onnx import artifact, manifest

CANARY='TEST_ONLY_ERROR_CANARY'
PATH='/private-test-fixture/model.onnx'


def test_A08_same_fault_injection(apps,caplog,record_property,request,monkeypatch):
    observed=[]
    original_request=TestClient.request
    def capture(self,method,url,*args,**kwargs):
        response=original_request(self,method,url,*args,**kwargs)
        observed.append({'method':method,'path':str(url),'status':response.status_code})
        return response
    monkeypatch.setattr(TestClient,'request',capture)
    evidence=[]
    def record(key,value):
        record_property(key,value)
        if key=='security_result':evidence.append(json.loads(value))
    original_a08(apps,caplog,record)
    record_property('a08_http',json.dumps(observed))
    assert len(observed)==4 and all(r['status']==503 for r in observed)
    expected='WEAKNESS_FOUND' if request.config.getoption('--a08-phase')=='before' else 'ATTACK_BLOCKED'
    assert evidence[-1]['assessment']==expected


@pytest.mark.parametrize('backend',['baseline','hybrid_local','hybrid_remote'])
@pytest.mark.parametrize('kind',['runtime','bootstrap'])
def test_startup_safe_even_for_exception_subclasses(backend,kind,monkeypatch,caplog,record_property):
    if backend=='hybrid_remote':monkeypatch.setenv('HYBRID_HF_REPO','test-only/private')
    caplog.set_level(logging.DEBUG)
    def fail():raise (BootstrapError if kind=='bootstrap' else RuntimeError)(CANARY+' '+PATH)
    app=(baseline_app if backend=='baseline' else hybrid_app)(fail)
    with TestClient(app) as client:
        if backend=='hybrid_remote':
            async def settled():
                for _ in range(100):
                    if 'TC_MODEL_INIT_FAILED' in (app.state.error or ''):return
                    await asyncio.sleep(.01)
                pytest.fail('Mock initialization did not settle')
            client.portal.call(settled)
        responses=[client.get('/health'),client.post('/predict',files={'file':('leaf.png',photo(),'image/png')})]
        assert all(r.status_code==503 and 'TC_MODEL_INIT_FAILED' in r.text for r in responses)
        assert all(CANARY not in r.text and PATH not in r.text for r in responses)
    assert CANARY not in caplog.text and PATH not in caplog.text
    assert 'TC_MODEL_INIT_FAILED' in caplog.text
    assert not any(r.exc_info for r in caplog.records if r.name in ('backend.main','hybrid_backend.main'))
    record_property('a08_startup',json.dumps({'backend':backend,'exception_kind':kind,'status':503,'response_leak':False,'log_leak':False}))


@pytest.mark.parametrize('backend',['Baseline','Hybrid'])
@pytest.mark.parametrize('kind',['runtime','http'])
def test_inference_error_is_safe_and_service_recovers(apps,backend,kind,caplog,record_property):
    caplog.set_level(logging.DEBUG)
    with TestClient(apps[backend]) as client:
        original=client.app.state.predictor.predict
        def fail(image):
            if kind=='http':raise HTTPException(503,CANARY+' '+PATH)
            raise RuntimeError(CANARY+' '+PATH)
        client.app.state.predictor.predict=fail
        response=client.post('/predict',files={'file':('leaf.png',photo(),'image/png')})
        assert response.status_code==503 and 'TC_PREDICTION_FAILED' in response.text
        assert CANARY not in response.text and PATH not in response.text
        assert CANARY not in caplog.text and PATH not in caplog.text
        assert 'TC_PREDICTION_FAILED' in caplog.text
        assert not any(r.exc_info for r in caplog.records if r.name in ('backend.main','hybrid_backend.main'))
        client.app.state.predictor.predict=original
        healthy=client.post('/predict',files={'file':('leaf.png',photo(),'image/png')})
        assert healthy.status_code==200 and healthy.json()['predicted_class']=='Early Blight'
        assert client.get('/health').status_code==200
    record_property('a08_inference',json.dumps({'backend':backend,'kind':kind,'error_status':503,'recovery_status':200,'response_leak':False,'log_leak':False}))


def test_decode_error_does_not_echo_decoder_exception(monkeypatch,apps,caplog,record_property):
    from PIL import Image
    def fail(*args,**kwargs):raise OSError(CANARY+' '+PATH)
    monkeypatch.setattr(Image,'open',fail)
    for backend,app in apps.items():
        with TestClient(app) as client:
            response=client.post('/predict',files={'file':('leaf.png',b'fixture','image/png')})
            assert response.status_code==422 and 'TC_INVALID_IMAGE' in response.text
            assert CANARY not in response.text and PATH not in response.text
    assert CANARY not in caplog.text and PATH not in caplog.text
    record_property('a08_decode',json.dumps({'status':422,'response_leak':False,'log_leak':False}))


def test_normal_prediction_contract_unchanged(apps,record_property):
    logits=np.array([0.,3.,1.,-1.]);expected=np.exp(logits-logits.max());expected/=expected.sum()
    for backend,app in apps.items():
        with TestClient(app) as client:
            assert client.get('/health').status_code==200
            for fmt,suffix,mime in [('PNG','png','image/png'),('JPEG','jpg','image/jpeg')]:
                response=client.post('/predict',files={'file':('leaf.'+suffix,photo(fmt),mime)})
                assert response.status_code==200
                result=response.json()
                assert result['predicted_class']=='Early Blight'
                assert list(result['probabilities'])==list(FixtureEngine.class_names)
                np.testing.assert_allclose(list(result['probabilities'].values()),expected,atol=1e-6)
                assert result['confidence']==pytest.approx(max(result['probabilities'].values()))
                assert {'predicted_class','confidence','probabilities',
                        'model_identifier' if backend=='Baseline' else 'model_version'} <= result.keys()
    record_property('a08_normal',json.dumps({'jpg_png_status':200,'softmax_and_labels_unchanged':True}))
