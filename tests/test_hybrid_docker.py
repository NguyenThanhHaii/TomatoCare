"""Static packaging checks always run; real image audit is explicit opt-in."""

import json
import os
import shlex
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]


def instructions():
    source=(ROOT/'Dockerfile.hybrid').read_text().replace('\\\n',' ')
    return [(line.split(None,1)[0],line.split(None,1)[1]) for line in source.splitlines()
            if line.strip() and not line.lstrip().startswith('#')]


def test_python_slim_nonroot_and_one_worker():
    items=instructions()
    assert items[0]==('FROM','python:3.12-slim')
    assert ('USER','10001:10001') in items
    assert ('EXPOSE','8000') in items
    command=json.loads(next(value for key,value in items if key=='CMD'))
    assert command[:2]==['sh','-c']
    assert command[2].startswith('exec uvicorn hybrid_backend.main:app ')
    assert '--host 0.0.0.0' in command[2] and '--workers 1' in command[2]
    assert '${PORT:-8000}' in command[2]


def test_no_secret_build_args_or_baked_model_configuration():
    from scripts.check_hybrid_docker import RUNTIME_ONLY_ENV
    items=instructions()
    assert not any(key=='ARG' for key,_ in items)
    pairs=[pair for key,value in items if key=='ENV' for pair in shlex.split(value)]
    assert not ({pair.split('=',1)[0] for pair in pairs} & RUNTIME_ONLY_ENV)
    assert all('HF_TOKEN' not in value for key,value in items if key in {'RUN','CMD'})
    assert not any('hf_hub_download' in value or 'snapshot_download' in value for key,value in items if key=='RUN')


def test_exact_runtime_copy_allowlist_and_all_sources_exist():
    sources=[]
    for key,value in instructions():
        if key=='COPY':sources.extend(item for item in shlex.split(value)[:-1] if not item.startswith("--"))
    assert set(sources)=={
        'hybrid_backend/requirements.txt','hybrid_backend/__init__.py','hybrid_backend/main.py',
        'backend/__init__.py','backend/main.py','tomatocare/__init__.py','tomatocare/inference.py',
        'tomatocare/hybrid_onnx.py','tomatocare/hybrid_bootstrap.py','docs/third-party/DINOv3-LICENSE.md'}
    assert all((ROOT/path).is_file() for path in sources)
    assert not any(path.endswith(('.onnx','.pth','.zip','.ipynb')) for path in sources)


def test_light_dependencies_and_no_model_loaded_during_build():
    requirements=(ROOT/'hybrid_backend/requirements.txt').read_text().lower()
    assert 'torch>=' not in requirements and 'transformers>=' not in requirements
    source=(ROOT/'Dockerfile.hybrid').read_text()
    assert '--no-cache-dir --no-compile' in source
    assert "rglob('*.onnx')" in source  # Strip ORT's bundled demo graphs as well.
    assert 'HEALTHCHECK' in source and "'/health'" in source
    assert 'HF_HUB_DISABLE_XET=1' in source


def test_dockerignore_covers_sensitive_build_context():
    rules=set((ROOT/'.dockerignore').read_text().splitlines())
    for rule in ['.git','**/.env*','**/.venv*','**/node_modules','**/__pycache__',
                 '**/*.onnx','**/*.pth','**/*.safetensors','**/*.zip','**/*.ipynb',
                 '**/.cache','**/huggingface','**/.staging-*','**/verified-*',
                 'frontend','notebooks','tests','outputs','data','hybrid_artifacts']:
        assert rule in rules
    # Existing baseline Dockerfile COPY directories remain available.
    assert 'tomatocare' not in rules and 'backend' not in rules


@pytest.mark.skipif(os.getenv('RUN_HYBRID_DOCKER_TESTS')!='1',reason='Real Docker audit requires explicit opt-in and a prebuilt image')
def test_real_prebuilt_hybrid_image():
    from scripts.check_hybrid_docker import check_image
    result=check_image(os.getenv('HYBRID_TEST_IMAGE','tomatocare-hybrid:phase7a'))
    assert result['audit']['runtime_imports']=='passed'
    assert result['no_model_http_smoke']=={'health':503,'predict':503}


def test_optional_proxy_trust_does_not_disable_tls_or_bake_credentials():
    source=(ROOT/'Dockerfile.hybrid').read_text()
    assert '--mount=type=secret,id=pip_ca,required=false' in source
    assert '--cert /run/secrets/pip_ca' in source
    assert '--trusted-host' not in source and 'PIP_TRUSTED_HOST' not in source
