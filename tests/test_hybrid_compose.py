"""Compose config uses synthetic credentials only; real containers are opt-in."""

import json
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
FAKE_TOKEN = 'compose-test-fixture-never-real-credentials'


def compose_env():
    env = dict(os.environ)
    env.update(HYBRID_HF_REVISION='a' * 40, HF_TOKEN=FAKE_TOKEN,
               HYBRID_HF_REPO='unit-test/private', HYBRID_MODEL_DIR='/tmp/hybrid-test',
               HYBRID_EXPECTED_ONNX_SHA256='b' * 64)
    return env


def compose(*args, env=None):
    return subprocess.run(['docker', 'compose', '-f', str(ROOT / 'compose.yaml'), *args],
                          cwd=ROOT, env=env or compose_env(), capture_output=True, text=True, timeout=90)


@pytest.fixture
def compose_available():
    if not shutil.which('docker'):
        pytest.skip('Docker Compose CLI not installed')
    result = subprocess.run(['docker', 'compose', 'version'], capture_output=True)
    if result.returncode:
        pytest.skip('Docker Compose plugin not installed')


def test_compose_resolved_config_and_secret_not_environment(compose_available):
    result = compose('config', '--format', 'json')
    assert result.returncode == 0, 'Compose config must validate'
    assert FAKE_TOKEN not in result.stdout + result.stderr
    config = json.loads(result.stdout)
    assert 'version' not in config and set(config['services']) == {'hybrid'}
    service = config['services']['hybrid']
    assert service['build']['dockerfile'] == 'Dockerfile.hybrid'
    assert int(service['mem_limit']) == 512 * 1024 * 1024 and float(service['cpus']) == 1
    assert service['ports'][0]['host_ip'] == '127.0.0.1'
    assert service['ports'][0]['target'] == 8000 and str(service['ports'][0]['published']) == '8001'
    assert service['environment']['HF_TOKEN_FILE'] == '/run/secrets/hf_token'
    assert 'HF_TOKEN' not in service['environment']
    assert config['secrets']['hf_token']['environment'] == 'HF_TOKEN'
    assert '/health' in ' '.join(service['healthcheck']['test'])
    assert not config.get('volumes') and not service.get('volumes')
    assert not service['build'].get('args')


def test_compose_requires_revision(compose_available):
    env = compose_env()
    env.pop('HYBRID_HF_REVISION')
    result = compose('config', '--quiet', env=env)
    assert result.returncode != 0 and 'HYBRID_HF_REVISION' in result.stderr
    assert FAKE_TOKEN not in result.stdout + result.stderr


def test_complete_hash_locked_requirements():
    source = (ROOT / 'hybrid_backend/requirements.lock.txt').read_text()
    records = [line.split('#', 1)[0].strip() for line in source.replace('\\\n', ' ').splitlines()]
    records = [line for line in records if line]
    names = set()
    for record in records:
        assert re.fullmatch(r'[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+(?:\s+--hash=sha256:[0-9a-f]{64})+', record)
        names.add(record.split('==')[0].lower().replace('_', '-'))
    assert {'fastapi', 'uvicorn', 'python-multipart', 'pillow', 'numpy', 'onnxruntime', 'huggingface-hub'} <= names
    assert {'pydantic', 'pydantic-core', 'starlette', 'requests', 'urllib3', 'protobuf', 'typing-extensions'} <= names
    assert not {'torch', 'transformers'} & names
    assert '--index-url' not in source and 'https://' not in source


@pytest.mark.skipif(os.getenv('RUN_HYBRID_DOCKER_TESTS') != '1', reason='Requires Docker daemon/prebuilt Phase 7B image')
def test_real_compose_secret_and_no_model_smoke(compose_available, tmp_path):
    from scripts.check_hybrid_docker import HTTP_PROBE
    override = tmp_path / 'no-model.yaml'
    # Keep production Compose intact; override only this fixture's remote loading.
    override.write_text('services:\n  hybrid:\n    environment:\n      HYBRID_HF_REPO: ""\n      HYBRID_HF_REVISION: ""\n      HYBRID_EXPECTED_ONNX_SHA256: ""\n      HYBRID_MODEL_DIR: ""\n')
    project = 'tomatocare-test-' + uuid.uuid4().hex[:8]
    prefix = ['-p', project, '-f', str(override)]
    try:
        result = compose(*prefix, 'up', '-d', '--no-build')
        assert result.returncode == 0, 'Compose no-model smoke must start'
        container = compose(*prefix, 'ps', '-q', 'hybrid').stdout.strip()
        inspected = subprocess.run(['docker', 'inspect', '--format',
                                    '{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}', container],
                                   capture_output=True, text=True, check=True)
        assert inspected.stdout.strip() == '536870912 1000000000'
        audit = compose(*prefix, 'exec', '-T', 'hybrid', 'python', '-c',
                        "import os; from pathlib import Path; from tomatocare.hybrid_bootstrap import _runtime_token; "
                        "assert 'HF_TOKEN' not in os.environ; "
                        "assert _runtime_token() == 'compose-test-fixture-never-real-credentials'; "
                        "assert Path('/run/secrets/hf_token').is_file(); print('secret mount passed')")
        assert audit.returncode == 0 and 'secret mount passed' in audit.stdout
        import time
        for _ in range(30):
            probe = compose(*prefix, 'exec', '-T', 'hybrid', 'python', '-c', HTTP_PROBE)
            if probe.returncode == 0:
                break
            time.sleep(0.2)
        assert probe.returncode == 0, 'No-model endpoints must report 503'
        assert json.loads(probe.stdout) == {'health': 503, 'predict': 503}
        logs = compose(*prefix, 'logs', '--no-color')
        assert FAKE_TOKEN not in logs.stdout + logs.stderr
    finally:
        compose(*prefix, 'down', '--remove-orphans')


@pytest.mark.skipif(os.getenv('RUN_HYBRID_DOCKER_TESTS') != '1', reason='Requires daemon/image')
def test_compose_missing_secret_rejected_before_start(compose_available):
    env = compose_env()
    env.pop('HF_TOKEN')
    project = 'tomatocare-missing-secret-' + uuid.uuid4().hex[:8]
    try:
        result = compose('-p', project, 'up', '-d', '--no-build', env=env)
        assert result.returncode != 0 and 'HF_TOKEN' in result.stderr
        assert FAKE_TOKEN not in result.stdout + result.stderr
    finally:
        compose('-p', project, 'down', '--remove-orphans', env=env)
