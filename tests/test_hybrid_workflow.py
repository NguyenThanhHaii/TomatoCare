"""Offline checks for the publication boundary; real Actions scheduling is external."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]
WORKFLOW = ROOT / '.github/workflows/hybrid-ci-cd.yml'


def workflow():
    # YAML 1.1's SafeLoader treats `on` as True; BaseLoader preserves Actions keys.
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


def test_triggers_and_least_privilege():
    config = workflow()
    assert set(config['on']) == {'push', 'pull_request'}
    assert config['on']['push']['branches'] == ['feature/hybrid-dinov3']
    assert config['on']['pull_request']['branches'] == ['main', 'feature/hybrid-dinov3']
    assert config['permissions'] == {'contents': 'read'}
    for node in walk(config):
        assert 'continue-on-error' not in node
        assert 'permissions' not in node or node['permissions'] == {'contents': 'read'}
        if 'run' in node:
            assert '|| true' not in node['run'] and 'set +e' not in node['run']


def test_actions_are_pinned_and_checkout_does_not_persist_credentials():
    allowed = {'actions/checkout', 'actions/setup-python', 'actions/upload-artifact',
               'actions/download-artifact', 'docker/setup-buildx-action',
               'docker/build-push-action', 'docker/login-action'}
    for node in walk(workflow()):
        if 'uses' in node:
            repo, sha = node['uses'].split('@')
            assert repo in allowed and re.fullmatch('[0-9a-f]{40}', sha)
            if repo == 'actions/checkout':
                assert node['with']['persist-credentials'] == 'false'


def test_ci_builds_then_runs_every_test_and_audit_before_export():
    ci = workflow()['jobs']['ci']
    assert ci['runs-on'] == 'ubuntu-24.04'
    steps = ci['steps']
    build = next(i for i, step in enumerate(steps) if step.get('id') == 'build')
    audit = next(i for i, step in enumerate(steps) if step.get('id') == 'audit')
    tests = next(i for i, step in enumerate(steps) if step.get('id') == 'tests')
    export = next(i for i, step in enumerate(steps) if 'docker save' in step.get('run', ''))
    upload = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('actions/upload-artifact@'))
    assert build < audit < tests < export < upload
    assert steps[build]['with']['push'] == 'false' and steps[build]['with']['load'] == 'true'
    assert steps[build]['with']['file'] == 'Dockerfile.hybrid'
    assert not any(key in steps[build]['with'] for key in ['build-args', 'secrets', 'secret-envs', 'secret-files'])
    assert steps[tests]['run'] == 'python -m pytest -q'
    assert steps[tests]['env']['RUN_HYBRID_DOCKER_TESTS'] == '1'
    assert 'scripts.check_hybrid_docker' in steps[audit]['run']
    assert any('config --quiet' in step.get('run', '') for step in steps)
    assert any('"$LOCAL_IMAGE" -m pip check' in step.get('run', '') for step in steps)
    assert steps[upload]['with']['if-no-files-found'] == 'error'
    assert steps[upload]['with']['retention-days'] == '1'
    for index in (export, upload):
        assert "github.event_name == 'push'" in steps[index]['if']
        assert "github.repository == 'NguyenThanhHaii/TomatoCare'" in steps[index]['if']


@pytest.mark.parametrize('event,repository,ref,result,expected', [
    ('push', 'NguyenThanhHaii/TomatoCare', 'refs/heads/feature/hybrid-dinov3', 'success', True),
    ('pull_request', 'NguyenThanhHaii/TomatoCare', 'refs/heads/feature/hybrid-dinov3', 'success', False),
    ('push', 'fork/TomatoCare', 'refs/heads/feature/hybrid-dinov3', 'success', False),
    ('push', 'NguyenThanhHaii/TomatoCare', 'refs/heads/main', 'success', False),
    ('push', 'NguyenThanhHaii/TomatoCare', 'refs/heads/feature/hybrid-dinov3', 'failure', False),
    ('push', 'NguyenThanhHaii/TomatoCare', 'refs/heads/feature/hybrid-dinov3', 'cancelled', False),
    ('workflow_dispatch', 'NguyenThanhHaii/TomatoCare', 'refs/heads/feature/hybrid-dinov3', 'success', False),
])
def test_publish_gate(event, repository, ref, result, expected):
    publish = workflow()['jobs']['publish']
    assert publish['needs'] == 'ci'
    # Evaluate only the deliberately small allowlisted equality/conjunction gate.
    values = {'needs.ci.result': result, 'github.event_name': event,
              'github.repository': repository, 'github.ref': ref}
    clauses = publish['if'].split(' && ')
    assert len(clauses) == 4
    matches = []
    for clause in clauses:
        match = re.fullmatch(r"([a-z_.]+) == '([^']+)'", clause)
        assert match and match[1] in values
        matches.append(values[match[1]] == match[2])
    assert all(matches) is expected


def test_secrets_gating_and_exact_image_publication():
    config = workflow()
    assert config['env']['IMAGE'] == 'nguyenthanhhaiz/tomatocare-hybrid'
    assert 'secrets.' not in str(config['jobs']['ci'])
    assert 'HF_TOKEN' not in WORKFLOW.read_text()
    steps = config['jobs']['publish']['steps']
    guard = steps[0]
    assert guard['id'] == 'credentials'
    assert guard['env']['HAS_DOCKERHUB_TOKEN'] == "${{ secrets.DOCKERHUB_TOKEN != '' }}"
    assert 'available=false' in guard['run'] and 'no tags pushed' in guard['run']
    for step in steps[1:]:
        assert step['if'] == "steps.credentials.outputs.available == 'true'"
    login_index = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('docker/login-action@'))
    audit_index = next(i for i, step in enumerate(steps) if 'scripts.check_hybrid_docker' in step.get('run', ''))
    assert audit_index < login_index
    login = steps[login_index]
    assert login['with']['username'] == '${{ secrets.DOCKERHUB_USERNAME }}'
    assert login['with']['password'] == '${{ secrets.DOCKERHUB_TOKEN }}'
    assert login['with']['logout'] == 'true'
    download = next(step for step in steps if step.get('uses', '').startswith('actions/download-artifact@'))
    assert set(download['with']) == {'name', 'path'}  # Same run, no cross-run/repo artifacts.
    push = steps[-1]['run']
    assert 'docker push "$IMAGE:sha-$GITHUB_SHA"' in push
    assert 'docker push "$IMAGE:phase7c"' in push and ':latest' not in push
    assert 'docker build' not in push.replace('docker buildx imagetools inspect', '')
