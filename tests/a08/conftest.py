"""Separate, append-only A08 remediation evidence; no historical report overwrite."""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
REPORTS = []


def pytest_addoption(parser):
    parser.addoption('--a08-phase', choices=('before','after'), default='after')
    parser.addoption('--a08-report-dir', default=None)


def pytest_configure(config):
    supplied=config.getoption('--a08-report-dir')
    if supplied and Path(supplied).exists():
        raise pytest.UsageError('Choose a new A08 report directory; existing evidence is immutable')


def pytest_runtest_logreport(report):
    if '/test_a08_remediation.py::' in report.nodeid:
        REPORTS.append(report)


def pytest_sessionfinish(session,exitstatus):
    if not REPORTS:
        return
    phase=session.config.getoption('--a08-phase')
    stamp=datetime.now(timezone.utc)
    supplied=session.config.getoption('--a08-report-dir')
    out=Path(supplied) if supplied else Path('docs/security/a08-remediation')/f'{phase}-{stamp.strftime("%Y%m%dT%H%M%S%fZ")}'
    out.mkdir(parents=True,exist_ok=False)
    tests={}
    primary=None
    for r in REPORTS:
        row=tests.setdefault(r.nodeid,{'test':r.nodeid,'outcome':'NOT_RUN','observations':[]})
        if r.failed:row['outcome']='FAIL'
        elif r.skipped and row['outcome']!='FAIL':row['outcome']='SKIP'
        elif r.when=='call' and row['outcome']!='FAIL':row['outcome']='PASS'
        if r.when=='call':
            for key,value in r.user_properties:
                if key=='security_result':primary=json.loads(value)
                elif key.startswith('a08_'):row['observations'].append({key:json.loads(value)})
    counts={k:sum(r['outcome']==k for r in tests.values()) for k in ('PASS','FAIL','SKIP')}
    assessment=primary['assessment'] if primary and counts['FAIL']==0 else 'INCONCLUSIVE'
    result={'phase':phase,'generated_at_utc':stamp.isoformat(),'pytest_exit_code':int(exitstatus),
            'scope':'Offline TestClient with synthetic canary/path; no real token or model',
            'assessment':assessment,'test_counts':counts,'primary_a08':primary,'tests':list(tests.values())}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'tests.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['test','outcome'],lineterminator='\n');writer.writeheader()
        writer.writerows({k:r[k] for k in writer.fieldnames} for r in tests.values())
