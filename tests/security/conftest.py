"""Reuse offline fixtures and export observed security outcomes only."""
import csv
import json
import re
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
OBSERVED = {}


def pytest_runtest_logreport(report):
    if '/test_security_scenarios.py::' not in report.nodeid:
        return
    match=re.search(r'::test_(A\d{2})_',report.nodeid)
    if not match:
        return
    code=match[1]
    if report.failed or report.skipped:
        OBSERVED[code]=dict(code=code,attack='Xem kế hoạch README',actual=f'pytest {report.when}: {report.outcome}; chưa kết luận phòng thủ',
                            assessment='INCONCLUSIVE',defense='Kiểm tra lỗi pytest trước khi phân loại')
    elif report.when=='call':
        for key,value in report.user_properties:
            if key=='security_result':
                OBSERVED[code]=json.loads(value)
        if code not in OBSERVED:
            OBSERVED[code]=dict(code=code,attack='Xem kế hoạch README',actual='Không có quan sát đủ để kết luận',
                                assessment='INCONCLUSIVE',defense='Kiểm tra evidence recorder')


def pytest_sessionfinish(session, exitstatus):
    # A08 remediation keeps all historical reports read-only; its reporter is separate.
    if session.config.getoption('--a08-report-dir', default=None):
        return
    if not OBSERVED:
        return
    out = Path(__file__).resolve().parents[2] / 'docs/security'
    out.mkdir(exist_ok=True)
    rows = [OBSERVED[k] for k in sorted(OBSERVED)]
    rows.append(dict(code='A09', attack='Ảnh đối kháng giới hạn', actual='Chưa có artifact/ảnh thật tại Codex; chỉ chuẩn bị script Kaggle',
                     assessment='NOT_RUN', defense='Adversarial evaluation, kiểm tra miền dữ liệu và cơ chế từ chối khi bất định'))
    for number in range(1,11):
        code=f'A{number:02d}'
        if not any(r['code']==code for r in rows):
            rows.append(dict(code=code,attack='Xem kế hoạch README',actual='Không được thực thi trong phiên này',assessment='NOT_RUN',defense='Chạy scenario trong môi trường cô lập'))
    rows.sort(key=lambda r:r['code'])
    (out/'results.json').write_text(json.dumps({'scope':'In-process TestClient + synthetic fixture only; no remote targets',
        'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'pytest_exit_code':int(exitstatus),
        'assessment_counts':{label:sum(r['assessment']==label for r in rows) for label in ('ATTACK_BLOCKED','WEAKNESS_FOUND','NOT_RUN','INCONCLUSIVE')}, 'scenarios':rows}, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    with (out/'results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=['# Security — kết quả thực nghiệm local', '', '| Mã | Loại tấn công | Kết quả thực tế | Đánh giá | Phòng thủ |', '| --- | --- | --- | --- | --- |']
    for r in rows:
        lines.append('| '+' | '.join(str(r[k]).replace('|','/').replace('\n',' ') for k in ('code','attack','actual','assessment','defense'))+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
