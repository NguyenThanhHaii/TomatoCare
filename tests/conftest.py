"""Generate Week 2 reports from actual pytest outcomes, never assumed PASS."""
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

REPORTS = []
CASES = {
'T01': ('API sẵn sàng', 'Cả hai API: health 200, status ready'),
'T02': ('Ảnh JPG/PNG hợp lệ', '200; đúng schema, thứ tự 4 lớp và softmax logits mock'),
'T03': ('Ảnh hỏng/rỗng', '422, không có dự đoán'),
'T04': ('File quá 10 MiB', '413, không có dự đoán'),
'T05': ('CORS và headers', 'Origin cho phép được chấp nhận; origin khác bị từ chối; no-store/nosniff'),
'T06': ('Mô hình không tải được', 'Health/predict 503, không fallback dự đoán'),
'T07': ('ONNX giả lập bị thay đổi', 'SHA-256 mismatch trước session; Hybrid health/predict 503'),
'T08': ('Logits NaN/Inf/sai shape', 'Cả hai API: 503; không trả nhãn hoặc xác suất'),
'T09': ('Độ chính xác trên 40 ảnh Test thật', '10 ảnh/lớp, seed 42; số đúng/sai, accuracy và từng lớp'),
'T10': ('Độ ổn định trên cùng 40 ảnh Test', 'Độ sáng 0.8/1.2, JPEG Q70; tỷ lệ giữ nhãn và các lần đổi nhãn'),
}


def pytest_runtest_logreport(report):
    REPORTS.append(report)


def pytest_sessionfinish(session, exitstatus):
    # A08 remediation keeps all historical reports read-only; its reporter is separate.
    if session.config.getoption('--a08-report-dir', default=None):
        return
    selected = {}
    for r in REPORTS:
        match = re.search(r'::test_(T0[1-8])_', r.nodeid)
        if match:
            selected.setdefault(match[1], []).append(r)
    if not selected:
        return
    rows = []
    for code, (description, expected) in CASES.items():
        reports = selected.get(code, [])
        failed = [r for r in reports if r.failed]
        calls = [r for r in reports if r.when == 'call']
        status = 'FAIL' if failed else ('PASS' if calls and all(r.passed for r in reports) else 'NOT_RUN')
        actual = [value for r in reports for key, value in r.user_properties if key == 'actual']
        detail = '; '.join(json.loads(actual[-1])) if actual else 'Chưa thực hiện'
        if failed:
            detail += ' | ' + failed[0].longreprtext.splitlines()[-1]
        if code in ('T09', 'T10'):
            detail = 'Chưa có Hybrid ONNX và ảnh Test có nhãn thật trong workspace; không dùng mock để báo AI Test'
        rows.append(dict(code=code, description=description, expected=expected, actual=detail, status=status))
    out = Path(__file__).resolve().parents[1] / 'docs/week2'
    out.mkdir(exist_ok=True)
    failures = {r.nodeid for r in REPORTS if r.failed}
    totals = {'passed': sum(r.when == 'call' and r.passed and r.nodeid not in failures for r in REPORTS),
              'failed': len(failures), 'skipped': sum(r.skipped for r in REPORTS)}
    (out / 'automation-results.json').write_text(json.dumps({'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'pytest_exit_code': int(exitstatus), 'execution_scope': 'TestClient nội bộ, synthetic engines; không model Private hoặc API production', 'suite_totals': totals, 'cases': rows}, ensure_ascii=False, indent=2)+'\n')
    with (out / 'automation-results.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, lineterminator='\n', fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    lines = ['# Kết quả automation/mock tuần 2', '',
             'Nguồn: pytest thực tế; không dùng model Private hoặc API production.', '',
             '| Mã | Kiểm tra | Trạng thái | Kết quả thực tế |',
             '| --- | --- | --- | --- |']
    for row in rows:
        cells = [str(row[key]).replace('|', '/').replace('\n', ' ') for key in ('code', 'description', 'status', 'actual')]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', f'Tổng phiên pytest: {totals["passed"]} passed; {totals["failed"]} failed; {totals["skipped"]} skipped.',
              '', 'PASS mock chỉ xác nhận hành vi phần mềm, không xác nhận độ chính xác AI thật.']
    (out / 'automation-summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
