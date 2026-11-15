# Tuần 2 — Bước 2: Automation/Mock T01–T08

Phạm vi: cả Baseline và Hybrid qua FastAPI TestClient trong process; riêng T07
kiểm tra loader Hybrid. Không mở kết nối đến API production, không dùng token
thật, không tải model Private và không chạy chiến dịch tấn công. ONNX fixture
là bytes giả được ghi rõ, không phải mô hình đã huấn luyện. FakeSession trả
logits cố định để kiểm tra **logic thực** preprocessing/softmax/API; không thay
mock vào production. Không có số liệu Accuracy/latency mô hình thật ở bước này.

## Tái sử dụng

`tests/week2/test_automation.py` dùng lại `FixtureEngine`, `photo` của
`tests/test_api.py`, `FakeSession`, fixtures `artifact`/`manifest` của
`tests/test_hybrid_onnx.py`. Không sửa tests cũ hoặc logic ứng dụng.
T01–T08 là tám test functions; mỗi function kiểm tra các biến thể mô tả trong
báo cáo. Lỗi bất kỳ assertion nào làm cả mã T tương ứng FAIL.

- T01: health ready 200 cho cả hai API.
- T02: JPG và PNG; schema, thứ tự bốn nhãn, giá trị hữu hạn [0,1], tổng bằng 1,
  nhãn/confidence và xác suất khớp softmax tính độc lập từ logits mock.
- T03: ảnh hỏng và file rỗng → 422, không có dự đoán.
- T04: 10 MiB + 1 byte → 413, không có dự đoán.
- T05: preflight origin cho phép/không cho phép; no-store/nosniff trên health,
  predict thành công và lỗi. Đây không phải xác minh authentication, HSTS/CSP
  hoặc cấu hình proxy triển khai; CORS không thay thế xác thực.
- T06: factory giả lập lỗi tải model; health và predict → 503, không fallback.
- T07: thay bytes ONNX fixture sau khi ghi manifest/hash; SHA-256 mismatch,
  session không được tạo, Hybrid health/predict → 503.
- T08: NaN, Inf và shape [1,3] thay cho [1,4]; cả hai API → 503, không nhãn/
  confidence/probabilities giả.

## Môi trường kiểm thử cô lập

Đã tạo `/tmp/tomatocare-week2`, không cài dependencies vào runtime production.
Để tái tạo trên Linux Python 3.12 tại root repository:

```bash
python -m venv /tmp/tomatocare-week2
/tmp/tomatocare-week2/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
/tmp/tomatocare-week2/bin/python -m pip install --require-hashes --only-binary=:all: -r hybrid_backend/requirements.lock.txt
python - <<'PY'
import re
from pathlib import Path
s = Path('hybrid_backend/requirements.lock.txt').read_text()
Path('/tmp/week2-constraints.txt').write_text('\n'.join(re.findall(r'^[A-Za-z0-9_.-]+==[^\s]+', s, re.M)))
PY
/tmp/tomatocare-week2/bin/python -m pip install -c /tmp/week2-constraints.txt -r requirements-ci.txt
/tmp/tomatocare-week2/bin/python -m pip check
MPLCONFIGDIR=/tmp/tomatocare-week2-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/tomatocare-week2/bin/python -m pytest -q tests/week2
# Regression: tất cả tests mặc định, lý do skip được in rõ
MPLCONFIGDIR=/tmp/tomatocare-week2-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /tmp/tomatocare-week2/bin/python -m pytest -q -rs
```

Giữ `RUN_HYBRID_DOCKER_TESTS` chưa bật cho lần kiểm tra mặc định này. Ba test
Docker integration opt-in cũ không được tính PASS nếu skipped; chúng cần chuẩn
bị image local phase7b và bật biến theo hướng dẫn Docker trước khi chạy riêng.
Không truyền credentials thật vào pytest. Fixtures mới xóa cấu hình remote/token
trong phạm vi test và chặn các hàm HF download đã biết. Cơ chế này không phải
sandbox mạng toàn hệ điều hành; chỉ TestClient/factory giả được sử dụng.

## Báo cáo tự động

- [Bảng tiếng Việt](automation-summary.md).
- [JSON](automation-results.json): timestamp UTC, exit code pytest, tổng kết phiên
  chạy và từng mã test.
- [CSV UTF-8 BOM](automation-results.csv): mở bằng Excel để đối chiếu báo cáo PDF.

`tests/conftest.py` lấy kết quả call/setup/teardown từ pytest. Không điền PASS
thủ công. Khi chạy có T01–T08, báo cáo được ghi lại; timestamp và suite_totals
thuộc phiên pytest gần nhất có các test này. Chạy suite con chỉ có test cũ sẽ
không thay báo cáo tuần 2. Nếu pytest không khởi động/collection lỗi trước khi
chạy bất kỳ T01–T08 nào, kiểm tra exit code/terminal và timestamp; không dùng
report cũ làm bằng chứng phiên lỗi. Xem báo cáo JSON để biết biến thể thực tế;
PASS mock không chứng minh mô hình chẩn đoán chính xác hoặc bảo mật toàn hệ thống.

## AI Test chưa chạy

- **T09 — NOT_RUN:** accuracy trên 40 ảnh Test thật, 10 ảnh/lớp, seed 42.
- **T10 — NOT_RUN:** độ ổn định trên chính 40 ảnh đó với độ sáng 0.8/1.2 và
  JPEG Q70. Đây là định nghĩa Bước 3 mới, thay cho đề xuất parity ở Bước 2.

Xem [hướng dẫn AI Test thật](ai-testing.md) và [protocol đã định trước](ai-preparation/protocol.json).

Workspace chưa có bundle/model thật hoặc dataset có nhãn. Cung cấp artifact
qua đường dẫn local/môi trường bí mật được cấp quyền khi thực hiện bước AI
Test; không gửi token qua chat, không đưa weights vào Git. Giữ giới hạn dataset:
trùng ảnh trong split có thể gây thiên lệch và chưa xác minh độc lập leaf_id.
Không phát hiện lỗi mới chỉ từ PASS các ca này; các kịch bản tấn công/phòng thủ
và thử nghiệm AI thật vẫn ngoài phạm vi bước hiện tại.
