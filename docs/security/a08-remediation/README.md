# Bước 11 — Khắc phục A08 và kiểm thử lại

Chỉ thay đổi source trên `feature/ai-security-testing`, chưa commit/push/deploy.
Không sửa main/nhánh Hybrid gốc, Dockerfile/Compose/CI/frontend, model, training,
preprocessing hoặc ngưỡng AI. Không dùng token thật, không gọi production,
không chạy lại A09 hoặc AI Test T09–T10 trên Kaggle.

## Nguyên nhân và thay đổi

- Baseline startup dùng `str(exc)` làm detail của health/predict và log exception.
- Hybrid local startup echo exception; remote trước đây tin nội dung BootstrapError.
- Baseline inference dùng logger.exception, ghi traceback có exception/path giả.
- Decoder cũng đưa str(exc) vào 422. Nay lỗi decoder trả thông báo cố định.

Client vẫn nhận `detail` là string, HTTP 503 khi model không sẵn sàng/suy luận
lỗi, HTTP 422 khi ảnh không hợp lệ. Không chèn exception, đường dẫn hoặc giá trị
credential. Các mã tham chiếu cố định:

- `TC_MODEL_INIT_FAILED`: không khởi tạo được model.
- `TC_PREDICTION_FAILED`: không hoàn tất suy luận.
- `TC_INVALID_IMAGE`: ảnh/định dạng/kích thước pixels không hợp lệ.

Đây là mã loại lỗi, không phải ID duy nhất cho mỗi request. Log ứng dụng chỉ
có câu cố định và mã; không exc_info/traceback/exception object. Exception từ
predictor được chặn ngay tại ranh giới gọi model, kể cả HTTPException chứa dữ
liệu riêng tư. Các HTTPException do validation của API vẫn giữ status tương
ứng. BootstrapError cũng không được tin để đưa trực tiếp ra HTTP response.
Cơ chế remote download/hash/cache không thay đổi.

Không thêm bộ lọc toàn cục làm mất log của thư viện khác; phạm vi chứng minh
là logger/HTTP qua các đường lỗi đã inject và regression mock transport sẵn
có. Chưa xác minh log nền tảng Render/Docker với credentials thật.

## Bằng chứng bất biến

- [before-snapshot](before-snapshot/provenance.json): copy chính xác ba báo cáo
  security cũ, kèm commit nguồn và SHA-256; giữ cả các đánh giá lịch sử khác.
- [before-run](before-run/results.json): chạy **chính hàm probe A08 cũ** trước
  sửa; 1 test đạt nghĩa là tái hiện WEAKNESS_FOUND, không phải phòng thủ đạt.
- [after-run](after-run/results.json): 13 tests AFTER đạt.
- [after-full-regression](after-full-regression/results.json): probe + kiểm tra
  subclasses, model HTTPException, decoder và khôi phục API sau lỗi; 13 tests
  riêng A08 đạt trong phiên chạy toàn bộ suite.
- [regression-final.json](regression-final.json): tổng suite **188 passed,
  0 failed, 3 skipped** từ JUnit XML thật; 3 cảnh báo Starlette/timm deprecation.
  Ba Docker integration tests là opt-in/cần image, chưa build/chạy Docker mới.
- [Bảng tiếng Việt](comparison.md), [CSV](comparison.csv), [JSON](comparison.json).

Regression đầu tiên sau sửa có **90 passed, 5 failed**. Năm assertion chờ chuỗi
lỗi cũ như exception download/path/config, trong khi HTTP 503 vẫn đúng. Danh
sách được giữ tại [regression-initial.json](regression-initial.json). Chỉ thay
các assertion đó sang mã lỗi công khai cố định và kiểm tra không có thông tin
cũ; không bỏ test, không đổi status để làm đẹp kết quả. Regression liên quan
sau cập nhật: **108 passed** (95 cũ + 13 A08). Full suite sau đó đạt như trên.

`scripts/check_hybrid_docker.py` chỉ cập nhật assertion detail sang mã an toàn,
không sửa Dockerfile hoặc build/deploy image. Image cũ chưa rebuild sẽ vẫn chạy
code cũ; không được suy diễn source đã sửa đồng nghĩa dịch vụ live đã khắc phục.

Các file báo cáo A01–A10 và T01–T10 đã lưu được kiểm tra giữ nguyên từng byte,
không đổi A08 lịch sử thành ATTACK_BLOCKED hoặc sửa các trạng thái khác. Kết
quả A09/T09/T10 người dùng đã lưu riêng trên Kaggle không được chạy/ghi lại.
Snapshot phản ánh báo cáo lịch sử trong Git, không tự nhập kết quả Kaggle mới.

## Chạy lại AFTER mà không ghi đè bằng chứng

Tại root source, dùng venv Python 3.12 đã chuẩn bị:

```bash
MPLCONFIGDIR=/tmp/tomatocare-a08-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/tomatocare-week2/bin/python -m pytest -q tests/a08 \
  --a08-phase after --a08-report-dir /tmp/a08-after-NEW
```

Thư mục report phải chưa tồn tại; recorder từ chối ghi đè. Truyền rõ
`--a08-report-dir` bật chế độ không ghi các báo cáo automation/security cũ;
A08 ghi báo cáo riêng. Không dùng `--a08-phase before` trên code đã sửa để
cố tạo lại BEFORE; bằng chứng BEFORE được giữ nguyên, không tái sinh.

Test dùng lại `test_A08_information_disclosure`, FixtureEngine, FakeSession và
ảnh giả từ bộ cũ. BEFORE/AFTER probe đều gửi 4 request có lỗi; HTTP đều 503.
Các test bổ sung kiểm tra startup local/remote với RuntimeError và
BootstrapError, inference RuntimeError/HTTPException và decoder lỗi. Không
canary/path giả trong response/log; API giải phóng lock và hoạt động lại sau
lỗi. JPG/PNG thành công vẫn đúng schema, nhãn và softmax chuẩn của engine
kiểm thử. Điều này không phải benchmark/đánh giá lại model Private thật.
