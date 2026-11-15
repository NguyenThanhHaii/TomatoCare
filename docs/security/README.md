# Week 2 — Phần II: Security & AI Attack Testing

Chỉ `feature/ai-security-testing`, TestClient nội bộ và fixture synthetic trong
pytest temporary directory. Không gọi Render, Vercel, HF hoặc API Internet;
không token thật, không stress kéo dài, không sửa production/Docker/CI/frontend.
T09–T10 trên Kaggle đã PASS **theo thông tin người dùng**; không chạy lại và
không ghi vào các báo cáo T09–T10 cũ. A09 là thí nghiệm riêng.

## Kế hoạch trước thực thi

| Mã | Mục tiêu | Điều kiện / mô phỏng có giới hạn | Kết quả phòng thủ mong đợi | Phòng thủ phù hợp |
| --- | --- | --- | --- | --- |
| A01 | Tránh tiêu tốn tài nguyên bằng upload lớn | Cả hai API, 1 file giả 10 MiB + 1 byte/API | HTTP 413, không suy luận | Giới hạn byte trong ASGI/multipart và reverse proxy |
| A02 | Chặn ảnh khai báo giải nén quá lớn | PNG header <128 bytes, 10000×10000 pixels, không tạo buffer pixel; chặn Image.load trong test | 422 do pixel/bomb limit trước khi load | Pixel limit, decoder timeout, isolation RAM |
| A03 | Vượt kiểm tra loại tệp | 3 ca/API: PNG tên .exe, PNG MIME GIF, BMP giả PNG | 415/415/422 | Whitelist extension/MIME và format decoder thực |
| A04 | Làm lỗi bộ xử lý ảnh | Rỗng, bytes bất thường, PNG cắt còn 30 bytes; 3 upload + health/API | 422, health vẫn 200 | Decode/load đầy đủ, exception handling, cập nhật Pillow |
| A05 | Kiểm tra quyền đọc chéo Origin | Origin không được phép: OPTIONS và POST hợp lệ/API | Preflight 400; không Allow-Origin; POST trực tiếp vẫn có thể xử lý | CORS cho browser; auth riêng nếu cần giới hạn người gọi |
| A06 | Vượt admission khi bận | 1 request đang giữ ở mock predictor, 2 request thử trong lúc bận/API; tối đa 2 HTTP request đồng thời | Request đầu 200, hai request khi bận 429 | Admission lock + timeout; rate/quota là cơ chế khác |
| A07 | Đánh giá nguy cơ thu thập đầu ra | 5 truy vấn tuần tự/API với ảnh giả; không train surrogate | Quan sát output/quota; 5 lần không đủ kết luận extraction hoặc hiệu lực mọi rate limit | Auth, quota, phát hiện pattern; cân nhắc mức chi tiết xác suất |
| A08 | Phát hiện lộ đường dẫn/secret khi lỗi | Inject canary và path **giả** vào factory/predictor, startup health + predict/API | Không canary trong response hoặc logs | Generic error + error ID, redaction, quyền đọc logs tối thiểu |
| A09 | Gây phân loại sai trong giới hạn nhiễu | ONNX thật, dữ liệu thật và selection gốc; 1 ảnh/lớp; ≤36 model queries | Đo label flip từ ảnh gốc đúng; không coi tìm không thấy là phòng thủ đầy đủ | Adversarial eval/training phù hợp, OOD/uncertainty review, giám sát drift |
| A10 | Phát hiện model bị sửa | Chỉ copy fixture tmp: đổi model; tiếp tục đổi manifest, giữ trusted hash ngoài bundle | Từ chối trước session trong cả hai ca | Trusted external SHA, immutable revision, quyền file read-only |

Toàn bộ A01–A08 có **42 HTTP calls nội bộ**; A10 không có HTTP call. Mỗi file
lớn chỉ khoảng 10 MiB, các ca tuần tự. A02 không tạo ảnh 100M pixels, chỉ thử
header bị khai báo; guard chứng minh không đi tới pixel load. A06 dùng events,
max_workers=2, timeout tối đa 3 giây để tránh treo; mô hình giả không tiêu CPU
như model thật. Các kết quả không phải benchmark DDoS/RSS của hệ thống triển khai.
Không có model/ảnh Private nào bị sửa ở A10.

## Đánh giá và bằng chứng

- **ATTACK_BLOCKED:** quan sát được cơ chế chặn trong **đúng phạm vi ca thử**.
  Không phải chứng nhận hệ thống chống mọi biến thể tấn công.
- **WEAKNESS_FOUND:** có bằng chứng điểm yếu cụ thể trong môi trường đã thử.
- **NOT_RUN:** chưa chạy (A09 thật ở Codex).
- **INCONCLUSIVE:** kết quả chưa đủ kết luận. A07 năm truy vấn không thể chứng
  minh đã sao chép được model hoặc hệ thống có/không có giới hạn dài hạn.

`pytest passed` nghĩa là scenario chạy và thu thập/kiểm tra được quan sát, không
đồng nghĩa mọi tấn công bị chặn. Vì vậy A08 có thể có test passed nhưng đánh giá
WEAKNESS_FOUND. Recorder chỉ dùng user_properties từ pytest; test lỗi/skip không
được gán ATTACK_BLOCKED. Setup/call/teardown fail ghi INCONCLUSIVE.

[Bảng kết quả tiếng Việt](summary.md), [JSON](results.json), [CSV Excel](results.csv).
JSON ghi timestamp/exit code; bảng có đủ 10 loại, bao gồm A09 NOT_RUN. Không tự
đổi hàng A09 thành ATTACK_BLOCKED khi chỉ unit test chuẩn bị đạt.

### Điểm yếu đã quan sát / phạm vi của kết luận

A08 fault injection cho thấy Baseline và Hybrid **local-bundle startup** đưa
exception text chứa path/canary giả ra `/health`. Baseline inference trả lỗi
client chung nhưng ghi exception đầy đủ vào logs. Hybrid inference đang trả
lỗi chung và logs chung trong ca đã thử. Đây là bằng chứng local với lỗi giả,
không phải chứng minh token thật đã bị lộ hoặc đã khai thác production.

Hybrid remote-HF startup có cơ chế redaction riêng đã có tests từ Phase 6;
không được suy diễn kết quả local startup thành mọi remote-HF lỗi đều lộ.
Đề xuất thống nhất lỗi client bằng error ID, không echo str(exc); redaction log
và chỉ cho người cần thiết đọc log. Chưa thực hiện sửa production ở bước này.

A05 preflight bị chặn và không Allow-Origin chỉ bảo vệ quyền đọc của browser.
Multipart POST từ site khác vẫn có thể tới server; HTTP client không bị CORS
ngăn gửi. Nếu cần kiểm soát người dùng/quota thì phải bổ sung auth; kiểm tra CSRF
riêng nếu sau này dùng cookie. Không gọi CORS là authentication.

A06 admission lock không phải rate limiting. A07 vẫn nhận đủ bốn xác suất trong
5/5 lần/API: có bề mặt thông tin để cân nhắc, chưa chứng minh model theft. Đề
xuất quota theo tài khoản/IP và theo thời gian, theo dõi bất thường; chỉ giảm
chi tiết probabilities khi phù hợp yêu cầu sản phẩm, không tự sửa UI/API.

A10 đổi model và manifest chỉ bị chặn chắc chắn khi trusted expected hash nằm
ngoài quyền sửa của kẻ tấn công. Hash do chính manifest cung cấp không tự thiết
lập nguồn tin cậy. Read-only file/container và quyền phân phối artifact vẫn cần.

## Chạy lại local

Dùng venv `/tmp/tomatocare-week2` đã cài runtime lock và requirements-ci ở bước
trước. Tại root source:

```bash
MPLCONFIGDIR=/tmp/tomatocare-security-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/tomatocare-week2/bin/python -m pytest -q tests/security
# Regression API/inference liên quan, không ghi lại báo cáo T09–T10:
MPLCONFIGDIR=/tmp/tomatocare-security-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp/tomatocare-week2/bin/python -m pytest -q \
  tests/test_api.py tests/test_hybrid_bootstrap.py tests/test_hybrid_onnx.py tests/test_inference.py
```

Windows: activate venv Python 3.12 rồi dùng `python -m pytest -q tests/security`.
Không truyền token. Tests chặn các hàm HF download hiện có, sử dụng các factory
injection/FakeSession/FixtureEngine/photo/manifest đã có; không thêm mock mode
vào production. Đây không phải sandbox egress cấp OS: dùng máy cô lập để bảo
đảm không có luồng ngoài ý muốn từ mã khác. Không chạy nguyên suite để tự ghi
đè báo cáo tuần 2 đã xác minh trên Kaggle. File kết quả security được tái tạo
chỉ bởi run có scenario Axx; unit test A09 helper đơn lẻ không ghi đè bảng này.

## A09 thật trên Kaggle

Script: [scripts/week2_adversarial.py](../../scripts/week2_adversarial.py).

Cần bundle local đã cấp quyền gồm `tomatocare_hybrid.onnx` và manifest, giữ kèm
license; dataset gốc đầy đủ; **selection.json gốc** đã cố định cho T09–T10.
Không chọn lại cohort, không điều chỉnh ngưỡng hoặc kết quả T09–T10. Script tái
sử dụng validator/hash/cohort/preprocessing/inference hiện có, không sửa model.
SHA ONNX mặc định là SHA đã xác minh từ Phase 5. Script không tải HF, không có
API URL mode và không yêu cầu token trong chat hoặc CLI.

```bash
# Checkout nhánh có script A09, giữ nguyên selection.json của T09–T10.
git clone --single-branch --branch feature/ai-security-testing https://github.com/NguyenThanhHaii/TomatoCare.git
cd TomatoCare
python -m scripts.week2_adversarial \
  --data-root /kaggle/input/PATH_TO_ORIGINAL_DATASET \
  --selection /kaggle/working/ORIGINAL_COHORT/selection.json \
  --model-dir /kaggle/working/PRIVATE_LOCAL_ONNX_BUNDLE \
  --out /kaggle/working/a09-run-001
```

Dùng nguồn nhánh `feature/ai-security-testing` có cả scripts/week2_ai_tests.py
và scripts/week2_adversarial.py. Nếu đã clone nhánh này, cập nhật source trước khi chạy.
Kaggle cần dependencies như hướng dẫn [AI tests](../week2/ai-testing.md); CPU
ORT đủ, không cần gradient/Transformers/pretrained tải thêm. Nếu cohort/model
đã có thì chỉ cần chuyển script/source mới sang workspace và chạy lệnh trên.
Không commit/export image hay weights đã tải vào repository.

Protocol A09 cố định trong SETTINGS:

- Chỉ **4 ảnh**: ảnh đầu tiên của mỗi lớp trong cohort 40 ảnh gốc, không thay
  ảnh sau khi nhìn dự đoán. Ảnh gốc sai nhãn thì ghi không đủ điều kiện, không
  tính là attack success và không thử tìm ảnh khác để thay.
- Tối đa 8 truy vấn biến thể/ảnh + 1 ảnh gốc = **36 model queries tổng**.
  Seed 2026; ảnh ≤1MP, file ≤10 MiB; xử lý tuần tự một ảnh.
- Black-box score-guided search bằng coarse random signs, mỗi bước ±2 mức
  uint8 RGB, luôn clamp về ảnh gốc trong **L∞ ≤8/255** (pixel 0–255 sai khác
  tối đa 8). Không đổi model, không training hoặc brute force kéo dài.
- Chỉ success khi ảnh gốc đúng, biến thể sai so với nhãn metadata và vẫn trong
  giới hạn L∞. Confidence chỉ là score tìm kiếm, không phải chứng minh đúng.
- Có ca thành công: WEAKNESS_FOUND cho phạm vi ảnh/model đó. Không có ca thành
  công, hoặc không có ảnh đủ điều kiện: INCONCLUSIVE, **không ATTACK_BLOCKED**.
  Sai artifact/validation/inference: ERROR + INCONCLUSIVE, không công bố kết
  quả chạy dở như một thử nghiệm hoàn tất. Thiếu artifact: NOT_RUN, exit 2.

Output mới riêng: a09-results.json/CSV, a09-summary.md; nếu có success thì thêm
PNG lossless ở output Kaggle, kèm hash. Trace ghi các query, nhãn/probability
lớp thật, L∞; báo số ảnh eligible và success rate trên ảnh eligible. Khi chạy
hoàn tất exit 0 chỉ báo công cụ đã chạy, không đồng nghĩa ATTACK_BLOCKED.
Không dùng kết quả từ model giả để bổ sung hàng A09. Các PNG này là dữ liệu
thử nghiệm, không stage vào Git. Dùng output mới, script từ chối ghi đè thư
mục tồn tại. [Báo cáo chuẩn bị tại Codex](a09-preparation/a09-results.json) chỉ
chứng minh điều kiện thiếu và 0 query, không phải kết quả tấn công thật.

Kết quả T09–T10 do người dùng cung cấp vẫn giữ nguyên; A09 không thay đổi chúng.
Chưa chạy A09 thật; chưa sửa A08/production hoặc deploy. Source được lưu trên
nhánh kiểm thử riêng, không merge vào main hay nhánh Hybrid nền.

## Kết quả kiểm tra bộ mã tại Codex

Đã chạy thực tế 11 tests mới (9 scenario local + 2 unit checks chuẩn bị A09)
và 95 tests regression API/inference liên quan: **106 passed, 0 failed,
0 skipped**, 3 cảnh báo deprecation Starlette/timm. Run riêng tests/security
cũng đạt 11 passed. Các con số này là kết quả pytest, không phải số tấn công bị
chặn: bảng security có 7 ATTACK_BLOCKED theo phạm vi thử, 1 WEAKNESS_FOUND,
1 INCONCLUSIVE và A09 NOT_RUN. Không chạy lại AI Test T09–T10.

Bước 9 kiểm tra lại toàn bộ regression mặc định: **175 passed, 0 failed,
3 skipped**, 3 cảnh báo deprecation. Ba bài Docker integration opt-in chưa
chuẩn bị image nên skipped, không tính PASS. Hook của regression có thể tái
sinh báo cáo automation tuần 2; lần rà soát này giữ nguyên bản báo cáo T01–T10
đã lưu theo từng byte, chỉ cập nhật bằng chứng security. A07 INCONCLUSIVE,
A08 WEAKNESS_FOUND, A09 NOT_RUN; chưa thực hiện sửa A08.
