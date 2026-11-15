# Kết quả automation/mock tuần 2

Nguồn: pytest thực tế; không dùng model Private hoặc API production.

| Mã | Kiểm tra | Trạng thái | Kết quả thực tế |
| --- | --- | --- | --- |
| T01 | API sẵn sàng | PASS | Baseline: HTTP 200, status=ready; Hybrid: HTTP 200, status=ready |
| T02 | Ảnh JPG/PNG hợp lệ | PASS | Baseline/jpg: HTTP 200; Baseline/png: HTTP 200; Hybrid/jpg: HTTP 200; Hybrid/png: HTTP 200 |
| T03 | Ảnh hỏng/rỗng | PASS | Baseline/hỏng: HTTP 422; Baseline/rỗng: HTTP 422; Hybrid/hỏng: HTTP 422; Hybrid/rỗng: HTTP 422 |
| T04 | File quá 10 MiB | PASS | Baseline: 10 MiB + 1 byte → HTTP 413; Hybrid: 10 MiB + 1 byte → HTTP 413 |
| T05 | CORS và headers | PASS | Baseline/CORS hợp lệ: HTTP 200; Baseline/CORS khác origin: HTTP 400; Hybrid/CORS hợp lệ: HTTP 200; Hybrid/CORS khác origin: HTTP 400; no-store và nosniff có trên health, predict thành công và lỗi |
| T06 | Mô hình không tải được | PASS | Baseline: HTTP 503; Baseline: HTTP 503; Hybrid: HTTP 503; Hybrid: HTTP 503 |
| T07 | ONNX giả lập bị thay đổi | PASS | SHA-256 mismatch; session chưa tạo=True; health=503; predict=503 |
| T08 | Logits NaN/Inf/sai shape | PASS | Baseline/NaN: HTTP 503; Hybrid/NaN: HTTP 503; Baseline/Inf: HTTP 503; Hybrid/Inf: HTTP 503; Baseline/sai shape: HTTP 503; Hybrid/sai shape: HTTP 503 |
| T09 | Độ chính xác trên 40 ảnh Test thật | NOT_RUN | Chưa có Hybrid ONNX và ảnh Test có nhãn thật trong workspace; không dùng mock để báo AI Test |
| T10 | Độ ổn định trên cùng 40 ảnh Test | NOT_RUN | Chưa có Hybrid ONNX và ảnh Test có nhãn thật trong workspace; không dùng mock để báo AI Test |

Tổng phiên pytest: 164 passed; 0 failed; 3 skipped.

PASS mock chỉ xác nhận hành vi phần mềm, không xác nhận độ chính xác AI thật.
