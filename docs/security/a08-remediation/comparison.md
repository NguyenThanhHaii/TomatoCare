# A08 — So sánh BEFORE/AFTER

| Vấn đề | Trước khi sửa | Sau khi sửa | Kết luận |
| --- | --- | --- | --- |
| Baseline khởi tạo local | Response: có lộ; log: có lộ | Response: không lộ; log: không lộ | Đã khắc phục trong probe |
| Baseline suy luận lỗi | Response: không lộ; log: có lộ | Response: không lộ; log: không lộ | Đã khắc phục trong probe |
| Hybrid khởi tạo local | Response: có lộ; log: có lộ | Response: không lộ; log: không lộ | Đã khắc phục trong probe |
| Hybrid suy luận lỗi | Response: không lộ; log: không lộ | Response: không lộ; log: không lộ | Duy trì xử lý an toàn |
| Trạng thái HTTP khi model/init hoặc suy luận lỗi | 503 trên 4 request của probe | 503 trên cùng 4 request | Không đổi trạng thái lỗi |
| Dự đoán JPG/PNG bình thường (test engine) | T02 PASS trong bằng chứng đã lưu | 200; nhãn và softmax khớp giá trị chuẩn | Hợp đồng thành công giữ nguyên; không chạy lại AI thật |

BEFORE: WEAKNESS_FOUND. AFTER: ATTACK_BLOCKED trong fault injection local đã thử.
Canary/path là giả; không dùng token thật. Không thay các kết quả lịch sử khác hoặc chứng nhận log production.
BEFORE probe: 1 test đạt; AFTER: 13 tests đạt; full regression: 188 passed, 0 failed, 3 skipped.
