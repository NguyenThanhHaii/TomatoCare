# TomatoCare — Tổng kết kiểm thử và bảo mật tuần 2

## Kiểm thử chức năng: 10/10 PASS

- T01–T08: Automation/Mock xác nhận API sẵn sàng, nhận ảnh hợp lệ, từ chối
  đầu vào lỗi/quá lớn, CORS/headers và xử lý lỗi model/hash/logits.
- T09: Hybrid ONNX dự đoán đúng **40/40 ảnh Test**, 10 ảnh mỗi lớp.
- T10: giảm sáng 20%, tăng sáng 20% và JPEG chất lượng 70 đều giữ nhãn
  **40/40 ảnh** so với ảnh gốc.

T09–T10 dùng mô hình thật trên Kaggle **theo kết quả người dùng cung cấp**;
Codex không tự chạy hoặc chạy lại các thử nghiệm này. Kết quả 40 ảnh không
đại diện đầy đủ cho toàn bộ tập Test; giữ nhãn hoặc confidence cao không tự
chứng minh dự đoán chính xác.

## Kiểm thử bảo mật: 10 loại

Sau khắc phục A08: **8 bị chặn trong phạm vi đã thử**, **2 chưa đủ căn cứ
kết luận** (A07 truy vấn lặp và A09 ảnh đối kháng). A09 đã chạy trên Kaggle theo
kết quả người dùng cung cấp, không phải do Codex thực hiện. Các báo cáo cũ được
lưu nguyên trạng; trạng thái lịch sử không bị sửa để khớp bảng tổng kết này.

| A08 | Trước sửa | Sau sửa |
| --- | --- | --- |
| Baseline/Hybrid startup local | Lộ exception/path giả qua response/log | Thông báo chung, mã lỗi an toàn |
| Baseline inference | Log chứa chi tiết exception | Không lộ canary/path trong log đã kiểm tra |
| Hoạt động API | Lỗi trả 503 | Vẫn 503; ảnh hợp lệ vẫn trả kết quả bình thường |

Probe A08 AFTER và 13 tests liên quan đã PASS tại Codex bằng TestClient/mock.
[Bằng chứng BEFORE/AFTER](../security/a08-remediation/comparison.md) lưu riêng,
không ghi đè kết quả gốc. Bản phòng thủ mới chỉ nằm trên nhánh kiểm thử, chưa
triển khai lên dịch vụ production.

## Hạn chế còn lại

- CORS không phải xác thực; giới hạn đồng thời không phải rate limiting.
  Cần xem xét auth/quota và giám sát truy vấn nếu mở API cho nhiều người dùng.
- A07/A09 INCONCLUSIVE không chứng minh hệ thống chống được sao chép hành vi
  hoặc mọi loại ảnh đối kháng.
- Tập thử nhỏ, chưa bảo đảm độc lập theo leaf_id; ảnh trùng trong split vẫn có
  thể gây thiên lệch. Không dùng kết quả Test để điều chỉnh mô hình.
- Kiểm tra A08 bằng canary giả không thay thế kiểm tra log nền tảng khi triển
  khai thật. Ba Docker integration tests vẫn được skip trong regression mặc
  định; không có triển khai/image mới được kiểm chứng ở bước này.
