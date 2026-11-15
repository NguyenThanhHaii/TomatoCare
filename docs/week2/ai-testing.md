# Bước 3 — Chuẩn bị AI Test T09–T10

**Chưa chạy AI thật.** `ai-preparation/results.json`, `case-results.csv` và
`summary.md` giữ T09/T10 NOT_RUN, chất lượng NOT_EVALUATED. Không dùng mock hoặc
kết quả cũ để tính accuracy/stability. Unit tests mới chỉ kiểm tra sampling,
biến đổi ảnh và trạng thái thiếu điều kiện, không phải T09/T10 đạt.

## Dữ liệu và protocol trước khi chạy

Dùng nguyên bundle TomatoCare có `metadata.csv`, `labels.json`, `images/`, dưới
một thư mục duy nhất (script tự tìm sâu từ --data-root). Schema đã xác minh từ
source: `image_path,label,label_id,split,sha256,source_image_path`. Path tính từ
root bundle, không từ thư mục PlantVillage gốc. Không đoán nhãn từ filename
hoặc dự đoán. Thứ tự nhãn: Healthy, Early Blight, Late Blight, Leaf Mold.

Reuse `discover_bundle()` để kiểm tra path/nhãn/split/duplicate leakage và số
lượng train=3816, val=816, test=820. Kiểm tra hash bytes thực tế của mọi ảnh với
metadata, sau đó chỉ **chọn từ records['test']**. Train/Val chỉ dùng để kiểm tra
bundle/leakage, không suy luận, không tham gia chọn ngưỡng hoặc 40 ảnh.

Chọn 10 ảnh khác SHA-256/lớp, seed Python Random 42, sort path trước khi sample.
Ảnh trùng hash cùng split/nhãn chỉ lấy một đại diện khi chọn cohort; không xóa
hoặc sửa dataset gốc. `selection.json` lưu 40 path, split, label ID/tên, hash,
hash metadata/labels và protocol. Chạy lại xác minh toàn bộ cohort, không đổi
seed/ảnh sau khi xem kết quả. Không tự loại ảnh sai/hỏng rồi thay ảnh khác:
prepare sẽ lỗi nếu ảnh được chọn không decode được.

Ngưỡng minh họa cho bài thực hành, được chốt **trước khi chạy** trong protocol:

- T09: tổng accuracy ≥90% (ít nhất 36/40) **và** mỗi lớp đúng ≥8/10.
- T10: tỷ lệ giữ nhãn ≥90% (ít nhất 36/40) cho **từng** biến thể.

Đây không phải ngưỡng được suy ra từ benchmark hay đảm bảo đạt. Muốn đổi ngưỡng
phải thiết kế protocol mới trước inference, review source và tạo selection/output
mới; không thay sau khi nhìn kết quả để biến FAIL thành PASS.

T10 sử dụng cùng 40 ảnh, mỗi ảnh tạo riêng từ ảnh RGB gốc đã sửa EXIF:
Brightness ×0.8, ×1.2 (Pillow, uint8 clipping); JPEG quality=70, subsampling=2,
optimize=False, progressive=False rồi decode RGB. Không nối tiếp ba biến đổi.
Tổng 160 lượt inference: 40 gốc + 120 biến thể, batch size 1. Cả hai nhánh dùng
nguyên preprocessing/softmax của `HybridOnnxPredictor`; không viết lại inference.

## Artifact cần có

1. Bundle dữ liệu đầy đủ trên, bao gồm ảnh thật; chỉ 40 ảnh rời không đủ để xác
   minh test membership và kiểm tra leakage với metadata gốc.
2. Bundle model local đã được cấp quyền: `tomatocare_hybrid.onnx`,
   `inference_manifest.json`, giữ kèm `DINOv3-LICENSE.md`.
3. ONNX trusted SHA-256 mặc định:
   `196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4`.
   Session chỉ nạp sau kiểm tra manifest/hash qua code Hybrid hiện có. Với
   model phiên bản khác, chỉ dùng --expected-onnx-sha256 sau khi xác minh độc lập.

Không cần checkpoint PyTorch/DINOv3 access token cho T10 độ ổn định. Script
không gọi HF Hub, không tải weights, không fallback random/mock và không gọi
API production. Không gửi token qua chat. Không public bundle chứa DINOv3.

## Chạy ở đâu, lệnh nào?

**Ít thao tác nhất: Kaggle**, nếu đã có dataset gốc và bundle ONNX private tại
đó. Attach dataset TomatoCare, dùng bundle đã có quyền truy cập local, mở source
nhánh `feature/ai-security-testing` từ GitHub trong workspace Kaggle.
Không clone nhánh hybrid cũ rồi kỳ vọng có script mới:

```bash
git clone --single-branch --branch feature/ai-security-testing https://github.com/NguyenThanhHaii/TomatoCare.git
cd TomatoCare
```
Kaggle có Torch; cài runtime lock nếu cần. Không cần GPU cho ONNX CPU.

**Codex:** chạy được sau khi cung cấp dataset/model local theo cơ chế được cấp
quyền; hiện chưa có. Venv `/tmp/tomatocare-week2` đã có dependencies.

**Windows/Docker local:** có thể chạy script trên host venv Python 3.12 với
requirements-ci và runtime lock như README tuần 2. Docker runtime tối giản không
có script/Torch loader, nên không chạy trực tiếp lệnh này trong image production.
Không sửa Docker để phục vụ bài test. Không triển khai HTTP API mode ở bước này.

Tại root source; thay các đường dẫn placeholder bằng đường dẫn thật:

```bash
# Kaggle: DATA_ROOT có thể là /kaggle/input nếu chỉ attach một bundle phù hợp.
# Dùng thư mục output mới cho mỗi lần chạy; không ghi đè bằng chứng cũ.
python -m scripts.week2_ai_tests prepare \
  --data-root /path/to/dataset-search-root --out /path/to/cohort-seed42
python -m scripts.week2_ai_tests run \
  --data-root /path/to/dataset-search-root \
  --selection /path/to/cohort-seed42/selection.json \
  --model-dir /path/to/local-private-onnx-bundle \
  --out /path/to/ai-run-001
```

Trên Codex dùng `/tmp/tomatocare-week2/bin/python` thay python. PowerShell có
thể viết mỗi lệnh trên một dòng hoặc thay dấu `\` bằng backtick. Không truyền
HF_TOKEN vào các lệnh này. Chỉ chạy prepare/run thật sau khi có đủ artifact.

## Kết quả và diễn giải

- `results.json`: model/version/ONNX hash, manifest hash, provenance, cohort,
  protocol, số đúng/sai, accuracy toàn bộ/từng lớp, stability từng biến thể và
  danh sách thay đổi nhãn.
- `predictions.csv`: 160 dòng khi hoàn tất, có nhãn thật, nhãn dự đoán,
  confidence, probabilities, correct, same_as_original và original_prediction.
- `case-results.csv`, `stability.csv`, `summary.md`: tổng hợp cho báo cáo PDF.
- Execution COMPLETED khác quality PASS/FAIL. Chạy đủ nhưng không đạt ngưỡng
  vẫn COMPLETED + FAIL; exit 1. Đạt tất cả ngưỡng: exit 0. Thiếu điều kiện:
  NOT_RUN; dữ liệu/model/hash không hợp lệ hoặc inference lỗi: ERROR, không
  đánh giá chất lượng, exit 2; không công bố metrics từ phần chạy dở.
- `prepare` lỗi validation thì không tạo cohort/output kết quả giả. Sửa nguyên
  nhân rồi dùng output mới. `plan` chỉ tạo kế hoạch NOT_RUN, không tạo selection.

Giữ nguyên nhãn sai vẫn được tính ổn định; confidence cao không chứng minh đúng.
Bốn mươi ảnh là mẫu nhỏ có cân bằng lớp, không thay thế metrics trên 820 ảnh.
Không điều chỉnh mô hình/hyperparameters theo tập Test này. Dataset chưa bảo
đảm độc lập leaf_id; ảnh trùng trong split có thể gây thiên lệch dù cohort đã
loại trùng hash. Không gọi đây là chứng nhận robustness hay chạy tấn công.
