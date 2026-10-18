# TomatoCare

Tomato leaf disease classification. Phase 1 implements the original **UniRepLKNet-A** architecture for four classes: Healthy, Early Blight, Late Blight, Leaf Mold. DINOv3 integration is reserved for a later phase.

## Setup and training tests

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests/test_model.py tests/test_pipeline.py
```

For a CPU-only environment, install CPU PyTorch first, then the remaining dependencies:

```bash
python -m pip install "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q tests/test_model.py tests/test_pipeline.py
```

The test constructs the model with randomly initialized weights and runs a CPU forward pass on a `[1, 3, 224, 224]` tensor. It checks for four raw logits and compatibility with `CrossEntropyLoss`. No pretrained weights are downloaded or needed. The optional CUDA iGEMM extension is disabled.

Application code can import `create_unireplknet_a` and `CLASS_NAMES` from `tomatocare.model`. The factory is the intended place to add later backbone choices, including DINOv3 after its weights become available. Phase 2 adds the head-only training workflow described below. No real-dataset accuracy is reported before a Kaggle run.

## Source and license

`tomatocare/vendor/unireplknet.py` is copied without modification from [AILab-CVC/UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet), commit `b5677d8fb0e7b8af61eac39c1bf02727f5afc1d5`, specifically its root `unireplknet.py`. The authors license that source under Apache License 2.0; the corresponding license text is in `tomatocare/vendor/LICENSE`. UniRepLKNet credits RepLKNet, ConvNeXt, timm, DINO and DeiT in its source header. No DINOv3 source or pretrained weights are included.

## Phase 2 — Huấn luyện trên Kaggle GPU

Notebook: [`notebooks/train_unireplknet_a_kaggle.ipynb`](notebooks/train_unireplknet_a_kaggle.ipynb).

1. Sau khi kiểm tra và xuất bản mã Giai đoạn 2, tải notebook vào Kaggle bằng **Import Notebook**. Notebook clone repository từ GitHub. Nếu mã chưa xuất bản, đính kèm toàn bộ source checkout dưới dạng Kaggle input và sửa `REPO_DIR` sang vị trí thực tế đó.
2. Bật **Settings → Accelerator → GPU** (T4 hoặc P100 nếu có), bật **Internet** để cài dependencies và tải weights. Thêm dataset [TomatoCare 4 classes](https://www.kaggle.com/datasets/thanhhainguyn/tomatocare-tomato-leaf-disease-4-classes) bằng **Add Input**.
3. Chạy các ô setup và kiểm tra dữ liệu. Bộ đọc tìm cụm `metadata.csv`, `labels.json`, `images/` trong `/kaggle/input/`, kể cả các thư mục lồng nhau; không dự đoán slug thư mục. Nếu có nhiều bản dataset, đặt `DATA_ROOT` vào đúng nhánh thư mục cần dùng.
4. Chọn cấu hình trước lần chạy cuối: mặc định 5 epochs, batch size 32, AdamW (learning rate 0.001, weight decay 0.05), seed 42. Nếu thiếu bộ nhớ GPU, dùng batch size 16. Backbone và running statistics của BatchNorm đóng băng; chỉ head được huấn luyện.
5. Dùng **Save Version → Save & Run All** để chạy phiên cuối từ đầu. Không dùng kết quả test để thay đổi hyperparameters. Notebook chọn checkpoint theo **validation Macro-F1**, sau đó khôi phục checkpoint này và đánh giá test đúng một lần. Khi xem lại kết quả, chỉ đọc các file đã lưu. Nếu đã chạy tương tác, hãy giữ kết quả đó hoặc chủ động lưu trữ output trước một thí nghiệm khác; script từ chối ghi đè checkpoint/kết quả test có sẵn.
6. Tải `/kaggle/working/best_unireplknet_a.pth` từ tab **Output** của phiên chạy đã lưu, cùng `history.json`, `validation_metrics.json`, `test_metrics.json`, `*_confusion_matrix.csv` và `*_confusion_matrix.png`.

Có thể chạy script trực tiếp từ thư mục repository:

```bash
python -m tomatocare.train --data-root /kaggle/input --epochs 5 --batch-size 32
```

### Schema và kiểm tra dữ liệu

Schema được người tạo dataset cung cấp:

```text
image_path,label,label_id,split,sha256,source_image_path
```

`image_path` tương đối với thư mục chứa `metadata.csv` (ví dụ `images/train/Healthy/filename.JPG`). Loader kiểm tra ảnh tồn tại trong `images/`, cấm đường dẫn thoát ra ngoài, nhãn hợp lệ, `label_id` khớp tên nhãn, split hợp lệ, mỗi split có đủ bốn lớp, và không lặp đường dẫn trong metadata. SHA-256 được ánh xạ tới `(split, label_id)`: cho phép lặp hash nếu cùng split và cùng nhãn; khác split hoặc khác nhãn sẽ báo lỗi. Loader cảnh báo số ảnh thuộc các nhóm trùng, số bản sao ngoài ảnh đầu tiên và số bản sao theo split; không xóa ảnh hoặc thay đổi metadata. Giữ nguyên 5.452 ảnh (train=3816, val=816, test=820). Hash được kiểm tra từ metadata, không tính lại SHA-256 của toàn bộ file ảnh. `source_image_path` chỉ là thông tin nguồn; không được dùng làm đầu vào mô hình. Với schema chuẩn, loader chọn `image_path`, `label`, `split`; có thể chỉ định `--path-column`, `--label-column`, `--split-column` cho schema khác đã kiểm tra.

`labels.json` là nguồn duy nhất quyết định thứ tự logits:

```json
{"Healthy": 0, "Early Blight": 1, "Late Blight": 2, "Leaf Mold": 3}
```

Không sắp xếp tên lớp theo alphabet. Pipeline xác nhận số lượng **train 3.816, validation 816, test 820**; nếu sai thì dừng để kiểm tra phiên bản dataset. Cờ `--allow-count-mismatch` chỉ dành cho bundle khác được chọn có chủ ý, như smoke test. Ảnh được chuyển RGB, resize bilinear 224×224, normalize ImageNet (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). Chỉ train có random horizontal flip. Validation và test không augmentation.

### Pretrained weights và checkpoint

Sử dụng đúng nguồn trong mã gốc của tác giả: [DingXiaoH/UniRepLKNet](https://huggingface.co/DingXiaoH/UniRepLKNet/tree/main), file `unireplknet_a_in1k_224_acc77.03.pth`. Loader nhận checkpoint bọc trong `model`/`state_dict` hoặc state dict trực tiếp, xác nhận head ImageNet 1.000 lớp và toàn bộ keys/shapes backbone. Head ImageNet được loại bỏ, giữ head 4 lớp từ `tomatocare/model.py`. Không sửa mã vendor và không dùng CUDA iGEMM extension.

Nếu tải thất bại, chương trình dừng với lỗi cụ thể. Tải checkpoint từ nguồn chính thức, đính kèm dưới dạng Kaggle input, rồi đặt `PRETRAINED_CHECKPOINT` trong notebook hoặc dùng:

```bash
python -m tomatocare.train --pretrained-checkpoint /actual/path/unireplknet_a_in1k_224_acc77.03.pth
```

Không tự chuyển sang random initialization. File `.pth` cục bộ phải là file đáng tin từ nguồn chính thức; loader dùng `torch.load(..., weights_only=True)`. Không thêm checkpoint hoặc dữ liệu vào Git. Checkpoint đầu ra chứa `state_dict`, `class_names`, `class_to_idx`, cấu hình preprocessing, cấu hình thí nghiệm, nguồn weights, epoch tốt nhất và validation Macro-F1. Tạo lại mô hình bằng `create_unireplknet_a()`, nạp `checkpoint["state_dict"]` với `strict=True`, gọi `eval()` và áp dụng preprocessing đã lưu để inference.

Metrics xuất ra gồm Accuracy, Precision/Recall/Macro-F1 với macro averaging, confusion matrix (hàng là nhãn thật, cột là nhãn dự đoán) và classification report từng lớp. Các lớp không có dự đoán được xử lý bằng `zero_division=0`.

### Kiểm thử và giới hạn

```bash
python -m pytest -q tests/test_model.py tests/test_pipeline.py
```

Tests ngoại tuyến tạo ảnh và checkpoint **ngẫu nhiên giả lập định dạng ImageNet** để kiểm tra schema, thứ tự nhãn, batch, forward 4 logits, validation metrics, đóng băng backbone/BatchNorm, train loop, lưu/nạp checkpoint, và test chỉ chạy sau khi chọn best epoch. Không tải weights trong test. Kết quả trên fixture không phản ánh chất lượng phân loại bệnh.

**Giới hạn dữ liệu:** các ảnh có cùng SHA-256 được nhóm vào cùng split, không nhất thiết bị xóa. Ảnh trùng hệt trong cùng split vẫn có thể gây thiên lệch: lặp mẫu trong train làm tăng trọng số của mẫu đó, còn lặp mẫu trong validation/test làm lệch trọng số đánh giá. Chưa bảo đảm các split độc lập theo `leaf_id`; các ảnh khác nhau của cùng lá hoặc ảnh gần trùng có thể nằm ở các split khác nhau. Vì vậy test metrics có thể đánh giá quá cao khả năng tổng quát sang lá mới. Cần tách theo định danh lá/nguồn và kiểm tra ảnh gần trùng trong nghiên cứu tiếp theo.

Môi trường Codex hiện bị chặn truy cập Kaggle và Hugging Face (HTTP 403), nên chưa xác minh trực tiếp dataset thật, chưa tải checkpoint chính thức và chưa huấn luyện trên GPU Kaggle. Notebook không lưu sẵn bất kỳ kết quả accuracy nào. Giai đoạn này chưa tích hợp DINOv3 hoặc Streamlit.

## Phase 3 — React PWA + FastAPI

The mobile application follows the supplied four-state design: Upload → Preview → Analyzing → Result. It uses the existing **UniRepLKNet-A baseline** and exactly four classes. DINOv3 remains planned. No Streamlit or paid API/database is used.

- [`frontend/`](frontend/): React + Vite, custom mobile CSS, Lucide icons and `vite-plugin-pwa`.
- [`backend/main.py`](backend/main.py): `/health`, multipart `/predict`, upload limits, EXIF/RGB checks, explicit CORS and one cached predictor per process.
- [`tomatocare/inference.py`](tomatocare/inference.py): ONNX CPU loading of `best_unireplknet_a.onnx` from `ThanhHai25/TomatoCare-UniRepLKNet-A`, fixed batch-one preprocessing and real softmax probabilities. PyTorch remains available for research. Failures never return fabricated predictions.
- [`scripts/export_onnx.py`](scripts/export_onnx.py) and [`scripts/benchmark.py`](scripts/benchmark.py): verified ONNX export and CPU/RSS measurement commands.
- [`docs/deployment.md`](docs/deployment.md): complete local setup, environment variables, free Vercel/Render preparation, artifact export and resource gates.
- [`docs/design-review.md`](docs/design-review.md), [`docs/screenshots/`](docs/screenshots/): visual comparison and 12 screenshots at 375/390/430px. Screenshot results are test fixtures, not trained-model output.

Quick local run on Linux/Python 3.12 (two terminals; see deployment guide for installation details):

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r backend/requirements.txt
CORS_ORIGINS=http://localhost:5173 INFERENCE_RUNTIME=onnx CPU_THREADS=1 \
  python -m uvicorn backend.main:app --port 8000 --workers 1
```

```bash
cd frontend
npm ci
VITE_API_URL=http://localhost:8000 npm run dev
```

The app shell may be installed and cached, but AI analysis needs a network connection. Photos/results are not cached by the PWA or saved permanently by the API. The supplied project evaluation figures are attributed in the deployment guide, not manufactured as current predictions.

**Current blocker:** ONNX download still returns HTTP 403 in Codex. The owner reports Kaggle parity on 12 validation images (maximum logit error 8.106e-6), Default 108.76 MB / 28.63 ms and Low Memory 90.42 MB / 91.33 ms. These measurements have not been reproduced here or on Render. The full FastAPI attempt had zero successful inferences and 88.90 MiB failed-startup RSS only. See [runtime evidence](docs/onnx-runtime-validation.md). `render.yaml` now targets native Python ONNX with seven runtime dependencies, one worker and one CPU thread. No deployment has been performed.

Full application tests require `python -m pip install -r requirements-dev.txt`, then `python -m pytest -q`; browser tests and production build run with `cd frontend && npm ci && npx playwright install chromium && npm test && npm run build`. Existing training-only tests can be run with `python -m pytest -q tests/test_model.py tests/test_pipeline.py`.


Production defaults to `ONNX_PROFILE=default`; `ONNX_PROFILE=low_memory` is optional. The public ONNX file does not require an extra sidecar: embedded metadata is validated if present, otherwise the model owner's confirmed four-class/ImageNet preprocessing contract is used explicitly. The frontend API origin is configured using **`VITE_API_URL`**. Full-service resource checks are available with `python -m scripts.benchmark_api --image /actual/path/leaf.jpg --profile default`. Refer to the deployment guide for Render start/build commands and Vercel settings.
