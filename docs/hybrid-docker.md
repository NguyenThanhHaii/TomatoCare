# Phase 7A — Docker cho Hybrid backend thử nghiệm

Chỉ dùng nhánh `feature/hybrid-dinov3`. Không thay backend baseline, frontend,
nhánh main hoặc dịch vụ đang chạy. Giai đoạn này không publish Docker Hub,
không thêm CI/CD và không deploy Render. Cần Docker Engine hoặc Docker Desktop
(chế độ Linux containers), Docker BuildKit và mạng để tải Python/packages.
Chạy mọi lệnh từ thư mục gốc repository.

## 1. Build image không có model hoặc secret

```bash
git branch --show-current
docker version
docker build --progress=plain -f Dockerfile.hybrid -t tomatocare-hybrid:phase7a .
docker image inspect --format 'Id={{.Id}} SizeBytes={{.Size}} User={{.Config.User}}' tomatocare-hybrid:phase7a
```

Kết quả kỳ vọng: build hoàn thành, user `10001:10001`. Python 3.12 slim,
chỉ dependencies từ `hybrid_backend/requirements.txt`; không PyTorch/Transformers.
COPY theo danh sách file runtime cụ thể, không `COPY . .`. Một số helper từ
`backend.main` và `tomatocare.inference` được dùng để kiểm tra upload; ứng dụng
chạy vẫn là `hybrid_backend.main:app`, không khởi động backend baseline.
Nguồn được cấp quyền đọc cho user không phải root. Các ONNX ví dụ đóng kèm
onnxruntime cũng được loại bỏ. Giữ bản license DINOv3 trong `/app/licenses`.

`.dockerignore` loại weights, notebook, dataset, frontend, cache, `.env`, key và
thư mục môi trường khỏi context. Không truyền HF_TOKEN bằng build argument,
không thêm token vào Dockerfile. Image gốc không có model/HF cache; khi chạy,
model và cache được tạo trong filesystem tạm của container, không trong image.
Không dùng `docker commit` hoặc export container đã tải model để chia sẻ.

Mạng doanh nghiệp có proxy/CA riêng có thể cần cấu hình Docker proxy/DNS.
Dockerfile hỗ trợ CA công khai tạm thời cho pip bằng BuildKit:

```bash
docker build --secret id=pip_ca,src=/path/to/public-ca-bundle.crt -f Dockerfile.hybrid -t tomatocare-hybrid:phase7a .
```

CA phải là certificate bundle công khai, không chứa private key. Mount này
không được giữ trong image. Không tắt TLS hoặc dùng trusted-host để bỏ xác minh.
Trong Codex, build cần proxy và CA công khai của môi trường; build thông thường
không có cấu hình mạng đó đã lỗi DNS/TLS. Máy có mạng bình thường không cần mount.

## 2. Cấu hình runtime Private HF — Bash

Lấy **full immutable 40-character commit SHA của HF repo**, không dùng `main`
hoặc GitHub commit SHA. Theo [hướng dẫn Phase 6](hybrid-phase6-private-bootstrap.md)
để tạo fine-grained read token chỉ cho private repo. Không ghi token vào lệnh,
file `.env` được track, ảnh chụp, log hoặc checkpoint.

```bash
export HYBRID_HF_REPO=ThanhHai25/TomatoCare-Hybrid-ONNX-Private
export HYBRID_HF_REVISION='<FULL_IMMUTABLE_HF_COMMIT_SHA>'
export HYBRID_EXPECTED_ONNX_SHA256=196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4
export HYBRID_MODEL_DIR=/tmp/tomatocare-hybrid
read -rsp 'HF read token: ' HF_TOKEN; echo
export HF_TOKEN
docker run -d --name tomatocare-hybrid \
  -p 127.0.0.1:8001:8000 --memory=512m --cpus=1 \
  --env HYBRID_HF_REPO --env HYBRID_HF_REVISION \
  --env HYBRID_EXPECTED_ONNX_SHA256 --env HYBRID_MODEL_DIR --env HF_TOKEN \
  tomatocare-hybrid:phase7a
unset HF_TOKEN
```

Thay placeholder revision trước khi chạy. `--env HF_TOKEN` lấy giá trị từ môi
trường shell, không đưa token literal vào lịch sử lệnh. Docker daemon/admin vẫn
có thể đọc runtime environment: chỉ dùng máy tin cậy, không chụp toàn bộ
`docker inspect` hoặc env. Xóa container sau khi thực hành để bỏ runtime token,
cache và model tải xuống. Không cần persistent volume. Không dùng tmpfs cho
model nếu chưa đo RAM vì tmpfs cũng tính vào giới hạn bộ nhớ.

## 3. Windows PowerShell

Bật Docker Desktop Linux containers, mở PowerShell ở repository root:

```powershell
$env:HYBRID_HF_REPO = 'ThanhHai25/TomatoCare-Hybrid-ONNX-Private'
$env:HYBRID_HF_REVISION = '<FULL_IMMUTABLE_HF_COMMIT_SHA>'
$env:HYBRID_EXPECTED_ONNX_SHA256 = '196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4'
$env:HYBRID_MODEL_DIR = '/tmp/tomatocare-hybrid'
$secureToken = Read-Host 'HF read token' -AsSecureString
$env:HF_TOKEN = [System.Net.NetworkCredential]::new('', $secureToken).Password
docker run -d --name tomatocare-hybrid `
  -p 127.0.0.1:8001:8000 --memory=512m --cpus=1 `
  --env HYBRID_HF_REPO --env HYBRID_HF_REVISION `
  --env HYBRID_EXPECTED_ONNX_SHA256 --env HYBRID_MODEL_DIR --env HF_TOKEN `
  tomatocare-hybrid:phase7a
Remove-Item Env:\HF_TOKEN
Remove-Variable secureToken
```

Không có khoảng trắng sau dấu backtick cuối dòng. Dùng `curl.exe` thay `curl`
trong các lệnh dưới để tránh alias PowerShell. Không in `$env:HF_TOKEN`.

## 4. Logs, health và dự đoán thật

```bash
docker logs --tail 100 tomatocare-hybrid
curl -i http://127.0.0.1:8001/health
curl -i -F 'file=@/absolute/path/to/validation-leaf.jpg' http://127.0.0.1:8001/predict
```

Windows: `curl.exe -i -F "file=@C:\images\leaf.jpg" http://127.0.0.1:8001/predict`.
Trong cold start, `/health` trả 503; sau tải đủ ba file cùng HF revision, xác minh
hash với manifest và trusted hash, rồi tạo session thành công mới trả 200.
Lỗi auth/mạng/hash không fallback sang mô hình giả hoặc baseline. Thiếu cấu hình
model cũng trả 503, không chứng minh inference sẵn sàng. Docker healthcheck có
start-period 300 giây; mạng chậm có thể khiến container unhealthy dù process
vẫn chạy. Xem logs đã lọc lỗi và sửa cấu hình, không công khai token.

Predict nhận JPG/PNG tối đa 10 MiB, sửa EXIF orientation, kiểm tra ảnh hỏng.
Kết quả thật gồm nhãn, confidence, xác suất Healthy/Early Blight/Late Blight/
Leaf Mold đúng thứ tự và model_version. Không dùng ảnh giả để báo accuracy.
Lỗi input trả lỗi HTTP rõ ràng; cần model ready trước khi kiểm thử input.
Một Uvicorn worker, ORT một thread, session được tái sử dụng, memory arena và
memory pattern tắt theo cấu hình Hybrid hiện có.

Port mặc định 8000 trong container. Có thể dùng `--env PORT=8080` cùng mapping
`-p 127.0.0.1:8001:8080`; healthcheck tự đọc PORT. Không chạy thêm worker.
CORS không được bật wildcard; dùng cấu hình origin hiện có nếu cần thử client.

## 5. Kiểm thử packaging và tài nguyên

```bash
python -m scripts.check_hybrid_docker --image tomatocare-hybrid:phase7a
# Có thể thêm --build để build trước khi audit trên mạng thông thường.
python -m pytest -q
# Opt-in integration: yêu cầu Docker daemon và image đã build.
RUN_HYBRID_DOCKER_TESTS=1 python -m pytest -q
```

PowerShell opt-in: `$env:RUN_HYBRID_DOCKER_TESTS='1'`, chạy pytest, sau đó
`Remove-Item Env:\RUN_HYBRID_DOCKER_TESTS`. Có thể đổi image bằng HYBRID_TEST_IMAGE.

Checker dùng container riêng, không token/volume, không public port; xác minh
runtime imports/CPU provider, không Torch/Transformers, không weights/cache/env
file trong image và user không root. Nó chạy Uvicorn dưới 512 MiB/1 CPU và yêu
cầu `/health` lẫn `/predict` trả **503 vì không có model**. Đây là kiểm thử
packaging/error path, không phải inference Private thành công. Container kiểm
thử được dọn sau chạy. Không có daemon thì checker báo blocked/failed, không
khẳng định build thành công; test integration mặc định skip nếu chưa opt-in.

Với container đã tải model thật, đo riêng:

```bash
docker stats --no-stream tomatocare-hybrid
docker exec tomatocare-hybrid sh -c "grep -E 'VmRSS|VmHWM' /proc/1/status"
docker inspect --format 'OOMKilled={{.State.OOMKilled}}' tomatocare-hybrid
```

PID 1 là Uvicorn nhờ exec. VmHWM là peak RSS (kB; chia 1024 thành MiB);
docker stats là memory cgroup, không đồng nhất với RSS. Đo từ cold start qua
tải model và nhiều request, bao gồm peak khi xác minh/session load. Kích thước
image trên disk không phải RAM runtime. Có thể thử `--cpus=0.1` để đánh giá CPU
hạn chế, nhưng cold start/latency sẽ khác Kaggle. Không suy diễn Render Free
512 MB đủ chỉ từ smoke không model. Kết quả Kaggle 275.65 MiB, 10/10 request do
chủ dự án cung cấp không phải số đo container này.

Kiểm thử private thật/parity nhiều ảnh: xem
[Phase 6 integration](hybrid-phase6-private-bootstrap.md) và
[Phase 5 parity](hybrid-phase5.md). Codex không có HF_TOKEN/artifact Private,
nên chưa xác minh Docker tải Private, inference thật, peak RAM với model hoặc
latency thật. Không thay đổi quyền phân phối DINOv3; giữ model private và rà
soát license trước bất kỳ đề xuất public image chứa model. Image Phase 7A chỉ
đóng gói code/dependencies, không publish ở bước này.

Đã thực chạy trong Codex ngày 2026-10-31: Docker build (với proxy/CA của
môi trường), audit imports/image và Uvicorn HTTP smoke không model đều passed.
Image ID: `sha256:addad77174b1195a8f41d94761383f142ab93964b6c3252b5a42dbf5afcdc35b`,
size Docker báo `293122477` bytes (khoảng 279.54 MiB trên disk). Không có
weights/token được truyền vào kiểm thử này.

## 6. Dừng và ảnh chụp báo cáo PDF

```bash
docker stop --time 30 tomatocare-hybrid
docker rm tomatocare-hybrid
```

Chụp terminal theo từng bước, ghi kết quả thực tế và ngày/máy thử nghiệm:

1. Branch feature, commit SHA, Docker client/server version.
2. Dockerfile, COPY allowlist, .dockerignore và giải thích token chỉ runtime.
3. Build thành công; image ID/size/user qua inspect chọn trường như trên.
4. JSON checker và pytest: phân biệt 503 không model với health 200 thật.
5. Khi đã thử Private: logs đã che dữ liệu nhạy cảm, health 200, predict có
   bốn xác suất/model_version. Nếu chưa chạy, ghi chưa kiểm chứng, không giả ảnh.
6. docker stats, VmHWM và OOMKilled sau cold start/request thật; chỉ rõ đơn vị.
7. docker stop/rm hoàn thành. Không có ảnh Docker Hub/deployment trong Phase 7A.

Không chụp token, secret entry, toàn bộ environment hoặc container inspect.
