# Phase 7A/7B — Docker và Compose cho Hybrid backend thử nghiệm

Chỉ dùng nhánh `feature/hybrid-dinov3`. Không thay backend baseline, frontend,
nhánh main hoặc dịch vụ đang chạy. Giai đoạn này không publish Docker Hub,
không thêm CI/CD và không deploy Render. Cần Docker Engine hoặc Docker Desktop
(chế độ Linux containers), Docker BuildKit và mạng để tải Python/packages.
Chạy mọi lệnh từ thư mục gốc repository.

## 1. Build image không có model hoặc secret

```bash
git branch --show-current
docker version
docker build --progress=plain -f Dockerfile.hybrid -t tomatocare-hybrid:phase7b .
docker image inspect --format 'Id={{.Id}} SizeBytes={{.Size}} User={{.Config.User}}' tomatocare-hybrid:phase7b
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
docker build --secret id=pip_ca,src=/path/to/public-ca-bundle.crt -f Dockerfile.hybrid -t tomatocare-hybrid:phase7b .
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
  tomatocare-hybrid:phase7b
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
  tomatocare-hybrid:phase7b
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
python -m scripts.check_hybrid_docker --image tomatocare-hybrid:phase7b
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
Image lịch sử Phase 7A: `sha256:addad77174b1195a8f41d94761383f142ab93964b6c3252b5a42dbf5afcdc35b`,
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

## 7. Phase 7B — Compose và dependency lock

Compose Specification không có trường `version`. Chỉ một service `hybrid`,
`platform: linux/amd64`, port loopback `127.0.0.1:8001:8000`, RAM 512 MiB,
CPU 1.0, một worker từ Dockerfile và GET /health healthcheck. Không có model
volume hoặc frontend/service production. Dừng container Docker thủ công ở
mục trước nếu nó đang giữ port 8001 rồi mới chạy Compose.

Base được pin theo manifest Linux amd64 đã xác minh trên Docker Hub:

```text
python:3.12.16-slim-trixie@sha256:1148dbb014615300fa7e713b2f94f550c0d1c69780c0805f84f093ceee34dd0b
```

Tag và digest được xác minh bằng `docker buildx imagetools inspect
python:3.12.16-slim-trixie` và build thật. Không dùng digest giả; digest trỏ đến
manifest amd64 (không phải multi-platform index). Máy ARM có thể cần Docker
emulation; chưa đo performance ARM. `requirements.txt` vẫn là khoảng phiên bản
đầu vào để resolve; image **chỉ cài requirements.lock.txt** với `--require-hashes
--only-binary=:all:` và chạy `pip check`. Không thay requirements Kaggle.

### PowerShell: chạy Compose với secret từ môi trường

Đã kiểm thử Compose v2.40.3. Cập nhật Docker Desktop nếu plugin không hỗ trợ
secret source `environment`. Không ghi token vào file; không dùng `env_file`
chứa secret. Compose lấy biến shell HF_TOKEN làm **nội dung secret runtime**,
mount read-only tại `/run/secrets/hf_token`; app đọc qua `HF_TOKEN_FILE`.
Token không nằm trong container environment, build arguments hoặc image.
Local Compose secrets không phải kho bí mật mã hóa: người có quyền Docker/host
vẫn có thể đọc secret. Chỉ thực hành trên máy tin cậy và `down` sau khi xong.

```powershell
docker compose version
$env:HYBRID_HF_REPO = 'ThanhHai25/TomatoCare-Hybrid-ONNX-Private'
$env:HYBRID_HF_REVISION = '<FULL_IMMUTABLE_HF_COMMIT_SHA>'
$env:HYBRID_EXPECTED_ONNX_SHA256 = '196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4'
$env:HYBRID_MODEL_DIR = '/tmp/tomatocare-hybrid'
if ($env:HYBRID_HF_REVISION -cnotmatch '^[0-9a-f]{40}$') { throw 'Set immutable HF revision first' }
$secureToken = Read-Host 'Fine-grained HF read token' -AsSecureString
$env:HF_TOKEN = [System.Net.NetworkCredential]::new('', $secureToken).Password
if ([string]::IsNullOrWhiteSpace($env:HF_TOKEN)) { throw 'HF_TOKEN is required' }
docker compose config --quiet
docker compose build
# Chỉ tiếp tục nếu config và build exit code 0.
docker compose up -d
docker compose ps
docker compose logs --tail 100 hybrid
curl.exe -i http://127.0.0.1:8001/health
curl.exe -i -F "file=@C:\images\validation-leaf.jpg" http://127.0.0.1:8001/predict
$container = docker compose ps -q hybrid
docker stats --no-stream $container
docker compose exec -T hybrid sh -c "grep -E 'VmRSS|VmHWM' /proc/1/status"
# Khi thực hành xong:
docker compose down
Remove-Item Env:\HF_TOKEN
Remove-Variable secureToken
```

Giữ biến shell HF_TOKEN đến khi `down` xong vì mỗi lần Compose có thể cần đọc
lại secret source, đặc biệt khi recreate. Không mở cửa sổ shell đó cho người
khác, không in biến. Repo/hash/model dir có default rõ ràng trong Compose;
revision không có default. Thiếu revision khiến `config --quiet` lỗi ngay.
Thiếu HF_TOKEN khiến **`up` lỗi trước khi service khởi chạy**; `config --quiet`
không xác minh nội dung secret hoặc quyền truy cập HF. Revision sai định dạng,
secret rỗng/không có quyền vẫn bị bootstrap từ chối và /health giữ 503.

Không dùng `docker compose config` với output đầy đủ, `config --environment`,
`printenv` hoặc toàn bộ `docker inspect` khi quay/chụp báo cáo. Không commit
`.env`, secret file hoặc token. `.secrets/`, `secrets/`, `hf_token.txt` được
ignore, nhưng không dựa riêng vào ignore để bảo vệ một tên secret tùy ý.
`.dockerignore` chặn các thư mục secret trước khi gửi build context.

Bash dùng các biến/read ẩn ở mục 2 rồi chạy cùng các lệnh `docker compose`;
thay `curl.exe` bằng `curl`. Giữ biến token đến sau `compose down`, không unset
ngay sau `up` như cách `docker run` thủ công.

`HF_TOKEN` trực tiếp vẫn được hỗ trợ cho Docker run/Render/Kaggle hiện có.
`HF_TOKEN_FILE` là lựa chọn mới khi remote bootstrap bật; nếu cấu hình cả hai
nguồn thì app từ chối để tránh dùng nhầm credential. File phải đọc được,
UTF-8, không rỗng, không khoảng trắng bên trong, tối đa 4096 bytes; newline
cuối file được bỏ. Lỗi không ghi nội dung hoặc đường dẫn secret vào log.
Local bundle mode không đọc token. Không cần hoặc đề xuất file token trên disk
với Compose mặc định này.

### Tái tạo lock trên Python 3.12 Linux amd64

Lock được resolve thực tế bằng pip-tools **7.6.2**, pip **25.0.1**, Python
3.12 Linux; đã cài/build trên base Python **3.12.16** Linux amd64. Dùng một môi
trường riêng, không thêm công cụ resolve vào runtime image:

```bash
python3.12 -m venv /tmp/tomatocare-lock
/tmp/tomatocare-lock/bin/python -m pip install 'pip==25.0.1' 'pip-tools==7.6.2'
PIP_TOOLS_CACHE_DIR=/tmp/tomatocare-piptools-cache /tmp/tomatocare-lock/bin/python -m piptools compile \
  --generate-hashes --resolver=backtracking --no-emit-index-url --no-emit-trusted-host \
  --output-file=hybrid_backend/requirements.lock.txt hybrid_backend/requirements.txt
```

Chạy trên Linux Python 3.12, không resolve lại trên Python/OS khác rồi coi là
cùng lock. Windows dùng Docker/WSL Linux để cập nhật lock. Khi cập nhật chủ động
có thể thêm `--upgrade`, review diff và build/test lại. Build hàng ngày không
resolve lại dependencies. Lock gồm exact versions và hash của các distribution
được resolver ghi nhận; pip trong image chỉ chọn wheel tương thích Linux amd64.
Không có index URL/credential trong lock. Giữ package index đáng tin cậy;
lock/hash xác minh artifact, không tự bảo đảm an toàn/license của dependency.

Kiểm tra lại image và Compose:

```powershell
docker run --rm --entrypoint python tomatocare-hybrid:phase7b -m pip check
python -m scripts.check_hybrid_docker --image tomatocare-hybrid:phase7b
$env:RUN_HYBRID_DOCKER_TESTS = '1'
python -m pytest -q
Remove-Item Env:\RUN_HYBRID_DOCKER_TESTS
```

Integration tests dùng token **giả rõ ràng**, không gọi HF thật; override remote
bootstrap chỉ trong fixture để kiểm thử `compose up/down`, secret mount và
HTTP 503 không model. Test riêng xác nhận thiếu secret khiến `up` bị từ chối.
Các test này cần daemon, image Phase 7B đã build và port 8001 rảnh. Không truyền
token thật vào pytest. Docker audit kiểm tra pip check, imports/CPU provider,
không weights, cache, secret hoặc Torch/Transformers trong image sạch.

Private inference/cold-start RAM/latency vẫn cần tự chạy với revision thật,
token hợp lệ và ONNX thật. Không suy diễn từ token giả hoặc HTTP 503. Image
code/dependencies có thể được xem xét public ở giai đoạn sau; model chứa
DINOv3 vẫn Private, không upload/đóng gói weights để public. Phase 7B chưa tạo
workflow CI/CD, push Docker Hub hay deploy.

Bổ sung ảnh PDF: Compose version; base tag/digest; lock exact versions/hash;
config --quiet exit code 0; compose build và pip check; pytest; compose ps/logs
đã lọc; HTTP health/predict thật nếu đã chạy; stats/VmHWM; compose down. Với
smoke fixture ghi rõ “không model, 503 kỳ vọng”. Tuyệt đối không chụp token,
SecureString conversion hoặc output environment đầy đủ.

Các packages đã khóa (đầy đủ, kể cả transitive):

```text
annotated-doc==0.0.5
annotated-types==0.8.0
anyio==4.15.1
certifi==2026.10.23
charset-normalizer==3.5.2
click==8.5.0
fastapi==0.162.0
filelock==4.2.5
flatbuffers==25.12.19
fsspec==2026.9.0
h11==0.16.0
hf-xet==1.8.0
huggingface-hub==0.36.2
idna==3.20
numpy==2.5.4
onnxruntime==1.31.1
opentelemetry-api==1.45.2
packaging==26.3
pillow==12.3.0
protobuf==7.36.2
pydantic==2.14.2
pydantic-core==2.50.2
python-multipart==0.0.32
pyyaml==6.0.3
requests==2.34.2
starlette==1.10.0
tqdm==4.70.1
typing-extensions==4.16.0
typing-inspection==0.4.4
urllib3==2.8.0
uvicorn==0.54.0
```

Kiểm chứng Phase 7B trong Codex ngày 2026-10-31: 145 pytest passed (gồm real
Docker/Compose opt-in), hai cảnh báo timm deprecation; Compose config quiet,
lockfile build, pip check và image audit passed. `compose up/down` thực tế với
fixture secret giả và remote loading tắt đã passed, xác nhận RAM 536870912 bytes
và NanoCPUs 1000000000. Không phải kiểm chứng Private inference.
Compose build thông thường bị giới hạn proxy/DNS của Codex; build đã thành
công khi dùng override **ngoài Git** cho proxy/CA công khai giống mục 1.
Final image từ Compose: `sha256:4c68c9f08cee3f1118e2b85582e7ec3f73d7df942de370648ffa442543ef023c`,
Docker báo size 293181069 bytes. Chưa đo RAM/inference với ONNX thật trong Docker.
