# Phase 7C — GitHub Actions CI/CD và Docker Hub

Workflow: [hybrid-ci-cd.yml](../.github/workflows/hybrid-ci-cd.yml), tên hiển thị
**Hybrid Docker CI/CD**. Chỉ source/code image của Hybrid thử nghiệm được public
ở `nguyenthanhhaiz/tomatocare-hybrid`. Không merge main, deploy Render/Vercel,
chuyển frontend hoặc thay backend baseline production.

CI (continuous integration) kiểm thử source và image mỗi thay đổi phù hợp.
CD ở giai đoạn này là continuous **delivery vào Docker Hub**, không phải tự
triển khai một dịch vụ nhận request. Push source có thể tự kích hoạt publish
khi các cổng kiểm tra đạt và credentials hợp lệ.

## 1. Triggers và ranh giới quyền

- `push` vào `feature/hybrid-dinov3`: chạy CI. Chỉ repository gốc
  `NguyenThanhHaii/TomatoCare` được đi tiếp sang publication.
- `pull_request` có **base branch** `main` hoặc `feature/hybrid-dinov3`: chạy CI
  trên merge ref của PR, kể cả PR từ fork; tuyệt đối không login/publish.
- Push main không kích hoạt workflow này. Không dùng `pull_request_target`,
  `workflow_run` hoặc workflow_dispatch để vượt cổng publish.
- GITHUB_TOKEN chỉ có `contents: read`, không cấp packages/actions/contents write
  hoặc id-token. Checkout đặt `persist-credentials: false`.
- Concurrency theo workflow ref, không tự hủy run đang chạy. Tag phase7c có thể
  thay đổi; khi cần tái lập phải dùng commit tag hoặc registry digest.

## 2. Job `ci`: tests, build, audit không có mô hình Private

Runner Ubuntu 24.04, Python 3.12, timeout 30 phút. Các bước bắt buộc:

1. Checkout đúng source của event, thiết lập Python.
2. Cài Torch 2.5.1 / torchvision 0.20.1 từ index CPU của PyTorch **trên runner
   kiểm thử**. Cài runtime lock với hash verification; tạo constraints exact
   versions từ lock rồi cài `requirements-ci.txt`. Pins cho test host đã được
   đối chiếu môi trường Python 3.12 đang chạy. Chạy pip check. Không thay lock
   hoặc requirements huấn luyện; các thư viện nghiên cứu không vào Docker image.
3. `docker compose version` và `docker compose config --quiet` với revision/hash
   giả rõ ràng. Không truyền HF credential, không chạy remote bootstrap.
4. Docker official setup-buildx/build-push Actions build `Dockerfile.hybrid`,
   `linux/amd64`, **load:true, push:false**, không login/build secret/build args.
   Base Python 3.12.16 slim/digest và requirements.lock.txt giữ nguyên Phase 7B.
   Thêm OCI source/revision labels để truy nguồn GitHub SHA.
5. `pip check` trong image; chạy `python -m scripts.check_hybrid_docker` để kiểm
   imports, CPU provider, user non-root, không Torch/Transformers, không weights,
   secret, HF cache hoặc .env. Chạy Uvicorn thật không model dưới 512 MiB/1 CPU.
   `/health` và `/predict` **phải trả 503**. Đây là packaging/error-path smoke,
   không được ghi là AI ready, không thay thế kiểm thử ONNX thật.
6. Toàn bộ `pytest -q` với `RUN_HYBRID_DOCKER_TESTS=1`: bao gồm real Docker và
   Compose integration. Compose tests tạo token fixture giả và tắt remote
   loading chỉ trong override của test, không gọi HF Private. Plugin Compose
   phải có sẵn vì bước 3 thất bại nếu thiếu. Source test bao gồm cổng publish.
7. Chỉ với push feature của repo gốc, **sau khi audit và pytest đạt**, docker save
   image đã kiểm thử, upload một artifact tar riêng cho run (retention 1 ngày).
   Không upload thư mục workspace/log/env/cache. PR và fork không export image.

Không có continue-on-error, không bỏ qua exit code của bài kiểm tra bắt buộc.
Bất kỳ bước lỗi nào sẽ làm CI fail, ngăn job publish. Runtime image không có
model hoặc token; tar trao đổi giữa hai job cũng chỉ chứa code/dependencies.
Không cache Docker build trên runner chung hoặc tải cache không tin cậy.
`requirements-ci.txt` pin direct test tools; không phải lock toàn bộ transitive
research dependencies. Lock production runtime 31 packages vẫn đầy đủ/hash-verified.

## 3. Job `publish`: phát hành chính image đã audit

Điều kiện job đồng thời là `needs.ci.result == success`, event push, repository
chính xác `NguyenThanhHaii/TomatoCare`, ref chính xác
`refs/heads/feature/hybrid-dinov3`. Không publish từ fork/PR/main.

1. Chỉ kiểm tra **boolean có/không** của hai repository secrets. Không in hoặc
   truyền giá trị vào bước kiểm tra. Thiếu secret: các bước publication skip,
   Step Summary/notice ghi không có tags được push. Job xanh trong trường hợp
   này **không có nghĩa đã publish**; phải kiểm tra step Push và Docker Hub.
2. Khi có đủ secret, checkout source, setup Python rồi download artifact từ
   job CI **cùng run**. Không chọn artifact từ PR/run/repo khác.
3. Docker load image; đối chiếu OCI revision label với GITHUB_SHA; chạy lại
   toàn bộ image audit/smoke trước login. Không build lại sau audit.
4. Docker official login-action dùng hai secrets, tự logout cuối job. Token chỉ
   được đưa vào action login, không dùng làm ARG/ENV của image/container.
5. Tag và push `sha-<FULL_GITHUB_SHA>` và `phase7c`, rồi inspect manifest tag SHA.
   Cả hai trỏ đến image đã kiểm thử. Không dùng hoặc sửa `latest`.

Nếu auth/token thiếu quyền Write, quota, registry/network, artifact retention
hoặc push lỗi: job fail/skip tương ứng, không tuyên bố đã publish. Có thể một tag
đã push trước khi tag thứ hai lỗi; xác minh cả hai rồi rerun. Rerun all jobs
được hỗ trợ bằng artifact overwrite cùng tên trong run; rerun failed publish
có thể tái dùng artifact CI nếu chưa hết retention. Sau 1 ngày hãy rerun all jobs
để tạo lại artifact. Docker Hub không tự bảo đảm bất biến chỉ vì tên có SHA;
kiểm tra digest và cấu hình immutable tags nếu muốn ràng buộc ở registry.

## 4. Secrets và image public/model Private

Trong GitHub: Settings → Secrets and variables → Actions → Repository secrets:

- `DOCKERHUB_USERNAME`: tài khoản có quyền push repository đích.
- `DOCKERHUB_TOKEN`: Docker Hub access token có quyền Read/Write phù hợp, hạn
  chế phạm vi khi loại tài khoản/token hỗ trợ. Không dùng password đăng nhập.

Không đưa token vào Git, `.env`, workflow literal, logs, build args hoặc image.
Không thêm HF_TOKEN secret vào Actions. CI không có quyền model Private.
Giá trị secret chỉ ở login-action; bước kiểm tra hiện diện chỉ nhận boolean.
Không dump environment/config Docker hoặc bật shell `set -x` quanh auth.
Không chụp màn hình giá trị token, trang tạo token hoặc log xác thực chi tiết.
Có thể chụp **tên** secrets với giá trị bị GitHub ẩn.

Image public chứa backend code + ONNX Runtime + dependency/license cần thiết,
không chứa ONNX/PyTorch weights, safetensors, dataset, HF cache hoặc token.
Không docker commit container đã tải mô hình. HF Private bundle/DINOv3 weights
vẫn Private; không upload weights lên Docker Hub hoặc HF Public. Giữ license
DINOv3 và rà soát điều khoản phân phối trước mọi thay đổi phạm vi công khai.

Khi chạy image đã pull, mô hình chỉ tải khi người chạy cung cấp runtime token,
immutable HF revision và expected SHA theo [hướng dẫn Docker](hybrid-docker.md).
Truy cập Docker Hub public không cấp quyền HF Private. 503 không model là đúng,
200 AI ready chỉ sau tải artifact thật và kiểm tra hash/session thành công.

## 5. Actions đã pin SHA

Đã đối chiếu tag cụ thể với remote Git chính thức bằng `git ls-remote` và kiểm
tra action.yml cho các action build/artifact. Các Actions đều là GitHub/Docker
chính thức, chạy Node 24 trên hosted runner hiện đại. SHA pin code Action;
không có nghĩa runner OS/services ngoài GitHub/Docker Hub cũng bất biến.

| Action | Version | Immutable commit SHA |
| --- | --- | --- |
| actions/checkout | v7.0.1 | 3d3c42e5aac5ba805825da76410c181273ba90b1 |
| actions/setup-python | v7.1.0 | 6d4fd22b20465ccc0f7354476588e8df7c9ca0e0 |
| docker/setup-buildx-action | v4.4.1 | f87e5991a6d7451dcb8d9637bfbc97413f497069 |
| docker/build-push-action | v7.4.0 | c3c9e263c25d99ce0380d002d59b67737d91b0dc |
| actions/upload-artifact | v7.0.2 | cf430e030ddbb5b0abf93d22962f4752f3646cd9 |
| actions/download-artifact | v8.0.2 | 9000827ccba6bdab643e8b6fd33ac0654aef8333 |
| docker/login-action | v4.7.0 | 81c6f93d4bc991e510b342a00b419b3bb4efc5c6 |

Ví dụ tự đối chiếu: `git ls-remote https://github.com/docker/login-action.git
refs/tags/v4.7.0`. Cập nhật pin phải review version/source và chạy validation.

## 6. Tìm run, tags và pull image

1. Mở [GitHub Actions workflow](https://github.com/NguyenThanhHaii/TomatoCare/actions/workflows/hybrid-ci-cd.yml).
   Chọn run thuộc branch feature và commit vừa push. Mở job ci/publish, xem
   từng bước và Step Summary. PR chỉ có CI, publish phải skipped.
2. Nếu workflow không chạy: kiểm tra Settings → Actions có bật, policy cho phép
   official Actions đã pin, runner/billing/quota và tên branch. Không đổi sang
   token quyền rộng để vượt lỗi. Workflow mới trên feature chạy bằng push;
   không cần merge main để kích hoạt push event.
3. Mở [Docker Hub tags](https://hub.docker.com/r/nguyenthanhhaiz/tomatocare-hybrid/tags).
   Xác minh **cả hai tag**, thời gian, digest và full Git commit. Không chụp token.
4. Pull và audit trên PowerShell (không cần HF token cho bước này):

```powershell
$commit = '<FULL_SUCCESSFUL_GITHUB_COMMIT_SHA>'
docker pull "nguyenthanhhaiz/tomatocare-hybrid:sha-$commit"
docker image inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' "nguyenthanhhaiz/tomatocare-hybrid:sha-$commit"
docker run --rm --entrypoint python "nguyenthanhhaiz/tomatocare-hybrid:sha-$commit" -m pip check
python -m scripts.check_hybrid_docker --image "nguyenthanhhaiz/tomatocare-hybrid:sha-$commit"
# Tag thử nghiệm có thể đổi ở lần push thành công tiếp theo:
docker pull nguyenthanhhaiz/tomatocare-hybrid:phase7c
```

Có thể pull `nguyenthanhhaiz/tomatocare-hybrid@sha256:<REGISTRY_DIGEST>` để chọn
chính xác nội dung registry. Không nhầm Docker image ID với registry digest.
Để chạy AI thật, dùng lệnh docker run ở tài liệu Docker, thay tên image cuối
lệnh bằng tag SHA vừa pull. Truyền token chỉ runtime. Compose hiện vẫn có build
local/tag phase7b; phase này không tự đổi Compose thành pull-only/production.

## 7. Kiểm thử local và ảnh PDF

```powershell
# Chuẩn bị dependencies test theo workflow hoặc dùng môi trường test hiện có.
python -m pytest -q tests/test_hybrid_workflow.py
# Cần image local phase7b và Docker/Compose để bật integration đầy đủ:
$env:RUN_HYBRID_DOCKER_TESTS = '1'
python -m pytest -q
Remove-Item Env:\RUN_HYBRID_DOCKER_TESTS
```

Có thể lint Actions YAML bằng actionlint 1.7.7 (ngoài runtime image). Trong Codex
đã chạy validator thật từ image tool riêng, bên cạnh pytest kiểm tra trigger,
permissions, pins, publish gates và secret boundaries. Các kiểm tra local không
mô phỏng đầy đủ GitHub scheduler, artifact service, secret injection hoặc quyền
Docker Hub. GitHub API trong Codex trả HTTP 403; phải xác minh run/tag thực tế
trực tiếp trên GitHub Actions/Docker Hub nếu không đọc được từ môi trường này.

Ảnh cho báo cáo PDF, theo thứ tự:

1. Branch/commit và file workflow; giải thích CI khác delivery/deployment.
2. Repository Secrets **chỉ tên** DOCKERHUB_USERNAME/DOCKERHUB_TOKEN, không giá trị.
3. Run overview đúng commit/event/branch, job CI xanh; các bước pip check,
   compose config, build, audit 503 kỳ vọng và toàn bộ pytest.
4. Job publish: download đúng artifact, revision/audit đạt, login thành công
   ở mức trạng thái (không chụp log xác thực), push hai tag thành công.
5. Docker Hub tags SHA/phase7c cùng digest; pull image, inspect revision/pip check.
6. Nếu thử AI thật sau pull: health 200 hash verified, predict 200 và RAM,
   che mọi thông tin xác thực. Ghi riêng khỏi smoke CI không model.
7. Nếu muốn minh chứng PR không publish, mở PR thủ công khi được phép rồi chụp
   publish skipped; Phase 7C không tự tạo PR hoặc merge main.

Sinh viên đã cung cấp minh chứng Windows trước Phase 7C: Private model healthy,
health/predict 200, memory usage 235.7 MiB/512 MiB, OOMKilled=false và down an
toàn. Đây là kết quả do sinh viên báo cáo, không phải kết quả GitHub CI hoặc
peak RSS mới đo trong Codex. Không bịa hình, số liệu hoặc trạng thái publication.

Kiểm chứng local ngày 31/10/2026: tạo venv Python 3.12 sạch và chạy đúng các
lệnh cài dependencies của CI, pip check passed; 156 tests passed, 3 cảnh báo
Starlette/timm deprecation. Actionlint 1.7.7 và YAML/policy tests passed.
Compose config với revision/hash giả không token, Docker build với OCI labels,
pip check/image audit, Uvicorn 503 không model, image save/load và đối chiếu
revision đều passed. Build tại Codex dùng proxy/CA công khai tạm ngoài Git
như Phase 7B; không đưa cấu hình mạng riêng đó vào workflow Ubuntu.
Chưa có bằng chứng GitHub-hosted execution, artifact upload/download qua service,
Docker Hub authentication/push hoặc tag registry từ các kiểm thử local này.
