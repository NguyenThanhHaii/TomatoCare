# Phase 6 — private HF bootstrap for the experimental Hybrid API

Source branch: `feature/hybrid-dinov3`. No baseline production, frontend, main branch, baseline checkpoint or existing `render.yaml` is changed. `render.hybrid.yaml` is a separate, unapplied configuration with automatic deployment disabled. This phase does not create a service, deploy, upload weights or store a token in Git.

## Evidence and what is not reproduced here

The owner has verified the following on Kaggle with the real Hybrid artifact:

- Accuracy 98.2927%, Macro-F1 98.0715%.
- PyTorch/ONNX parity passed on 12 distinct-SHA validation images.
- Real Hybrid FastAPI/Uvicorn passed; peak RSS 240.07 MB and mean HTTP latency 180.18 ms on Kaggle CPU.
- Read permission for the private HF manifest was verified using HF_TOKEN.
- Tested ONNX SHA-256: `196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4`.

These are owner-supplied Kaggle results, not measurements on Render or fresh Codex results. Codex has no private HF_TOKEN/artifact or operator-selected HF revision. Private downloads, real cold-start time, peak memory during authenticated download/session construction and Render's 0.1 CPU latency therefore remain unverified here. Mock tests do not establish those claims.

## Local compatibility and remote startup

Phase 5 remains supported: with `HYBRID_HF_REPO` unset, `HYBRID_MODEL_DIR` points directly at the local bundle. No HF request or token is needed. `HYBRID_EXPECTED_ONNX_SHA256` is optional in local mode and adds a trusted external digest check if supplied.

Set `HYBRID_HF_REPO` to enable remote mode. Required remote settings:

| Environment variable | Value/source |
| --- | --- |
| `HYBRID_HF_REPO` | `ThanhHai25/TomatoCare-Hybrid-ONNX-Private` |
| `HYBRID_HF_REVISION` | Full 40-character lowercase immutable commit SHA from this HF model repo; required, no `main`, tags or short SHA |
| `HF_TOKEN` | Fine-grained read credential, entered securely in Kaggle/Render, never code or CLI arguments |
| `HYBRID_EXPECTED_ONNX_SHA256` | `196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4` |
| `HYBRID_MODEL_DIR` | Ephemeral bootstrap parent, e.g. `/tmp/tomatocare-hybrid`; this is also the remote default |

In remote mode the parent is not used as a direct preexisting bundle. The downloader creates a private staging directory, downloads exactly `tomatocare_hybrid.onnx`, `inference_manifest.json` and `DINOv3-LICENSE.md` with **the same pinned revision and explicit token**, and copies only those files into a candidate bundle. SDK metadata and credentials are not promoted. Model SHA-256 must equal both the manifest and trusted expected digest. The manifest schema and accompanying nonempty UTF-8 DINOv3 license are checked. The license is sourced from the pinned revision; no separate trusted license digest was supplied, so it is not described as independently hash-pinned.

Only after those checks does one atomic same-filesystem rename publish a unique `verified-...` directory. The ONNX constructor rechecks manifest/trusted digest and actual file hash before creating its CPU session. Neither partial downloads nor previous verified folders are used as fallback on an error. Legacy local files in the parent are not overwritten. The session is then reused for all predictions.

Each new process/startup deliberately authenticates and downloads the three files again (`force_download=True`), even if old files remain. There is no network model refresh per request. This avoids hiding revoked/invalid credentials or network errors by silently reusing an old local bundle. A killed process may leave an unused staging directory; later startup never treats it as activated. All files live on ephemeral storage; no persistent disk is needed.

Remote initialization runs in a background startup task. `/health` returns 503 while loading or after an error; it becomes 200 only when the fully initialized session is available. Ready health includes `artifact_source=private-hf`, the pinned `hf_revision`, `onnx_sha256` and model version. `/predict` returns 503 before readiness and otherwise preserves the Phase 5 response/preprocessing contract. One worker, one session, one intra/inter-op thread and the existing low-memory options are retained.

## Fine-grained read token and immutable revision

1. In the HF account authorized for the private repository, open **Settings → Access Tokens → Create new token**, choose **Fine-grained**.
2. Grant read/download access only to `ThanhHai25/TomatoCare-Hybrid-ONNX-Private`. Do not grant write, admin, broad account access or unrelated gated model permissions for this runtime. The server downloads the already exported ONNX; it does not load the original Facebook encoder.
3. Store the value as Kaggle Secret `HF_TOKEN` for integration testing and as a secure runtime Environment Variable in Render later. Do not put it in source, `.env` committed to Git, screenshots, build commands or URLs. No global `hf auth login` is necessary.
4. In the private repository's Files/Versions commit history, locate the commit containing **all three tested files** and copy its full commit SHA. This is an **HF repository SHA**, not the GitHub application commit, ONNX file hash, or a branch name.

Alternatively resolve the current HF snapshot in an authorized Kaggle cell, printing only the resulting SHA:

```python
import re
from kaggle_secrets import UserSecretsClient
from huggingface_hub import HfApi
secret = UserSecretsClient().get_secret("HF_TOKEN")
try:
    try:
        revision = HfApi().model_info(
            "ThanhHai25/TomatoCare-Hybrid-ONNX-Private", token=secret
        ).sha
    except Exception:
        raise RuntimeError("Private revision lookup failed; check scoped read permission") from None
finally:
    del secret
assert re.fullmatch(r"[0-9a-f]{40}", revision)
print(revision)  # Nonsecret immutable revision only.
```

The lookup is only to choose a snapshot. Deployment must use the resulting immutable SHA and must pass the expected ONNX hash gate. If the latest commit is not the tested bundle, choose the correct historical commit instead; never bypass the checksum to make startup pass. Token rotation requires a service restart. Do not enable HTTP/SDK debug tracing of Authorization headers or signed download URLs.

## Render Free configuration (prepared only)

`render.hybrid.yaml` describes an independent Python 3.12 experimental service. Its `HF_TOKEN` and `HYBRID_HF_REVISION` have no saved value (`sync: false`); supply them securely when manually configuring the service after review. Set `HYBRID_CORS_ORIGINS` to explicit approved client origins or leave it empty for no browser cross-origin access. Wildcard `*` is rejected; CORS is not authentication and does not make a public endpoint private.

Build command:

```bash
pip install -r hybrid_backend/requirements.txt
```

Start command:

```bash
uvicorn hybrid_backend.main:app --host 0.0.0.0 --port "$PORT" --workers 1
```

The `:app` suffix is required by Uvicorn; `uvicorn hybrid_backend.main` alone is not a valid import string. Use one worker only. The runtime has seven direct dependencies including `huggingface_hub`; neither PyTorch nor Transformers is installed. The configuration sets ephemeral HF storage, one-thread environment settings, disables progress bars and disables optional Xet acceleration to favor the standard streaming path. It does not guarantee a particular memory peak.

Free storage is ephemeral, so instances can download the 90 MiB model again after a cold start. Startup includes transfer, validation and graph construction; measure its duration separately from steady-state HTTP latency. The previously measured 240.07 MB Uvicorn peak does not automatically include this new bootstrap path. Download buffers, file/page cache, session initialization and concurrent HTTP traffic must still fit the actual 512 MB service/container budget. Kaggle latency is not Render 0.1 CPU latency. Readiness must remain 503 on failure; do not remove hash checks or substitute the baseline to obtain a green health check.

## Actual private integration test on Kaggle

Clone/update the feature source. Prefer a fresh ONNX-only environment for measuring the complete service, not the training notebook process with PyTorch already loaded. The private artifact is fetched by the child server; do not inject a mocked predictor.

```bash
python -m venv /kaggle/working/hybrid-private-runtime
/kaggle/working/hybrid-private-runtime/bin/python -m pip install -r hybrid_backend/requirements.txt
```

From the checkout root, configure nonsecret values and retrieve the secret only in memory:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient

os.environ["HYBRID_HF_REPO"] = "ThanhHai25/TomatoCare-Hybrid-ONNX-Private"
os.environ["HYBRID_HF_REVISION"] = revision  # The verified full HF commit SHA above.
os.environ["HYBRID_EXPECTED_ONNX_SHA256"] = "196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4"
os.environ["HYBRID_MODEL_DIR"] = "/kaggle/working/hybrid-private-cache"
os.environ["HF_HOME"] = "/kaggle/working/hybrid-hf-cache"
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
try:
    subprocess.run([
        "/kaggle/working/hybrid-private-runtime/bin/python", "-m", "scripts.check_private_hybrid_api",
        "--image", "/actual/path/to/a/validation/leaf.jpg",
        "--runs", "10", "--startup-timeout", "300",
        "--report", "/kaggle/working/private_hybrid_api_report.json",
    ], check=True)
finally:
    os.environ.pop("HF_TOKEN", None)
```

The checker starts a fresh actual Uvicorn process with one worker, observes pending 503/readiness, validates the private revision and full ONNX hash, then posts real uploads. It records download/startup time, HTTP latencies and Linux process high-water RSS including model initialization. No SDK exception text, credentials or raw server log is written to the report. It exits nonzero with `status=blocked` on failure and never invents prediction timing. RSS from a failed startup is not successful loaded-model RSS. Use a fresh report path, and `--port` if the default 18002 is occupied.

For repeat cold-start trials, use new process/report/cache directories and record network/cache conditions; there is intentionally no old-bundle fallback. Peak process RSS is not total cgroup/container memory: inspect Render memory metrics and repeated HTTP behavior at the target limit before declaring readiness. Retest failure cases in a disposable environment (missing/invalid token, unavailable revision, trusted digest mismatch), never by changing a working production service. Review only credential-redacted health messages; do not paste secrets in diagnostics.

## Security, distribution and validation boundary

Bootstrap emits curated status messages for missing credentials, HTTP 401/403/404, timeout, interrupted downloads, invalid manifest/license and digest mismatch. Raw SDK errors and causal chains are suppressed because they can contain auth headers/signed URLs. Requests have no model download path. The API has no file-serving route, and images are not stored permanently. The staged download stores only private artifact bytes and normal SDK metadata, not a login token; only the three validated files are activated.

Private HF storage is not automatic permission for public redistribution or public model hosting. Preserve `DINOv3-LICENSE.md`, review the current accepted DINOv3 agreement/model terms (including derivative distribution and acknowledgment conditions) before any public-facing deployment, and decide access control separately from CORS. This source-preparation step does not publish weights, provide license clearance, create a Render service or deploy anything.

Offline tests mock private reads and use synthetic tiny artifact bytes only. They exercise missing/invalid credentials, immutable revisions, wrapped 401/403 errors, timeout/interruption, corrupt files, trusted-hash mismatches, atomic activation, startup pending/error health, session reuse, secret redaction and Phase 5 local compatibility. No private weights or real token are included. Run the full suite with the existing dev requirements plus the updated Hybrid requirements:

```bash
python -m pip install -r requirements-dev.txt -r hybrid_backend/requirements.txt
python -m pytest -q
```
