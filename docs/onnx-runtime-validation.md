# ONNX production handoff evidence

## Model-owner Kaggle report (not reproduced in Codex)

Model repo: `ThanhHai25/TomatoCare-UniRepLKNet-A`, artifact `best_unireplknet_a.onnx`.

| Check | Owner-reported result |
| --- | --- |
| Validation parity | 12 images, maximum absolute logit error 8.106e-6 |
| Default | Peak RSS 108.76 MB, mean latency 28.63 ms |
| Low Memory | Peak RSS 90.42 MB, mean latency 91.33 ms |
| Input | float32 [1,3,224,224] |
| Logit order | Healthy, Early Blight, Late Blight, Leaf Mold |

This establishes the owner's Kaggle observation. It does not measure Render's 0.1 CPU or total API/container memory. The implementation's configurable Low Memory profile may differ from the owner's benchmark settings.

## Codex real-artifact attempt

The HF API and actual production downloader both return HTTP 403. Running the full-process benchmark with production dependencies only produced:

- Status: **blocked**, exit code 2.
- Successful HTTP predictions: **0**.
- Inference latency: **not measured**.
- Peak API process RSS: **88.90 MiB during failed startup**, with no loaded model/session inference.

This number must not be presented as successful real-model memory usage. Production returned an explicit 503 model-unavailable response. No mock fallback occurred.

## Dependency footprint

`backend/requirements.txt`: seven direct dependencies, 217 bytes of declarations. The tested isolated ONNX environment uses approximately 219 MiB on disk (including package tooling). PyTorch is absent; backend import succeeds. Disk usage is not RAM usage.

## Remaining gates

Real inference and full API RSS must be checked where the HF model is accessible, then on the actual Render Free service. UI, architecture and training pipeline remain unchanged. No commit, push or deploy has been performed as part of this handoff.
