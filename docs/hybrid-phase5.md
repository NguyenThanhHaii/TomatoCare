# Phase 5 — experimental private-bundle Hybrid ONNX API

This is an independent research service on `feature/hybrid-dinov3`. It does not replace `backend.main:app`, change the production checkpoint, alter React, change training, merge into main or deploy anything. The existing frontend remains pointed at its baseline API.

## Private artifact contract

Set `HYBRID_MODEL_DIR` to the actual directory containing:

```text
private-bundle/
  tomatocare_hybrid.onnx
  inference_manifest.json
  DINOv3-LICENSE.md
```

Unpack the trusted Kaggle bundle outside the checkout. Do not place it in public source control or publicly upload its weights. If the ZIP contains a nested folder, configure that inner folder, not the ZIP itself or its parent. No model is automatically downloaded or uploaded.

The manifest supplied by the owner is retained as a text-only [example](examples/hybrid_inference_manifest.json). Its expected SHA-256 is `196c46f385414800a9233650661ef102d3407bbaca0a69da908222c4f504f4c4`. The runtime reads the actual local manifest and checks the **entire ONNX file hash before constructing any session**. A missing file, hash mismatch, unsupported preprocessing or invalid graph signature keeps health unavailable with a clear error; there is no baseline/random/mock fallback.

The fixed graph contract is two float32 inputs, named `baseline_images` and `dino_images`, both `[1,3,224,224]`, and one float32 `[1,4]` logits output. Logit order is strictly Healthy, Early Blight, Late Blight, Leaf Mold. It is never sorted alphabetically. Softmax is computed on the actual graph logits. Since the supplied manifest has no `model_version`, the version defaults to `hybrid-` plus the first 12 SHA characters; an explicit nonempty manifest version is also supported. Historical test metrics in the manifest are never used to generate predictions.

## Exact two-path preprocessing

Both branches start with the same decoded, EXIF-oriented RGB image. Inference has no random training augmentation.

- **UniRepLKNet:** PIL uint8 bilinear resize exactly 224×224, divide by 255, normalize with that branch's manifest `mean`/`std`.
- **DINOv3ViTImageProcessorFast:** float32 rescale first using `rescale_factor`, antialiased float tensor resize using the manifest `size`/`resample`, optional center crop, then normalize using its own `image_mean`/`image_std`. Nullable crop/RGB flags from the supplied manifest are accepted; RGB is already established at decode. No crop is applied for the supplied null crop flag.

The two normalizations are read independently even though the current manifest has equal values. The tensors can still differ because PIL rounds resized uint8 pixels while the Fast processor resizes floats after rescaling. Reusing one tensor for both encoders would not preserve this behavior.

A NumPy separable resize implements float32 half-pixel coordinates and antialias filtering without a Torch dependency. It has oracle tests against Torchvision float resize for bilinear, bicubic, nearest, up/downsampling, shortest-edge resize, cropping and padding. Sparse filter support avoids dense BLAS multiplications/extra CPU pools. The supported Fast-processor contract requires rescale-before-resize (`do_rescale=true`); an unsupported processor/order fails instead of silently approximating. Final tensors must be finite contiguous float32 `[1,3,224,224]`.

These unit comparisons do not establish full trained-model parity. The Kaggle integration gate below independently uses the actual pretrained processor/training baseline transform and tests both pixel tensors and logits across multiple validation images.

## Run the experimental backend

Use a fresh Python 3.12 environment, separate from the training environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r hybrid_backend/requirements.txt
HYBRID_MODEL_DIR=/absolute/path/to/private-bundle \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m uvicorn hybrid_backend.main:app --host 127.0.0.1 --port 8001 --workers 1
```

Only six direct runtime dependencies are needed: FastAPI, Uvicorn, python-multipart, Pillow, NumPy and ONNX Runtime. There is no mandatory Torch, Transformers or HF SDK. The runtime was installed/imported in an isolated environment with all three absent. `.env.example` documents variables; it is not automatically loaded. Set explicit `HYBRID_CORS_ORIGINS` only for an intentional experimental client; the current React application is not changed or switched.

One session is initialized during app lifespan and reused. Settings match the stated low-memory controls: CPUExecutionProvider, one intra-op and inter-op thread, sequential execution, CPU arena disabled and memory pattern disabled. Other graph optimizations retain the runtime default; no quantization or replacement model is introduced. One worker is required. One prediction is admitted at a time; concurrent predictions receive 429.

```bash
curl -f http://localhost:8001/health
curl -f -F 'file=@/actual/path/leaf.jpg' http://localhost:8001/predict
```

`/health` returns 200 only with a loaded, SHA-verified graph and includes `experimental`, `model_version`, runtime and hash-verification status. `/predict` returns `predicted_class`, `confidence`, all four `probabilities` and `model_version`. Unsupported format is 415, corrupt/empty image 422, files above 10 MiB 413, and unavailable/failed inference 503. The baseline's image validator and body limiter are reused without starting its app lifespan or changing its source. JPG/PNG content, EXIF, RGB and the shared 20-megapixel safety limit and an experimental 8192-pixel per-side limit are checked. The latter bounds resampling workspace for extreme aspect ratios; no target-host memory guarantee is implied. Uploads are closed, images are not stored permanently, and responses use `Cache-Control: no-store`.

## Real multi-image parity gate on Kaggle

The code is in `scripts/check_hybrid_onnx.py`. Work from the feature branch source, attach the private ONNX bundle and the original TomatoCare dataset, and enable authorized gated access only for the **reference PyTorch reconstruction**, not for the ONNX service.

```bash
git clone --branch feature/hybrid-dinov3 --single-branch \
  https://github.com/NguyenThanhHaii/TomatoCare.git /kaggle/working/TomatoCare-phase5
cd /kaggle/working/TomatoCare-phase5
python -m pip install -r requirements-hybrid.txt -r hybrid_backend/requirements.txt
```

If the trained PyTorch `model` and its actual HF `processor` are still present in the Kaggle notebook, use them directly:

```python
from scripts.check_hybrid_onnx import compare_loaded_model
report = compare_loaded_model(
    model, processor,
    model_dir="/actual/private/bundle-directory",
    data_root="/kaggle/input",
    output_path="/kaggle/working/hybrid_parity_multi_image.json",
    per_class=3, device="cpu",
)
print(report["passed"], report["images_checked"], report["max_absolute_logits_error"])
```

The reference must be the genuinely trained HybridModel with matching recorded encoder provenance. Alternatively reconstruct the original Phase 4 head-only checkpoint; it is **not** contained in the three-file ONNX bundle. Use a Kaggle secret in memory, never print it:

```python
import os
from kaggle_secrets import UserSecretsClient
# For CLI use in a notebook subprocess, expose the secret only temporarily.
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
try:
    import subprocess, sys
    subprocess.run([
        sys.executable, "-m", "scripts.check_hybrid_onnx",
        "--model-dir", "/actual/private/bundle-directory",
        "--head-checkpoint", "/actual/path/best_hybrid_unireplknet_a_dinov3.pth",
        "--data-root", "/kaggle/input",
        "--report", "/kaggle/working/hybrid_parity_multi_image.json",
        "--per-class", "3", "--device", "cpu",
    ], check=True)
finally:
    os.environ.pop("HF_TOKEN", None)
```

A trusted original baseline file may be supplied with `--baseline-checkpoint`; otherwise the baseline and DINO sources are reconstructed at the manifest's resolved revisions. `restore_head()` checks the trained head format/source identities. The CLI never invents a reference model. If only the ONNX bundle is available with no trained PyTorch head/reference, ONNX inference can be tried but PyTorch parity cannot be verified.

The default gate checks **12 distinct-SHA validation images, three from each class**, and never samples test images. It checks independent original Torchvision baseline preprocessing and the real HF processor against NumPy inputs (`atol=rtol=1e-5`), then actual PyTorch/ONNX logits (`atol=rtol=1e-4`) plus the same predicted class on every sample. It records per-image errors, maximum logit error and artifact hash. Any mismatch writes a failed report and raises an error/nonzero CLI exit. Fewer than four images are refused; an existing report path is not reused. Do not publish successful multi-image parity until this gate actually passes with the trained artifact. Passing this finite set is not a proof for every possible input.

## Tests and evidence boundary

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

New tests cover supplied manifest schema, independent branch transforms, exact class order, checksum-before-session, session options/cache, invalid graph/logits, API errors, EXIF, and the validation harness. Most tests mock the session explicitly. One tiny synthetic graph verifies real ORT two-input plumbing only; it is not the trained Hybrid ONNX and is created only in temporary test storage. The parity harness is tested with an explicit mock reference, not presented as trained-model evidence.

Owner-reported results: Accuracy 98.2927%, Macro-F1 98.0715%, 90.21 MiB ONNX, 177.84 MB Kaggle CPU peak RSS and 139.07 ms mean latency. The reported 4.77e-6 max logit error is from **one validation image** only. None of those measurements have been independently reproduced in Codex. The actual ZIP/ONNX is absent here, so its SHA, signature, end-to-end accuracy, real multi-image parity, full FastAPI RSS and target-host performance remain unverified. Actual startup without the private bundle returns 503 rather than a mock prediction. No Render/Vercel deployment or frontend switch has occurred.

## Distribution and dataset limitations

The private bundle includes DINOv3 material and its license; retain the accompanying agreement. The [reviewed DINOv3 License](third-party/DINOv3-LICENSE.md) is Meta's custom agreement, not a blanket Apache/MIT grant for the model. Section 1.b.i subjects redistribution of materials/derivatives to its terms and requires a copy; research publication acknowledgment is addressed separately. Review the current accepted gated/model-specific terms before proposing any public model hosting. This phase proposes only private/local experimentation, no public HF upload or model hosting. A checksum proves file identity, not distribution permission or parity.

Exact duplicate images within a split/class and unverified `leaf_id` independence remain dataset limitations; the new accuracy figures do not remove potential dataset bias. Test data must not be used to choose preprocessing tolerances or model settings. No weights, ZIP, dataset, token or secret `.env` should enter Git; the source ignore rules cover model/archive artifacts.
