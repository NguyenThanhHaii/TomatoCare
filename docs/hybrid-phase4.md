# Phase 4 — frozen UniRepLKNet-A + DINOv3 research variant

## Architecture and isolation

Baseline production remains UniRepLKNet-A. No baseline model/vendor code, Phase 2 data/training pipeline, deployed ONNX, FastAPI, React design, deployment requirements or hosting settings are modified by this phase.

```text
one original RGB photo, one shared optional train-time horizontal flip
  ├─ baseline RGB exact bilinear 224×224 + ImageNet normalization
  │   UniRepLKNet-A downsample/stages 0,1,2 → GAP → [B,160]
  └─ original DINOv3 AutoImageProcessor from its pretrained config
      pretrained DINOv3 ViT-S/16 → last_hidden_state[:,0,:] → [B,384]
concat [B,544] → LayerNorm(544) → Linear(544,4) → raw logits
```

The class order is **Healthy, Early Blight, Late Blight, Leaf Mold**. It must agree with both `labels.json` and the trained baseline checkpoint. No alphabetic reorder or softmax is added to the CrossEntropyLoss forward path.

The UniRepLKNet encoder physically owns only `downsample_layers[0:3]` and `stages[0:3]`. It has depths (2,2,6), stage channels (40,80,160), and the original stem/downsampling/block implementations. `downsample_layers[3]`, `stages[3]`, final 320-dimensional norm and baseline classifier are not registered in the hybrid model. The checkpoint's retained keys/shapes must exactly match, then load with `strict=True`; `module.` prefixes, missing retained weights or incompatible class/preprocessing metadata fail explicitly. No other CNN is substituted.

DINOv3 loads with `AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m", use_safetensors=True)`. The model config must be `dinov3_vit`, hidden size 384, patch size 16. The source embedding sequence is `[CLS, register tokens, patch tokens]`; only position zero is used, never register/patch averaging. The model validates the complete token count from input H/W and configured register count. DINOv3 resize/rescale/crop/normalization order is controlled by its own downloaded `AutoImageProcessor`, not the baseline transform. No DINO size or preprocessing is guessed from UniRepLKNet.

Both encoders have `requires_grad=False`, stay eval even when `model.train()` is called, and execute in `torch.no_grad()`. Fusion remains outside that context so its **3,268** parameters receive gradients. Baseline BatchNorm running statistics and DINO dropout stay frozen. No full-encoder fine-tuning, DINOv3 integration into production, ONNX changes or UI work is performed.

## Kaggle notebook

Import `notebooks/train_hybrid_dinov3_kaggle.ipynb`. Enable Internet, select a T4 GPU and attach `thanhhainguyn/tomatocare-tomato-leaf-disease-4-classes`. Enable notebook access to Kaggle Secret `HF_TOKEN`, using the account with the already granted DINOv3 access. Never paste its value in code or outputs.

The notebook checks source availability, installs `requirements-hybrid.txt`, validates all 3,816/816/820 records, loads the public four-class baseline with no auth token, and separately loads gated DINOv3 using the secret in memory. It does not call global HF login, print auth values or put them in checkpoint/config fields. Loading exceptions are redacted instead of echoing SDK exceptions. Missing/mismatched pretrained weights abort creation; no random-pretrained fallback exists.

The notebook clones `feature/hybrid-dinov3` with `--single-branch` into `/kaggle/working/TomatoCare-hybrid`. Production main is not used. Alternatively attach that source checkout as a source-code dataset and point `REPO_DIR` there. If an old/mismatched checkout is present, the notebook fails clearly instead of running a different baseline notebook.

The real-pretrained smoke cell prints both branch shapes, DINO input size, logits and actual GPU peak allocated memory. **T4 batch size 4 is a starting configuration, not a measured guarantee.** If necessary, set 2 or 1 before starting training. Both encoders are frozen/no-grad; no activation graph is kept for them. The FP32 default avoids assumptions about pretrained mixed-precision compatibility.

`RUN_TRAINING=False` and `RUN_FINAL_TEST=False` by default. Running all cells loads weights and performs smoke only; it does not start training or touch held-out test images. Importing Python modules does neither. When explicitly enabled later, training uses AdamW (LR 0.001, weight decay 0.05), five epochs, seed 42, batch 4, zero loader workers, CrossEntropyLoss and validation Macro-F1 selection. Test evaluation is a separate explicit action after settings are frozen; repeated test evaluation is refused if its metrics already exist. Validation/test have no random augmentation. Duplicate-image rows within the same split/class remain intact.

A CLI smoke is also available, with token passed through the process environment securely:

```bash
python -m pip install -r requirements-hybrid.txt
python -m tomatocare.train_hybrid --data-root /kaggle/input --batch-size 4
# --train is required to opt into an actual training run later.
```

This research feature branch does not merge into main or deploy the app. No Kaggle training or model-weight publication is performed by the source-preparation workflow.

## Artifacts and reconstruction

The optional training run writes private outputs under `/kaggle/working/tomatocare_hybrid/`:

- `best_hybrid_unireplknet_a_dinov3.pth`: format `tomatocare-hybrid-head-v1`, **classifier-only** tensors, label mapping, feature dimensions, whitelisted preprocessing, baseline SHA-256, requested/resolved model revisions, epoch and validation metrics.
- `history.json`, validation metrics and confusion-matrix CSV/PNG.
- Test metrics and confusion matrix only when the final-test opt-in is explicitly enabled after model selection.

The artifact intentionally contains **neither encoder's weights**. This is not a standalone full-model checkpoint, and cannot be passed to the deployed baseline inference loader. Recreate the encoders with `create_hybrid_model(...)` using the recorded resolved revisions (or the original local baseline file), verify the baseline hash and DINO source revision, then call `restore_head(model, path)`. Requesting a pinned SHA instead of `main` is allowed when the resolved encoder identities match. Authorized DINOv3 access is still required. No credentials are serialized, and the processor is represented by a transform-only whitelist rather than the complete object/config.

## Fair baseline/hybrid comparison

Keep the existing dataset version, split assignments and label order. Compare the same metrics and macro averaging, same held-out examples and fixed preprocessing per branch. Select both variants using validation only; do not use the reported test baseline to repeatedly tune the hybrid. Record seeds, number of epochs/optimizer steps, batch-size differences, augmentations, source checkpoint hashes/revisions, parameter counts, GPU memory and timing. Several predetermined seeds can support a stronger comparison, with the selection/evaluation protocol fixed before consulting test results.

The hybrid adds a second large pretrained encoder and different pretraining data; any future gain cannot be attributed solely to fusion architecture. The supplied baseline Accuracy 91.46% / Macro-F1 90.51% is a historical user-reported result, not a new run here. No hybrid accuracy or improvement is fabricated.

Dataset SHA-256 groups may contain repeated identical images within a split/class. These can bias sample weighting and evaluation. `leaf_id` independence is not established and near-duplicate leaves may span splits. The hybrid does not resolve these dataset limitations.

## Sources and license review

Reviewed on 2026-10-19:

- UniRepLKNet code: [AILab-CVC/UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet), existing vendored Apache License 2.0. Trained baseline source: [ThanhHai25/TomatoCare-UniRepLKNet-A](https://huggingface.co/ThanhHai25/TomatoCare-UniRepLKNet-A).
- DINOv3 model: [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
- Official [DINOv3 README](https://github.com/facebookresearch/dinov3#license) says both code and model weights use the **custom DINOv3 License**. A reviewed copy of the official [LICENSE.md](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md), last updated August 19, 2025, is retained at [third-party/DINOv3-LICENSE.md](third-party/DINOv3-LICENSE.md). Its contents were verified against official commit `6876159a11b4df116f30f667f8c9888617df0751`.
- Token semantics are verified against [Transformers v4.57.1 DINOv3 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/dinov3_vit/modeling_dinov3_vit.py): embeddings concatenate CLS/registers/patches, and normalized position zero is the class embedding. Official DINOv3 documentation states Transformers support from 4.56.0; research dependencies select 4.57.x.

License section 1.b.i states that redistribution of DINO Materials or derivatives is subject to the agreement and requires providing a copy. Section 1.b.ii requires acknowledging DINO Materials in research publications. Its other applicable-use and distribution terms also remain relevant. It is not appropriate to label a full hybrid containing DINOv3 weights as simply Apache/MIT or assume gated download access alone authorizes unrestricted redistribution.

This phase does **not** propose or automatically publish a full DINOv3-containing checkpoint. Saving only the fusion head avoids bundling the gated encoder weights; it is not a claim that every possible head artifact is exempt from derivative-work terms. Recheck the current model card/accepted gated conditions and the agreement before any public release, include required license/acknowledgment material, and retain the original source notices. The HF model card could not be read in Codex because access is blocked; the official GitHub license was accessible and inspected.

## Validation boundary

Unit tests are entirely offline. The UniRepLKNet branch uses the original architecture with a synthetic baseline state fixture; DINOv3 and its processor are explicitly mocked. They check token selection, shapes, frozen parameters/BatchNorm, head gradients, strict key mapping, dual RGB preprocessing/shared flip, no-fallback errors, token exclusion from checkpoints and opt-in notebook cells. No real pretrained forward or T4 execution is claimed from those tests.

Codex HF access remains HTTP 403. The user's successful gated download on Kaggle is acknowledged; real-pretrained smoke must be run in that authorized Kaggle session. There are no measured hybrid VRAM, accuracy, test results or latency yet.
