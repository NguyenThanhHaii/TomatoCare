# Changelog

## Unreleased — Phase 4 research

- Add a physically truncated UniRepLKNet-A Stage 1–3 encoder loaded strictly from the trained four-class baseline.
- Add gated real-pretrained DINOv3 ViT-S/16 CLS extraction, separate original processor, and 160+384-dimensional fusion with LayerNorm/Linear.
- Freeze both encoders and BatchNorm/dropout modes; keep gradients only for the 3,268-parameter fusion head.
- Add dual-branch data loading with shared train flip, opt-in Kaggle smoke/training/final-test notebook, validation selection and head-only artifact/provenance saving.
- Add offline mock tests, source/license review and a copy of the official DINOv3 License. No gated weights or secrets are redistributed.
- Preserve baseline training, production ONNX, FastAPI, React and hosting configuration. Research source is prepared on feature/hybrid-dinov3; no actual training, model-weight release, merge to main or deployment performed.

## Unreleased — Phase 3

- Add a mobile-first React/Vite PWA matching the supplied four-state design.
- Add image upload/camera input, real API waiting, error/retry flows, four-class probability bars and readable care information.
- Add FastAPI image validation, EXIF normalization, configured CORS, cached CPU inference and explicit model-unavailable errors.
- Add trained-checkpoint loading using the existing UniRepLKNet-A factory, with no architecture or training changes.
- Add ONNX export with multi-input logit/argmax checks, artifact integrity metadata and a PyTorch-free runtime dependency set.
- Add CPU benchmark tooling, Vercel/Render configuration, deployment guidance and visual/API tests.
- Preserve the existing training modules and Kaggle notebook. DINOv3 remains future work.
- Default production inference to the owner's `best_unireplknet_a.onnx`, batch one, CPU provider and one thread; add Default/Low Memory profiles.
- Accept validated embedded metadata or the owner's confirmed checkpoint contract; no required sidecar for the public ONNX.
- Configure Vercel via `VITE_API_URL`; cover HTML cold-start errors, retry and the 120-second request timeout without changing the React design.
- Add whole-FastAPI-process RSS/HTTP benchmark tooling and native Python Render Free configuration.
- Attribute the owner's Kaggle parity/performance report separately from Codex evidence. Real trained inference and 512 MB free-host readiness remain unverified in Codex due to HF HTTP 403; only failed-startup RSS was measured. No deployment performed.
