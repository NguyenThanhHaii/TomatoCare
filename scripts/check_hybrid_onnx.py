"""Kaggle-only real-artifact validation: independent training preprocessing and logits.

No weights are generated. Use compare_loaded_model() with an existing trained
HybridModel/processor, or CLI with a real Phase 4 head checkpoint and HF access.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tomatocare.hybrid_onnx import CLASS_NAMES, HybridOnnxPredictor, preprocess, sha256_file


def compare_loaded_model(model, processor, model_dir, data_root, output_path, *, per_class=3,
                         device="cpu", predictor=None, check_counts=True):
    import torch
    from torchvision import transforms
    from backend.main import decode_image
    from tomatocare.data import MEAN, STD
    from tomatocare.train_hybrid import validate_hybrid_bundle
    if per_class < 1:
        raise ValueError("Use at least one distinct validation image per class (at least four total)")
    path = Path(output_path)
    if path.exists():
        raise FileExistsError("Choose a new report path to avoid reusing stale parity evidence")
    predictor = predictor or HybridOnnxPredictor(model_dir)
    expected = predictor.manifest.get("provenance", {})
    actual = getattr(model, "provenance", {})
    if (not expected.get("baseline", {}).get("sha256") or
        expected["baseline"]["sha256"] != actual.get("baseline", {}).get("sha256") or
        expected.get("dinov3", {}).get("model_id") != actual.get("dinov3", {}).get("model_id") or
        not expected.get("dinov3", {}).get("resolved_revision") or
        expected["dinov3"]["resolved_revision"] != actual.get("dinov3", {}).get("resolved_revision")):
        raise ValueError("PyTorch encoder provenance does not match the ONNX manifest")
    bundle = validate_hybrid_bundle(data_root, check_counts=check_counts)
    rng = np.random.default_rng(42)
    selected, seen_hashes = [], set()
    for target in range(4):
        candidates = [r for r in bundle.records["val"] if r.target == target]
        chosen = []
        for index in rng.permutation(len(candidates)):
            record = candidates[index]
            digest = sha256_file(record.path)
            if digest not in seen_hashes:
                chosen.append(record)
                seen_hashes.add(digest)
            if len(chosen) == per_class:
                break
        if len(chosen) < per_class:
            raise ValueError(f"Not enough distinct validation images for {CLASS_NAMES[target]}")
        selected.extend(chosen)
    # Independent reference: original Torchvision baseline transform plus the
    # actual HF processor loaded from the trained DINO revision, not our NumPy path.
    baseline_transform = transforms.Compose([transforms.Resize((224, 224), antialias=True),
                                              transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    model.to(device).eval()
    rows = []
    for record in selected:
        image = decode_image(record.path.read_bytes())
        try:
            values = preprocess(image, predictor.manifest)
            baseline = baseline_transform(image).unsqueeze(0)
            dino = processor(images=image, return_tensors="pt")["pixel_values"]
            if list(dino.shape) != [1, 3, 224, 224]:
                raise ValueError("Actual pretrained processor does not match manifest input shape")
            pixel_errors = {"baseline": float(np.max(np.abs(values["baseline_images"] - baseline.numpy()))),
                            "dino": float(np.max(np.abs(values["dino_images"] - dino.numpy())))}
            pixels_match = (np.allclose(values["baseline_images"], baseline.numpy(), atol=1e-5, rtol=1e-5) and
                            np.allclose(values["dino_images"], dino.numpy(), atol=1e-5, rtol=1e-5))
            with torch.inference_mode():
                reference = model(baseline.to(device), dino.to(device)).float().cpu().numpy()
            actual_logits = predictor.logits(image)
            if reference.shape != (1, 4) or not np.isfinite(reference).all():
                raise ValueError("Invalid PyTorch reference logits")
            rows.append({"validation_image": str(record.path.relative_to(bundle.root)),
                         "preprocessing_max_abs_error": pixel_errors, "preprocessing_matches": bool(pixels_match),
                         "logits_max_abs_error": float(np.max(np.abs(reference - actual_logits))),
                         "logits_match": bool(np.allclose(reference, actual_logits, atol=1e-4, rtol=1e-4)),
                         "same_argmax": bool(reference.argmax() == actual_logits.argmax())})
        finally:
            image.close()
    passed = len(rows) >= 4 and all(r["preprocessing_matches"] and r["logits_match"] and r["same_argmax"] for r in rows)
    report = {"passed": passed, "images_checked": len(rows), "split": "val", "distinct_sha256": True,
              "model_version": predictor.model_version, "onnx_sha256": predictor.manifest["model_sha256"],
              "max_absolute_logits_error": max(row["logits_max_abs_error"] for row in rows),
              "preprocessing_atol": 1e-5, "preprocessing_rtol": 1e-5, "logits_atol": 1e-4, "logits_rtol": 1e-4,
              "rows": rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError(f"Multi-image parity failed; inspect {path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=os.getenv("HYBRID_MODEL_DIR"))
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--head-checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--report", default="/kaggle/working/hybrid_parity.json")
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.model_dir:
        parser.error("Set HYBRID_MODEL_DIR or --model-dir")
    predictor = HybridOnnxPredictor(args.model_dir)
    from tomatocare.hybrid_model import create_hybrid_model
    from tomatocare.train_hybrid import restore_head
    sources = predictor.manifest.get("provenance", {})
    model, processor = create_hybrid_model(
        baseline_checkpoint=args.baseline_checkpoint, hf_token=os.getenv("HF_TOKEN"),
        baseline_revision=sources["baseline"]["resolved_revision"],
        dino_revision=sources["dinov3"]["resolved_revision"])
    restore_head(model, args.head_checkpoint)
    report = compare_loaded_model(model, processor, args.model_dir, args.data_root, args.report,
                                  per_class=args.per_class, device=args.device, predictor=predictor)
    print(json.dumps({"passed": report["passed"], "images_checked": report["images_checked"],
                      "max_absolute_logits_error": report["max_absolute_logits_error"], "report": args.report}))


if __name__ == "__main__":
    main()
