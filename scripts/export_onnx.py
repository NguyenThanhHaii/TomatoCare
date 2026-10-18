"""Export the trained checkpoint and require multi-input parity before publishing artifacts.

Run from repo root: python -m scripts.export_onnx --output-dir outputs/onnx
"""

import argparse
import json
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

from tomatocare.inference import TorchEngine, preprocess, sha256_file


def export_verified(checkpoint_path, output_dir, image_paths=()):
    import onnxruntime as ort
    import torch
    engine = TorchEngine(checkpoint_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "best_unireplknet_a.onnx"
    manifest = output / "inference.json"
    if path.exists() or manifest.exists():
        raise FileExistsError("Use an empty output directory; never overwrite a previously verified artifact")
    temporary = output / "unireplknet_a.pending.onnx"
    sample = torch.zeros(1, 3, 224, 224)
    # Newer PyTorch defaults to Dynamo and requires extra exporter packages.
    # Use the tested legacy exporter explicitly when that switch is available.
    export_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(engine.model, sample, str(temporary), input_names=["image"], output_names=["logits"],
                      opset_version=17, do_constant_folding=True, **export_options)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    inputs = [Image.fromarray(rng.integers(0, 256, (260 + i * 7, 300, 3), dtype=np.uint8)) for i in range(5)]
    inputs.extend(Image.new("RGB", (224, 224), color) for color in ("black", "white"))
    for image_path in image_paths:
        from PIL import ImageOps
        with Image.open(image_path) as image:
            inputs.append(ImageOps.exif_transpose(image).convert("RGB"))
    errors = []
    for image in inputs:
        values = preprocess(image, engine.config)
        expected = engine.logits(values)
        actual = session.run(None, {"image": values})[0]
        errors.append(float(np.max(np.abs(expected - actual))))
        if not np.isfinite(actual).all() or not np.allclose(expected, actual, rtol=1e-4, atol=1e-4) or expected.argmax() != actual.argmax():
            raise RuntimeError(f"ONNX parity failed: max_abs_error={errors[-1]}; pending file is not production-verified")
    temporary.replace(path)
    metadata = {
        "class_names": list(engine.class_names),
        "class_to_idx": {name: i for i, name in enumerate(engine.class_names)},
        "preprocessing": engine.config, "checkpoint_sha256": engine.checkpoint_sha256,
        "onnx_sha256": sha256_file(path),
        "verification": {"passed": True, "input_count": len(inputs), "max_absolute_logit_error": max(errors),
                         "same_argmax_all": True, "rtol": 1e-4, "atol": 1e-4},
    }
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return path, manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Trusted trained checkpoint; omit to download from the configured HF repo")
    parser.add_argument("--output-dir", default="outputs/onnx")
    parser.add_argument("--image", action="append", default=[], help="Optional leaf images for additional parity checks")
    args = parser.parse_args()
    export_verified(args.checkpoint, args.output_dir, args.image)
