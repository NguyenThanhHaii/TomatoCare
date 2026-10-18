"""Measure cold-load, CPU latency and process peak RSS using configured real artifacts.

Run each runtime in a fresh process. No fake checkpoints are generated here.
"""

import argparse
import json
import os
import resource
import time

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["pytorch", "onnx"], required=True)
    parser.add_argument("--image", help="Optional real leaf image; default is a generated RGB gradient for timing only")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    os.environ["INFERENCE_RUNTIME"] = args.runtime
    started = time.perf_counter()
    from tomatocare.inference import load_predictor
    predictor = load_predictor()
    load_ms = (time.perf_counter() - started) * 1000
    if args.image:
        from PIL import ImageOps
        with Image.open(args.image) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
    else:
        import numpy as np
        image = Image.fromarray(np.random.default_rng(42).integers(0, 256, (300, 300, 3), dtype=np.uint8))
    durations = []
    for _ in range(args.runs):
        tick = time.perf_counter()
        predictor.predict(image)
        durations.append((time.perf_counter() - tick) * 1000)
    # Linux ru_maxrss is KiB; run inside the deployment container for meaningful RSS.
    peak_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps({"runtime": args.runtime, "model_identifier": predictor.engine.identifier,
                      "load_ms": load_ms, "latency_ms": durations, "peak_rss_mib": peak_mib,
                      "below_512_mib_process_only": peak_mib < 512,
                      "note": "Process-only measurement, not proof of Render service/container headroom or 0.1 CPU latency."}, indent=2))


if __name__ == "__main__":
    main()
