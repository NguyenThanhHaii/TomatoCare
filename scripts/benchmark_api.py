"""Benchmark the whole FastAPI/Uvicorn child process with the configured ONNX model.

Uses Linux process high-water RSS (MiB), including HTTP, decoding and session.
A missing model reports blocked with zero successful requests, never fake timings.
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def process_peak_mib(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="JPG/PNG used for real HTTP inference requests")
    parser.add_argument("--profile", choices=("default", "low_memory"), default="default")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--startup-timeout", type=float, default=120)
    args = parser.parse_args()
    if args.runs < 1 or args.startup_timeout <= 0:
        parser.error("runs and startup-timeout must be positive")
    image = Path(args.image)
    payload = image.read_bytes()
    if image.suffix.lower() not in (".jpg", ".jpeg", ".png") or len(payload) > 10 * 1024 * 1024:
        parser.error("--image must be JPG/PNG, at most 10 MiB")
    boundary = "TomatoCareBenchmarkBoundary"
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="leaf{image.suffix.lower()}"\r\n'
            f'Content-Type: {mime}\r\n\r\n').encode() + payload + f'\r\n--{boundary}--\r\n'.encode()
    env = {**os.environ, "INFERENCE_RUNTIME": "onnx", "ONNX_PROFILE": args.profile,
           "CPU_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    # The proxy is bypassed only for requests to our own loopback test server.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{args.port}"
    started = time.perf_counter()
    report = {"status": "blocked", "runtime": "onnx", "profile": args.profile,
              "successful_requests": 0, "latency_ms": [], "peak_api_process_rss_mib": 0,
              "note": "Linux whole-process RSS; not container/cgroup peak or a measurement on Render's 0.1 CPU."}
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1",
                                    "--port", str(args.port), "--workers", "1"], env=env, stdout=log, stderr=log)
        try:
            deadline = started + args.startup_timeout
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Uvicorn exited during startup; check model and port configuration")
                try:
                    with opener.open(base + "/health", timeout=1) as response:
                        health = json.load(response)
                    break
                except urllib.error.HTTPError as exc:
                    detail = json.load(exc).get("detail", str(exc))
                    raise RuntimeError(f"API model unavailable: {detail}") from exc
                except (urllib.error.URLError, TimeoutError):
                    if time.perf_counter() > deadline:
                        raise RuntimeError("API startup timed out")
                    time.sleep(0.2)
            report["startup_ms"] = (time.perf_counter() - started) * 1000
            report["model_identifier"] = health["model_identifier"]
            for _ in range(args.runs):
                tick = time.perf_counter()
                request = urllib.request.Request(base + "/predict", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
                with opener.open(request, timeout=120) as response:
                    prediction = json.load(response)
                if len(prediction.get("probabilities", {})) != 4:
                    raise RuntimeError("Invalid inference response")
                report["latency_ms"].append((time.perf_counter() - tick) * 1000)
                report["successful_requests"] += 1
            report["mean_latency_ms"] = sum(report["latency_ms"]) / args.runs
            report["status"] = "completed"
        except Exception as exc:
            report["reason"] = str(exc)
        finally:
            report["peak_api_process_rss_mib"] = process_peak_mib(process.pid)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            report["peak_api_process_rss_mib"] = max(report["peak_api_process_rss_mib"], resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
