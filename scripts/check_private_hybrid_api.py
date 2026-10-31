"""Fresh-process private HF cold-start + real HTTP smoke/benchmark on Kaggle.

Secrets are inherited only as environment variables, never CLI arguments or logs.
Run in a fresh ONNX-only environment to measure the complete Uvicorn process.
"""

import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def _rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, ProcessLookupError):
        pass
    return 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Real validation JPG/PNG (never a secret/weight file)")
    parser.add_argument("--port", type=int, default=18002)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--report", default="/kaggle/working/private_hybrid_api_report.json")
    args = parser.parse_args()
    if args.runs < 1 or args.startup_timeout <= 0:
        parser.error("runs and startup-timeout must be positive")
    report_path = Path(args.report)
    if report_path.exists():
        parser.error("Choose a fresh report path rather than reusing stale evidence")
    from tomatocare.hybrid_bootstrap import BootstrapError, _expected_sha
    # Validate required names without logging values; remote bootstrap validates revision/repo.
    if any(not os.getenv(k) for k in ("HYBRID_HF_REPO", "HYBRID_HF_REVISION", "HF_TOKEN", "HYBRID_EXPECTED_ONNX_SHA256")):
        parser.error("Set all private bootstrap environment variables securely before running")
    try:
        expected = _expected_sha(os.environ["HYBRID_EXPECTED_ONNX_SHA256"])
    except BootstrapError as exc:
        parser.error(str(exc))
    image = Path(args.image)
    data = image.read_bytes()
    if image.suffix.lower() not in (".jpg", ".jpeg", ".png") or not 0 < len(data) <= 10 * 1024 * 1024:
        parser.error("Use a nonempty JPG/PNG up to 10 MiB")
    # Refuse an occupied port instead of accidentally testing some other service.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    boundary = "TomatoCarePrivateApiValidation"
    mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="leaf{image.suffix.lower()}"\r\n'
            f'Content-Type: {mime}\r\n\r\n').encode() + data + f'\r\n--{boundary}--\r\n'.encode()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # Loopback only.
    base = f"http://127.0.0.1:{args.port}"
    report = {"status": "blocked", "successful_requests": 0, "pending_health_503": 0,
              "latency_ms": [], "peak_uvicorn_rss_mib": 0,
              "note": "Whole Linux Uvicorn process incl. bootstrap, not container/cgroup peak or Render 0.1 CPU performance."}
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
           "HF_HUB_DISABLE_XET": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    # A new subprocess loads the private model; no reference/mock model is injected.
    with tempfile.TemporaryFile(mode="w+") as logs:
        start = time.perf_counter()
        child = subprocess.Popen([sys.executable, "-m", "uvicorn", "hybrid_backend.main:app", "--host", "127.0.0.1",
                                  "--port", str(args.port), "--workers", "1"], env=env, stdout=logs, stderr=logs)
        try:
            deadline = start + args.startup_timeout
            while True:
                if child.poll() is not None:
                    raise RuntimeError("Experimental Uvicorn process exited during startup")
                if time.perf_counter() > deadline:
                    raise RuntimeError("Private bootstrap did not become ready before startup timeout")
                try:
                    with opener.open(base + "/health", timeout=2) as response:
                        health = json.load(response)
                    break
                except urllib.error.HTTPError as exc:
                    report["pending_health_503"] += 1
                    detail = json.load(exc).get("detail", "")
                    if detail and detail != "Hybrid model initializing":
                        raise RuntimeError("Private bootstrap failed; inspect the credential-redacted /health error") from None
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.1)
            if (health.get("artifact_source") != "private-hf" or health.get("hf_revision") != env["HYBRID_HF_REVISION"] or
                health.get("onnx_sha256") != expected or not health.get("sha256_verified")):
                raise RuntimeError("Health identity does not match the expected private artifact/revision")
            report["startup_ms"] = (time.perf_counter() - start) * 1000
            report["hf_revision"] = health["hf_revision"]
            report["model_version"] = health["model_version"]
            report["onnx_sha256"] = health["onnx_sha256"]
            for _ in range(args.runs):
                tick = time.perf_counter()
                request = urllib.request.Request(base + "/predict", data=body,
                                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
                with opener.open(request, timeout=120) as response:
                    prediction = json.load(response)
                probabilities = prediction.get("probabilities", {})
                from tomatocare.hybrid_onnx import CLASS_NAMES
                if (list(probabilities) != list(CLASS_NAMES) or
                    prediction.get("model_version") != health["model_version"] or
                    any(not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in probabilities.values()) or
                    abs(sum(probabilities.values()) - 1) > 1e-5 or
                    prediction.get("predicted_class") not in probabilities or
                    abs(prediction.get("confidence", -1) - max(probabilities.values())) > 1e-5 or
                    abs(prediction.get("confidence", -1) - probabilities[prediction["predicted_class"]]) > 1e-5):
                    raise RuntimeError("Invalid private hybrid prediction response")
                report["latency_ms"].append((time.perf_counter() - tick) * 1000)
                report["successful_requests"] += 1
            report["mean_http_latency_ms"] = sum(report["latency_ms"]) / args.runs
            report["status"] = "passed"
        except Exception:
            # Do not serialize arbitrary SDK/HTTP exception text or service logs.
            report["reason"] = "Private startup/HTTP validation did not complete; check credential-redacted service health"
        finally:
            report["peak_uvicorn_rss_mib"] = _rss(child.pid)
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
            report["peak_uvicorn_rss_mib"] = max(report["peak_uvicorn_rss_mib"], resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
