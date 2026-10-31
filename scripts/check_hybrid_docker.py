"""Build/audit the Hybrid image and smoke its real Uvicorn without model secrets.

A 503 is EXPECTED in this packaging smoke because no private token/model is
injected. It verifies server startup, never trained-model inference/readiness.
"""

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ONLY_ENV = {"HF_TOKEN", "HF_TOKEN_FILE", "HYBRID_HF_REPO", "HYBRID_HF_REVISION", "HYBRID_MODEL_DIR", "HYBRID_EXPECTED_ONNX_SHA256"}

AUDIT_CODE = r'''
import importlib.util,json,os,site
from pathlib import Path
import numpy,PIL,fastapi,uvicorn,onnxruntime,huggingface_hub
import hybrid_backend.main
assert os.getuid()!=0, 'Runtime must not be root'
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('transformers') is None
assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()
assert not Path('/run/secrets/hf_token').exists(), 'Runtime HF secret must not be baked'
assert not Path('/run/secrets/pip_ca').exists(), 'Temporary build CA must not be persisted'
root=Path('/app')
files=[p for p in root.rglob('*') if p.is_file()]
for p in files:
 assert p.suffix.lower() not in {'.onnx','.pth','.pt','.ckpt','.safetensors','.zip','.ipynb','.png','.jpg'}, p.name
 assert not p.name.startswith('.env'), 'Environment values must not be in image'
 assert not any(part in {'node_modules','.venv','tests','notebooks','__pycache__'} for part in p.parts)
for package_root in site.getsitepackages():
 assert not list(Path(package_root).rglob('*.onnx')), 'Bundled ONNX examples should be stripped'
cache=Path(os.environ['HF_HOME'])
assert not cache.exists() or not any(p.is_file() for p in cache.rglob('*')), 'HF cache must not be baked'
assert (root/'licenses/DINOv3-LICENSE.md').is_file()
assert not any(p.name in {'token','stored_tokens'} for p in Path('/home').rglob('*'))
print(json.dumps({'uid':os.getuid(),'runtime_imports':'passed','torch':False,'transformers':False,
 'weights_and_secret_files':'absent','hf_cache':'empty','cpu_provider':True,
 'app_files':sorted(str(p.relative_to(root)) for p in files)}))
'''

HTTP_PROBE = r'''
import json,os,urllib.request,urllib.error
base='http://127.0.0.1:'+os.environ.get('PORT','8000')
result={}
for route in ('health','predict'):
 data=None;headers={}
 if route=='predict':
  import io
  from PIL import Image
  picture=io.BytesIO();Image.new('RGB',(32,32),'green').save(picture,format='PNG')
  data=b'--test\r\nContent-Disposition: form-data; name="file"; filename="leaf.png"\r\nContent-Type: image/png\r\n\r\n'+picture.getvalue()+b'\r\n--test--\r\n'
  headers={'Content-Type':'multipart/form-data; boundary=test'}
 request=urllib.request.Request(base+'/'+route,data=data,headers=headers)
 try:
  with urllib.request.urlopen(request,timeout=3) as response:
   status=response.status;body=json.load(response)
 except urllib.error.HTTPError as error:
  status=error.code;body=json.load(error)
 assert status==503, 'Without an artifact this must never claim model readiness'
 assert 'HYBRID_MODEL_DIR' in body.get('detail',''), 'Expected explicit missing model configuration'
 result[route]=status
print(json.dumps(result))
'''


def docker(*args, timeout=120):
    return subprocess.run(["docker", *args], cwd=ROOT, text=True, capture_output=True, timeout=timeout, check=True).stdout.strip()


def check_image(image, build=False):
    docker("info", "--format", "{{.ServerVersion}}", timeout=15)
    if build:
        print(docker("build", "--progress=plain", "-f", "Dockerfile.hybrid", "-t", image, ".", timeout=1200))
    config = json.loads(docker("image", "inspect", "--format", "{{json .Config}}", image))
    keys = {item.split("=", 1)[0] for item in config.get("Env", [])}
    if keys & RUNTIME_ONLY_ENV:
        raise RuntimeError("Image contains runtime-only model/auth environment keys")
    if config.get("User") != "10001:10001" or "--workers 1" not in " ".join(config.get("Cmd", [])):
        raise RuntimeError("Incorrect nonroot/one-worker image configuration")
    if "hybrid_backend.main:app" not in " ".join(config.get("Cmd", [])):
        raise RuntimeError("Image does not run the experimental Hybrid app")
    pip_check = docker("run", "--rm", "--entrypoint", "python", image, "-m", "pip", "check")
    audit = json.loads(docker("run", "--rm", "--entrypoint", "python", image, "-c", AUDIT_CODE))
    name = "tomatocare-packaging-check-" + uuid.uuid4().hex[:10]
    try:
        # No env/token/volume/published port: safe framework-only smoke, not inference.
        docker("run", "--detach", "--name", name, "--memory=512m", "--cpus=1", image)
        deadline = time.monotonic() + 45
        last_error = None
        while time.monotonic() < deadline:
            try:
                responses = json.loads(docker("exec", name, "python", "-c", HTTP_PROBE, timeout=10))
                break
            except subprocess.CalledProcessError as exc:
                last_error = exc
                if docker("inspect", "--format", "{{.State.Running}}", name) != "true":
                    raise RuntimeError("Uvicorn container exited before packaging smoke") from None
                time.sleep(0.25)
        else:
            raise RuntimeError("Container did not serve the expected no-model responses") from last_error
        oom = docker("inspect", "--format", "{{.State.OOMKilled}}", name)
        if oom != "false":
            raise RuntimeError("Container was OOM-killed even without a model")
        return {"image":image, "image_id":docker("image", "inspect", "--format", "{{.Id}}", image),
                "pip_check":pip_check, "audit":audit, "no_model_http_smoke":responses,
                "private_inference":"not tested; no credentials/artifact injected",
                "resource_scope":"512 MiB / 1 CPU framework-only smoke, not real-model memory proof"}
    finally:
        subprocess.run(["docker","rm","--force",name], capture_output=True, text=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image",default="tomatocare-hybrid:phase7b")
    parser.add_argument("--build",action="store_true")
    args=parser.parse_args()
    try:
        report=check_image(args.image,args.build)
    except (OSError,subprocess.SubprocessError,RuntimeError,ValueError) as error:
        print(json.dumps({"status":"blocked_or_failed", "reason":type(error).__name__,
                          "next_step":"Check daemon/build output and run the documented Docker commands; no success claimed."}))
        return 2
    print(json.dumps({"status":"passed",**report},indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
