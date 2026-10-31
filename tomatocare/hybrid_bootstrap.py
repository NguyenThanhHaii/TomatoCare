"""Private HF bootstrap: immutable revision, staged verification, atomic activation.

No credentials are persisted and no SDK exception text is exposed to logs/clients.
Local HYBRID_MODEL_DIR loading remains independent of this optional transport.
"""

import logging
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from contextlib import contextmanager

FILES = ("tomatocare_hybrid.onnx", "inference_manifest.json", "DINOv3-LICENSE.md")


class BootstrapError(RuntimeError):
    """Only source-controlled, credential-free messages may be used here."""


def _expected_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise BootstrapError("HYBRID_EXPECTED_ONNX_SHA256 must be a full lowercase SHA-256 digest")
    return value


def _verify_bundle(directory, expected):
    from .hybrid_onnx import load_manifest, sha256_file
    root = Path(directory).resolve()
    for filename in FILES:
        path = root / filename
        if not path.is_file() or path.is_symlink():
            raise BootstrapError("Private bundle is incomplete; all three regular artifact files are required")
    try:
        _, manifest = load_manifest(root)
    except Exception:
        raise BootstrapError("Private inference_manifest.json is invalid; bundle was not activated") from None
    if manifest["model_sha256"] != expected:
        raise BootstrapError("Manifest ONNX SHA-256 differs from trusted HYBRID_EXPECTED_ONNX_SHA256")
    if sha256_file(root / FILES[0]) != expected:
        raise BootstrapError("Downloaded ONNX SHA-256 mismatch; bundle was not activated")
    license_path = root / FILES[2]
    try:
        if not 0 < license_path.stat().st_size <= 1024 * 1024:
            raise ValueError()
        text = license_path.read_text(encoding="utf-8")
        if "DINOv3" not in text or "License" not in text:
            raise ValueError()
    except Exception:
        raise BootstrapError("DINOv3-LICENSE.md is missing, empty or invalid") from None


@contextmanager
def _quiet_private_transport():
    """Suppress SDK/transport diagnostics that can contain signed private URLs.

    Application status logs and health polling remain enabled. Filters are scoped
    to the synchronous download and are removed even on network exceptions.
    """
    prefixes = ("huggingface_hub", "urllib3", "requests", "httpx", "httpcore", "hf_xet")
    class TransportFilter(logging.Filter):
        def filter(self, record):
            return not record.name.startswith(prefixes)
    guard = TransportFilter()
    loggers = [logging.getLogger()]
    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if isinstance(logger, logging.Logger) and name.startswith(prefixes):
            loggers.append(logger)
    targets = set(loggers)
    for logger in loggers:
        targets.update(logger.handlers)
    for target in targets:
        target.addFilter(guard)
    try:
        yield
    finally:
        for target in targets:
            target.removeFilter(guard)


def _download(repo, revision, token, filename, directory):
    # Import lazily: local Phase 5 mode does not call HF or require credentials.
    from huggingface_hub import hf_hub_download
    with _quiet_private_transport():
        return hf_hub_download(repo_id=repo, repo_type="model", revision=revision,
                               filename=filename, token=token, local_dir=str(directory),
                               force_download=True)


def _safe_download_error(exc):
    # force_download may wrap an HTTP/auth error in ValueError. Inspect only
    # numeric status/type through the cause chain, never exception text or URLs.
    current, seen, statuses, timed_out = exc, set(), [], False
    for _ in range(10):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        statuses.append(getattr(getattr(current, "response", None), "status_code", None))
        timed_out |= isinstance(current, TimeoutError) or "Timeout" in type(current).__name__
        current = current.__cause__ or current.__context__
    if any(status in (401, 403) for status in statuses):
        return "Private HF read denied (HTTP 401/403); check the fine-grained HF_TOKEN permission"
    if 404 in statuses:
        return "Private HF repository, pinned revision or required file is unavailable; check read permission and configuration"
    if timed_out:
        return "Private HF download timed out; retry startup after checking network connectivity"
    return "Private HF download failed or was interrupted; check connectivity and scoped read access"


def bootstrap_private_bundle(repo, revision, token, expected_sha256, model_dir):
    """Download exactly three same-revision files; never reuse an old bundle on failure."""
    # All configuration is validated before touching the network or disk.
    if not isinstance(repo, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise BootstrapError("HYBRID_HF_REPO must be a model repository owner/name")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise BootstrapError("HYBRID_HF_REVISION must be a full immutable 40-character HF commit SHA; main/tags/short SHAs are refused")
    if not isinstance(token, str) or not token.strip():
        raise BootstrapError("HF_TOKEN is required for private HF bootstrap; set it securely in the service environment")
    expected = _expected_sha(expected_sha256)
    try:
        root = Path(model_dir).resolve()
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix=".staging-", dir=root) as temporary:
            work = Path(temporary)
            downloads, candidate = work / "downloads", work / "candidate"
            candidate.mkdir(mode=0o700)
            for filename in FILES:
                try:
                    source = Path(_download(repo, revision, token, filename, downloads))
                    if not source.is_file() or not source.resolve().is_relative_to(downloads.resolve()):
                        raise OSError()
                    shutil.copyfile(source, candidate / filename)
                    (candidate / filename).chmod(0o600)
                except Exception as exc:
                    # No raw error/response/URL, repr(token), traceback chain or auth headers.
                    raise BootstrapError(_safe_download_error(exc)) from None
            _verify_bundle(candidate, expected)
            final = root / f"verified-{revision}-{uuid.uuid4().hex}"
            # Same-filesystem rename publishes the entire verified bundle at once.
            # Unique names avoid overwriting another process or a legacy local bundle.
            candidate.rename(final)
            return final
    except BootstrapError:
        raise
    except Exception:
        raise BootstrapError("Private artifact staging/verification failed; check writable ephemeral storage") from None


def resolve_hybrid_model_dir():
    repo = os.getenv("HYBRID_HF_REPO", "")
    if not repo:
        if os.getenv("HYBRID_HF_REVISION"):
            raise BootstrapError("Set HYBRID_HF_REPO to enable remote loading, or remove the remote revision for local mode")
        directory = os.getenv("HYBRID_MODEL_DIR")
        if not directory:
            raise ValueError("Set HYBRID_MODEL_DIR to the private directory containing inference_manifest.json and tomatocare_hybrid.onnx")
        expected = os.getenv("HYBRID_EXPECTED_ONNX_SHA256")
        if expected:
            # Optional trusted hash strengthens local mode without changing its default contract.
            from .hybrid_onnx import load_manifest, sha256_file
            expected = _expected_sha(expected)
            try:
                root, manifest = load_manifest(directory)
                if manifest["model_sha256"] != expected or sha256_file(root / manifest["model_file"]) != expected:
                    raise BootstrapError("Local ONNX SHA-256 differs from trusted expected digest")
            except BootstrapError:
                raise
            except Exception:
                raise BootstrapError("Local trusted-hash verification failed") from None
        return Path(directory)
    return bootstrap_private_bundle(
        repo, os.getenv("HYBRID_HF_REVISION"), os.getenv("HF_TOKEN"),
        os.getenv("HYBRID_EXPECTED_ONNX_SHA256"),
        os.getenv("HYBRID_MODEL_DIR") or "/tmp/tomatocare-hybrid",
    )
