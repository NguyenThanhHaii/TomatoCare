"""Frozen UniRepLKNet-A stages 1–3 + gated pretrained DINOv3 ViT-S/16.

Importing this module never downloads weights or starts training.
"""

import hashlib
from pathlib import Path

import torch
from torch import nn

from .data import PREPROCESSING
from .model import CLASS_NAMES, create_unireplknet_a

BASELINE_REPO = "ThanhHai25/TomatoCare-UniRepLKNet-A"
BASELINE_FILE = "best_unireplknet_a.pth"
DINO_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UniRepLKNetStage3(nn.Module):
    """Own only downsampling/stages 0,1,2: stage 3/norm/head are absent."""
    def __init__(self, baseline):
        super().__init__()
        self.downsample_layers = nn.ModuleList(list(baseline.downsample_layers)[:3])
        self.stages = nn.ModuleList(list(baseline.stages)[:3])

    def forward(self, images):
        for downsample, stage in zip(self.downsample_layers, self.stages):
            images = stage(downsample(images))
        return images.mean(dim=(-2, -1))


def load_baseline_encoder(checkpoint_path=None, revision="main"):
    if checkpoint_path is None:
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(repo_id=BASELINE_REPO, filename=BASELINE_FILE,
                                               revision=revision, token=False)
        except Exception:
            raise RuntimeError("Cannot download the public trained baseline. Provide baseline_checkpoint; no random-weight fallback.") from None
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "unireplknet_a" or checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Baseline architecture/class_names do not match the confirmed four-class checkpoint")
    mapping = checkpoint.get("class_to_idx")
    if mapping is not None and mapping != {name: i for i, name in enumerate(CLASS_NAMES)}:
        raise ValueError("Baseline class_to_idx disagrees with class_names")
    preprocessing = checkpoint.get("preprocessing", {})
    for key in ("color_mode", "image_size", "mean", "std", "resize", "interpolation"):
        if preprocessing.get(key) != PREPROCESSING[key]:
            raise ValueError(f"Baseline preprocessing mismatch: {key}")
    encoder = UniRepLKNetStage3(create_unireplknet_a())
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("Baseline requires a state_dict")
    kept_prefixes = tuple(f"{part}.{i}." for part in ("downsample_layers", "stages") for i in range(3))
    ignored_prefixes = ("downsample_layers.3.", "stages.3.", "norm.", "head.")
    if any(not k.startswith(kept_prefixes + ignored_prefixes) for k in state):
        raise ValueError("Unrecognized baseline key mapping (for example, a module. prefix)")
    selected = {key: value for key, value in state.items() if key.startswith(kept_prefixes)}
    expected = encoder.state_dict()
    if selected.keys() != expected.keys() or any(not isinstance(selected[k], torch.Tensor) or selected[k].shape != expected[k].shape for k in expected):
        raise ValueError("Baseline Stage 1–3 keys/shapes do not exactly match UniRepLKNet-A")
    encoder.load_state_dict(selected, strict=True)
    snapshot = checkpoint_path.parent.name if checkpoint_path.parent.parent.name == "snapshots" else None
    return encoder, {"repo_id": BASELINE_REPO, "filename": BASELINE_FILE,
                     "requested_revision": revision, "resolved_revision": snapshot,
                     "sha256": _file_hash(checkpoint_path)}


class HybridModel(nn.Module):
    """Injected encoders allow offline tests; the factory below requires real weights."""
    def __init__(self, unireplknet_encoder, dinov3_encoder, provenance=None):
        super().__init__()
        config = dinov3_encoder.config
        if config.model_type != "dinov3_vit" or config.hidden_size != 384 or config.patch_size != 16:
            raise ValueError("Expected pretrained DINOv3 ViT-S/16 with hidden size 384")
        self.num_register_tokens = config.num_register_tokens
        self.unireplknet_encoder = unireplknet_encoder
        self.dinov3_encoder = dinov3_encoder
        self.classifier = nn.Sequential(nn.LayerNorm(544), nn.Linear(544, 4))
        self.provenance = provenance or {}
        for encoder in (self.unireplknet_encoder, self.dinov3_encoder):
            encoder.requires_grad_(False)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        self.unireplknet_encoder.eval()
        self.dinov3_encoder.eval()
        return self

    def forward_features(self, baseline_images, dino_images):
        if baseline_images.ndim != 4 or baseline_images.shape[1:] != (3, 224, 224):
            raise ValueError("UniRepLKNet branch expects [B,3,224,224]")
        if dino_images.ndim != 4 or dino_images.shape[0] != baseline_images.shape[0] or dino_images.shape[1] != 3:
            raise ValueError("DINOv3 branch expects matching [B,3,H,W]")
        height, width = dino_images.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("DINOv3 processor output dimensions must be divisible by patch size 16")
        # no_grad, not inference_mode: head backward must be able to save features.
        with torch.no_grad():
            first = self.unireplknet_encoder(baseline_images)
            output = self.dinov3_encoder(pixel_values=dino_images, return_dict=True)
            tokens = output.last_hidden_state
            expected_tokens = 1 + self.num_register_tokens + (height // 16) * (width // 16)
            if tokens.shape != (baseline_images.shape[0], expected_tokens, 384):
                raise ValueError("Unexpected DINOv3 CLS/register/patch token layout")
            second = tokens[:, 0, :]  # CLS first, then registers, then patches. No token averaging.
        if first.shape != (baseline_images.shape[0], 160):
            raise ValueError("Stage 3 GAP must return [B,160]")
        return first, second, torch.cat((first, second), dim=1)

    def forward(self, baseline_images, dino_images):
        _, _, features = self.forward_features(baseline_images, dino_images)
        return self.classifier(features)


def processor_configuration(processor):
    """Whitelist image transforms only; never serialize auth kwargs or model objects."""
    allowed = ("image_processor_type", "do_resize", "size", "resample", "do_center_crop", "crop_size",
               "do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std", "do_convert_rgb")
    config = processor.to_dict()
    return {key: config[key] for key in allowed if key in config}


def create_hybrid_model(baseline_checkpoint=None, hf_token=None, baseline_revision="main", dino_revision="main"):
    """Load both real pretrained encoders; any download/key failure stops creation."""
    encoder, baseline_source = load_baseline_encoder(baseline_checkpoint, baseline_revision)
    try:
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_ID, revision=dino_revision,
                                                       token=hf_token, trust_remote_code=False)
        dino, loading = AutoModel.from_pretrained(DINO_MODEL_ID, revision=dino_revision, token=hf_token,
                                                 trust_remote_code=False, use_safetensors=True,
                                                 output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("Incomplete pretrained checkpoint")
    except Exception as exc:
        # Do not echo SDK exceptions: access credentials must never reach logs/checkpoints.
        raise RuntimeError(f"DINOv3 pretrained loading failed ({type(exc).__name__}). Check gated access, Kaggle HF_TOKEN and requirements-hybrid.txt. No random-pretrained fallback.") from None
    provenance = {"baseline": baseline_source,
                  "dinov3": {"model_id": DINO_MODEL_ID, "requested_revision": dino_revision,
                             "resolved_revision": getattr(dino.config, "_commit_hash", None)}}
    return HybridModel(encoder, dino, provenance), processor
