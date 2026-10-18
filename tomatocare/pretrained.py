"""Load the author's ImageNet-1K checkpoint without replacing the four-class head."""

from pathlib import Path

import torch

REPO_ID = "DingXiaoH/UniRepLKNet"
FILENAME = "unireplknet_a_in1k_224_acc77.03.pth"


def load_imagenet_backbone(model, checkpoint_path=None):
    guidance = (f"Provide the official {FILENAME} via --pretrained-checkpoint /path/to/file.pth. "
                f"Source: https://huggingface.co/{REPO_ID}/tree/main . No random-init fallback is allowed.")
    try:
        if checkpoint_path is None:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        path = Path(checkpoint_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        if state["head.weight"].shape != (1000, 320) or state["head.bias"].shape != (1000,):
            raise ValueError("Expected the original 1000-class UniRepLKNet-A head")
        backbone = {key: value for key, value in state.items() if not key.startswith("head.")}
        # Validate keys and shapes before changing model parameters.
        expected = {k: v for k, v in model.state_dict().items() if not k.startswith("head.")}
        if backbone.keys() != expected.keys() or any(backbone[k].shape != expected[k].shape for k in expected):
            raise ValueError("Checkpoint backbone keys/shapes do not exactly match UniRepLKNet-A")
        result = model.load_state_dict(backbone, strict=False)
        if set(result.missing_keys) != {"head.weight", "head.bias"} or result.unexpected_keys:
            raise ValueError(f"Unexpected checkpoint loading result: {result}")
        return {"repo_id": REPO_ID, "filename": FILENAME, "local_filename": path.name}
    except Exception as exc:
        raise RuntimeError(f"Cannot load official ImageNet-1K backbone: {exc}. {guidance}") from exc
