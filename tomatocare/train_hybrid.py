"""Opt-in hybrid head training and separately gated held-out test evaluation.

No training/downloads happen on import. CLI defaults to pretrained smoke only;
--train is required for a real training run. HF_TOKEN is never part of config.
"""

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import PREPROCESSING, discover_bundle
from .hybrid_data import HybridLeafDataset
from .hybrid_model import create_hybrid_model, processor_configuration
from .model import CLASS_NAMES
from .train import _write_metrics, summarize_metrics


@dataclass
class HybridTrainConfig:
    data_root: str = "/kaggle/input"
    output_dir: str = "/kaggle/working/tomatocare_hybrid"
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_hybrid_bundle(data_root, check_counts=True):
    bundle = discover_bundle(data_root)
    if bundle.class_names != tuple(CLASS_NAMES):
        raise ValueError("Dataset labels.json order disagrees with the pretrained baseline checkpoint")
    counts = {name: len(rows) for name, rows in bundle.records.items()}
    if check_counts and counts != {"train": 3816, "val": 816, "test": 820}:
        raise ValueError(f"Expected 3816/816/820 images, found {counts}")
    return bundle


def make_loader(records, processor, config, train=False):
    return DataLoader(HybridLeafDataset(records, processor, train=train), batch_size=config.batch_size,
                      shuffle=train, num_workers=config.num_workers, pin_memory=config.device.startswith("cuda"),
                      generator=torch.Generator().manual_seed(config.seed))


def _inputs(batch, device):
    inputs, labels = batch
    return inputs["unireplknet"].to(device), inputs["dinov3"].to(device), labels.to(device)


def smoke_test(model, batch, device):
    previous = model.training
    model.to(device).eval()
    try:
        with torch.no_grad():
            first, second, labels = _inputs(batch, device)
            a, b, fused = model.forward_features(first, second)
            logits = model.classifier(fused)
            if logits.shape != (len(labels), 4) or not torch.isfinite(logits).all():
                raise ValueError("Hybrid smoke logits must be finite [B,4]")
            nn.functional.cross_entropy(logits, labels)
            return {"unireplknet": list(a.shape), "dinov3_cls": list(b.shape),
                    "fusion": list(fused.shape), "logits": list(logits.shape),
                    "dino_pixels": list(second.shape),
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    finally:
        model.train(previous)


def train_head_epoch(model, loader, optimizer, device):
    model.train()  # HybridModel.train keeps both encoders eval, including BatchNorm.
    total, count = 0.0, 0
    for batch in loader:
        first, second, labels = _inputs(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(first, second)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite hybrid loss")
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate_hybrid(model, loader, device):
    model.eval()
    truth, predictions, total = [], [], 0.0
    for batch in loader:
        first, second, labels = _inputs(batch, device)
        logits = model(first, second)
        total += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        truth.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = summarize_metrics(truth, predictions, CLASS_NAMES)
    metrics["loss"] = total / len(truth)
    return metrics


def save_head_checkpoint(path, model, processor, config, epoch, metrics):
    """Private/local artifact: no DINOv3 weights, token, model object or raw config."""
    payload = {
        "format": "tomatocare-hybrid-head-v1", "architecture": "unireplknet_a_stage3_dinov3_vits16",
        "classifier_state_dict": {k: v.detach().cpu() for k, v in model.classifier.state_dict().items()},
        "class_names": list(CLASS_NAMES), "class_to_idx": {n: i for i, n in enumerate(CLASS_NAMES)},
        "feature_dims": [160, 384, 544], "provenance": model.provenance,
        "preprocessing": {"unireplknet": PREPROCESSING, "dinov3": processor_configuration(processor)},
        "config": asdict(config), "epoch": epoch, "validation_metrics": metrics,
        "license_note": "DINOv3 weights excluded. Review the DINOv3 License before any public distribution.",
    }
    path = Path(path)
    temporary = path.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_head(model, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != "tomatocare-hybrid-head-v1" or checkpoint.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Not a compatible TomatoCare hybrid head checkpoint")
    saved = checkpoint.get("provenance", {})
    current = model.provenance
    baseline_matches = saved.get("baseline", {}).get("sha256") == current.get("baseline", {}).get("sha256")
    old_dino, new_dino = saved.get("dinov3", {}), current.get("dinov3", {})
    dino_matches = (old_dino.get("model_id") == new_dino.get("model_id") and
                    (old_dino.get("resolved_revision") or old_dino.get("requested_revision")) ==
                    (new_dino.get("resolved_revision") or new_dino.get("requested_revision")))
    if not baseline_matches or not dino_matches:
        raise ValueError("Encoder sources/revisions differ; recreate encoders from the saved provenance first")
    model.classifier.load_state_dict(checkpoint["classifier_state_dict"], strict=True)
    return checkpoint


def run_hybrid_training(config, model, processor):
    """Explicitly called only after a successful real-pretrained Kaggle smoke test."""
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0:
        raise ValueError("Invalid epochs, batch size or worker count")
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Enable the Kaggle T4 GPU; CUDA is unavailable")
    seed_everything(config.seed)
    bundle = validate_hybrid_bundle(config.data_root)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_path = output / "best_hybrid_unireplknet_a_dinov3.pth"
    if best_path.exists() or (output / "history.json").exists():
        raise FileExistsError("Archive the existing experiment or choose another output_dir")
    train = make_loader(bundle.records["train"], processor, config, train=True)
    validation = make_loader(bundle.records["val"], processor, config)
    print(json.dumps({"pretrained_smoke": smoke_test(model, next(iter(train)), config.device)}))
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best = [], -1.0
    for epoch in range(1, config.epochs + 1):
        loss = train_head_epoch(model, train, optimizer, config.device)
        metrics = evaluate_hybrid(model, validation, config.device)
        row = {"epoch": epoch, "train_loss": loss, "validation": metrics}
        history.append(row)
        print(json.dumps(row))
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            save_head_checkpoint(best_path, model, processor, config, epoch, metrics)
        (output / "history.json").write_text(json.dumps(history, indent=2))
    checkpoint = restore_head(model, best_path)
    _write_metrics(output, "validation", checkpoint["validation_metrics"], CLASS_NAMES)
    return best_path


def evaluate_final_test(config, model, processor, checkpoint_path):
    """Separate explicit action after validation choices are locked; never tune on test."""
    output = Path(config.output_dir)
    if (output / "test_metrics.json").exists():
        raise FileExistsError("Final test already recorded; read saved results instead of repeating it")
    restore_head(model, checkpoint_path)
    bundle = validate_hybrid_bundle(config.data_root)
    model.to(config.device)
    loader = make_loader(bundle.records["test"], processor, config)
    metrics = evaluate_hybrid(model, loader, config.device)
    _write_metrics(output, "test", metrics, CLASS_NAMES)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--baseline-checkpoint")
    parser.add_argument("--output-dir", default="/kaggle/working/tomatocare_hybrid")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train", action="store_true", help="Opt in; otherwise perform smoke only")
    args = parser.parse_args()
    config = HybridTrainConfig(data_root=args.data_root, output_dir=args.output_dir,
                               batch_size=args.batch_size, epochs=args.epochs, device=args.device)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA unavailable; enable Kaggle GPU")
    bundle = validate_hybrid_bundle(config.data_root)
    seed_everything(config.seed)
    model, processor = create_hybrid_model(args.baseline_checkpoint, hf_token=os.getenv("HF_TOKEN"))
    if args.train:
        run_hybrid_training(config, model, processor)
    else:
        print(json.dumps(smoke_test(model, next(iter(make_loader(bundle.records["train"], processor, config))), config.device)))


if __name__ == "__main__":
    main()
