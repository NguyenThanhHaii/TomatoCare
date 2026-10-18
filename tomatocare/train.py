"""Head-only UniRepLKNet-A training. Run: python -m tomatocare.train --help."""

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

from .data import LeafDataset, PREPROCESSING, discover_bundle
from .model import create_unireplknet_a
from .pretrained import load_imagenet_backbone


@dataclass
class TrainConfig:
    data_root: str = "/kaggle/input"
    output_dir: str = "/kaggle/working"
    pretrained_checkpoint: str | None = None
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.05
    num_workers: int = 2
    seed: int = 42
    device: str = "cuda"
    path_column: str | None = None
    label_column: str | None = None
    split_column: str | None = None
    check_expected_counts: bool = True


def summarize_metrics(targets, predictions, class_names):
    labels = list(range(len(class_names)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision), "recall_macro": float(recall), "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=labels).tolist(),
        "classification_report": classification_report(targets, predictions, labels=labels,
                                                        target_names=list(class_names), output_dict=True, zero_division=0),
    }


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.head.parameters():
        param.requires_grad_(True)
    # Keep backbone BatchNorm running statistics frozen as well as parameters.
    model.eval()
    model.head.train()


def train_one_epoch(model, loader, optimizer, device):
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss()
    total, count = 0.0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(targets)
        count += len(targets)
    return total / count


@torch.inference_mode()
def evaluate(model, loader, device, class_names):
    model.eval()
    targets_all, predictions = [], []
    total = 0.0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        total += nn.functional.cross_entropy(logits, targets, reduction="sum").item()
        targets_all.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = summarize_metrics(targets_all, predictions, class_names)
    metrics["loss"] = total / len(targets_all)
    return metrics


def _write_metrics(output, split, metrics, class_names):
    (output / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    with (output / f"{split}_confusion_matrix.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true / predicted", *class_names])
        writer.writerows([name, *row] for name, row in zip(class_names, metrics["confusion_matrix"]))
    # Headless rendering works on Kaggle and local CPU test environments.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]), display_labels=class_names).plot(ax=ax, colorbar=False)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / f"{split}_confusion_matrix.png")
    plt.close(fig)


def run_training(config):
    if config.epochs < 1 or config.batch_size < 1 or config.num_workers < 0:
        raise ValueError("epochs/batch_size must be positive and num_workers nonnegative")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Enable a Kaggle GPU accelerator, or explicitly use --device cpu for a smoke test.")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    bundle = discover_bundle(config.data_root, config.path_column, config.label_column, config.split_column)
    counts = {s: len(rows) for s, rows in bundle.records.items()}
    if config.check_expected_counts and counts != {"train": 3816, "val": 816, "test": 820}:
        raise ValueError(f"Dataset counts differ from expected 3816/816/820: {counts}. Inspect the attached dataset/version before training; --allow-count-mismatch is only for intentional alternate bundles.")
    print(json.dumps({"dataset": str(bundle.root), "classes_in_logit_order": bundle.class_names, "split_counts": counts}))
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best_unireplknet_a.pth"
    # Avoid silently overwriting an experiment or reusing its held-out test results.
    if checkpoint_path.exists() or (output / "test_metrics.json").exists():
        raise FileExistsError(f"Previous experiment exists in {output}; archive it or choose a new --output-dir")
    generator = torch.Generator().manual_seed(config.seed)
    loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                         pin_memory=device.type == "cuda")
    train_loader = DataLoader(LeafDataset(bundle.records["train"], train=True), shuffle=True,
                              generator=generator, **loader_kwargs)
    val_loader = DataLoader(LeafDataset(bundle.records["val"]), shuffle=False, **loader_kwargs)
    model = create_unireplknet_a()
    source = load_imagenet_backbone(model, config.pretrained_checkpoint)
    model.to(device)
    freeze_backbone(model)
    # Sanity check reads only training samples, never the held-out test split.
    images, targets = next(iter(train_loader))
    with torch.inference_mode():
        logits = model(images.to(device))
    if images.shape[1:] != (3, 224, 224) or logits.shape != (len(targets), 4) or not torch.isfinite(logits).all():
        raise RuntimeError("Training batch/forward sanity check failed")
    print(f"Batch verified: images={tuple(images.shape)}, labels={tuple(targets.shape)}, logits={tuple(logits.shape)}")
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history, best_f1 = [], -1.0
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, bundle.class_names)
        row = {"epoch": epoch, "train_loss": train_loss, "validation": val_metrics}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            checkpoint = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "architecture": "unireplknet_a", "depths": [2, 2, 6, 2], "dims": [40, 80, 160, 320],
                "class_names": list(bundle.class_names),
                "class_to_idx": {name: i for i, name in enumerate(bundle.class_names)},
                "preprocessing": PREPROCESSING, "config": asdict(config), "pretrained_source": source,
                "epoch": epoch, "best_validation_macro_f1": best_f1, "validation_metrics": val_metrics,
                "split_counts": counts,
            }
            temporary = checkpoint_path.with_suffix(".pth.tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(checkpoint_path)
        (output / "history.json").write_text(json.dumps(history, indent=2))
    # Select solely on validation, restore the winner, then evaluate test exactly once.
    best = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["state_dict"], strict=True)
    _write_metrics(output, "validation", best["validation_metrics"], bundle.class_names)
    test_loader = DataLoader(LeafDataset(bundle.records["test"]), shuffle=False, **loader_kwargs)
    test_metrics = evaluate(model, test_loader, device, bundle.class_names)
    _write_metrics(output, "test", test_metrics, bundle.class_names)
    print(json.dumps({"best_epoch": best["epoch"], "test": test_metrics, "checkpoint": str(checkpoint_path)}))
    return checkpoint_path, test_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/kaggle/input")
    parser.add_argument("--output-dir", default="/kaggle/working")
    parser.add_argument("--pretrained-checkpoint")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for role in ("path", "label", "split"):
        parser.add_argument(f"--{role}-column")
    parser.add_argument("--allow-count-mismatch", dest="check_expected_counts", action="store_false")
    run_training(TrainConfig(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
