"""Validated readers for the Kaggle metadata, labels, and images bundle."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PREPROCESSING = {
    "color_mode": "RGB", "image_size": [224, 224], "mean": list(MEAN),
    "std": list(STD), "resize": "exact", "interpolation": "bilinear",
    "train_augmentation": "random horizontal flip (p=0.5)",
}


def read_labels(path):
    """Accept ordered lists, class->index, index->class, or named wrappers.

    Never alphabetically sort class names: numeric IDs define logit order.
    """
    obj = json.loads(Path(path).read_text())
    if isinstance(obj, dict):
        wrappers = [key for key in ("class_to_idx", "idx_to_class", "labels", "classes") if key in obj]
        if len(wrappers) > 1:
            raise ValueError("Ambiguous labels.json wrappers")
        if wrappers:
            obj = obj[wrappers[0]]
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        names = obj
    elif isinstance(obj, dict) and all(isinstance(v, int) and not isinstance(v, bool) for v in obj.values()):
        if sorted(obj.values()) != list(range(len(obj))):
            raise ValueError("Label IDs must be unique, contiguous, and start at zero")
        names = [k for k, _ in sorted(obj.items(), key=lambda item: item[1])]
    elif isinstance(obj, dict) and all(str(k).isdigit() and isinstance(v, str) for k, v in obj.items()):
        if sorted(map(int, obj)) != list(range(len(obj))):
            raise ValueError("Label IDs must be contiguous and start at zero")
        names = [obj[str(i)] for i in range(len(obj))]
    else:
        raise ValueError("Unsupported labels.json: expected ordered names or class/index mapping")
    normalize = lambda x: "".join(c for c in x.lower() if c.isalnum())
    if len(names) != 4 or {normalize(n) for n in names} != {"healthy", "earlyblight", "lateblight", "leafmold"}:
        raise ValueError(f"Expected the four TomatoCare classes; got {names}")
    return tuple(names)


@dataclass(frozen=True)
class Record:
    path: Path
    target: int
    split: str


@dataclass
class Bundle:
    root: Path
    class_names: tuple
    records: dict


def _column(columns, candidates, explicit, role):
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Missing {role} column {explicit!r}; available: {columns}")
        return explicit
    matches = [c for c in columns if c.lower() in candidates]
    if len(matches) != 1:
        raise ValueError(f"Cannot choose {role} column from {columns}; specify --{role}-column")
    return matches[0]


def discover_bundle(search_root, path_column=None, label_column=None, split_column=None):
    search_root = Path(search_root).resolve()
    candidates = [p.parent for p in search_root.rglob("metadata.csv")
                  if (p.parent / "labels.json").is_file() and (p.parent / "images").is_dir()]
    if len(candidates) != 1:
        raise ValueError(f"Expected one bundle containing metadata.csv, labels.json, images/ under {search_root}; found {candidates}. Pass --data-root to narrow the search.")
    root = candidates[0]
    names = read_labels(root / "labels.json")
    class_to_idx = {name: i for i, name in enumerate(names)}
    records = {s: [] for s in ("train", "val", "test")}
    aliases = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val", "test": "test", "testing": "test"}
    seen = set()
    seen_hashes = set()
    with (root / "metadata.csv").open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        pc = _column(columns, {"path", "image_path", "filepath", "file_path", "filename", "file_name", "image"}, path_column or ("image_path" if "image_path" in columns else None), "path")
        lc = _column(columns, {"label", "class", "class_name", "label_name", "label_id", "class_id", "target", "disease"}, label_column or ("label" if "label" in columns else None), "label")
        sc = _column(columns, {"split", "subset", "partition"}, split_column, "split")
        for line, row in enumerate(reader, start=2):
            try:
                label = row[lc].strip()
                if lc == "label" and label not in class_to_idx:
                    raise ValueError(f"Unknown class name {label!r}")
                target = class_to_idx[label] if label in class_to_idx else int(label)
                if "label_id" in columns and int(row["label_id"]) != target:
                    raise ValueError("label and label_id disagree with labels.json")
                if "sha256" in columns:
                    digest = row["sha256"].strip().lower()
                    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                        raise ValueError("Invalid sha256 digest")
                    if digest in seen_hashes:
                        raise ValueError("Duplicate sha256 in metadata; inspect deduplication/split leakage")
                    seen_hashes.add(digest)
                if not 0 <= target < len(names):
                    raise ValueError(f"Out-of-range label {label!r}")
                split = aliases[row[sc].strip().lower()]
                rel = Path(row[pc].strip())
                if rel.is_absolute() or ".." in rel.parts:
                    raise ValueError("Image path must be relative and stay within images/")
                options = {(root / rel).resolve(), (root / "images" / rel).resolve()}
                options = [p for p in options if p.is_relative_to((root / "images").resolve()) and p.is_file()]
                if len(options) != 1:
                    raise ValueError(f"Missing or ambiguous image path {rel}")
                path = options[0]
                if path in seen:
                    raise ValueError(f"Image appears more than once in metadata: {rel}")
                seen.add(path)
                records[split].append(Record(path, target, split))
            except (ValueError, KeyError, AttributeError) as exc:
                raise ValueError(f"Invalid metadata.csv line {line}: {exc}") from exc
    for split, rows in records.items():
        if {r.target for r in rows} != set(range(4)):
            raise ValueError(f"Split {split} must contain all four classes")
    return Bundle(root, names, records)


class LeafDataset(Dataset):
    def __init__(self, records, train=False):
        self.records = records
        steps = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)]
        if train:
            steps.append(transforms.RandomHorizontalFlip())
        steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as image:
            image = self.transform(image.convert("RGB"))
        return image, record.target
