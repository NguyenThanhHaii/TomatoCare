"""Offline pipeline tests. Synthetic images/weights are NOT training evidence."""

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from tomatocare.data import LeafDataset, discover_bundle
from tomatocare.model import create_unireplknet_a
from tomatocare.pretrained import load_imagenet_backbone
from tomatocare.train import TrainConfig, evaluate, freeze_backbone, run_training, summarize_metrics, train_one_epoch


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def bundle_path(tmp_path):
    root = tmp_path / "input" / "dataset-slug" / "tomatocare_data"
    root.mkdir(parents=True)
    # Deliberately different insertion and ID order to catch alphabetical ordering.
    labels = {"Healthy": 1, "Early Blight": 3, "Late Blight": 0, "Leaf Mold": 2}
    (root / "labels.json").write_text(json.dumps(labels))
    rows = []
    for split_index, split in enumerate(("train", "val", "test")):
        for name, idx in labels.items():
            rel = Path("images") / split / name / "sample.png"
            path = root / rel
            path.parent.mkdir(parents=True)
            Image.new("RGB", (37, 41), (idx * 50, split_index * 70, 20)).save(path)
            rows.append({"image_path": str(rel), "label": name, "label_id": idx, "split": split,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "source_image_path": f"PlantVillage/{name}/sample.png"})
    _write_rows(root, rows)
    return root


def _write_rows(root, rows):
    with (root / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "label", "label_id", "split", "sha256", "source_image_path"])
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(root):
    with (root / "metadata.csv").open() as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def synthetic_checkpoint(tmp_path):
    # Mimics the original checkpoint format only; weights are random test fixtures.
    model = create_unireplknet_a()
    model.head = nn.Linear(320, 1000)
    path = tmp_path / "synthetic_imagenet_format.pth"
    torch.save({"model": model.state_dict()}, path)
    return path


def test_schema_batch_and_label_order(bundle_path):
    bundle = discover_bundle(bundle_path.parents[1])
    assert bundle.class_names == ("Late Blight", "Healthy", "Leaf Mold", "Early Blight")
    assert {s: len(r) for s, r in bundle.records.items()} == {"train": 4, "val": 4, "test": 4}
    images, labels = next(iter(DataLoader(LeafDataset(bundle.records["train"], train=True), batch_size=4)))
    assert images.shape == (4, 3, 224, 224)
    assert labels.tolist() == [1, 3, 0, 2]
    assert torch.isfinite(images).all()
    val = LeafDataset(bundle.records["val"])
    assert torch.equal(val[0][0], val[0][0])


@pytest.mark.parametrize("field,value,match", [
    ("label", "Unknown", "Unknown class"),
    ("label_id", "3", "disagree"),
    ("split", "holdout", "Invalid metadata"),
    ("image_path", "../outside.png", "relative"),
    ("image_path", "images/missing.png", "Missing or ambiguous"),
    ("sha256", "bad", "Invalid sha256"),
])
def test_reject_invalid_metadata(bundle_path, field, value, match):
    rows = _read_rows(bundle_path)
    rows[0][field] = value
    _write_rows(bundle_path, rows)
    with pytest.raises(ValueError, match=match):
        discover_bundle(bundle_path)


def test_reject_duplicate_hash(bundle_path):
    rows = _read_rows(bundle_path)
    rows[4]["sha256"] = rows[0]["sha256"]
    _write_rows(bundle_path, rows)
    with pytest.raises(ValueError, match="Duplicate sha256"):
        discover_bundle(bundle_path)


def test_pretrained_loading_strict_and_head_preserved(synthetic_checkpoint, tmp_path):
    model = create_unireplknet_a()
    head = model.head.weight.detach().clone()
    load_imagenet_backbone(model, synthetic_checkpoint)
    assert torch.equal(head, model.head.weight)
    bad = torch.load(synthetic_checkpoint, weights_only=True)
    bad["model"].pop("norm.weight")
    path = tmp_path / "invalid.pth"
    torch.save(bad, path)
    with pytest.raises(RuntimeError, match="No random-init fallback"):
        load_imagenet_backbone(create_unireplknet_a(), path)


def test_download_failure_is_fatal(monkeypatch):
    import huggingface_hub
    def fail(**kwargs):
        raise OSError("offline test")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fail)
    with pytest.raises(RuntimeError, match="--pretrained-checkpoint"):
        load_imagenet_backbone(create_unireplknet_a())


def test_freezing_includes_buffers_and_only_head_changes(bundle_path):
    model = create_unireplknet_a()
    freeze_backbone(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loader = DataLoader(LeafDataset(discover_bundle(bundle_path).records["train"], train=True), batch_size=4)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    train_one_epoch(model, loader, optimizer, torch.device("cpu"))
    after = model.state_dict()
    assert not torch.equal(before["head.weight"], after["head.weight"])
    assert all(torch.equal(v, after[k]) for k, v in before.items() if not k.startswith("head."))
    assert not model.training
    assert model.head.training


def test_metrics_known_predictions():
    metrics = summarize_metrics([0, 1, 2, 3], [0, 1, 2, 3], ["a", "b", "c", "d"])
    assert metrics["accuracy"] == metrics["macro_f1"] == 1.0
    assert metrics["confusion_matrix"] == torch.eye(4, dtype=torch.int).tolist()


def test_small_training_checkpoint_and_single_test_pass(bundle_path, synthetic_checkpoint, tmp_path, monkeypatch):
    import tomatocare.train as training
    calls = []
    def track(model, loader, device, names):
        calls.append(loader.dataset.records[0].split)
        return evaluate(model, loader, device, names)
    monkeypatch.setattr(training, "evaluate", track)
    output = tmp_path / "working"
    config = TrainConfig(data_root=str(bundle_path), output_dir=str(output),
                         pretrained_checkpoint=str(synthetic_checkpoint), epochs=2,
                         batch_size=4, num_workers=0, device="cpu", check_expected_counts=False)
    path, metrics = run_training(config)
    saved = torch.load(path, weights_only=True)
    assert calls == ["val", "val", "test"]
    assert saved["class_to_idx"] == {"Late Blight": 0, "Healthy": 1, "Leaf Mold": 2, "Early Blight": 3}
    assert saved["preprocessing"]["image_size"] == [224, 224]
    assert saved["state_dict"]["head.weight"].shape == (4, 320)
    history = json.loads((output / "history.json").read_text())
    assert saved["best_validation_macro_f1"] == max(row["validation"]["macro_f1"] for row in history)
    assert (output / "test_confusion_matrix.png").is_file()
    assert (output / "test_confusion_matrix.csv").is_file()
    assert "classification_report" in metrics
    # Restoring the saved checkpoint can reproduce inference without downloads.
    restored = create_unireplknet_a().eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    with torch.inference_mode():
        assert restored(torch.randn(1, 3, 224, 224)).shape == (1, 4)
    with pytest.raises(FileExistsError):
        run_training(config)


def test_notebook_cells_compile():
    notebook = json.loads((Path(__file__).parents[1] / "notebooks/train_unireplknet_a_kaggle.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")
            assert not cell["outputs"]
