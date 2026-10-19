"""Phase 4 offline tests. DINOv3 is explicitly mocked; no pretrained claim."""

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

from tomatocare.data import LeafDataset, PREPROCESSING
from tomatocare.hybrid_data import HybridLeafDataset
from tomatocare.hybrid_model import HybridModel, create_hybrid_model, load_baseline_encoder
from tomatocare.model import CLASS_NAMES, create_unireplknet_a
from tomatocare.train_hybrid import HybridTrainConfig, restore_head, save_head_checkpoint, smoke_test, train_head_epoch, validate_hybrid_bundle


@pytest.fixture(autouse=True)
def small_cpu_pool():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


class MockDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.3))
        self.bn = nn.BatchNorm2d(3)
        self.dropout = nn.Dropout(0.9)
        self.config = SimpleNamespace(model_type="dinov3_vit", hidden_size=384, patch_size=16,
                                      num_register_tokens=4, _commit_hash="MOCK-DINO-REVISION")

    def forward(self, pixel_values, return_dict):
        assert return_dict and not torch.is_grad_enabled() and not self.training
        pixel_values = self.dropout(self.bn(pixel_values))
        b, _, h, w = pixel_values.shape
        tokens = torch.full((b, 1 + 4 + (h // 16) * (w // 16), 384), -1000.0, device=pixel_values.device)
        tokens[:, 0] = self.scale * 3
        tokens[:, 1:5] = 1000.0  # Registers must never be treated as CLS.
        return SimpleNamespace(last_hidden_state=tokens, pooler_output=tokens[:, 1])


class MockProcessor:
    def __call__(self, images, return_tensors):
        assert images.mode == "RGB" and return_tensors == "pt"
        return {"pixel_values": transforms.ToTensor()(F.resize(images, [256, 256])).unsqueeze(0)}

    def to_dict(self):
        return {"image_processor_type": "MockProcessor", "size": {"height": 256, "width": 256},
                "token": "secret-for-unit-tests-only"}


@pytest.fixture
def baseline_checkpoint(tmp_path):
    path = tmp_path / "random_baseline_fixture.pth"
    torch.save({"architecture": "unireplknet_a", "state_dict": create_unireplknet_a().state_dict(),
                "class_names": list(CLASS_NAMES), "class_to_idx": {n: i for i, n in enumerate(CLASS_NAMES)},
                "preprocessing": PREPROCESSING}, path)
    return path


@pytest.fixture
def model(baseline_checkpoint):
    encoder, provenance = load_baseline_encoder(baseline_checkpoint)
    return HybridModel(encoder, MockDino(), {"baseline": provenance, "dinov3": {"model_id": "MOCK-DINO"}})


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / "nested" / "tomatocare_data"
    root.mkdir(parents=True)
    (root / "labels.json").write_text(json.dumps({n: i for i, n in enumerate(CLASS_NAMES)}))
    rows = []
    for split_idx, split in enumerate(("train", "val", "test")):
        for label_id, label in enumerate(CLASS_NAMES):
            rel = Path("images") / split / label / "leaf.png"
            path = root / rel
            path.parent.mkdir(parents=True)
            pixels = np.zeros((41, 37, 4), dtype=np.uint8)
            pixels[:, :, 0] = np.arange(37)
            pixels[:, :, 1] = label_id * 40
            pixels[:, :, 2] = split_idx * 50
            pixels[:, :, 3] = 255
            Image.fromarray(pixels).save(path)
            rows.append({"image_path": str(rel), "label": label, "label_id": label_id, "split": split,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source_image_path": str(rel)})
    with (root / "metadata.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return validate_hybrid_bundle(tmp_path, check_counts=False)


def test_shapes_cls_register_handling_and_stage4_absent(model):
    images = torch.randn(2, 3, 224, 224)
    a, b, fused = model.forward_features(images, images)
    assert a.shape == (2, 160)
    assert b.shape == (2, 384)
    assert fused.shape == (2, 544)
    assert torch.allclose(b, torch.full_like(b, 0.9))
    assert model(images, images).shape == (2, 4)
    assert tuple(len(stage) for stage in model.unireplknet_encoder.stages) == (2, 2, 6)
    assert len(model.unireplknet_encoder.downsample_layers) == 3
    assert not hasattr(model.unireplknet_encoder, "head") and not hasattr(model.unireplknet_encoder, "norm")
    assert all(not k.startswith(("stages.3.", "downsample_layers.3.", "head.", "norm.")) for k in model.unireplknet_encoder.state_dict())
    assert isinstance(model.classifier[0], nn.LayerNorm) and model.classifier[0].normalized_shape == (544,)
    assert model.classifier[1].in_features == 544 and model.classifier[1].out_features == 4


def test_only_fusion_head_gradients_and_frozen_buffers(model):
    model.train()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    pixels = torch.randn(2, 3, 224, 224, requires_grad=True)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    logits = model(pixels, pixels)
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert pixels.grad is None
    assert not model.unireplknet_encoder.training and not model.dinov3_encoder.training
    assert all(not module.training for module in model.unireplknet_encoder.modules())
    assert all(p.grad is None and not p.requires_grad for p in model.unireplknet_encoder.parameters())
    assert all(p.grad is None and not p.requires_grad for p in model.dinov3_encoder.parameters())
    assert all(p.grad is not None for p in model.classifier.parameters())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 3268
    optimizer.step()
    after = model.state_dict()
    assert all(torch.equal(v, after[k]) for k, v in before.items() if not k.startswith("classifier."))
    assert not torch.equal(before["classifier.1.weight"], after["classifier.1.weight"])


def test_loaded_baseline_keys_match_exactly(baseline_checkpoint):
    encoder, _ = load_baseline_encoder(baseline_checkpoint)
    source = torch.load(baseline_checkpoint, weights_only=True)["state_dict"]
    assert all(torch.equal(v, source[k]) for k, v in encoder.state_dict().items())


@pytest.mark.parametrize("kind", ["missing", "shape", "prefix", "labels", "mapping", "preprocessing"])
def test_reject_incompatible_baseline(baseline_checkpoint, tmp_path, kind):
    checkpoint = torch.load(baseline_checkpoint, weights_only=True)
    key = next(iter(checkpoint["state_dict"]))
    if kind == "missing": checkpoint["state_dict"].pop(key)
    elif kind == "shape": checkpoint["state_dict"][key] = torch.zeros(1)
    elif kind == "prefix": checkpoint["state_dict"] = {"module." + k: v for k, v in checkpoint["state_dict"].items()}
    elif kind == "labels": checkpoint["class_names"] = list(reversed(CLASS_NAMES))
    elif kind == "mapping": checkpoint["class_to_idx"]["Healthy"] = 3
    else: checkpoint["preprocessing"] = {**PREPROCESSING, "interpolation": "bicubic"}
    path = tmp_path / "bad_baseline.pth"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        load_baseline_encoder(path)


def _fake_transformers(monkeypatch, *, fail=False, missing=False):
    module = ModuleType("transformers")
    calls = []
    def processor_loader(model_id, **kwargs):
        calls.append(("processor", model_id, kwargs))
        return MockProcessor()
    def model_loader(model_id, **kwargs):
        calls.append(("model", model_id, kwargs))
        if fail: raise RuntimeError("do not log secret-for-unit-tests-only")
        return MockDino(), {"missing_keys": ["block.weight"] if missing else [], "unexpected_keys": []}
    module.AutoImageProcessor = SimpleNamespace(from_pretrained=processor_loader)
    module.AutoModel = SimpleNamespace(from_pretrained=model_loader)
    monkeypatch.setitem(sys.modules, "transformers", module)
    return calls


def test_factory_requires_pretrained_safetensors_and_keeps_token_out_of_provenance(baseline_checkpoint, monkeypatch):
    calls = _fake_transformers(monkeypatch)
    model, processor = create_hybrid_model(baseline_checkpoint, hf_token="secret-for-unit-tests-only")
    assert all(call[1] == "facebook/dinov3-vits16-pretrain-lvd1689m" for call in calls)
    options = calls[1][2]
    assert options["use_safetensors"] and options["output_loading_info"] and not options["trust_remote_code"]
    assert "secret-for-unit-tests-only" not in str(model.provenance)


@pytest.mark.parametrize("kind", ["gated", "missing_weights"])
def test_no_dino_random_pretrained_fallback(baseline_checkpoint, monkeypatch, kind):
    _fake_transformers(monkeypatch, fail=kind == "gated", missing=kind == "missing_weights")
    with pytest.raises(RuntimeError, match="No random-pretrained fallback") as caught:
        create_hybrid_model(baseline_checkpoint, hf_token="secret-for-unit-tests-only")
    assert "secret-for-unit-tests-only" not in str(caught.value)


def test_public_baseline_download_failure_is_fatal(monkeypatch):
    import huggingface_hub
    def blocked(**kwargs):
        assert kwargs["token"] is False
        raise OSError("offline")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", blocked)
    with pytest.raises(RuntimeError, match="no random-weight fallback"):
        load_baseline_encoder()


def test_dual_loader_preserves_baseline_preprocessing(bundle):
    dataset = HybridLeafDataset(bundle.records["val"], MockProcessor())
    data, labels = next(iter(DataLoader(dataset, batch_size=2)))
    assert data["unireplknet"].shape == (2, 3, 224, 224)
    assert data["dinov3"].shape == (2, 3, 256, 256)  # Independently chosen by processor.
    assert labels.tolist() == [0, 1]
    assert torch.equal(dataset[0][0]["unireplknet"], LeafDataset(bundle.records["val"])[0][0])
    assert torch.equal(dataset[0][0]["dinov3"], dataset[0][0]["dinov3"])


def test_shared_train_flip_only(bundle, monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.0))
    train = HybridLeafDataset(bundle.records["train"], MockProcessor(), train=True)
    plain = HybridLeafDataset(bundle.records["train"], MockProcessor())
    assert torch.allclose(train[0][0]["unireplknet"], F.hflip(plain[0][0]["unireplknet"]))
    assert torch.allclose(train[0][0]["dinov3"], F.hflip(plain[0][0]["dinov3"]), atol=1e-6)


def test_smoke_and_one_offline_head_step(model, bundle):
    loader = DataLoader(HybridLeafDataset(bundle.records["train"], MockProcessor()), batch_size=2)
    report = smoke_test(model, next(iter(loader)), "cpu")
    assert report["fusion"] == [2, 544] and report["logits"] == [2, 4]
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    assert np.isfinite(train_head_epoch(model, loader, optimizer, "cpu"))


def test_checkpoint_has_only_head_not_dino_weights_or_credentials(model, tmp_path):
    path = tmp_path / "best_hybrid.pth"
    save_head_checkpoint(path, model, MockProcessor(), HybridTrainConfig(device="cpu"), 1, {"macro_f1": 0.0})
    checkpoint = restore_head(model, path)
    assert set(checkpoint["classifier_state_dict"]) == {"0.weight", "0.bias", "1.weight", "1.bias"}
    assert "secret-for-unit-tests-only" not in str(checkpoint)
    assert "token" not in checkpoint["preprocessing"]["dinov3"]
    assert path.stat().st_size < 100_000
    checkpoint["provenance"] = {"different": True}
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="sources/revisions differ"):
        restore_head(model, path)


def test_dataset_counts_stay_strict(bundle):
    with pytest.raises(ValueError, match="3816/816/820"):
        validate_hybrid_bundle(bundle.root)


def test_notebook_is_opt_in_and_contains_no_outputs():
    path = Path(__file__).parents[1] / "notebooks/train_hybrid_dinov3_kaggle.ipynb"
    notebook = json.loads(path.read_text())
    source = "".join("".join(cell["source"]) for cell in notebook["cells"])
    assert "RUN_TRAINING = False" in source and "RUN_FINAL_TEST = False" in source
    assert 'get_secret("HF_TOKEN")' in source
    assert '"--branch", "feature/hybrid-dinov3", "--single-branch"' in source
    assert "print(token" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(path), "exec")
            assert cell["outputs"] == []


def test_restore_accepts_pinned_revision_for_same_encoder_weights(model, tmp_path):
    model.provenance["dinov3"].update(requested_revision="main", resolved_revision="test-commit")
    path = tmp_path / "head.pth"
    save_head_checkpoint(path, model, MockProcessor(), HybridTrainConfig(device="cpu"), 1, {"macro_f1": 0.0})
    model.provenance["dinov3"]["requested_revision"] = "test-commit"
    model.provenance["baseline"]["requested_revision"] = "another-selector-for-identical-file"
    assert restore_head(model, path)["epoch"] == 1
