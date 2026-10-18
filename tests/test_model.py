import torch
from torch import nn

from tomatocare.model import CLASS_NAMES, create_unireplknet_a


def test_unireplknet_a_forward():
    torch.set_num_threads(2)
    model = create_unireplknet_a().eval()
    assert len(CLASS_NAMES) == 4
    assert tuple(len(stage) for stage in model.stages) == (2, 2, 6, 2)
    assert model.head.in_features == 320
    assert model.head.out_features == 4
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 224, 224))
    assert logits.shape == (1, 4)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(nn.CrossEntropyLoss()(logits, torch.tensor([0])))
