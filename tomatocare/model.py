"""TomatoCare model construction; future backbones can be added here."""

from torch import nn

from .vendor.unireplknet import UniRepLKNet

CLASS_NAMES = ("Healthy", "Early Blight", "Late Blight", "Leaf Mold")


def create_unireplknet_a() -> nn.Module:
    """Create the author's UniRepLKNet-A architecture with four raw logits.

    Weights are initialized from scratch. This function never downloads weights.
    """
    return UniRepLKNet(
        in_chans=3,
        num_classes=len(CLASS_NAMES),
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        attempt_use_lk_impl=False,
    )
