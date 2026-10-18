# TomatoCare

Tomato leaf disease classification. Phase 1 implements the original **UniRepLKNet-A** architecture for four classes: Healthy, Early Blight, Late Blight, Leaf Mold. DINOv3 integration is reserved for a later phase.

## Setup and test

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
```

For a CPU-only environment, install CPU PyTorch first, then the remaining dependencies:

```bash
python -m pip install "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
```

The test constructs the model with randomly initialized weights and runs a CPU forward pass on a `[1, 3, 224, 224]` tensor. It checks for four raw logits and compatibility with `CrossEntropyLoss`. No pretrained weights are downloaded or needed. The optional CUDA iGEMM extension is disabled.

Application code can import `create_unireplknet_a` and `CLASS_NAMES` from `tomatocare.model`. The factory is the intended place to add later backbone choices, including DINOv3 after its weights become available. This phase does not train a model or report disease recognition accuracy.

## Source and license

`tomatocare/vendor/unireplknet.py` is copied without modification from [AILab-CVC/UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet), commit `b5677d8fb0e7b8af61eac39c1bf02727f5afc1d5`, specifically its root `unireplknet.py`. The authors license that source under Apache License 2.0; the corresponding license text is in `tomatocare/vendor/LICENSE`. UniRepLKNet credits RepLKNet, ConvNeXt, timm, DINO and DeiT in its source header. No DINOv3 source or pretrained weights are included.
