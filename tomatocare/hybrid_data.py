"""One photo, shared training flip, two independent pretrained preprocessing paths."""

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F

from .data import MEAN, STD


class HybridLeafDataset(Dataset):
    def __init__(self, records, dino_processor, train=False):
        self.records = records
        self.dino_processor = dino_processor
        self.train = train
        self.baseline = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record.path) as original:
            image = original.convert("RGB")
            # Keep baseline Resize -> flip -> normalize behavior exactly as Phase 2.
            baseline = F.resize(image, [224, 224], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            if self.train and torch.rand(()).item() < 0.5:
                baseline, image = F.hflip(baseline), F.hflip(image)
            baseline = self.baseline(baseline)
            # Pass the original RGB view, not the baseline normalized tensor, to HF.
            pixels = self.dino_processor(images=image, return_tensors="pt")["pixel_values"]
        if pixels.ndim != 4 or pixels.shape[0] != 1 or pixels.shape[1] != 3:
            raise ValueError("DINOv3 processor must return [1,3,H,W] pixel_values")
        return {"unireplknet": baseline, "dinov3": pixels[0]}, record.target
