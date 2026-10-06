from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


GTSRB_MEAN = (0.3403, 0.3121, 0.3214)
GTSRB_STD = (0.2724, 0.2608, 0.2669)


def build_transforms(image_size: int = 224, train: bool = True):
    ops: list[object] = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [transforms.RandomRotation(10), transforms.ColorJitter(brightness=0.15, contrast=0.15)]
    ops += [transforms.ToTensor(), transforms.Normalize(GTSRB_MEAN, GTSRB_STD)]
    return transforms.Compose(ops)


def gtsrb_loaders(root: str | Path = "data", batch_size: int = 64, image_size: int = 224, seed: int = 42, num_workers: int = 4):
    train_full = datasets.GTSRB(str(root), split="train", download=True, transform=build_transforms(image_size, True))
    test = datasets.GTSRB(str(root), split="test", download=True, transform=build_transforms(image_size, False))
    val_size = max(1, int(0.1 * len(train_full)))
    train_size = len(train_full) - val_size
    train, val = random_split(train_full, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    # Validation must use deterministic transform, so replace the underlying dataset for a matched split.
    val_base = datasets.GTSRB(str(root), split="train", download=False, transform=build_transforms(image_size, False))
    val.dataset = val_base
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return DataLoader(train, shuffle=True, **kwargs), DataLoader(val, shuffle=False, **kwargs), DataLoader(test, shuffle=False, **kwargs)
