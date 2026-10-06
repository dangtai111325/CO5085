from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def fashion_mnist_loaders(root: str | Path = "data", batch_size: int = 128, seed: int = 42, num_workers: int = 2):
    tfm = transforms.ToTensor()
    train_full = datasets.FashionMNIST(root=str(root), train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(root=str(root), train=False, download=True, transform=tfm)
    train, val = random_split(train_full, [55_000, 5_000], generator=torch.Generator().manual_seed(seed))
    kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (
        DataLoader(train, shuffle=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
    )
