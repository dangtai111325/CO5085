from __future__ import annotations

import torch
from torch.utils.data import TensorDataset


def classification_dataset(n: int = 32, channels: int = 1, height: int = 28, width: int = 28, classes: int = 10) -> TensorDataset:
    x = torch.randn(n, channels, height, width)
    y = torch.randint(0, classes, (n,))
    return TensorDataset(x, y)
