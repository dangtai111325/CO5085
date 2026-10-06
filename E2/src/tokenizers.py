from __future__ import annotations

import torch
from torch import nn


class RowTokenizer(nn.Module):
    """Treat each image row as one token."""
    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(28, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)  # [B, 28, 28]
        return self.proj(x)


class PatchTokenizer(nn.Module):
    """Non-overlapping image patches implemented explicitly with unfold."""
    def __init__(self, patch_size: int = 4, d_model: int = 64) -> None:
        super().__init__()
        if 28 % patch_size != 0:
            raise ValueError("patch_size must divide 28 for Fashion-MNIST")
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)  # B,C,H/P,W/P,p,p
        patches = patches.contiguous().view(x.size(0), -1, p * p)
        return self.proj(patches)


class CNNStemTokenizer(nn.Module):
    """CNN stem followed by spatial flattening into a token sequence."""
    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, d_model, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)  # [B,D,7,7]
        return z.flatten(2).transpose(1, 2)
