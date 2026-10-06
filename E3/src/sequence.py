from __future__ import annotations

import torch


def image_to_sequence(x: torch.Tensor, mode: str = "rows", patch_size: int = 4) -> torch.Tensor:
    if mode == "rows":
        return x.squeeze(1)
    if mode == "columns":
        return x.squeeze(1).transpose(1, 2)
    if mode == "patch4":
        p = patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        return patches.contiguous().view(x.size(0), -1, p * p)
    raise ValueError(f"Unknown sequence mode: {mode}")
