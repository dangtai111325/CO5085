from __future__ import annotations

import math
import torch


class AnchorGenerator:
    """Small explicit anchor generator for a fixed resized image canvas."""

    def __init__(self, sizes: tuple[tuple[int, ...], ...], aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)) -> None:
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

    def num_anchors_per_location(self) -> list[int]:
        return [len(s) * len(self.aspect_ratios) for s in self.sizes]

    def _base_anchors(self, sizes: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        anchors: list[list[float]] = []
        for size in sizes:
            area = float(size * size)
            for ratio in self.aspect_ratios:
                w = math.sqrt(area / ratio)
                h = ratio * w
                anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(anchors, device=device, dtype=dtype)

    def __call__(self, features: list[torch.Tensor], image_size: tuple[int, int]) -> list[torch.Tensor]:
        if len(features) != len(self.sizes):
            raise ValueError(f"Expected {len(self.sizes)} feature levels, got {len(features)}")
        image_h, image_w = image_size
        anchors_all: list[torch.Tensor] = []
        for feature, sizes in zip(features, self.sizes):
            _, _, h, w = feature.shape
            stride_y = image_h / h
            stride_x = image_w / w
            shifts_x = (torch.arange(w, device=feature.device, dtype=feature.dtype) + 0.5) * stride_x
            shifts_y = (torch.arange(h, device=feature.device, dtype=feature.dtype) + 0.5) * stride_y
            yy, xx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shifts = torch.stack((xx, yy, xx, yy), dim=-1).reshape(-1, 4)
            base = self._base_anchors(sizes, feature.device, feature.dtype)
            anchors = shifts[:, None, :] + base[None, :, :]
            anchors_all.append(anchors.reshape(-1, 4))
        return anchors_all
