from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F


class FeaturePyramidNetwork(nn.Module):
    """FPN top-down pathway with lateral 1x1 and output 3x3 convolutions."""
    def __init__(self, in_channels_list: list[int], out_channels: int = 256) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.output = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])

    def forward(self, features: list[torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        if len(features) != len(self.lateral):
            raise ValueError("Feature count does not match FPN configuration")
        results: list[torch.Tensor] = [torch.empty(0)] * len(features)
        last_inner = self.lateral[-1](features[-1])
        results[-1] = self.output[-1](last_inner)
        for i in range(len(features) - 2, -1, -1):
            lateral = self.lateral[i](features[i])
            top_down = F.interpolate(last_inner, size=lateral.shape[-2:], mode="nearest")
            last_inner = lateral + top_down
            results[i] = self.output[i](last_inner)
        return OrderedDict((str(i), feat) for i, feat in enumerate(results))
