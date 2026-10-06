from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .fpn import FeaturePyramidNetwork


class ResNetStages(nn.Module):
    def __init__(self, pretrained: bool = True, trainable_layers: int = 2) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        trainable = {"layer4"}
        if trainable_layers >= 2:
            trainable.add("layer3")
        if trainable_layers >= 3:
            trainable.add("layer2")
        if trainable_layers >= 4:
            trainable.add("layer1")
        if trainable_layers >= 5:
            trainable.add("stem")
        for name, module in self.named_children():
            req = name in trainable
            for p in module.parameters():
                p.requires_grad = req

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class SingleScaleResNetBackbone(nn.Module):
    out_channels = 256

    def __init__(self, pretrained: bool = True, trainable_layers: int = 2) -> None:
        super().__init__()
        self.stages = ResNetStages(pretrained, trainable_layers)
        self.proj = nn.Conv2d(2048, self.out_channels, 1)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict({"0": self.proj(self.stages(x)[-1])})


class ResNetFPNBackbone(nn.Module):
    out_channels = 256

    def __init__(self, pretrained: bool = True, trainable_layers: int = 2) -> None:
        super().__init__()
        self.stages = ResNetStages(pretrained, trainable_layers)
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        return self.fpn(self.stages(x))
