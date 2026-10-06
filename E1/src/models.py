from __future__ import annotations

import torch
from torch import nn


class SoftmaxClassifier(nn.Module):
    def __init__(self, in_features: int = 28 * 28, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))


class MLPClassifier(nn.Module):
    def __init__(self, in_features: int = 28 * 28, hidden: tuple[int, ...] = (256, 128), num_classes: int = 10, dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        d = in_features
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_bn: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers += [nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNClassifier(nn.Module):
    def __init__(self, channels: tuple[int, ...] = (32, 64), num_classes: int = 10, use_bn: bool = False, dropout: float = 0.25) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        in_ch = 1
        for out_ch in channels:
            blocks.append(ConvBlock(in_ch, out_ch, use_bn=use_bn))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        scale = 2 ** len(channels)
        h = 28 // scale
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch * h * h, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_e1_models() -> dict[str, nn.Module]:
    return {
        "softmax": SoftmaxClassifier(),
        "mlp": MLPClassifier(),
        "cnn_2block": CNNClassifier((32, 64), use_bn=False),
        "cnn_3block_bn": CNNClassifier((32, 64, 128), use_bn=True, dropout=0.35),
    }
