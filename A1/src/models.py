from __future__ import annotations

from torch import nn

from .resnet import ResNet18, load_imagenet_resnet18, set_resnet_trainability
from .vit import VisionTransformerTiny, load_timm_deit_tiny, set_vit_trainability


def build_a1_model(family: str, mode: str, num_classes: int = 43, pretrained: bool = False) -> nn.Module:
    if family == "resnet18":
        model = ResNet18(num_classes=num_classes)
        if pretrained:
            load_imagenet_resnet18(model)
        set_resnet_trainability(model, mode)
        return model
    if family == "vit_tiny":
        model = VisionTransformerTiny(num_classes=num_classes)
        if pretrained:
            load_timm_deit_tiny(model)
        set_vit_trainability(model, mode)
        return model
    raise ValueError(family)
