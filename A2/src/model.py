from __future__ import annotations

from torch import nn

from .anchors import AnchorGenerator
from .backbone import ResNetFPNBackbone, SingleScaleResNetBackbone
from .detector import CustomFasterRCNN
from .rpn import RegionProposalNetwork
from .roi_heads import RoIHeads


def build_detector(
    variant: str = "fpn",
    num_classes: int = 11,
    pretrained_backbone: bool = True,
    trainable_backbone_layers: int = 2,
    image_size: tuple[int, int] = (640, 1024),
    small_object_anchors: bool = False,
    rpn_pre_nms_top_n: int = 1000,
    rpn_post_nms_top_n: int = 300,
) -> nn.Module:
    """Build the custom Faster R-CNN pipeline from explicit modules.

    No torchvision FasterRCNN / fasterrcnn_resnet50_fpn constructor is used.
    torchvision low-level NMS and RoIAlign operators remain intentionally allowed.
    """
    if variant == "single_scale":
        backbone = SingleScaleResNetBackbone(pretrained_backbone, trainable_backbone_layers)
        sizes = ((32, 64, 128, 256, 512),)
        featmap_names = ["0"]
    elif variant == "fpn":
        backbone = ResNetFPNBackbone(pretrained_backbone, trainable_backbone_layers)
        sizes = ((16,), (32,), (64,), (128,)) if small_object_anchors else ((32,), (64,), (128,), (256,))
        featmap_names = ["0", "1", "2", "3"]
    else:
        raise ValueError(variant)

    anchors = AnchorGenerator(sizes=sizes, aspect_ratios=(0.5, 1.0, 2.0))
    rpn = RegionProposalNetwork(
        in_channels=backbone.out_channels,
        anchor_generator=anchors,
        pre_nms_top_n=rpn_pre_nms_top_n,
        post_nms_top_n=rpn_post_nms_top_n,
    )
    roi_heads = RoIHeads(
        featmap_names=featmap_names,
        out_channels=backbone.out_channels,
        num_classes=num_classes,
        representation_size=512,
    )
    return CustomFasterRCNN(backbone, rpn, roi_heads, image_size=image_size)
