from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F

from .rpn import RegionProposalNetwork
from .roi_heads import RoIHeads


class DetectionTransform(nn.Module):
    """Resize all images to a fixed canvas, scale boxes, normalize, then undo box scaling at inference."""

    def __init__(self, image_size: tuple[int, int], mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> None:
        super().__init__()
        self.image_size = image_size
        self.register_buffer("mean", torch.tensor(mean).view(3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(3, 1, 1), persistent=False)

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None):
        resized, transformed_targets, original_sizes = [], [], []
        out_h, out_w = self.image_size
        for i, image in enumerate(images):
            h, w = image.shape[-2:]
            original_sizes.append((h, w))
            img = F.interpolate(image.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
            img = (img - self.mean.to(img)) / self.std.to(img)
            resized.append(img)
            if targets is not None:
                target = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in targets[i].items()}
                boxes = target["boxes"].clone()
                if boxes.numel():
                    boxes[:, [0, 2]] *= out_w / w
                    boxes[:, [1, 3]] *= out_h / h
                target["boxes"] = boxes
                if "area" in target:
                    target["area"] = target["area"] * (out_w / w) * (out_h / h)
                transformed_targets.append(target)
        return torch.stack(resized, dim=0), (transformed_targets if targets is not None else None), original_sizes

    def postprocess(self, detections: list[dict[str, torch.Tensor]], original_sizes: list[tuple[int, int]]):
        out_h, out_w = self.image_size
        for det, (h, w) in zip(detections, original_sizes):
            boxes = det["boxes"]
            if boxes.numel():
                boxes[:, [0, 2]] *= w / out_w
                boxes[:, [1, 3]] *= h / out_h
        return detections


class CustomFasterRCNN(nn.Module):
    """Educational Faster R-CNN assembled from explicit transform, backbone, RPN and RoI heads.

    The code intentionally does not call torchvision.models.detection.FasterRCNN or any
    one-line detector constructor. Optimized low-level torchvision operators are used only
    for NMS and RoIAlign inside the custom components.
    """

    def __init__(self, backbone: nn.Module, rpn: RegionProposalNetwork, roi_heads: RoIHeads, image_size: tuple[int, int]) -> None:
        super().__init__()
        self.transform = DetectionTransform(image_size)
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.image_size = image_size

    def forward(self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None):
        if self.training and targets is None:
            raise ValueError("targets are required in training mode")
        batch, targets_t, original_sizes = self.transform(images, targets)
        features = self.backbone(batch)
        if not isinstance(features, OrderedDict):
            raise TypeError("backbone must return OrderedDict[str, Tensor]")
        proposals, rpn_losses = self.rpn(features, self.image_size, targets_t)
        detections, roi_losses = self.roi_heads(features, proposals, self.image_size, targets_t)
        if self.training:
            return {**roi_losses, **rpn_losses}
        return self.transform.postprocess(detections, original_sizes)
