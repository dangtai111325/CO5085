from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import batched_nms

from .anchors import AnchorGenerator
from .box_ops import box_iou, clip_boxes_to_image, decode_boxes, encode_boxes, remove_small_boxes


class RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.objectness = nn.Conv2d(in_channels, num_anchors, 1)
        self.box_regression = nn.Conv2d(in_channels, num_anchors * 4, 1)
        for layer in [self.conv, self.objectness, self.box_regression]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, features: list[torch.Tensor]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        logits, deltas = [], []
        for feat in features:
            t = F.relu(self.conv(feat))
            logits.append(self.objectness(t))
            deltas.append(self.box_regression(t))
        return logits, deltas


def _flatten_level(logit: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    b, a, h, w = logit.shape
    logits = logit.permute(0, 2, 3, 1).reshape(b, -1)
    deltas = delta.view(b, a, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
    return logits, deltas


def _subsample(labels: torch.Tensor, batch_size: int, positive_fraction: float) -> torch.Tensor:
    pos = torch.where(labels == 1)[0]
    neg = torch.where(labels == 0)[0]
    n_pos = min(int(batch_size * positive_fraction), pos.numel())
    n_neg = min(batch_size - n_pos, neg.numel())
    if pos.numel() > n_pos:
        pos = pos[torch.randperm(pos.numel(), device=labels.device)[:n_pos]]
    if neg.numel() > n_neg:
        neg = neg[torch.randperm(neg.numel(), device=labels.device)[:n_neg]]
    return torch.cat((pos, neg))


class RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        anchor_generator: AnchorGenerator,
        pre_nms_top_n: int = 1000,
        post_nms_top_n: int = 300,
        nms_thresh: float = 0.7,
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        min_size: float = 1.0,
    ) -> None:
        super().__init__()
        counts = anchor_generator.num_anchors_per_location()
        if len(set(counts)) != 1:
            raise ValueError("This educational RPN requires the same anchors/location across levels")
        self.anchor_generator = anchor_generator
        self.head = RPNHead(in_channels, counts[0])
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.min_size = min_size

    def _assign_targets(self, anchors: torch.Tensor, gt_boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.full((anchors.shape[0],), -1, dtype=torch.int64, device=anchors.device)
        matched = torch.zeros((anchors.shape[0],), dtype=torch.int64, device=anchors.device)
        if gt_boxes.numel() == 0:
            labels[:] = 0
            return labels, matched
        iou = box_iou(anchors, gt_boxes)
        max_iou, matched = iou.max(dim=1)
        labels[max_iou < self.bg_iou_thresh] = 0
        labels[max_iou >= self.fg_iou_thresh] = 1
        # Guarantee at least one positive anchor for every GT box.
        best_anchor_per_gt = iou.argmax(dim=0)
        labels[best_anchor_per_gt] = 1
        matched[best_anchor_per_gt] = torch.arange(gt_boxes.shape[0], device=anchors.device)
        return labels, matched

    def _filter_proposals(
        self,
        logits_by_level: list[torch.Tensor],
        deltas_by_level: list[torch.Tensor],
        anchors_by_level: list[torch.Tensor],
        image_size: tuple[int, int],
        image_index: int,
    ) -> torch.Tensor:
        boxes_all, scores_all, levels_all = [], [], []
        for level, (logits, deltas, anchors) in enumerate(zip(logits_by_level, deltas_by_level, anchors_by_level)):
            scores = logits[image_index].sigmoid().detach()
            reg = deltas[image_index].detach()
            n = min(self.pre_nms_top_n, scores.numel())
            top_idx = scores.topk(n).indices
            boxes = decode_boxes(reg[top_idx], anchors[top_idx]).reshape(-1, 4)
            boxes = clip_boxes_to_image(boxes, image_size)
            keep = remove_small_boxes(boxes, self.min_size)
            boxes, scores = boxes[keep], scores[top_idx][keep]
            boxes_all.append(boxes)
            scores_all.append(scores)
            levels_all.append(torch.full((len(boxes),), level, dtype=torch.int64, device=boxes.device))
        if not boxes_all:
            return anchors_by_level[0].new_zeros((0, 4))
        boxes = torch.cat(boxes_all, dim=0)
        scores = torch.cat(scores_all, dim=0)
        levels = torch.cat(levels_all, dim=0)
        keep = batched_nms(boxes, scores, levels, self.nms_thresh)[: self.post_nms_top_n]
        return boxes[keep]

    def forward(
        self,
        features: OrderedDict[str, torch.Tensor],
        image_size: tuple[int, int],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        feats = list(features.values())
        anchors_by_level = self.anchor_generator(feats, image_size)
        obj_raw, reg_raw = self.head(feats)
        logits_by_level, deltas_by_level = zip(*[_flatten_level(o, d) for o, d in zip(obj_raw, reg_raw)])
        logits_by_level, deltas_by_level = list(logits_by_level), list(deltas_by_level)
        proposals = [
            self._filter_proposals(logits_by_level, deltas_by_level, anchors_by_level, image_size, i)
            for i in range(feats[0].shape[0])
        ]

        losses: dict[str, torch.Tensor] = {}
        if targets is not None:
            all_anchors = torch.cat(anchors_by_level, dim=0)
            obj = torch.cat(logits_by_level, dim=1)
            reg = torch.cat(deltas_by_level, dim=1)
            obj_losses, reg_losses = [], []
            for i, target in enumerate(targets):
                labels, matched = self._assign_targets(all_anchors, target["boxes"])
                sampled = _subsample(labels, self.batch_size_per_image, self.positive_fraction)
                obj_losses.append(F.binary_cross_entropy_with_logits(obj[i, sampled], labels[sampled].float()))
                pos = sampled[labels[sampled] == 1]
                if pos.numel() and target["boxes"].numel():
                    reg_targets = encode_boxes(target["boxes"][matched[pos]], all_anchors[pos])
                    reg_losses.append(F.smooth_l1_loss(reg[i, pos], reg_targets, beta=1.0 / 9.0, reduction="sum") / max(sampled.numel(), 1))
                else:
                    reg_losses.append(reg[i].sum() * 0.0)
            losses = {
                "loss_objectness": torch.stack(obj_losses).mean(),
                "loss_rpn_box_reg": torch.stack(reg_losses).mean(),
            }
        return proposals, losses
