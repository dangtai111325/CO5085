from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign, batched_nms

from .box_ops import box_iou, clip_boxes_to_image, decode_boxes, encode_boxes, remove_small_boxes


class TwoMLPHead(nn.Module):
    def __init__(self, in_channels: int, representation_size: int = 512) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cls_score(x), self.bbox_pred(x)


def _sample_indices(labels: torch.Tensor, batch_size: int, positive_fraction: float) -> torch.Tensor:
    pos = torch.where(labels > 0)[0]
    neg = torch.where(labels == 0)[0]
    n_pos = min(int(batch_size * positive_fraction), pos.numel())
    n_neg = min(batch_size - n_pos, neg.numel())
    if pos.numel() > n_pos:
        pos = pos[torch.randperm(pos.numel(), device=labels.device)[:n_pos]]
    if neg.numel() > n_neg:
        neg = neg[torch.randperm(neg.numel(), device=labels.device)[:n_neg]]
    return torch.cat((pos, neg))


class RoIHeads(nn.Module):
    def __init__(
        self,
        featmap_names: list[str],
        out_channels: int,
        num_classes: int,
        output_size: int = 7,
        representation_size: int = 512,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.25,
        fg_iou_thresh: float = 0.5,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
    ) -> None:
        super().__init__()
        self.pool = MultiScaleRoIAlign(featmap_names=featmap_names, output_size=output_size, sampling_ratio=2)
        self.box_head = TwoMLPHead(out_channels * output_size * output_size, representation_size)
        self.predictor = FastRCNNPredictor(representation_size, num_classes)
        self.num_classes = num_classes
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.fg_iou_thresh = fg_iou_thresh
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    def _match(self, proposals: torch.Tensor, target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        gt_boxes, gt_labels = target["boxes"], target["labels"]
        if gt_boxes.numel() == 0:
            return torch.zeros((len(proposals),), dtype=torch.int64, device=proposals.device), torch.zeros((len(proposals),), dtype=torch.int64, device=proposals.device)
        iou = box_iou(proposals, gt_boxes)
        max_iou, matched = iou.max(dim=1)
        labels = gt_labels[matched].clone()
        labels[max_iou < self.fg_iou_thresh] = 0
        return labels, matched

    def _select_training_samples(self, proposals: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]):
        sampled_props, labels_out, reg_targets_out = [], [], []
        for props, target in zip(proposals, targets):
            props = torch.cat((props, target["boxes"]), dim=0) if target["boxes"].numel() else props
            labels, matched = self._match(props, target)
            idx = _sample_indices(labels, self.batch_size_per_image, self.positive_fraction)
            props = props[idx]
            labels = labels[idx]
            matched = matched[idx]
            reg_targets = props.new_zeros((len(props), 4))
            pos = torch.where(labels > 0)[0]
            if pos.numel():
                reg_targets[pos] = encode_boxes(target["boxes"][matched[pos]], props[pos])
            sampled_props.append(props)
            labels_out.append(labels)
            reg_targets_out.append(reg_targets)
        return sampled_props, labels_out, reg_targets_out

    def _fastrcnn_loss(self, class_logits, box_regression, labels_list, reg_targets_list):
        labels = torch.cat(labels_list, dim=0)
        reg_targets = torch.cat(reg_targets_list, dim=0)
        loss_classifier = F.cross_entropy(class_logits, labels)
        box_regression = box_regression.reshape(box_regression.shape[0], self.num_classes, 4)
        pos = torch.where(labels > 0)[0]
        if pos.numel():
            selected = box_regression[pos, labels[pos]]
            loss_box = F.smooth_l1_loss(selected, reg_targets[pos], beta=1.0 / 9.0, reduction="sum") / max(labels.numel(), 1)
        else:
            loss_box = box_regression.sum() * 0.0
        return loss_classifier, loss_box

    @torch.no_grad()
    def _postprocess(self, class_logits: torch.Tensor, box_regression: torch.Tensor, proposals: list[torch.Tensor], image_size: tuple[int, int]):
        counts = [len(p) for p in proposals]
        logits_split = class_logits.split(counts, dim=0)
        reg_split = box_regression.split(counts, dim=0)
        results = []
        for logits, reg, props in zip(logits_split, reg_split, proposals):
            if props.numel() == 0:
                results.append({"boxes": props, "scores": props.new_zeros((0,)), "labels": torch.zeros((0,), dtype=torch.int64, device=props.device)})
                continue
            scores = F.softmax(logits, dim=-1)
            boxes = decode_boxes(reg, props).reshape(len(props), self.num_classes, 4)
            boxes = clip_boxes_to_image(boxes, image_size)
            boxes = boxes[:, 1:, :].reshape(-1, 4)
            scores = scores[:, 1:].reshape(-1)
            labels = torch.arange(1, self.num_classes, device=props.device).view(1, -1).expand(len(props), -1).reshape(-1)
            keep = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            keep_small = remove_small_boxes(boxes, 1.0)
            boxes, scores, labels = boxes[keep_small], scores[keep_small], labels[keep_small]
            keep_nms = batched_nms(boxes, scores, labels, self.nms_thresh)[: self.detections_per_img]
            results.append({"boxes": boxes[keep_nms], "scores": scores[keep_nms], "labels": labels[keep_nms]})
        return results

    def forward(
        self,
        features: OrderedDict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_size: tuple[int, int],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> tuple[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
        image_shapes = [image_size for _ in proposals]
        if targets is not None:
            proposals, labels, reg_targets = self._select_training_samples(proposals, targets)
        pooled = self.pool(features, proposals, image_shapes)
        rep = self.box_head(pooled)
        class_logits, box_regression = self.predictor(rep)
        if targets is not None:
            loss_cls, loss_box = self._fastrcnn_loss(class_logits, box_regression, labels, reg_targets)
            return [], {"loss_classifier": loss_cls, "loss_box_reg": loss_box}
        return self._postprocess(class_logits, box_regression, proposals, image_size), {}
