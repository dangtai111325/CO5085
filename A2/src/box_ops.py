from __future__ import annotations

import math
import torch


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-6)


def encode_boxes(reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """Encode reference boxes relative to proposals using Faster R-CNN parameterization."""
    px = (proposals[:, 0] + proposals[:, 2]) * 0.5
    py = (proposals[:, 1] + proposals[:, 3]) * 0.5
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1e-6)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1e-6)

    gx = (reference_boxes[:, 0] + reference_boxes[:, 2]) * 0.5
    gy = (reference_boxes[:, 1] + reference_boxes[:, 3]) * 0.5
    gw = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1e-6)
    gh = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1e-6)

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    return torch.stack((dx, dy, dw, dh), dim=1)


def decode_boxes(deltas: torch.Tensor, boxes: torch.Tensor, bbox_xform_clip: float = math.log(1000.0 / 16.0)) -> torch.Tensor:
    """Decode [N,4] or [N,K*4] deltas relative to boxes."""
    if boxes.numel() == 0:
        return deltas.new_zeros(deltas.shape)
    boxes = boxes.to(deltas.dtype)
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-6)
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    d = deltas.reshape(deltas.shape[0], -1, 4)
    dx, dy = d[..., 0], d[..., 1]
    dw = d[..., 2].clamp(max=bbox_xform_clip)
    dh = d[..., 3].clamp(max=bbox_xform_clip)

    pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
    pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred = torch.stack(
        (
            pred_ctr_x - 0.5 * pred_w,
            pred_ctr_y - 0.5 * pred_h,
            pred_ctr_x + 0.5 * pred_w,
            pred_ctr_y + 0.5 * pred_h,
        ),
        dim=-1,
    )
    return pred.reshape(deltas.shape[0], -1)


def clip_boxes_to_image(boxes: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    h, w = image_size
    x = boxes[..., 0::2].clamp(min=0, max=w)
    y = boxes[..., 1::2].clamp(min=0, max=h)
    out = boxes.clone()
    out[..., 0::2] = x
    out[..., 1::2] = y
    return out


def remove_small_boxes(boxes: torch.Tensor, min_size: float) -> torch.Tensor:
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    return torch.where((ws >= min_size) & (hs >= min_size))[0]
