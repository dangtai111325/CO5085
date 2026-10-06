from __future__ import annotations

from typing import Iterable

import torch


@torch.no_grad()
def mean_average_precision(model, loader, device: torch.device) -> dict[str, float]:
    """COCO-style mAP via torchmetrics. Install pycocotools as listed in requirements."""
    try:
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
    except Exception as exc:
        raise RuntimeError("torchmetrics + pycocotools are required for full detection evaluation") from exc
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    model.eval()
    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        outputs = model(images_dev)
        preds_cpu = [{k: v.detach().cpu() for k, v in out.items() if k in {"boxes", "scores", "labels"}} for out in outputs]
        targets_cpu = [{k: v.detach().cpu() for k, v in t.items() if k in {"boxes", "labels"}} for t in targets]
        metric.update(preds_cpu, targets_cpu)
    result = metric.compute()
    scalar_keys = ["map", "map_50", "map_75", "map_small", "map_medium", "map_large", "mar_100"]
    return {k: float(result[k].item()) for k in scalar_keys if k in result}
