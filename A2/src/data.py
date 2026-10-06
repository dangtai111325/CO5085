from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


BDD100K_CLASSES = [
    "pedestrian", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(BDD100K_CLASSES)}  # 0 = background


class BDD100KDetection(Dataset):
    """BDD100K detection loader for frame-level JSON annotations.

    Images stay at original resolution here. The custom detector transform performs
    the deterministic resize/normalization. Optional horizontal flip is implemented
    in this dataset because it must update bounding boxes explicitly.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_json: str | Path,
        max_images: int | None = None,
        horizontal_flip_prob: float = 0.0,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        with open(labels_json, "r", encoding="utf-8") as f:
            records = json.load(f)
        self.records = records[:max_images] if max_images is not None else records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        rec: dict[str, Any] = self.records[index]
        img_path = self.images_dir / rec["name"]
        image = F.to_tensor(Image.open(img_path).convert("RGB"))
        boxes: list[list[float]] = []
        labels: list[int] = []
        for ann in rec.get("labels", []):
            box = ann.get("box2d")
            category = ann.get("category")
            if box is None or category not in CLASS_TO_ID:
                continue
            x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_TO_ID[category])
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        label_tensor = torch.tensor(labels, dtype=torch.int64)

        if self.horizontal_flip_prob > 0 and random.random() < self.horizontal_flip_prob:
            image = torch.flip(image, dims=[2])
            width = image.shape[-1]
            if box_tensor.numel():
                old_x1 = box_tensor[:, 0].clone()
                old_x2 = box_tensor[:, 2].clone()
                box_tensor[:, 0] = width - old_x2
                box_tensor[:, 2] = width - old_x1

        area = (
            (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
            if len(box_tensor)
            else torch.zeros(0, dtype=torch.float32)
        )
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": torch.tensor(index),
            "area": area,
            "iscrowd": torch.zeros((len(label_tensor),), dtype=torch.int64),
        }
        return image, target


def detection_collate(batch):
    return tuple(zip(*batch))
