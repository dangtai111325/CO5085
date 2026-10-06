from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from A2.src.data import BDD100KDetection, detection_collate
from A2.src.engine import predict, train_one_epoch_detection
from A2.src.model import build_detector
from common.seed import resolve_device, seed_everything


class TinyDetectionDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        image = torch.rand(3, 128, 128)
        target = {
            "boxes": torch.tensor([[20.0, 20.0, 80.0, 90.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
            "image_id": torch.tensor(idx),
            "area": torch.tensor([4200.0]),
            "iscrowd": torch.tensor([0], dtype=torch.int64),
        }
        return image, target


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["single_scale", "fpn"], default="fpn")
    p.add_argument("--images-dir", type=Path)
    p.add_argument("--labels-json", type=Path)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--small-object-anchors", action="store_true")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-images", type=int)
    p.add_argument("--image-height", type=int, default=640)
    p.add_argument("--image-width", type=int, default=1024)
    p.add_argument("--trainable-backbone-layers", type=int, default=2)
    args = p.parse_args()

    seed_everything(42)
    device = resolve_device()
    if args.smoke:
        dataset = TinyDetectionDataset()
        model = build_detector(
            args.variant,
            pretrained_backbone=False,
            image_size=(128, 128),
            small_object_anchors=args.small_object_anchors,
            rpn_pre_nms_top_n=120,
            rpn_post_nms_top_n=60,
        )
        batch_size = 1
    else:
        if args.images_dir is None or args.labels_json is None:
            p.error("--images-dir and --labels-json are required outside --smoke")
        dataset = BDD100KDetection(
            args.images_dir,
            args.labels_json,
            max_images=args.max_images,
            horizontal_flip_prob=0.5,
        )
        model = build_detector(
            args.variant,
            pretrained_backbone=True,
            trainable_backbone_layers=args.trainable_backbone_layers,
            image_size=(args.image_height, args.image_width),
            small_object_anchors=args.small_object_anchors,
        )
        batch_size = args.batch_size

    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate)
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=0.0025,
        momentum=0.9,
        weight_decay=1e-4,
    )
    losses = train_one_epoch_detection(model, loader, optimizer, device)
    images, _ = next(iter(loader))
    preds = predict(model, list(images), device)
    print({"variant": args.variant, "losses": losses, "pred_boxes": [len(pred["boxes"]) for pred in preds]})


if __name__ == "__main__":
    main()
