from __future__ import annotations

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from A1.src.data import gtsrb_loaders
from A1.src.models import build_a1_model
from common.engine import fit_classifier
from common.seed import resolve_device, seed_everything
from common.synthetic import classification_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["resnet18", "vit_tiny"], default="resnet18")
    p.add_argument("--mode", choices=["scratch", "head", "partial", "full"], default="scratch")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    seed_everything(42)
    device = resolve_device()
    if args.smoke:
        ds = classification_dataset(n=8, channels=3, height=224, width=224, classes=43)
        train_loader = DataLoader(ds, batch_size=2, shuffle=True)
        val_loader = DataLoader(ds, batch_size=2)
        epochs = 1
        pretrained = False
    else:
        train_loader, val_loader, _ = gtsrb_loaders(batch_size=args.batch_size)
        epochs = args.epochs
        pretrained = args.pretrained
    model = build_a1_model(args.family, args.mode, pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-4 if args.family == "vit_tiny" else 1e-3, weight_decay=1e-4)
    history = fit_classifier(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    print({"family": args.family, "mode": args.mode, "pretrained": pretrained, "last": history[-1]})


if __name__ == "__main__":
    main()
