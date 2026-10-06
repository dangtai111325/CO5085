from __future__ import annotations

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.engine import fit_classifier
from common.seed import resolve_device, seed_everything
from common.synthetic import classification_dataset
from E1.src.data import fashion_mnist_loaders
from E2.src.models import build_e2_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", choices=["row", "patch4", "cnn_stem"], default="patch4")
    p.add_argument("--attention", choices=["manual", "torch"], default="manual")
    p.add_argument("--heads", type=int, choices=[2, 4, 8], default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    seed_everything(42)
    device = resolve_device()
    if args.smoke:
        ds = classification_dataset(n=32)
        train_loader = DataLoader(ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(ds, batch_size=8)
        epochs = 1
    else:
        train_loader, val_loader, _ = fashion_mnist_loaders(batch_size=128)
        epochs = args.epochs
    model = build_e2_model(args.tokenizer, args.attention, num_heads=args.heads).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history = fit_classifier(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    print({"tokenizer": args.tokenizer, "attention": args.attention, "heads": args.heads, "last": history[-1]})


if __name__ == "__main__":
    main()
