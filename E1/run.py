from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from common.engine import evaluate_classifier, fit_classifier
from common.seed import resolve_device, seed_everything
from common.synthetic import classification_dataset
from E1.src.data import fashion_mnist_loaders
from E1.src.models import build_e1_models


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(build_e1_models()), default="cnn_2block")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    seed_everything(42)
    device = resolve_device()
    if args.smoke:
        ds = classification_dataset(n=48)
        train_loader = DataLoader(ds, batch_size=16, shuffle=True)
        val_loader = DataLoader(ds, batch_size=16)
        test_loader = val_loader
        epochs = 1
    else:
        train_loader, val_loader, test_loader = fashion_mnist_loaders(batch_size=args.batch_size)
        epochs = args.epochs
    model = build_e1_models()[args.model].to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = fit_classifier(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    test_metrics, _, _ = evaluate_classifier(model, test_loader, criterion, device)
    print({"model": args.model, "history_last": history[-1], "test": test_metrics.__dict__})


if __name__ == "__main__":
    main()
