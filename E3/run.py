from __future__ import annotations

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from common.engine import fit_classifier
from common.seed import resolve_device, seed_everything
from common.synthetic import classification_dataset
from E1.src.data import fashion_mnist_loaders
from E3.src.models import build_e3_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["manual_lstm", "manual_gru", "torch_lstm", "torch_gru"], default="manual_lstm")
    p.add_argument("--sequence", choices=["rows", "columns", "patch4"], default="rows")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    seed_everything(42)
    device = resolve_device()
    if args.smoke:
        ds = classification_dataset(n=24)
        train_loader = DataLoader(ds, batch_size=8, shuffle=True)
        val_loader = DataLoader(ds, batch_size=8)
        epochs = 1
    else:
        train_loader, val_loader, _ = fashion_mnist_loaders(batch_size=128)
        epochs = args.epochs
    model = build_e3_model(args.kind, args.sequence).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = fit_classifier(model, train_loader, val_loader, optimizer, criterion, device, epochs)
    print({"kind": args.kind, "sequence": args.sequence, "last": history[-1]})


if __name__ == "__main__":
    main()
