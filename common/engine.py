from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import ClassificationMetrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    grad_clip: float | None = None,
) -> ClassificationMetrics:
    """Explicit PyTorch training loop required by the assignment."""
    model.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        batch = targets.size(0)
        loss_sum += float(loss.detach().item()) * batch
        correct += int((logits.argmax(1) == targets).sum().item())
        seen += batch
    return ClassificationMetrics(loss=loss_sum / max(seen, 1), accuracy=correct / max(seen, 1), n_samples=seen)


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[ClassificationMetrics, list[int], list[int]]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    seen = 0
    ys: list[int] = []
    ps: list[int] = []
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        preds = logits.argmax(1)
        batch = targets.size(0)
        loss_sum += float(loss.item()) * batch
        correct += int((preds == targets).sum().item())
        seen += batch
        ys.extend(targets.cpu().tolist())
        ps.extend(preds.cpu().tolist())
    metrics = ClassificationMetrics(loss=loss_sum / max(seen, 1), accuracy=correct / max(seen, 1), n_samples=seen)
    return metrics, ys, ps


def fit_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    *,
    scheduler: Any | None = None,
    callback: Callable[[int, ClassificationMetrics, ClassificationMetrics], None] | None = None,
) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va, _, _ = evaluate_classifier(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": tr.loss, "train_accuracy": tr.accuracy,
                        "val_loss": va.loss, "val_accuracy": va.accuracy})
        if scheduler is not None:
            scheduler.step()
        if callback is not None:
            callback(epoch, tr, va)
    return history
