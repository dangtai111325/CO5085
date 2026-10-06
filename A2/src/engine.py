from __future__ import annotations

import torch


def train_one_epoch_detection(model, loader, optimizer, device: torch.device) -> dict[str, float]:
    """Explicit Faster R-CNN training loop; no high-level trainer is used."""
    model.train()
    totals: dict[str, float] = {}
    steps = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().item())
        totals["loss_total"] = totals.get("loss_total", 0.0) + float(loss.detach().item())
        steps += 1
    return {k: v / max(steps, 1) for k, v in totals.items()}


@torch.no_grad()
def predict(model, images: list[torch.Tensor], device: torch.device):
    model.eval()
    return model([img.to(device) for img in images])
