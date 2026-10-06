from __future__ import annotations

import torch
from torch import nn

from .cells import ManualGRUCell, ManualLSTMCell
from .sequence import image_to_sequence


class ManualLSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 10, mode: str = "rows") -> None:
        super().__init__()
        self.mode = mode
        self.cell = ManualLSTMCell(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = image_to_sequence(x, self.mode)
        h = torch.zeros(x.size(0), self.cell.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        for t in range(seq.size(1)):
            h, c = self.cell(seq[:, t], (h, c))
        return self.head(h)


class ManualGRUClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, num_classes: int = 10, mode: str = "rows") -> None:
        super().__init__()
        self.mode = mode
        self.cell = ManualGRUCell(input_size, hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = image_to_sequence(x, self.mode)
        h = torch.zeros(x.size(0), self.cell.hidden_size, device=x.device, dtype=x.dtype)
        for t in range(seq.size(1)):
            h = self.cell(seq[:, t], h)
        return self.head(h)


class TorchRecurrentClassifier(nn.Module):
    """Reference model using nn.LSTM/nn.GRU, permitted by the assignment."""
    def __init__(self, kind: str, input_size: int, hidden_size: int = 128, num_classes: int = 10, mode: str = "rows") -> None:
        super().__init__()
        self.mode = mode
        rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = image_to_sequence(x, self.mode)
        out, state = self.rnn(seq)
        h = state[0][-1] if isinstance(state, tuple) else state[-1]
        return self.head(h)


def input_size_for_mode(mode: str) -> int:
    return 16 if mode == "patch4" else 28


def build_e3_model(kind: str = "manual_lstm", mode: str = "rows", hidden_size: int = 128) -> nn.Module:
    inp = input_size_for_mode(mode)
    if kind == "manual_lstm":
        return ManualLSTMClassifier(inp, hidden_size, mode=mode)
    if kind == "manual_gru":
        return ManualGRUClassifier(inp, hidden_size, mode=mode)
    if kind == "torch_lstm":
        return TorchRecurrentClassifier("lstm", inp, hidden_size, mode=mode)
    if kind == "torch_gru":
        return TorchRecurrentClassifier("gru", inp, hidden_size, mode=mode)
    raise ValueError(kind)
