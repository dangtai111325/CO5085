from __future__ import annotations

import torch
from torch import nn


class ManualLSTMCell(nn.Module):
    """LSTM cell written from gate equations for pedagogical transparency."""
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x_proj = nn.Linear(input_size, 4 * hidden_size)
        self.h_proj = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        h_prev, c_prev = state
        gates = self.x_proj(x_t) + self.h_proj(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class ManualGRUCell(nn.Module):
    """GRU cell written from update/reset/new gate equations."""
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.x_r = nn.Linear(input_size, hidden_size)
        self.h_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x_z = nn.Linear(input_size, hidden_size)
        self.h_z = nn.Linear(hidden_size, hidden_size, bias=False)
        self.x_n = nn.Linear(input_size, hidden_size)
        self.h_n = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        r = torch.sigmoid(self.x_r(x_t) + self.h_r(h_prev))
        z = torch.sigmoid(self.x_z(x_t) + self.h_z(h_prev))
        n = torch.tanh(self.x_n(x_t) + r * self.h_n(h_prev))
        return (1 - z) * n + z * h_prev
