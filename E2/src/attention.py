from __future__ import annotations

import math
import torch
from torch import nn


class ManualMultiHeadSelfAttention(nn.Module):
    """MSA from Linear, reshape, matmul, softmax and head concatenation."""
    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        return x.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        self.last_attention = attn.detach()
        context = self.dropout(attn) @ v
        context = context.transpose(1, 2).contiguous().view(x.size(0), x.size(1), self.d_model)
        return self.out_proj(context)


class TorchMultiHeadSelfAttention(nn.Module):
    """Reference implementation using nn.MultiheadAttention for comparison only."""
    def __init__(self, d_model: int = 64, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, attn = self.mha(x, x, x, need_weights=True, average_attn_weights=False)
        self.last_attention = attn.detach()
        return y
