from __future__ import annotations

import torch
from torch import nn

from .attention import ManualMultiHeadSelfAttention, TorchMultiHeadSelfAttention
from .tokenizers import CNNStemTokenizer, PatchTokenizer, RowTokenizer


class EncoderBlock(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int = 64, mlp_ratio: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = attention
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyAttentionClassifier(nn.Module):
    def __init__(self, tokenizer: nn.Module, attention: nn.Module, d_model: int = 64, depth: int = 2, num_classes: int = 10) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([EncoderBlock(attention if i == 0 else type(attention)(d_model=d_model, num_heads=getattr(attention, 'num_heads', 4))) for i in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.norm(tokens[:, 0]))


def build_e2_model(tokenizer_name: str = "patch4", attention_name: str = "manual", d_model: int = 64, num_heads: int = 4) -> TinyAttentionClassifier:
    tokenizers = {
        "row": RowTokenizer(d_model),
        "patch4": PatchTokenizer(4, d_model),
        "cnn_stem": CNNStemTokenizer(d_model),
    }
    attn_cls = ManualMultiHeadSelfAttention if attention_name == "manual" else TorchMultiHeadSelfAttention
    return TinyAttentionClassifier(tokenizers[tokenizer_name], attn_cls(d_model, num_heads), d_model=d_model)
