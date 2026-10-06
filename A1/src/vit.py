from __future__ import annotations

import math
import torch
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192) -> None:
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 3, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        self.last_attention = attn.detach()
        x = self.attn_drop(attn) @ v
        x = x.transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim: int = 192, num_heads: int = 3, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerTiny(nn.Module):
    """DeiT/ViT-style tiny model written layer by layer."""
    def __init__(self, image_size: int = 224, patch_size: int = 16, num_classes: int = 43, embed_dim: int = 192, depth: int = 12, num_heads: int = 3) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = self.pos_drop(torch.cat((cls, x), dim=1) + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x)[:, 0])


def load_timm_deit_tiny(model: VisionTransformerTiny) -> tuple[list[str], list[str]]:
    """Transfer matching ImageNet-pretrained DeiT-Tiny weights from timm into the custom model."""
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("Install timm to load pretrained DeiT-Tiny weights") from exc
    ref = timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=True)
    state = ref.state_dict()
    # Head shape differs for GTSRB; distillation-only keys may also be present depending on timm version.
    for key in list(state):
        if key.startswith("head") or key.startswith("head_dist") or key.startswith("dist_token"):
            state.pop(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return list(missing), list(unexpected)


def set_vit_trainability(model: VisionTransformerTiny, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad = True
    if mode in {"scratch", "full"}:
        return
    if mode == "head":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("head.")
        return
    if mode == "partial":
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("blocks.9") or name.startswith("blocks.10") or name.startswith("blocks.11") or name.startswith("norm.") or name.startswith("head.")
        return
    raise ValueError(mode)
