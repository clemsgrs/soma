"""HIPT — Hierarchical Image Pyramid Transformer (Chen et al., 2022).

Adapted from github.com/clemsgrs/hipt (MIT). Only LocalHIPT is ported:
the region-level ViT (VisionTransformer4K) and global transformer are
jointly trained on pre-extracted tile features.

This is a pure aggregator (no classifier, no loss). The classifier lives
in TaskHead, making HIPT composable with any task.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.attention_pool import AttentionPool
from soma.aggregators.registry import aggregator_registry


# ---------------------------------------------------------------------------
# ViT building blocks (simplified from DINO / hipt)
# ---------------------------------------------------------------------------


def _drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
    return x.div(keep_prob) * mask


class DropPath(nn.Module):
    """Stochastic Depth per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        return _drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# VisionTransformer4K — region-level ViT
# ---------------------------------------------------------------------------


class VisionTransformer4K(nn.Module):
    """Region-level Vision Transformer.

    Takes pre-extracted tile features arranged in a spatial grid and produces
    a single region-level embedding via a CLS token.

    Input shape:  ``(M, input_embed_dim, npatch, npatch)``
    Output shape: ``(M, output_embed_dim)``

    Args:
        input_embed_dim: Dimension of pre-extracted tile features.
        output_embed_dim: Transformer hidden / output dimension.
        npatch: Default spatial grid size (used to initialize positional encoding).
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        qkv_bias: Whether to use bias in QKV projection.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        input_embed_dim: int = 384,
        output_embed_dim: int = 192,
        npatch: int = 16,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = output_embed_dim

        # Project tile features to transformer dimension
        self.phi = nn.Sequential(
            nn.Linear(input_embed_dim, output_embed_dim),
            nn.GELU(),
            nn.Dropout(p=drop_rate),
        )

        num_patches = npatch * npatch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, output_embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, output_embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=output_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, drop_path=dpr[i],
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(output_embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        """Interpolate positional encoding for arbitrary grid sizes."""
        npatch_sq = x.shape[1] - 1  # exclude CLS
        N = self.pos_embed.shape[1] - 1

        if npatch_sq == N and w == h:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]  # (1, 1, D)
        patch_pos = self.pos_embed[:, 1:]  # (1, N, D)
        dim = x.shape[-1]

        sqrt_N = int(math.sqrt(N))
        patch_pos = patch_pos.reshape(1, sqrt_N, sqrt_N, dim).permute(0, 3, 1, 2)
        patch_pos = nn.functional.interpolate(
            patch_pos,
            scale_factor=((w + 0.1) / sqrt_N, (h + 0.1) / sqrt_N),
            mode="bicubic",
            align_corners=False,
            recompute_scale_factor=True,
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls_pos, patch_pos), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: ``(M, input_embed_dim, npatch, npatch)``

        Returns:
            CLS token embedding ``(M, output_embed_dim)``
        """
        B, _, w, h = x.shape
        # Flatten spatial dims and project
        x = x.flatten(2, 3).transpose(1, 2)  # (M, npatch², D_in)
        x = self.phi(x)  # (M, npatch², D_out)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)  # (M, npatch²+1, D_out)

        # Add positional encoding
        x = x + self._interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]  # CLS token


# ---------------------------------------------------------------------------
# HIPT aggregator
# ---------------------------------------------------------------------------


class HIPT(Aggregator):
    """Hierarchical Image Pyramid Transformer aggregator (Chen et al., 2022).

    Two-level hierarchy: a region-level ViT aggregates P tile features per
    region, then a global transformer + gated attention pools M region
    embeddings into a slide-level representation.

    Features are stored either flat as ``(B, N, D)`` where ``N = M × P``,
    or natively hierarchical as ``(B, M, P, D)`` where
    ``P = (region_size / patch_size)²``. HIPT reshapes internally.

    Args:
        input_dim: Feature dimension of input tiles (auto-resolved from FeatureStore).
        region_size: Region pixel size (e.g. 4096). **Required.**
        patch_size: Subtile pixel size within region (e.g. 256). **Required.**
        embed_dim_region: Region ViT output dimension.
        embed_dim_slide: Global transformer / output dimension.
        num_heads: Attention heads in region ViT.
        dropout: Dropout rate.
        pretrained_region_weights: Path to pretrained region ViT weights (optional).
    """

    def __init__(
        self,
        input_dim: int,
        region_size: int,
        patch_size: int,
        embed_dim_region: int = 192,
        embed_dim_slide: int = 192,
        num_heads: int = 6,
        dropout: float = 0.25,
        pretrained_region_weights: str | None = None,
    ) -> None:
        super().__init__()

        if region_size % patch_size != 0:
            msg = f"region_size ({region_size}) must be divisible by patch_size ({patch_size})"
            raise ValueError(msg)
        if region_size < 2 * patch_size:
            msg = (
                f"region_size ({region_size}) must be >= 2 * patch_size ({patch_size})"
            )
            raise ValueError(msg)

        self._npatch = region_size // patch_size
        self._P = self._npatch ** 2
        self._embed_dim_slide = embed_dim_slide

        # Region-level ViT
        self.vit_region = VisionTransformer4K(
            input_embed_dim=input_dim,
            output_embed_dim=embed_dim_region,
            npatch=self._npatch,
            num_heads=num_heads,
            drop_rate=dropout,
        )

        if pretrained_region_weights is not None:
            self._load_pretrained_region(pretrained_region_weights)

        # Global aggregation
        self.global_phi = nn.Sequential(
            nn.Linear(embed_dim_region, embed_dim_slide),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.global_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim_slide,
                nhead=3,
                dim_feedforward=embed_dim_slide,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=2,
        )

        self.global_attn_pool = AttentionPool(
            input_dim=embed_dim_slide,
            hidden_dim=embed_dim_slide,
            gated=True,
        )

        self.global_rho = nn.Sequential(
            nn.Linear(embed_dim_slide, embed_dim_slide),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def _load_pretrained_region(self, path: str) -> None:
        """Load pretrained weights for the region ViT."""
        if not Path(path).is_file():
            msg = f"Pretrained weights not found: {path}"
            raise FileNotFoundError(msg)

        state_dict = torch.load(path, weights_only=True, map_location="cpu")
        # Handle DINO-style checkpoints
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state_dict.items()
        }
        # Only load matching keys
        model_dict = self.vit_region.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        self.vit_region.load_state_dict(model_dict)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        """Forward pass.

        Args:
            X: Tile features ``(B, N, D)`` where ``N = M × P``, or
                hierarchical features ``(B, M, P, D)``.
            mask: Boolean mask ``(B, N)`` for flat inputs or ``(B, M)`` for
                hierarchical inputs. True = valid tile/region.

        Returns:
            AggregatorOutput with slide-level representation ``(B, D_slide)``.
        """
        P = self._P
        npatch = self._npatch

        if X.ndim == 4:
            B, M, P_in, D = X.shape
            if P_in != P:
                raise ValueError(f"Expected P={P} tiles per region, got {P_in}")
            if mask is None:
                region_mask = torch.ones(B, M, dtype=torch.bool, device=X.device)
            else:
                if mask.shape != (B, M):
                    raise ValueError(
                        f"Hierarchical mask must have shape (B, M), got {tuple(mask.shape)}"
                    )
                region_mask = mask
            X_vit = X.reshape(B * M, P, D).transpose(1, 2).reshape(B * M, D, npatch, npatch)
            region_embeds = self.vit_region(X_vit).reshape(B, M, -1)
        elif X.ndim == 3:
            B, N, D = X.shape

            if mask is None:
                mask = torch.ones(B, N, dtype=torch.bool, device=X.device)

            remainder = N % P
            if remainder != 0:
                pad_n = P - remainder
                X = torch.nn.functional.pad(X, (0, 0, 0, pad_n))
                mask = torch.nn.functional.pad(mask, (0, pad_n), value=False)
                N = N + pad_n

            M = N // P
            X_regions = X.reshape(B, M, P, D)
            tile_mask = mask.reshape(B, M, P)
            X_vit = X_regions.reshape(B * M, P, D)
            X_vit = X_vit.transpose(1, 2).reshape(B * M, D, npatch, npatch)
            region_embeds = self.vit_region(X_vit).reshape(B, M, -1)
            region_mask = tile_mask.any(dim=2)
        else:
            raise ValueError(f"HIPT expects rank-3 or rank-4 input, got rank {X.ndim}")

        z = self.global_phi(region_embeds)
        z = self.global_transformer(z, src_key_padding_mask=~region_mask)
        z, _attn_logits = self.global_attn_pool(z, mask=region_mask)
        z = self.global_rho(z)

        return AggregatorOutput(bag_representation=z, tile_attention=None)

    @property
    def output_dim(self) -> int:
        return self._embed_dim_slide


aggregator_registry.register("hipt", HIPT)
