"""TransMIL — Transformer-based Correlated MIL (Shao et al., 2021).

Adapted from torchmil. Unlike torchmil's monolithic TransMIL,
this is a pure aggregator (no classifier, no loss). The classifier lives
in TaskHead, making TransMIL composable with any task.
"""

from __future__ import annotations

from math import ceil, sqrt

import torch
from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.nystrom import NystromTransformerLayer
from soma.aggregators.registry import aggregator_registry


class PPEG(nn.Module):
    """Pyramid Positional Encoding Generator.

    Uses three depthwise convolutions (7x7, 5x5, 3x3) in parallel to
    inject spatial positional information into the feature sequence.

    Args:
        dim: Feature dimension.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """Apply pyramid positional encoding.

        Args:
            x: Input, shape (B, H*W+1, D). First token is cls_token.
            H: Grid height.
            W: Grid width.

        Returns:
            Positionally-encoded features, shape (B, H*W+1, D).
        """
        B, _, D = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, D, H, W)
        y = cnn_feat + self.proj(cnn_feat) + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        y = y.flatten(2).transpose(1, 2)
        return torch.cat((cls_token.unsqueeze(1), y), dim=1)


class TransMIL(Aggregator):
    """Transformer-based Correlated MIL aggregator.

    Uses Nystromformer layers with PPEG positional encoding and a
    learnable class token to aggregate tile features.

    Args:
        input_dim: Feature dimension of input tiles.
        att_dim: Transformer embedding dimension.
        n_layers: Number of Nystromformer layers (must be >= 2).
        n_heads: Number of attention heads.
        n_landmarks: Landmarks for Nystrom approximation (default: att_dim//2).
        pinv_iterations: Pseudo-inverse iterations.
        dropout: Dropout rate.
        use_mlp: Whether to use MLP blocks in transformer layers.
    """

    def __init__(
        self,
        input_dim: int,
        att_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 4,
        n_landmarks: int | None = None,
        pinv_iterations: int = 6,
        dropout: float = 0.0,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()

        if n_layers < 2:
            msg = f"n_layers must be at least 2, got {n_layers}"
            raise ValueError(msg)

        self._att_dim = att_dim

        if n_landmarks is None:
            n_landmarks = att_dim // 2

        self.fc1 = (
            nn.Linear(input_dim, att_dim) if input_dim != att_dim else nn.Identity()
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, att_dim))
        self.pos_layer = PPEG(dim=att_dim)

        self.layers = nn.ModuleList(
            [
                NystromTransformerLayer(
                    dim=att_dim,
                    n_heads=n_heads,
                    n_landmarks=n_landmarks,
                    pinv_iterations=pinv_iterations,
                    dropout=dropout,
                    use_mlp=use_mlp,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(att_dim)

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        B, bag_size, _ = X.shape

        X = self.fc1(X)  # (B, N, att_dim)

        # Pad to perfect square
        padded_size = int(ceil(sqrt(bag_size)))
        total = padded_size * padded_size
        add_length = total - bag_size

        if add_length > 0:
            # Repeat first tiles to fill the square
            X = torch.cat([X, X[:, :add_length, :]], dim=1)

        # Build mask for padded+cls sequence
        layer_mask = None
        if mask is not None:
            # Pad mask: duplicated tiles are marked as False (padding)
            pad_mask = torch.zeros(B, add_length, dtype=torch.bool, device=X.device)
            padded_mask = torch.cat([mask, pad_mask], dim=1) if add_length > 0 else mask
            # Prepend True for cls_token
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=X.device)
            layer_mask = torch.cat([cls_mask, padded_mask], dim=1)

        # Add cls_token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)  # (B, total+1, att_dim)

        # First transformer layer
        X = self.layers[0](X, mask=layer_mask)

        # Positional encoding
        X = self.pos_layer(X, padded_size, padded_size)

        # Remaining layers (except last)
        for layer in self.layers[1:-1]:
            X = layer(X, mask=layer_mask)

        # Last layer with attention extraction
        out = self.layers[-1](X, mask=layer_mask, return_att=True)
        X, attn = out  # attn: (B, H, total+1, total+1)

        # Average attention across heads, extract cls→instance attention
        attn = attn.mean(dim=1)  # (B, total+1, total+1)
        tile_attention = attn[:, 0, 1 : bag_size + 1]  # (B, bag_size)

        # Layer norm and extract cls_token
        X = self.norm(X)
        cls_output = X[:, 0]  # (B, att_dim)

        return AggregatorOutput(
            bag_representation=cls_output,
            tile_attention=tile_attention,
        )

    @property
    def output_dim(self) -> int:
        return self._att_dim


aggregator_registry.register("transmil", TransMIL)
