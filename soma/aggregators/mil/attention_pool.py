"""Gated attention pooling — adapted from torchmil (Apache 2.0).

Core building block for ABMIL and CLAM. Not an Aggregator itself — used
internally by MIL aggregator modules.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def masked_softmax(X: Tensor, mask: Tensor | None = None) -> Tensor:
    """Softmax with optional boolean mask.

    Args:
        X: Input tensor, shape (..., N, 1) or (..., N).
        mask: Boolean mask, same shape as X. True = valid, False = masked out.

    Returns:
        Softmax probabilities with masked positions set to ~0.
    """
    if mask is not None:
        X = X.masked_fill(~mask, float("-inf"))
    return torch.softmax(X, dim=1)


class AttentionPool(nn.Module):
    """Gated attention pooling (Ilse et al., 2018).

    Computes z = X^T @ softmax(MLP(X)), where MLP is a two-layer network
    with optional gating.

    Adapted from torchmil (Apache 2.0).

    Args:
        input_dim: Feature dimension of input tiles.
        hidden_dim: Attention bottleneck dimension.
        activation: Activation function ('tanh', 'relu', 'gelu').
        gated: If True, use gated attention (tanh * sigmoid).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        activation: str = "tanh",
        gated: bool = True,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1, bias=False)

        self.gated = gated
        if gated:
            self.fc_gate = nn.Linear(input_dim, hidden_dim)

        activations = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}
        if activation not in activations:
            msg = f"activation must be one of {list(activations)}, got '{activation}'"
            raise ValueError(msg)
        self.act = activations[activation]()

    def forward(
        self, X: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            X: Tile features, shape (B, N, D).
            mask: Boolean mask, shape (B, N). True = valid tile.

        Returns:
            z: Bag representation, shape (B, D).
            attn_logits: Raw attention logits, shape (B, N).
        """
        H = self.act(self.fc1(X))  # (B, N, hidden_dim)

        if self.gated:
            G = torch.sigmoid(self.fc_gate(X))  # (B, N, hidden_dim)
            H = H * G

        f = self.fc2(H)  # (B, N, 1)

        # Build mask for softmax: (B, N, 1)
        mask_3d = mask.unsqueeze(-1) if mask is not None else None
        s = masked_softmax(f, mask_3d)  # (B, N, 1)

        z = torch.bmm(X.transpose(1, 2), s).squeeze(-1)  # (B, D)

        return z, f.squeeze(-1)  # (B, D), (B, N)
