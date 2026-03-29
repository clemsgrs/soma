"""ABMIL — Attention-Based Multiple Instance Learning (Ilse et al., 2018).

Adapted from torchmil (Apache 2.0). Unlike torchmil's monolithic ABMIL,
this is a pure aggregator (no classifier, no loss). The classifier lives
in TaskHead, making ABMIL composable with any task.
"""

from __future__ import annotations

from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.mil.attention_pool import AttentionPool
from soma.aggregators.registry import aggregator_registry


class ABMIL(Aggregator):
    """Attention-Based MIL aggregator.

    Uses gated attention pooling to aggregate tile features into a
    slide-level representation, with per-tile attention weights for
    interpretability and heatmap generation.

    Args:
        input_dim: Feature dimension of input tiles.
        hidden_dim: Attention bottleneck dimension.
        activation: Activation function ('tanh', 'relu', 'gelu').
        gated: If True, use gated attention.
        dropout: Dropout rate applied before attention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        activation: str = "tanh",
        gated: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.pool = AttentionPool(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            gated=gated,
        )

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        X = self.dropout(X)
        z, attn_logits = self.pool(X, mask=mask)
        return AggregatorOutput(bag_representation=z, tile_attention=attn_logits)

    @property
    def output_dim(self) -> int:
        return self._input_dim


aggregator_registry.register("abmil", ABMIL)
