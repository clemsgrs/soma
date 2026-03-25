"""Simple pooling aggregators — MeanPool and MaxPool."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from soma.aggregators.base import Aggregator, AggregatorOutput
from soma.aggregators.registry import aggregator_registry


class MeanPool(Aggregator):
    """Masked mean pooling over tile features.

    Adapted from torchmil (Apache 2.0). No attention — returns tile_attention=None.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        if mask is None:
            z = X.mean(dim=1)
        else:
            mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)
            count = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
            z = (X * mask_expanded).sum(dim=1) / count  # (B, D)
        return AggregatorOutput(bag_representation=z)

    @property
    def output_dim(self) -> int:
        return self._input_dim


class MaxPool(Aggregator):
    """Masked max pooling over tile features.

    Adapted from torchmil (Apache 2.0). No attention — returns tile_attention=None.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self._input_dim = input_dim

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)
            X = X.masked_fill(~mask_expanded, float("-inf"))
        z = X.max(dim=1).values  # (B, D)
        return AggregatorOutput(bag_representation=z)

    @property
    def output_dim(self) -> int:
        return self._input_dim


# Register
aggregator_registry.register("mean_pool", MeanPool)
aggregator_registry.register("max_pool", MaxPool)
