"""Aggregator base class and structured output."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class AggregatorOutput:
    """Structured output from an aggregator.

    Attributes:
        bag_representation: Slide-level representation, shape (B, D_out).
        tile_attention: Per-tile attention weights, shape (B, N).
            None for non-attention methods (e.g. MeanPool, MaxPool).
    """

    bag_representation: Tensor
    tile_attention: Tensor | None = None


class Aggregator(ABC, nn.Module):
    """Abstract base class for MIL aggregators.

    Consumes a bag of tile features (B, N, D) and produces a slide-level
    representation (B, D_out) wrapped in an AggregatorOutput.
    """

    @abstractmethod
    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        """Aggregate tile features into a slide-level representation.

        Args:
            X: Tile features, shape (B, N, D).
            mask: Boolean mask, shape (B, N). True = valid tile.

        Returns:
            AggregatorOutput with bag_representation and optional tile_attention.
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the bag representation."""
        ...
