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
        auxiliary: Optional dict of auxiliary tensors for training-time
            losses (e.g. instance logits for DSMIL, embeddings for CLAM).
    """

    bag_representation: Tensor
    tile_attention: Tensor | None = None
    auxiliary: dict[str, Tensor] | None = None


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

    def compute_auxiliary_loss(
        self,
        auxiliary: dict[str, Tensor],
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor | None:
        """Compute auxiliary training loss from auxiliary output.

        Override in subclasses that produce auxiliary data
        (e.g. CLAM, DSMIL, DTFD-MIL). Returns None by default.

        Args:
            auxiliary: Dict of auxiliary tensors from forward().
            labels: Bag labels, shape (B,).
            mask: Boolean mask, shape (B, N). True = valid tile.

        Returns:
            Scalar auxiliary loss, or None if no auxiliary loss.
        """
        return None

    def combine_losses(
        self,
        task_loss: Tensor,
        auxiliary: dict[str, Tensor] | None,
        labels: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Combine task loss with any aggregator-specific auxiliary loss.

        Aggregators that use a custom weighting strategy can override this.
        By default, this preserves soma's existing behavior of adding any
        auxiliary loss to the task loss.
        """
        if auxiliary is None:
            return task_loss
        auxiliary_loss = self.compute_auxiliary_loss(auxiliary, labels, mask=mask)
        if auxiliary_loss is None:
            return task_loss
        return task_loss + auxiliary_loss
