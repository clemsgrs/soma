"""Decoder base class — maps a dense feature grid to per-pixel class logits.

The ``decoder`` is a first-class, swappable component parallel to ``aggregator``
(see segmentation-design §6): an aggregator pools a bag of tile features into a
slide vector, whereas a decoder expands a *single tile's* dense feature grid into
per-pixel logits.

Contract: ``forward`` maps ``(B, d, h, w) -> (B, num_classes, h', w')`` where
``d`` is the encoder's channel dim, ``(h, w)`` the token grid, and ``(h', w')``
the (possibly upsampled) decoder grid. The decoder owns the feature→logits mapping
and any *learned* upsampling; the ``SegmentationHead`` (a later slice) crops/resizes
the logits to the mask's target size before loss and metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Decoder(ABC, nn.Module):
    """Abstract base class for dense decoders."""

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Map a dense feature grid to class logits.

        Args:
            X: Dense feature grid, shape ``(B, d, h, w)``.

        Returns:
            Per-pixel class logits, shape ``(B, num_classes, h', w')``.
        """
        ...

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of output channels (segmentation classes)."""
        ...
