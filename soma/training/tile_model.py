"""TileClassifier — task head wrapper for encoded tile-image features."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from soma.tasks.base import TaskHead


@dataclass
class TileClassifierOutput:
    """Output of TileClassifier.forward."""

    logits: Tensor  # (B, num_classes)


class TileClassifier(nn.Module):
    """Classifier for encoded tile-image features.

    Takes per-tile embeddings ``(B, D)`` produced by a tile encoder and maps
    them directly to logits via a task head. No aggregation is applied — each
    sample is a single tile.

    Args:
        task_head: Task head (classification or regression) to apply.
    """

    def __init__(self, task_head: TaskHead) -> None:
        super().__init__()
        self.task_head = task_head

    def forward(self, X: Tensor) -> TileClassifierOutput:
        """Forward pass.

        Args:
            X: Encoded tile features of shape ``(B, D)``.

        Returns:
            TileClassifierOutput with logits of shape ``(B, num_classes)``.
        """
        return TileClassifierOutput(logits=self.task_head(X))
