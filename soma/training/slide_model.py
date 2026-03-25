"""SlideModel — task head wrapper for slide-level (pre-aggregated) features."""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from soma.tasks.base import TaskHead


@dataclass
class SlideModelOutput:
    """Output of SlideModel.forward."""

    logits: Tensor  # (B, num_classes)


class SlideModel(nn.Module):
    """Model for slide-level features: a task head with no aggregator.

    Takes pre-aggregated slide embeddings (B, D) directly and maps them
    to logits via a task head, bypassing MIL aggregation.
    """

    def __init__(self, task_head: TaskHead) -> None:
        super().__init__()
        self.task_head = task_head

    def forward(self, X: Tensor) -> SlideModelOutput:
        """Forward pass.

        Args:
            X: Slide-level features of shape (B, D).

        Returns:
            SlideModelOutput with logits of shape (B, num_classes).
        """
        return SlideModelOutput(logits=self.task_head(X))
