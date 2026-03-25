"""TaskHead base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class TaskHead(ABC, nn.Module):
    """Abstract base class for task heads.

    Maps an aggregated slide representation to task-specific predictions.
    """

    @abstractmethod
    def forward(self, X: Tensor) -> Tensor:
        """Produce predictions from aggregated representation.

        Args:
            X: Slide-level representation, shape (B, D).

        Returns:
            Predictions, shape depends on task (e.g. (B, num_classes)).
        """
        ...

    @abstractmethod
    def compute_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute task-specific loss.

        Args:
            logits: Model predictions.
            targets: Ground truth labels.

        Returns:
            Scalar loss tensor.
        """
        ...
