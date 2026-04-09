"""TaskHead base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from soma.dataset import Dataset


class TaskHead(ABC, nn.Module):
    """Abstract base class for task heads.

    Maps an aggregated slide representation to task-specific predictions.

    Subclasses must implement forward, compute_loss, postprocess, and
    compute_metrics, and should set label_dtype and auto_params appropriately.
    """

    label_dtype: torch.dtype = torch.long

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
    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute task-specific loss.

        Args:
            predictions: Model predictions.
            targets: Ground truth labels.

        Returns:
            Scalar loss tensor.
        """
        ...

    @abstractmethod
    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        """Convert raw model output to task-specific predictions.

        Args:
            raw_output: Raw output tensor from forward(), shape (B, ...).

        Returns:
            Dict with task-specific keys, e.g.:
              - classification: {"probabilities": np.ndarray, "predicted_labels": np.ndarray}
              - regression: {"predictions": np.ndarray}
        """
        ...

    @abstractmethod
    def compute_metrics(self, raw_output: Tensor, targets: Tensor) -> dict[str, float]:
        """Compute task-specific evaluation metrics.

        Args:
            raw_output: Raw output tensor from forward(), shape (B, ...).
            targets: Ground truth labels tensor, shape (B,).

        Returns:
            Dict mapping metric name to scalar value.
        """
        ...

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        """Return parameters to auto-inject from the dataset during instantiation.

        The pipeline merges these with user-provided task.params before calling
        the constructor. Override to inject dataset-derived values such as
        num_classes for classification.

        Args:
            dataset: The Dataset instance for the current run.

        Returns:
            Dict of keyword arguments to pass to __init__.
        """
        return {}
