"""TaskHead base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from soma.dataset import Dataset, SampleRecord


def build_input_dropout(dropout: float) -> nn.Dropout | None:
    """Build a task head's input-dropout module, or ``None`` at the default.

    ``None`` rather than an ``nn.Identity`` placeholder: assigning ``None`` to an
    ``nn.Module`` attribute registers no submodule, so a head at the default has
    exactly the modules, ``state_dict`` and random-number consumption of a head built
    before the knob existed. Heads guard their forward on ``is not None``.

    Args:
        dropout: Probability of zeroing an input element, in ``[0.0, 1.0)``.

    Returns:
        An ``nn.Dropout`` when ``dropout > 0``, else ``None``.
    """
    if isinstance(dropout, bool) or not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be a probability in [0.0, 1.0), got {dropout!r}.")
    return nn.Dropout(dropout) if dropout > 0.0 else None


class TaskHead(ABC, nn.Module):
    """Abstract base class for task heads.

    Maps an aggregated slide representation to task-specific predictions.

    Subclasses must implement forward, compute_loss, postprocess,
    compute_metrics, and extract_targets, and should set target_dtypes and
    auto_params appropriately.

    The head owns its target contract: ``target_dtypes`` declares the keys it
    consumes (and their tensor dtypes), ``extract_targets`` maps a
    ``SampleRecord`` to a dict of raw target values, and ``compute_loss`` /
    ``compute_metrics`` receive a ``dict[str, Tensor]`` keyed by those names.
    """

    target_dtypes: dict[str, torch.dtype] = {"label": torch.long}
    supports_branch_representation: bool = False
    task_family: str = "generic"
    # When True, the trainer computes the tune loss once over the whole tune
    # cohort (concatenated logits/targets) instead of averaging per-batch losses.
    # Required for losses that couple samples within a batch (e.g. CoxPH partial
    # likelihood), where the mean of per-batch losses is not the full-cohort loss.
    full_cohort_eval_loss: bool = False
    # When True, the pipeline builds the training loader with an event-balanced
    # BatchSampler so every batch contains at least one event (needed for Cox).
    needs_event_balanced_batches: bool = False
    # When True, the trainer uses a windowed training loop: it forwards each bag
    # in a window un-padded, keeps the per-bag outputs graph-connected, and
    # computes one loss over the window (Cox prediction accumulation for large
    # variable-size MIL bags). Instances set this; the class default is False.
    accumulates_predictions: bool = False

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
    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        """Map a sample record to this head's raw target values.

        Args:
            record: The SampleRecord for one sample (for patient-level
                pipelines, a representative record for the patient).

        Returns:
            Dict mapping each key in ``target_dtypes`` to a raw scalar value.
        """
        ...

    @abstractmethod
    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        """Compute task-specific loss.

        Args:
            predictions: Model predictions.
            targets: Ground truth targets keyed by ``target_dtypes``.

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
    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        """Compute task-specific evaluation metrics.

        Args:
            raw_output: Raw output tensor from forward(), shape (B, ...).
            targets: Ground truth targets keyed by ``target_dtypes``.

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
