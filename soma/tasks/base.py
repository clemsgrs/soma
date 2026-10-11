"""TaskHead base class."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from soma.evaluation.predictions import family_prediction_columns

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from soma.data import SampleRecord


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


def as_grid_geometry(geometry):
    """A grid head's bound geometry as a :class:`~soma.data.GridGeometry`.

    A bare :class:`~soma.data.geometry.DenseGridGeometry` (a frame layout without a
    level-0 anchor) anchors at the origin with level-0 pixels equal to frame pixels.
    """
    from soma.data.geometry import GridGeometry

    if isinstance(geometry, GridGeometry):
        return geometry
    return GridGeometry.from_dense(geometry)


_CLASSIFICATION_FAMILIES = frozenset({"binary_classification", "multiclass_classification"})
_LABEL_FAMILIES = _CLASSIFICATION_FAMILIES | {"ordinal_classification"}


class TaskHead(ABC, nn.Module):
    """Abstract base class for task heads.

    Maps an aggregated slide representation to task-specific predictions.

    Subclasses must implement forward, compute_loss, postprocess,
    compute_metrics, and extract_targets, and should set target_dtypes and
    auto_params appropriately. ``prediction_columns`` / ``to_columns`` declare what
    :func:`soma.predict` writes; the base implementation covers soma's set families.

    The head owns its target contract: ``target_dtypes`` declares the keys it
    consumes (and their tensor dtypes), ``extract_targets`` maps a
    ``SampleRecord`` to a dict of raw target values, and ``compute_loss`` /
    ``compute_metrics`` receive a ``dict[str, Tensor]`` keyed by those names.
    """

    target_dtypes: dict[str, torch.dtype] = {"label": torch.long}
    # The record targets this head reads and their Python kind (``object`` = any raw
    # value, e.g. a categorical label); ``Cohort`` validates presence and kind up front.
    # Dense heads read their supervision from a TargetSource and declare none.
    target_schema: dict[str, type] = {"label": object}
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
    def resolve_target_schema(cls, params: "Mapping[str, Any] | None" = None) -> dict[str, type]:
        """The ``target_schema`` of a head built from ``params`` (a config's ``task.params``).

        What cohort loading validates before any head exists. The class's own
        ``target_schema`` by default; a head whose target keys depend on its params (a
        multi-output regression) overrides it.
        """
        return dict(cls.target_schema)

    @property
    def target_keys(self) -> tuple[str, ...]:
        return tuple(self.target_schema)

    # --- predictions ------------------------------------------------------ #

    @property
    def classes(self) -> tuple[Any, ...] | None:
        """The raw label of each class index (``prob_<i>`` is ``classes[i]``), or ``None``.

        A head built with a ``label_map`` reports its labels in index order; a label
        head without one reads labels as class indices ``0..num_classes-1``.
        """
        label_map = getattr(self, "_label_map", None)
        if label_map:
            return tuple(sorted(label_map, key=label_map.__getitem__))
        num_classes = getattr(self, "num_classes", None)
        if self.task_family in _LABEL_FAMILIES and num_classes:
            return tuple(range(int(num_classes)))
        return None

    @property
    def prediction_columns(self) -> tuple[str, ...]:
        """The columns :meth:`to_columns` emits, keyed on ``task_family``: ``prob_<i>`` +
        ``predicted_label`` (classification), ``predicted_label`` + ``raw_score``
        (ordinal), ``predicted_value`` (regression), ``risk_score`` (survival). A head
        of another family overrides this and :meth:`to_columns`."""
        return family_prediction_columns(self.task_family, getattr(self, "num_classes", None))

    def to_columns(self, processed: dict[str, Any]) -> dict[str, np.ndarray]:
        """Map :meth:`postprocess` output to the declared prediction columns, one value per sample."""
        family = self.task_family
        if family in _CLASSIFICATION_FAMILIES:
            probabilities = np.asarray(processed["probabilities"])
            columns = {f"prob_{i}": probabilities[:, i] for i in range(probabilities.shape[1])}
            columns["predicted_label"] = np.asarray(processed["predicted_labels"]).astype(np.int64)
            return columns
        if family == "ordinal_classification":
            return {
                "predicted_label": np.asarray(processed["predicted_labels"]).astype(np.int64),
                "raw_score": np.asarray(processed["raw_scores"]),
            }
        if family == "regression":
            values = np.asarray(processed["predictions"])
            if values.ndim == 2 and values.shape[1] == 1:
                values = values[:, 0]
            if values.ndim != 1:
                raise ValueError(
                    f"regression predictions of shape {values.shape} have no single "
                    "predicted_value column; override prediction_columns and to_columns"
                )
            return {"predicted_value": values}
        if family == "survival":
            return {"risk_score": np.asarray(processed["risk_scores"])}
        raise NotImplementedError(
            f"{type(self).__name__} (task family {family!r}) declares no set-shape "
            "prediction columns; override prediction_columns and to_columns"
        )

    @classmethod
    def auto_params(cls, records: "Sequence[SampleRecord]") -> dict[str, Any]:
        """Return parameters to auto-inject from the cohort's records during instantiation.

        The pipeline merges these with user-provided task.params before calling
        the constructor. Override to inject cohort-derived values such as
        num_classes for classification.

        Args:
            records: The records of the current run (``cohort.records``).

        Returns:
            Dict of keyword arguments to pass to __init__.
        """
        return {}
