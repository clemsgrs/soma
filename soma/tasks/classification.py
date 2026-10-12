"""Binary and multi-class classification task heads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.evaluation.metrics import compute_metrics, resolve_metrics
from soma.params import check_positive, check_probability
from soma.tasks.base import TaskHead, build_input_dropout
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from soma.data import Cohort, SampleRecord


def categorical_label_map(records: "Sequence[SampleRecord]") -> dict[str | int, int]:
    """Build the sorted raw-label -> integer-index map for categorical tasks.

    Sorted raw labels, the ordering soma has always used, so
    class indices (and thus the positive class for binary metrics) are stable.
    """
    classes = sorted({r.targets["label"] for r in records})
    return {label: idx for idx, label in enumerate(classes)}


@dataclass
class ClassificationParams:
    """The params of the categorical heads (binary, multiclass, branch-aware, ordinal).

    ``num_classes`` and ``label_map`` left unset are inferred from the cohort's labels
    (:func:`infer_label_map`); ``input_dim`` comes from the model and ``metrics`` from
    the evaluation section.
    """

    num_classes: int | None = None
    label_map: dict[str | int, int] | None = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        check_positive("num_classes", self.num_classes, integer=True, optional=True)
        check_probability("dropout", self.dropout, below_one=True)


def infer_label_map(cohort: "Cohort", params: "Mapping[str, Any]") -> dict[str, Any]:
    """``num_classes`` and the sorted raw-label -> index ``label_map`` of the cohort."""
    label_map = categorical_label_map(cohort.records)
    return {"num_classes": len(label_map), "label_map": label_map}


def _check_label_map_width(num_classes: int, label_map: dict[str | int, int] | None) -> None:
    """Refuse a label map that names fewer (or more) classes than the head has outputs.

    ``prob_<i>`` is ``classes[i]``, so every probability column needs a raw label. A
    configured ``num_classes`` above the labels the cohort shows leaves columns unnamed.
    """
    if label_map is not None and len(label_map) != num_classes:
        raise ValueError(
            f"num_classes={num_classes} but the label map names {len(label_map)} class(es) "
            f"({sorted(label_map, key=label_map.__getitem__)}). Each probability column "
            "needs a label: set task.params.label_map to a {label: index} map naming every "
            "class, or drop num_classes to infer it from the cohort's labels."
        )


def _extract_categorical_target(
    label_map: dict[str | int, int] | None, record: "SampleRecord"
) -> dict[str, int | float]:
    raw = record.targets["label"]
    encoded = int(label_map[raw]) if label_map is not None else int(raw)
    return {"label": encoded}


class BinaryClassificationHead(TaskHead):
    """Linear classification head for binary (two-class) tasks.

    Args:
        input_dim: Dimension of the input representation.
        num_classes: Must be 2.
        metrics: Metrics to compute. Empty list uses the default set for
            binary_classification: auroc, balanced_accuracy, auprc, f1.
        dropout: Probability of zeroing an element of the head's input, applied
            before the linear layer. ``0.0`` (default) builds no dropout module.
    """

    target_dtypes = {"label": torch.long}
    task_family = "binary_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        metrics: list[str] | None = None,
        label_map: dict[str | int, int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError(
                f"BinaryClassificationHead requires num_classes=2, got {num_classes}. "
                "Use multiclass_classification for more than two classes."
            )
        _check_label_map_width(num_classes, label_map)
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = build_input_dropout(dropout)
        self.num_classes = num_classes
        self._label_map = label_map
        self.metrics = resolve_metrics("binary_classification", metrics or [])

    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        return _extract_categorical_target(self._label_map, record)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"BinaryClassificationHead expects input of shape (B, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        if self.dropout is not None:
            X = self.dropout(X)
        return self.fc(X)

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return F.cross_entropy(predictions, targets["label"])

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        return {"probabilities": probs, "predicted_labels": preds}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true = targets["label"].detach().cpu().numpy()
        return compute_metrics("binary_classification", self.metrics, y_true, preds, y_prob=probs)


task_registry.register("binary_classification", BinaryClassificationHead, params=ClassificationParams, infer=infer_label_map)


class MulticlassClassificationHead(TaskHead):
    """Linear classification head for multi-class tasks.

    Args:
        input_dim: Dimension of the input representation.
        num_classes: Number of output classes (>= 2).
        metrics: Metrics to compute. Empty list uses the default set for
            multiclass_classification: auroc_macro, balanced_accuracy, f1_macro.
        dropout: Probability of zeroing an element of the head's input, applied
            before the linear layer. ``0.0`` (default) builds no dropout module.
    """

    target_dtypes = {"label": torch.long}
    task_family = "multiclass_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        metrics: list[str] | None = None,
        label_map: dict[str | int, int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"MulticlassClassificationHead requires num_classes >= 2, got {num_classes}."
            )
        _check_label_map_width(num_classes, label_map)
        self.fc = nn.Linear(input_dim, num_classes)
        self.dropout = build_input_dropout(dropout)
        self.num_classes = num_classes
        self._label_map = label_map
        self.metrics = resolve_metrics("multiclass_classification", metrics or [])

    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        return _extract_categorical_target(self._label_map, record)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"MulticlassClassificationHead expects input of shape (B, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        if self.dropout is not None:
            X = self.dropout(X)
        return self.fc(X)

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return F.cross_entropy(predictions, targets["label"])

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        return {"probabilities": probs, "predicted_labels": preds}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true = targets["label"].detach().cpu().numpy()
        return compute_metrics(
            "multiclass_classification", self.metrics, y_true, preds, y_prob=probs
        )


task_registry.register("multiclass_classification", MulticlassClassificationHead, params=ClassificationParams, infer=infer_label_map)


class BranchAwareClassificationHead(TaskHead):
    """Classification head for branch-aware bag representations.

    Expects one representation per class branch: `(B, C, D) -> (B, C)`.
    This matches reference-style CLAM-MB semantics.

    Args:
        input_dim: Dimension of each branch representation.
        num_classes: Number of output classes.
        metrics: Metrics to compute. Empty list uses the default set for
            multiclass_classification: auroc_macro, balanced_accuracy, f1_macro.
        dropout: Probability of zeroing an element of the head's input, applied
            before the per-branch linear layers. ``0.0`` (default) builds no dropout module.
    """

    target_dtypes = {"label": torch.long}
    supports_branch_representation = True
    task_family = "multiclass_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        metrics: list[str] | None = None,
        label_map: dict[str | int, int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _check_label_map_width(num_classes, label_map)
        self.branch_fcs = nn.ModuleList(nn.Linear(input_dim, 1) for _ in range(num_classes))
        self.dropout = build_input_dropout(dropout)
        self.num_classes = num_classes
        self._label_map = label_map
        self.metrics = resolve_metrics("multiclass_classification", metrics or [])

    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        return _extract_categorical_target(self._label_map, record)

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 3:
            msg = f"BranchAwareClassificationHead expects input of shape (B, C, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        if X.shape[1] != self.num_classes:
            msg = (
                f"Branch-aware classification expects {self.num_classes} branches, "
                f"got {X.shape[1]}"
            )
            raise ValueError(msg)
        if self.dropout is not None:
            X = self.dropout(X)
        logits = torch.empty(X.shape[0], self.num_classes, device=X.device, dtype=X.dtype)
        for idx, fc in enumerate(self.branch_fcs):
            logits[:, idx] = fc(X[:, idx, :]).squeeze(-1)
        return logits

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return F.cross_entropy(predictions, targets["label"])

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        return {"probabilities": probs, "predicted_labels": preds}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true = targets["label"].detach().cpu().numpy()
        return compute_metrics(
            "multiclass_classification", self.metrics, y_true, preds, y_prob=probs
        )


task_registry.register("branch_aware_classification", BranchAwareClassificationHead, params=ClassificationParams, infer=infer_label_map)
