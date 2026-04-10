"""Binary and multi-class classification task heads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.evaluation.metrics import compute_metrics, resolve_metrics
from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import Dataset


class BinaryClassificationHead(TaskHead):
    """Linear classification head for binary (two-class) tasks.

    Args:
        input_dim: Dimension of the input representation.
        num_classes: Must be 2.
        metrics: Metrics to compute. Empty list uses the default set for
            binary_classification: auroc, balanced_accuracy, auprc, f1.
    """

    label_dtype = torch.long
    task_family = "binary_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError(
                f"BinaryClassificationHead requires num_classes=2, got {num_classes}. "
                "Use multiclass_classification for more than two classes."
            )
        self.fc = nn.Linear(input_dim, num_classes)
        self.num_classes = num_classes
        self.metrics = resolve_metrics("binary_classification", metrics or [])

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        return {"num_classes": dataset.num_classes}

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"BinaryClassificationHead expects input of shape (B, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        return self.fc(X)

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(predictions, targets)

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        return {"probabilities": probs, "predicted_labels": preds}

    def compute_metrics(self, raw_output: Tensor, targets: Tensor) -> dict[str, float]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true = targets.detach().cpu().numpy()
        return compute_metrics("binary_classification", self.metrics, y_true, preds, y_prob=probs)


task_registry.register("binary_classification", BinaryClassificationHead)


class MulticlassClassificationHead(TaskHead):
    """Linear classification head for multi-class tasks.

    Args:
        input_dim: Dimension of the input representation.
        num_classes: Number of output classes (>= 2).
        metrics: Metrics to compute. Empty list uses the default set for
            multiclass_classification: auroc_macro, balanced_accuracy, f1_macro.
    """

    label_dtype = torch.long
    task_family = "multiclass_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError(
                f"MulticlassClassificationHead requires num_classes >= 2, got {num_classes}."
            )
        self.fc = nn.Linear(input_dim, num_classes)
        self.num_classes = num_classes
        self.metrics = resolve_metrics("multiclass_classification", metrics or [])

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        return {"num_classes": dataset.num_classes}

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"MulticlassClassificationHead expects input of shape (B, D), got {tuple(X.shape)}"
            raise ValueError(msg)
        return self.fc(X)

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(predictions, targets)

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        return {"probabilities": probs, "predicted_labels": preds}

    def compute_metrics(self, raw_output: Tensor, targets: Tensor) -> dict[str, float]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true = targets.detach().cpu().numpy()
        return compute_metrics(
            "multiclass_classification", self.metrics, y_true, preds, y_prob=probs
        )


task_registry.register("multiclass_classification", MulticlassClassificationHead)


class BranchAwareClassificationHead(TaskHead):
    """Classification head for branch-aware bag representations.

    Expects one representation per class branch: `(B, C, D) -> (B, C)`.
    This matches reference-style CLAM-MB semantics.

    Args:
        input_dim: Dimension of each branch representation.
        num_classes: Number of output classes.
        metrics: Metrics to compute. Empty list uses the default set for
            multiclass_classification: auroc_macro, balanced_accuracy, f1_macro.
    """

    label_dtype = torch.long
    supports_branch_representation = True
    task_family = "multiclass_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        metrics: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.branch_fcs = nn.ModuleList(nn.Linear(input_dim, 1) for _ in range(num_classes))
        self.num_classes = num_classes
        self.metrics = resolve_metrics("multiclass_classification", metrics or [])

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        return {"num_classes": dataset.num_classes}

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
        logits = torch.empty(X.shape[0], self.num_classes, device=X.device, dtype=X.dtype)
        for idx, fc in enumerate(self.branch_fcs):
            logits[:, idx] = fc(X[:, idx, :]).squeeze(-1)
        return logits

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(predictions, targets)

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        return {"probabilities": probs, "predicted_labels": preds}

    def compute_metrics(self, raw_output: Tensor, targets: Tensor) -> dict[str, float]:
        probs = torch.softmax(raw_output, dim=1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        y_true = targets.detach().cpu().numpy()
        return compute_metrics(
            "multiclass_classification", self.metrics, y_true, preds, y_prob=probs
        )


task_registry.register("branch_aware_classification", BranchAwareClassificationHead)
