"""Classification task head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.evaluation.metrics import compute_classification_metrics
from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import Dataset


class ClassificationHead(TaskHead):
    """Linear classification head for binary or multi-class tasks.

    Args:
        input_dim: Dimension of the input representation.
        num_classes: Number of output classes.
    """

    label_dtype = torch.long
    task_family = "classification"

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.num_classes = num_classes

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        return {"num_classes": dataset.num_classes}

    def forward(self, X: Tensor) -> Tensor:
        if X.ndim != 2:
            msg = f"ClassificationHead expects input of shape (B, D), got {tuple(X.shape)}"
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
        return compute_classification_metrics(y_true, probs, preds)


task_registry.register("classification", ClassificationHead)


class BranchAwareClassificationHead(TaskHead):
    """Classification head for branch-aware bag representations.

    Expects one representation per class branch: `(B, C, D) -> (B, C)`.
    This matches reference-style CLAM-MB semantics.
    """

    label_dtype = torch.long
    supports_branch_representation = True
    task_family = "classification"

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.branch_fcs = nn.ModuleList(nn.Linear(input_dim, 1) for _ in range(num_classes))
        self.num_classes = num_classes

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
        return compute_classification_metrics(y_true, probs, preds)


task_registry.register("branch_aware_classification", BranchAwareClassificationHead)