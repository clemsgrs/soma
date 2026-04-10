"""Ordinal classification task head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.evaluation.metrics import compute_ordinal_metrics
from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import Dataset


class OrdinalClassificationHead(TaskHead):
    """Linear head for ordinal classification using MSE loss.

    Treats ordered integer labels as continuous values during training (MSE
    loss) while producing integer predictions at inference by rounding the
    continuous output to the nearest class. Both the rounded prediction and
    the raw continuous score are reported.

    Args:
        input_dim: Dimension of the input representation.
        num_classes: Number of ordinal classes (e.g. 6 for labels 0–5).
    """

    label_dtype = torch.long
    task_family = "ordinal_classification"

    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.num_classes = num_classes

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        return {"num_classes": dataset.num_classes}

    def forward(self, X: Tensor) -> Tensor:
        return self.fc(X)

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        return F.mse_loss(predictions.squeeze(-1), targets.float())

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        raw_scores = raw_output.squeeze(-1).detach().cpu().numpy()
        predicted_labels = np.clip(
            np.round(raw_scores), 0, self.num_classes - 1
        ).astype(int)
        return {"predicted_labels": predicted_labels, "raw_scores": raw_scores}

    def compute_metrics(self, raw_output: Tensor, targets: Tensor) -> dict[str, float]:
        processed = self.postprocess(raw_output)
        y_true = targets.detach().cpu().numpy()
        return compute_ordinal_metrics(y_true, processed["predicted_labels"])


task_registry.register("ordinal_classification", OrdinalClassificationHead)
