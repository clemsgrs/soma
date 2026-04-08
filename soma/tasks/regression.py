"""Regression task head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.evaluation.metrics import compute_regression_metrics
from soma.tasks.base import TaskHead
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import Dataset


class RegressionHead(TaskHead):
    """Linear regression head for single or multi-target regression.

    Args:
        input_dim: Dimension of the input representation.
        num_targets: Number of regression targets. Default 1.
    """

    label_dtype = torch.float

    def __init__(self, input_dim: int, num_targets: int = 1) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, num_targets)
        self.num_targets = num_targets

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        return {}

    def forward(self, X: Tensor) -> Tensor:
        return self.fc(X)

    def compute_loss(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if self.num_targets == 1:
            predictions = predictions.squeeze(-1)
        return F.mse_loss(predictions, targets)

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        preds = raw_output.squeeze(-1) if self.num_targets == 1 else raw_output
        return {"predictions": preds.detach().cpu().numpy()}

    def compute_metrics(self, raw_output: Tensor, targets: Tensor) -> dict[str, float]:
        preds = raw_output.squeeze(-1) if self.num_targets == 1 else raw_output
        y_pred = preds.detach().cpu().numpy()
        y_true = targets.detach().cpu().numpy()
        return compute_regression_metrics(y_true, y_pred)


task_registry.register("regression", RegressionHead)
