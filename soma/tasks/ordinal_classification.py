"""Ordinal classification task head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.evaluation.metrics import compute_metrics, resolve_metrics
from soma.tasks.base import TaskHead, build_input_dropout
from soma.tasks.classification import _categorical_auto_params, _extract_categorical_target
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from soma.dataset import Dataset, SampleRecord


class OrdinalClassificationHead(TaskHead):
    """Linear head for ordinal classification using MSE loss.

    Treats ordered integer labels as continuous values during training (MSE
    loss) while producing integer predictions at inference by rounding the
    continuous output to the nearest class. Both the rounded prediction and
    the raw continuous score are reported.

    Args:
        input_dim: Dimension of the input representation.
        num_classes: Number of ordinal classes (e.g. 6 for labels 0–5).
        metrics: Metrics to compute. Empty list uses the default set for
            ordinal_classification: qwk, balanced_accuracy.
        dropout: Probability of zeroing an element of the head's input, applied
            before the linear layer. ``0.0`` (default) builds no dropout module.
    """

    target_dtypes = {"label": torch.long}
    task_family = "ordinal_classification"

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        metrics: list[str] | None = None,
        label_map: dict[str | int, int] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.dropout = build_input_dropout(dropout)
        self.num_classes = num_classes
        self._label_map = label_map
        self.metrics = resolve_metrics("ordinal_classification", metrics or [])

    @classmethod
    def auto_params(cls, dataset: Dataset) -> dict[str, Any]:
        return _categorical_auto_params(dataset)

    def extract_targets(self, record: "SampleRecord") -> dict[str, int | float]:
        return _extract_categorical_target(self._label_map, record)

    def forward(self, X: Tensor) -> Tensor:
        if self.dropout is not None:
            X = self.dropout(X)
        return self.fc(X)

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return F.mse_loss(predictions.squeeze(-1), targets["label"].float())

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        raw_scores = raw_output.squeeze(-1).detach().cpu().numpy()
        predicted_labels = np.clip(
            np.round(raw_scores), 0, self.num_classes - 1
        ).astype(int)
        return {"predicted_labels": predicted_labels, "raw_scores": raw_scores}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        processed = self.postprocess(raw_output)
        y_true = targets["label"].detach().cpu().numpy()
        return compute_metrics(
            "ordinal_classification", self.metrics, y_true, processed["predicted_labels"]
        )


task_registry.register("ordinal_classification", OrdinalClassificationHead)
