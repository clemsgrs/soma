"""Regression task head."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from soma.evaluation.metrics import metric_registry, resolve_metrics
from soma.tasks.base import TaskHead, build_input_dropout
from soma.tasks.registry import task_registry

if TYPE_CHECKING:
    from collections.abc import Sequence

    from soma.data import SampleRecord

#: The metrics a multi-output regression reports (one value per sample and output).
_MULTI_OUTPUT_METRICS = ("mean_pearson",)


class RegressionHead(TaskHead):
    """Linear regression head for one or many outputs.

    Args:
        input_dim: Dimension of the input representation.
        num_outputs: Number of regressed values ``G``. With one output the head reads the
            cohort's ``value`` target and predicts ``predicted_value``; with ``G > 1`` it
            reads ``value_0 .. value_{G-1}`` and predicts the same-named columns (a gene
            panel, say), scored by ``mean_pearson``.
        metrics: Metrics to compute. Empty uses the regression defaults (``mae``, ``r2``)
            for one output and ``mean_pearson`` for several.
        dropout: Probability of zeroing an element of the head's input, applied
            before the linear layer. ``0.0`` (default) builds no dropout module.
    """

    target_dtypes = {"value": torch.float}
    target_schema = {"value": float}
    task_family = "regression"

    def __init__(
        self,
        input_dim: int,
        num_outputs: int = 1,
        metrics: list[str] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if int(num_outputs) < 1:
            raise ValueError(f"num_outputs must be >= 1, got {num_outputs}")
        self.num_outputs = int(num_outputs)
        self.fc = nn.Linear(input_dim, self.num_outputs)
        self.dropout = build_input_dropout(dropout)
        if self.num_outputs == 1:
            self.metrics = resolve_metrics("regression", metrics or [])
        else:
            self.target_schema = {f"value_{k}": float for k in range(self.num_outputs)}
            requested = list(metrics or _MULTI_OUTPUT_METRICS)
            unsupported = sorted(set(requested) - set(_MULTI_OUTPUT_METRICS))
            if unsupported:
                raise ValueError(
                    f"a {self.num_outputs}-output regression reports {list(_MULTI_OUTPUT_METRICS)}; "
                    f"got {unsupported}"
                )
            self.metrics = requested

    @classmethod
    def auto_params(cls, records: "Sequence[SampleRecord]") -> dict[str, Any]:
        return {}

    def extract_targets(self, record: "SampleRecord") -> dict[str, Any]:
        if self.num_outputs == 1:
            return {"value": float(record.targets["value"])}
        return {"value": [float(record.targets[f"value_{k}"]) for k in range(self.num_outputs)]}

    def forward(self, X: Tensor) -> Tensor:
        if self.dropout is not None:
            X = self.dropout(X)
        return self.fc(X)

    def _values(self, raw_output: Tensor) -> Tensor:
        return raw_output.squeeze(-1) if self.num_outputs == 1 else raw_output

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        return F.mse_loss(self._values(predictions), targets["value"])

    def postprocess(self, raw_output: Tensor) -> dict[str, Any]:
        return {"predictions": self._values(raw_output).detach().cpu().numpy()}

    @property
    def prediction_columns(self) -> tuple[str, ...]:
        if self.num_outputs == 1:
            return ("predicted_value",)
        return tuple(f"value_{k}" for k in range(self.num_outputs))

    def to_columns(self, processed: dict[str, Any]) -> dict[str, np.ndarray]:
        values = np.asarray(processed["predictions"])
        if self.num_outputs == 1:
            return {"predicted_value": values.reshape(-1)}
        return {f"value_{k}": values[:, k] for k in range(self.num_outputs)}

    def compute_metrics(self, raw_output: Tensor, targets: dict[str, Tensor]) -> dict[str, float]:
        y_pred = self._values(raw_output).detach().cpu().numpy()
        y_true = targets["value"].detach().cpu().numpy()
        if self.num_outputs == 1:
            predicted = pd.DataFrame({"predicted_value": y_pred.reshape(-1)})
            truth = pd.DataFrame({"value": y_true.reshape(-1)})
        else:
            columns = list(self.prediction_columns)
            predicted = pd.DataFrame(y_pred, columns=columns)
            truth = pd.DataFrame(y_true, columns=columns)
        return {name: float(metric_registry.get(name).fn(predicted, truth)) for name in self.metrics}


task_registry.register("regression", RegressionHead)
