"""Evaluation report dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SamplePrediction:
    """Per-sample prediction record for audit trail."""

    sample_id: str
    true_label: int | float
    predicted_label: int
    probabilities: list[float]


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation results for a single split.

    Attributes:
        split: Split name (e.g. 'test', 'tune').
        metrics: Metric name -> value.
        predictions: Per-sample prediction records.
    """

    split: str
    metrics: dict[str, float]
    predictions: list[SamplePrediction]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
