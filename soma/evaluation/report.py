"""Evaluation report dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SamplePrediction:
    """Per-sample prediction record for audit trail.

    For survival predictions, ``true_label`` holds the time-to-event, ``event``
    the 0/1 indicator, and ``risk_score`` the predicted risk (higher = earlier
    predicted event).
    """

    sample_id: str
    true_label: int | float
    predicted_label: int | None = None
    probabilities: list[float] | None = None
    predicted_value: float | None = None
    raw_score: float | None = None
    risk_score: float | None = None
    event: float | None = None
    is_placeholder: bool = False
    missing_reason: str | None = None


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
