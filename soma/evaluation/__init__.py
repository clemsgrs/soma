"""Evaluation module — metrics and reports."""

from soma.evaluation.metrics import (
    DEFAULT_METRICS,
    VALID_METRICS,
    compute_metrics,
    resolve_metrics,
)
from soma.evaluation.report import EvaluationReport, SamplePrediction

__all__ = [
    "DEFAULT_METRICS",
    "VALID_METRICS",
    "compute_metrics",
    "resolve_metrics",
    "EvaluationReport",
    "SamplePrediction",
]
