"""Evaluation module — metrics and reports."""

from soma.evaluation.metrics import compute_classification_metrics
from soma.evaluation.report import EvaluationReport, SamplePrediction

__all__ = [
    "compute_classification_metrics",
    "EvaluationReport",
    "SamplePrediction",
]
