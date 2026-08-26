"""Evaluation module — metrics and reports."""

from soma.evaluation.metrics import (
    DEFAULT_METRICS,
    VALID_METRICS,
    compute_metrics,
    resolve_metrics,
)
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.evaluation.confusion_evidence import (
    ConfusionMetrics,
    SegmentationConfusionRecord,
    aggregate_confusion_matrices,
    aggregate_confusion_records,
    load_confusion_records,
    validate_confusion_records,
    write_confusion_records,
)

__all__ = [
    "DEFAULT_METRICS",
    "VALID_METRICS",
    "compute_metrics",
    "resolve_metrics",
    "EvaluationReport",
    "SamplePrediction",
    "ConfusionMetrics",
    "SegmentationConfusionRecord",
    "aggregate_confusion_matrices",
    "aggregate_confusion_records",
    "load_confusion_records",
    "validate_confusion_records",
    "write_confusion_records",
]
