"""Evaluation: ``predict`` → ``Predictions`` → ``evaluate`` → ``Evaluation``.

Inference and scoring are two verbs with one noun between them. Targets never travel
inside predictions; ``evaluate`` joins them from the cohort by ``sample_id``.
"""

from soma.evaluation.baseline import Baseline
from soma.evaluation.confusion_evidence import (
    ConfusionMetrics,
    SegmentationConfusionRecord,
    aggregate_confusion_matrices,
    aggregate_confusion_records,
    confusion_dice_from_matrices,
    confusion_records,
    load_confusion_records,
    validate_confusion_records,
    write_confusion_records,
)
from soma.evaluation.grid import LabelMapPredictions, PointPredictions, sweep_threshold
from soma.evaluation.grid_callbacks import SaveHeatmaps, SaveOverlays, SaveProbabilities
from soma.evaluation.inference import predict
from soma.evaluation.metrics import (
    DEFAULT_METRICS,
    VALID_METRICS,
    MetricSpec,
    compute_metrics,
    metric_registry,
    resolve_metrics,
)
from soma.evaluation.predictions import Predictions
from soma.evaluation.scoring import Evaluation, compare, evaluate

__all__ = [
    "Baseline",
    "DEFAULT_METRICS",
    "Evaluation",
    "LabelMapPredictions",
    "MetricSpec",
    "PointPredictions",
    "Predictions",
    "SaveHeatmaps",
    "SaveOverlays",
    "SaveProbabilities",
    "VALID_METRICS",
    "compare",
    "compute_metrics",
    "evaluate",
    "metric_registry",
    "predict",
    "resolve_metrics",
    "sweep_threshold",
    "ConfusionMetrics",
    "SegmentationConfusionRecord",
    "aggregate_confusion_matrices",
    "aggregate_confusion_records",
    "confusion_dice_from_matrices",
    "confusion_records",
    "load_confusion_records",
    "validate_confusion_records",
    "write_confusion_records",
]
