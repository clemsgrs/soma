"""Evaluation module — metrics and reports."""

from soma.evaluation.metrics import (
    DEFAULT_METRICS,
    VALID_METRICS,
    compute_metrics,
    resolve_metrics,
)
from soma.evaluation.report import EvaluationReport, SamplePrediction
from soma.evaluation.patient_oof import (
    BootstrapResult,
    ConfusionMetrics,
    OOFReport,
    PatientConfusionRecord,
    SpacingSensitivityResult,
    aggregate_patient_oof,
    aggregate_patient_oof_files,
)

__all__ = [
    "DEFAULT_METRICS",
    "VALID_METRICS",
    "compute_metrics",
    "resolve_metrics",
    "EvaluationReport",
    "SamplePrediction",
    "BootstrapResult",
    "ConfusionMetrics",
    "OOFReport",
    "PatientConfusionRecord",
    "SpacingSensitivityResult",
    "aggregate_patient_oof",
    "aggregate_patient_oof_files",
]
