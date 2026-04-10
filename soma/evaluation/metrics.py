"""Metrics catalogue and dispatcher for all soma task families."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Metric catalogue
# ---------------------------------------------------------------------------

VALID_METRICS: dict[str, set[str]] = {
    "binary_classification": {
        "accuracy",
        "balanced_accuracy",
        "auroc",
        "auprc",
        "f1",
        "sensitivity",
        "specificity",
        "mcc",
    },
    "multiclass_classification": {
        "accuracy",
        "balanced_accuracy",
        "auroc_macro",
        "f1_macro",
        "f1_weighted",
        "mcc",
    },
    "ordinal_classification": {
        "qwk",
        "linear_wk",
        "accuracy",
        "balanced_accuracy",
        "mae",
        "spearman",
    },
    "regression": {
        "mse",
        "rmse",
        "mae",
        "r2",
        "pearson",
        "spearman",
    },
}

DEFAULT_METRICS: dict[str, list[str]] = {
    "binary_classification": ["auroc", "balanced_accuracy", "auprc", "f1"],
    "multiclass_classification": ["auroc_macro", "balanced_accuracy", "f1_macro"],
    "ordinal_classification": ["qwk", "balanced_accuracy"],
    "regression": ["mae", "r2"],
}

# ---------------------------------------------------------------------------
# Per-metric functions — signature: (y_true, y_pred, y_prob) -> float
# y_prob is None for ordinal/regression metrics
# ---------------------------------------------------------------------------

_Fn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]


def _accuracy(y_true, y_pred, y_prob) -> float:
    return float(accuracy_score(y_true, y_pred))


def _balanced_accuracy(y_true, y_pred, y_prob) -> float:
    return float(balanced_accuracy_score(y_true, y_pred))


def _auroc(y_true, y_pred, y_prob) -> float:
    try:
        value = float(roc_auc_score(y_true, y_prob[:, 1]))
    except ValueError:
        value = float("nan")
    return value if np.isfinite(value) else 0.5


def _auroc_macro(y_true, y_pred, y_prob) -> float:
    try:
        value = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        value = float("nan")
    return value if np.isfinite(value) else 0.5


def _auprc(y_true, y_pred, y_prob) -> float:
    try:
        return float(average_precision_score(y_true, y_prob[:, 1]))
    except ValueError:
        return float("nan")


def _f1_binary(y_true, y_pred, y_prob) -> float:
    return float(f1_score(y_true, y_pred, average="binary", zero_division=0))


def _f1_macro(y_true, y_pred, y_prob) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _f1_weighted(y_true, y_pred, y_prob) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def _sensitivity(y_true, y_pred, y_prob) -> float:
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def _specificity(y_true, y_pred, y_prob) -> float:
    return float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))


def _mcc(y_true, y_pred, y_prob) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def _qwk(y_true, y_pred, y_prob) -> float:
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        return 0.0


def _linear_wk(y_true, y_pred, y_prob) -> float:
    try:
        return float(cohen_kappa_score(y_true, y_pred, weights="linear"))
    except ValueError:
        return 0.0


def _mae(y_true, y_pred, y_prob) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def _spearman(y_true, y_pred, y_prob) -> float:
    result = spearmanr(y_true, y_pred)
    return float(result.statistic)


def _mse(y_true, y_pred, y_prob) -> float:
    return float(mean_squared_error(y_true, y_pred))


def _rmse(y_true, y_pred, y_prob) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _r2(y_true, y_pred, y_prob) -> float:
    return float(r2_score(y_true, y_pred))


def _pearson(y_true, y_pred, y_prob) -> float:
    result = pearsonr(y_true, y_pred)
    return float(result.statistic)


_METRIC_FUNS: dict[str, dict[str, _Fn]] = {
    "binary_classification": {
        "accuracy": _accuracy,
        "balanced_accuracy": _balanced_accuracy,
        "auroc": _auroc,
        "auprc": _auprc,
        "f1": _f1_binary,
        "sensitivity": _sensitivity,
        "specificity": _specificity,
        "mcc": _mcc,
    },
    "multiclass_classification": {
        "accuracy": _accuracy,
        "balanced_accuracy": _balanced_accuracy,
        "auroc_macro": _auroc_macro,
        "f1_macro": _f1_macro,
        "f1_weighted": _f1_weighted,
        "mcc": _mcc,
    },
    "ordinal_classification": {
        "qwk": _qwk,
        "linear_wk": _linear_wk,
        "accuracy": _accuracy,
        "balanced_accuracy": _balanced_accuracy,
        "mae": _mae,
        "spearman": _spearman,
    },
    "regression": {
        "mse": _mse,
        "rmse": _rmse,
        "mae": _mae,
        "r2": _r2,
        "pearson": _pearson,
        "spearman": _spearman,
    },
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_metrics(task_family: str, metrics: list[str]) -> list[str]:
    """Return the effective metric list for a task family.

    An empty list triggers the default set for the family. Raises ValueError
    if any name is not valid for the given task family.

    Args:
        task_family: One of the keys in VALID_METRICS.
        metrics: User-supplied metric names; empty means "use defaults".

    Returns:
        Non-empty list of validated metric names.
    """
    effective = metrics if metrics else DEFAULT_METRICS[task_family]
    invalid = set(effective) - VALID_METRICS[task_family]
    if invalid:
        raise ValueError(
            f"Invalid metrics for '{task_family}': {sorted(invalid)}. "
            f"Valid options: {sorted(VALID_METRICS[task_family])}"
        )
    return list(effective)


def compute_metrics(
    task_family: str,
    metrics: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the requested metrics for a given task family.

    Args:
        task_family: One of the keys in VALID_METRICS.
        metrics: Names of metrics to compute (must be valid for the family).
        y_true: Ground truth labels/values, shape (N,).
        y_pred: Predicted labels/values, shape (N,).
        y_prob: Predicted probabilities, shape (N, num_classes). Required for
            binary and multiclass classification families; ignored otherwise.

    Returns:
        Dict mapping each metric name to its scalar value.
    """
    funs = _METRIC_FUNS[task_family]
    return {name: funs[name](y_true, y_pred, y_prob) for name in metrics}
