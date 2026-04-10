"""Classification metrics for evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard regression metrics.

    Args:
        y_true: Ground truth values, shape (N,).
        y_pred: Predicted values, shape (N,).

    Returns:
        Dictionary with mse, mae, r2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    return {"mse": mse, "mae": mae, "r2": r2}


def compute_ordinal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute metrics for ordinal classification (integer predictions).

    Args:
        y_true: Ground truth integer labels, shape (N,).
        y_pred: Predicted integer labels, shape (N,).

    Returns:
        Dictionary with qwk, accuracy, balanced_accuracy.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    try:
        qwk = float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    except ValueError:
        qwk = 0.0

    metrics["qwk"] = qwk
    return metrics


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_prob: Predicted probabilities, shape (N, num_classes).
        y_pred: Predicted labels, shape (N,).

    Returns:
        Dictionary with accuracy, balanced_accuracy, auc, f1_macro.
    """
    num_classes = y_prob.shape[1]

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # AUC: binary uses column 1 probability, multi-class uses OvR
    try:
        if num_classes == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            auc = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
    except ValueError:
        auc = float("nan")

    if not np.isfinite(auc):
        # Tiny debug folds can have a single class in the validation split.
        auc = 0.5

    metrics["auc"] = auc

    return metrics
