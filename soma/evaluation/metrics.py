"""Classification metrics for evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


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
    if num_classes == 2:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))
    else:
        metrics["auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )

    return metrics
