"""Classification metrics for evaluation."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - preferred path when sklearn is installed
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )
except ModuleNotFoundError:  # pragma: no cover - lightweight fallback for tests
    def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

    def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(y_true)
        recalls: list[float] = []
        for label in labels:
            true_mask = y_true == label
            denom = int(true_mask.sum())
            if denom == 0:
                continue
            recalls.append(float(np.mean(y_pred[true_mask] == label)))
        return float(np.mean(recalls)) if recalls else 0.0

    def f1_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        average: str = "macro",
        zero_division: int = 0,
    ) -> float:
        del average, zero_division
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        scores: list[float] = []
        for label in labels:
            tp = float(np.sum((y_true == label) & (y_pred == label)))
            fp = float(np.sum((y_true != label) & (y_pred == label)))
            fn = float(np.sum((y_true == label) & (y_pred != label)))
            precision = tp / (tp + fp) if tp + fp > 0 else 0.0
            recall = tp / (tp + fn) if tp + fn > 0 else 0.0
            if precision + recall == 0:
                scores.append(0.0)
            else:
                scores.append(2 * precision * recall / (precision + recall))
        return float(np.mean(scores)) if scores else 0.0

    def roc_auc_score(
        y_true: np.ndarray,
        y_score: np.ndarray,
        *,
        multi_class: str | None = None,
        average: str | None = None,
    ) -> float:
        del average
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
            labels = labels.astype(int)
            positives = labels.sum()
            negatives = len(labels) - positives
            if positives == 0 or negatives == 0:
                return 0.5
            order = np.argsort(scores)
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(scores) + 1)
            sum_ranks_pos = float(ranks[labels == 1].sum())
            return (sum_ranks_pos - positives * (positives + 1) / 2.0) / (positives * negatives)

        if y_score.ndim == 1 or y_score.shape[1] == 1:
            return _binary_auc(y_true, y_score.ravel())

        if y_score.shape[1] == 2 and multi_class is None:
            return _binary_auc(y_true, y_score[:, 1])

        labels = np.unique(y_true)
        aucs: list[float] = []
        for idx, label in enumerate(labels):
            binary_true = (y_true == label).astype(int)
            if idx >= y_score.shape[1]:
                continue
            aucs.append(_binary_auc(binary_true, y_score[:, idx]))
        return float(np.mean(aucs)) if aucs else 0.5


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
