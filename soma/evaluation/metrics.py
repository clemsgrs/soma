"""Metrics catalogue and dispatcher for all soma task families."""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, pearsonr, spearmanr
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
        "qwk",
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
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        value = float(average_precision_score(y_true, y_prob[:, 1]))
    except ValueError:
        value = float("nan")
    return value if np.isfinite(value) else float("nan")


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


def _finite_correlation(statistic: float) -> float:
    value = float(statistic)
    return value if np.isfinite(value) else 0.0


def _spearman(y_true, y_pred, y_prob) -> float:
    result = spearmanr(y_true, y_pred)
    return _finite_correlation(result.statistic)


def _mse(y_true, y_pred, y_prob) -> float:
    return float(mean_squared_error(y_true, y_pred))


def _rmse(y_true, y_pred, y_prob) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _r2(y_true, y_pred, y_prob) -> float:
    return float(r2_score(y_true, y_pred))


def _pearson(y_true, y_pred, y_prob) -> float:
    result = pearsonr(y_true, y_pred)
    return _finite_correlation(result.statistic)


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
        "qwk": _qwk,
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


def bh_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction via scipy.stats.false_discovery_control.

    Returns adjusted p-values in the same order as the input.
    An empty list returns an empty list; a single p-value is returned unchanged.
    """
    if len(p_values) == 0:
        return []
    return list(false_discovery_control(p_values, method="bh"))


def compare_run_predictions(
    runs_predictions: list[pd.DataFrame],
    task_family: str,
    metric: str,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
) -> list[float | None]:
    """Sample-level paired permutation test comparing each run against the best.

    Aligns runs by sample_id, then for each permutation randomly swaps
    prediction columns between the best run and one other run for a random
    subset of samples, recomputes the metric, and builds a null distribution.
    Works for any number of folds (including single-fold) as long as
    predictions exist and share sample IDs.

    Args:
        runs_predictions: Pooled prediction DataFrames, one per run.
            Each must contain sample_id, true_label, and prediction columns.
        task_family: Task family string (e.g. 'binary_classification').
        metric: Metric name to use as the test statistic.
        n_permutations: Number of permutation iterations.
        seed: Random seed for reproducibility.

    Returns:
        List of p-values, one per run. The best run gets p-value 1.0.
        Returns None for a run when comparison is not possible.
    """
    n_runs = len(runs_predictions)
    if n_runs < 2:
        return [None] * n_runs

    funs = _METRIC_FUNS.get(task_family, {})
    if metric not in funs:
        return [None] * n_runs

    required_pred_col = "predicted_value" if task_family == "regression" else "predicted_label"
    missing_required = [i for i, df in enumerate(runs_predictions) if required_pred_col not in df.columns]
    if missing_required:
        raise ValueError(
            f"compare_run_predictions requires '{required_pred_col}' for task_family='{task_family}'. "
            f"Missing in runs: {missing_required}."
        )

    def _score(df: pd.DataFrame) -> float | None:
        if df.empty or len(df) < 2:
            return None
        try:
            y_true, y_pred, y_prob = _extract_arrays(df, task_family)
            return float(funs[metric](y_true, y_pred, y_prob))
        except Exception:
            logger.debug("Metric %s could not be computed for a run", metric, exc_info=True)
            return None

    scores = [_score(df) for df in runs_predictions]
    valid = [(i, s) for i, s in enumerate(scores) if s is not None]
    if not valid:
        return [None] * n_runs

    best_idx, best_score = max(valid, key=lambda x: x[1])
    rng = np.random.default_rng(seed)

    def _prediction_cols(df: pd.DataFrame) -> list[str]:
        return [
            c for c in df.columns
            if c in {required_pred_col, "raw_score"} or c.startswith("prob_")
        ]

    metric_needs_probabilities = task_family in {"binary_classification", "multiclass_classification"} and metric in {
        "auroc",
        "auprc",
        "auroc_macro",
    }

    pred_cols = _prediction_cols(runs_predictions[best_idx])
    prob_cols = [c for c in pred_cols if c.startswith("prob_")]

    best_df = runs_predictions[best_idx]
    best_ids = set(best_df["sample_id"])

    p_values: list[float | None] = []
    for i, df in enumerate(runs_predictions):
        if i == best_idx:
            p_values.append(1.0)
            continue
        if scores[i] is None:
            p_values.append(None)
            continue

        # Align both runs to a common sorted sample order
        common_ids = sorted(best_ids.intersection(df["sample_id"]))
        if len(common_ids) < 2:
            p_values.append(None)
            continue

        df_a = best_df[best_df["sample_id"].isin(common_ids)].sort_values("sample_id")
        df_b = df[df["sample_id"].isin(common_ids)].sort_values("sample_id")

        y_true = df_a["true_label"].to_numpy()
        shared_pred_cols = [c for c in pred_cols if c in df_b.columns]
        if not shared_pred_cols:
            p_values.append(None)
            continue
        shared_prob_cols = [c for c in prob_cols if c in shared_pred_cols]
        if metric_needs_probabilities and len(shared_prob_cols) != len(prob_cols):
            p_values.append(None)
            continue
        vals_a = df_a[shared_pred_cols].to_numpy()
        vals_b = df_b[shared_pred_cols].to_numpy()

        observed_delta = abs(best_score - scores[i])

        def _df_from(vals: np.ndarray, cols: list[str] = shared_pred_cols, yt: np.ndarray = y_true) -> pd.DataFrame:
            out = pd.DataFrame(vals, columns=cols)
            out["true_label"] = yt
            return out

        null_deltas = np.empty(n_permutations)
        for p_idx in range(n_permutations):
            swap = rng.random(len(common_ids)) < 0.5
            perm_a = np.where(swap[:, None], vals_b, vals_a)
            perm_b = np.where(swap[:, None], vals_a, vals_b)
            ya, ypa, yproba = _extract_arrays(_df_from(perm_a), task_family)
            yb, ypb, yprobab = _extract_arrays(_df_from(perm_b), task_family)
            null_deltas[p_idx] = abs(
                float(funs[metric](ya, ypa, yproba)) - float(funs[metric](yb, ypb, yprobab))
            )

        p_values.append(float(np.mean(null_deltas >= observed_delta)))

    return p_values


def _extract_arrays(
    df: pd.DataFrame,
    task_family: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract y_true, y_pred, y_prob from a predictions DataFrame.

    Supports the column formats written by pipeline._save_predictions:
    - Classification: true_label, predicted_label, prob_0, prob_1, ...
    - Ordinal: true_label, predicted_label, raw_score
    - Regression: true_label, predicted_value
    """
    y_true = df["true_label"].to_numpy()

    prob_cols = sorted(c for c in df.columns if c.startswith("prob_"))
    if prob_cols:
        y_pred = df["predicted_label"].to_numpy()
        y_prob = df[prob_cols].to_numpy()
    elif "predicted_label" in df.columns:
        y_pred = df["predicted_label"].to_numpy()
        y_prob = None
    else:
        y_pred = df["predicted_value"].to_numpy()
        y_prob = None

    return y_true, y_pred, y_prob


def compute_subgroup_metrics(
    task_family: str,
    metrics: list[str],
    predictions_df: pd.DataFrame,
    subgroup_columns: list[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Compute metrics per subgroup for each categorical column.

    For each column, groups predictions by unique values and computes all
    requested metrics within each group. Groups with fewer than 2 samples
    are skipped.

    Args:
        task_family: Task family (e.g. 'binary_classification').
        metrics: Metric names to compute (must be valid for the family).
        predictions_df: DataFrame with prediction columns + subgroup columns.
        subgroup_columns: Column names to group by.

    Returns:
        {column → {group_value → {metric: value, "n": count}}}
    """
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for col in subgroup_columns:
        if col not in predictions_df.columns:
            continue
        groups: dict[str, dict[str, float | int]] = {}
        for group_val, group_df in predictions_df.groupby(col, sort=True):
            if len(group_df) < 2:
                continue
            y_true, y_pred, y_prob = _extract_arrays(group_df, task_family)
            try:
                group_metrics = compute_metrics(task_family, metrics, y_true, y_pred, y_prob)
            except Exception:
                logger.debug("Subgroup metrics failed for %s=%s", col, group_val, exc_info=True)
                continue
            group_metrics["n"] = int(len(group_df))
            groups[str(group_val)] = group_metrics
        result[col] = groups
    return result


def compute_subgroup_stats(
    task_family: str,
    metrics: list[str],
    predictions_df: pd.DataFrame,
    subgroup_columns: list[str],
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    min_group_size: int = 10,
) -> dict[str, dict[str, dict[str, float]]]:
    """Permutation test for each subgroup vs. the rest of the data.

    For each (column, group, metric): computes the observed metric delta
    (group minus rest), then permutes group membership `n_permutations` times
    to estimate a two-sided p-value. Groups with fewer than `min_group_size`
    samples are skipped.

    Args:
        task_family: Task family (e.g. 'binary_classification').
        metrics: Metric names to test.
        predictions_df: DataFrame with prediction columns + subgroup columns.
        subgroup_columns: Column names to test.
        n_permutations: Number of permutation iterations.
        seed: Random seed for reproducibility.
        min_group_size: Minimum group size required to run the test.

    Returns:
        {column → {group_value → {metric: p_value}}}
        Only groups with n >= min_group_size are included.
    """
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, dict[str, float]]] = {}

    for col in subgroup_columns:
        if col not in predictions_df.columns:
            continue
        col_stats: dict[str, dict[str, float]] = {}
        for group_val, group_df in predictions_df.groupby(col, sort=True):
            if len(group_df) < min_group_size:
                continue
            group_mask = predictions_df[col] == group_val
            rest_df = predictions_df[~group_mask]
            if len(rest_df) < 2:
                continue

            metric_pvals: dict[str, float] = {}
            for metric in metrics:
                try:
                    y_true_g, y_pred_g, y_prob_g = _extract_arrays(group_df, task_family)
                    y_true_r, y_pred_r, y_prob_r = _extract_arrays(rest_df, task_family)
                    val_group = compute_metrics(task_family, [metric], y_true_g, y_pred_g, y_prob_g)[metric]
                    val_rest = compute_metrics(task_family, [metric], y_true_r, y_pred_r, y_prob_r)[metric]
                    observed_delta = abs(val_group - val_rest)
                except Exception:
                    logger.debug("Permutation test: could not compute %s for group %s", metric, group_val, exc_info=True)
                    continue

                # Permutation test: shuffle group membership, recompute delta
                n_group = int(group_mask.sum())
                n_total = len(predictions_df)
                null_deltas = []
                for _ in range(n_permutations):
                    perm_mask = np.zeros(n_total, dtype=bool)
                    perm_mask[rng.choice(n_total, size=n_group, replace=False)] = True
                    perm_group = predictions_df[perm_mask]
                    perm_rest = predictions_df[~perm_mask]
                    try:
                        yg_t, yg_p, yg_pr = _extract_arrays(perm_group, task_family)
                        yr_t, yr_p, yr_pr = _extract_arrays(perm_rest, task_family)
                        v_g = compute_metrics(task_family, [metric], yg_t, yg_p, yg_pr)[metric]
                        v_r = compute_metrics(task_family, [metric], yr_t, yr_p, yr_pr)[metric]
                        null_deltas.append(abs(v_g - v_r))
                    except Exception:
                        continue

                if null_deltas:
                    metric_pvals[metric] = float(np.mean(np.array(null_deltas) >= observed_delta))

            if metric_pvals:
                col_stats[str(group_val)] = metric_pvals
        result[col] = col_stats
    return result
