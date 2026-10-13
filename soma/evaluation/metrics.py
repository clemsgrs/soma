"""Metrics catalogue and dispatcher for all soma task families."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
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
        "mean_pearson",
    },
    "survival": {
        "c_index",
    },
    "segmentation": {
        "mean_dice",
        "dataset_global_mean_dice",
        "foreground_mean_dice",
        "mean_iou",
        "dice_per_class",
    },
    "detection": {
        "mean_f1",
        "f1_per_class",
        "precision",
        "recall",
        "mean_f1_per_image",
    },
}

DEFAULT_METRICS: dict[str, list[str]] = {
    "binary_classification": ["auroc", "balanced_accuracy", "auprc", "f1"],
    "multiclass_classification": ["auroc_macro", "balanced_accuracy", "f1_macro"],
    "ordinal_classification": ["qwk", "balanced_accuracy"],
    "regression": ["mae", "r2"],
    "survival": ["c_index"],
    "segmentation": ["mean_dice", "mean_iou"],
    "detection": ["mean_f1"],
}

#: The defaults of a multi-output regression (``value_<k>`` prediction columns), whose
#: predictions the single-``value`` regression defaults cannot read.
MULTI_OUTPUT_REGRESSION_METRICS: list[str] = ["mean_pearson"]

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
    return value if np.isfinite(value) else float("nan")


def _auroc_macro(y_true, y_pred, y_prob) -> float:
    if y_prob.shape[1] == 2:
        # sklearn rejects a (n, 2) score matrix for two classes; both one-vs-rest AUROCs
        # then equal the binary AUROC of the positive column, and so does their mean.
        return _auroc(y_true, y_pred, y_prob)
    try:
        value = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        value = float("nan")
    return value if np.isfinite(value) else float("nan")


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
    if len(y_true) < 2:
        return 0.0
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
        # One output: the mean over outputs is that output's Pearson.
        "mean_pearson": _pearson,
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


def _concordance_index(event: np.ndarray, time: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C-index via scikit-survival, ``nan`` on degenerate splits.

    ``concordance_index_censored`` *raises* when a split has no events or no
    comparable pairs (e.g. an all-censored or single-sample tune split). Like
    ``_auroc`` this reports ``nan`` rather than a fake chance-level score; the
    trainer fails fast when a monitored metric is non-finite, and the CV summary
    averages over the finite folds while counting the excluded ones.
    """
    from sksurv.metrics import concordance_index_censored

    try:
        value = float(
            concordance_index_censored(event.astype(bool), time, risk)[0]
        )
    except (ValueError, ZeroDivisionError):
        value = float("nan")
    return value if np.isfinite(value) else float("nan")


def compute_survival_metrics(
    metrics: list[str],
    event: np.ndarray,
    time: np.ndarray,
    risk: np.ndarray,
) -> dict[str, float]:
    """Compute survival metrics from event indicator, event time, and risk score.

    The survival family uses an ``(event, time, risk)`` signature rather than the
    generic ``(y_true, y_pred, y_prob)`` triple, since concordance needs
    censoring information. Higher ``risk`` must mean earlier predicted event.

    Args:
        metrics: Metric names to compute (must be valid for the survival family).
        event: Event indicator, 1 = event observed, 0 = censored, shape (N,).
        time: Event or last-follow-up time, shape (N,).
        risk: Predicted risk score, shape (N,).

    Returns:
        Dict mapping each metric name to its scalar value.
    """
    event = np.asarray(event)
    time = np.asarray(time)
    risk = np.asarray(risk)
    results: dict[str, float] = {}
    for name in metrics:
        if name == "c_index":
            results[name] = _concordance_index(event, time, risk)
        else:
            raise ValueError(f"Unknown survival metric: {name!r}")
    return results


def probability_columns(columns) -> list[str]:
    """Return ``prob_*`` columns sorted by numeric class index."""
    prob_cols = [c for c in columns if str(c).startswith("prob_")]

    def _key(name: str) -> tuple[int, int | str]:
        suffix = str(name).removeprefix("prob_")
        try:
            return (0, int(suffix))
        except ValueError:
            return (1, suffix)

    return sorted(prob_cols, key=_key)


# ---------------------------------------------------------------------------
# Metric registry — frame metrics ``(predictions, targets) -> float``
# ---------------------------------------------------------------------------

FrameMetric = Callable[[pd.DataFrame, pd.DataFrame], float]


@dataclass(frozen=True)
class MetricSpec:
    """One registered metric.

    For the set families ``fn`` scores the aligned frames :func:`soma.evaluate` builds:
    the predictions and the targets, indexed by ``sample_id`` in the same order, with
    ``label`` already mapped to class indices. For the grid families (``segmentation``,
    ``detection``) ``fn`` reduces per-image ``(N, C, 3)`` counts — ``(intersection,
    predicted, target)`` pixels or ``(tp, fp, fn)`` points — as
    ``fn(counts, *, num_classes) -> float | dict[str, float]`` (a dict expands into one
    key per class). It is ``None`` for metrics another path computes; those entries only
    declare a direction.
    """

    name: str
    fn: FrameMetric | None
    families: frozenset[str]
    higher_is_better: bool = True


class MetricRegistry:
    """Name → :class:`MetricSpec`; the single source of a metric's family and direction."""

    def __init__(self) -> None:
        self._entries: dict[str, MetricSpec] = {}

    def register(
        self,
        name: str,
        fn: FrameMetric | None = None,
        *,
        families,
        higher_is_better: bool = True,
    ) -> MetricSpec:
        if name in self._entries:
            raise ValueError(f"metric '{name}' is already registered")
        spec = MetricSpec(name, fn, frozenset(families), bool(higher_is_better))
        self._entries[name] = spec
        return spec

    def get(self, name: str) -> MetricSpec:
        try:
            return self._entries[name]
        except KeyError:
            raise KeyError(
                f"metric '{name}' is not registered; available: {sorted(self._entries)}"
            ) from None

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def names(self, task_family: str | None = None) -> list[str]:
        """Registered names, optionally only those declaring ``task_family``."""
        return [
            name
            for name, spec in self._entries.items()
            if task_family is None or task_family in spec.families
        ]

    def higher_is_better(self, metric: str) -> bool:
        """Direction of ``metric`` (a ``split/`` prefix is ignored); unregistered names,
        such as per-class keys or a benchmark's own scores, count as higher-is-better."""
        name = metric.rsplit("/", 1)[-1]
        spec = self._entries.get(name)
        return True if spec is None else spec.higher_is_better


def _label_arrays(
    predictions: pd.DataFrame, targets: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """``(y_true, y_pred, y_prob)`` for the array metrics above: ``label`` against
    ``predicted_label`` for the label families, ``value`` against ``predicted_value``."""
    if "predicted_value" in predictions.columns:
        y_true = targets["value"].to_numpy(dtype=float)
        y_pred = predictions["predicted_value"].to_numpy(dtype=float)
    else:
        y_true = targets["label"].to_numpy(dtype=np.int64)
        y_pred = predictions["predicted_label"].to_numpy(dtype=np.int64)
    prob_cols = probability_columns(predictions.columns)
    y_prob = predictions[prob_cols].to_numpy(dtype=float) if prob_cols else None
    return y_true, y_pred, y_prob


def _c_index(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    return _concordance_index(
        targets["event"].to_numpy(dtype=float),
        targets["time"].to_numpy(dtype=float),
        predictions["risk_score"].to_numpy(dtype=float),
    )


def _frame_metric(fn: _Fn) -> FrameMetric:
    return lambda predictions, targets: fn(*_label_arrays(predictions, targets))


def _output_columns(predictions: pd.DataFrame) -> list[str]:
    """``value_<k>`` columns of a multi-output regression, in output order."""
    columns = [c for c in predictions.columns if str(c).startswith("value_")]
    return sorted(columns, key=lambda c: int(str(c).removeprefix("value_")))


def _mean_pearson(predictions: pd.DataFrame, targets: pd.DataFrame) -> float:
    """Mean over outputs of the Pearson correlation pooled over samples.

    A multi-output regression (``value_<k>`` prediction columns, matching target
    columns) averages one correlation per output; a single output is its Pearson. A
    constant column contributes ``0.0``.
    """
    columns = _output_columns(predictions)
    if not columns:
        return _pearson(
            targets["value"].to_numpy(dtype=float), predictions["predicted_value"].to_numpy(dtype=float), None
        )
    return float(
        np.mean(
            [
                _pearson(targets[c].to_numpy(dtype=float), predictions[c].to_numpy(dtype=float), None)
                for c in columns
            ]
        )
    )


# --- grid reducers: per-image (N, C, 3) counts -> scalar(s) ------------------------


def _dice_iou(counts, num_classes: int, aggregation: str = "per_image_macro") -> dict[str, float]:
    from soma.models.heads.dense_metrics import reduce_dice_iou

    return reduce_dice_iou(counts, num_classes=num_classes, aggregation=aggregation)


def _f1(counts, num_classes: int, aggregation: str = "dataset_global") -> dict[str, float]:
    from soma.evaluation.detection.matching import reduce_f1

    return reduce_f1(counts, num_classes=num_classes, aggregation=aggregation)


def _foreground_dice(counts, *, num_classes: int) -> float:
    from soma.models.heads.dense_metrics import reduce_foreground_dice

    return reduce_foreground_dice(counts, num_classes=num_classes)


def _per_class(prefix: str, reduce):
    def per_class(counts, *, num_classes: int) -> dict[str, float]:
        values = reduce(counts, num_classes)
        return {f"{prefix}_class_{c}": values[f"{prefix}_class_{c}"] for c in range(num_classes)}

    return per_class


_GRID_REDUCERS: dict[str, Callable] = {
    "mean_dice": lambda counts, *, num_classes: _dice_iou(counts, num_classes)["mean_dice"],
    "dataset_global_mean_dice": lambda counts, *, num_classes: _dice_iou(
        counts, num_classes, "dataset_global"
    )["mean_dice"],
    "foreground_mean_dice": _foreground_dice,
    "mean_iou": lambda counts, *, num_classes: _dice_iou(counts, num_classes)["mean_iou"],
    "dice_per_class": _per_class("dice", _dice_iou),
    "mean_f1": lambda counts, *, num_classes: _f1(counts, num_classes)["mean_f1"],
    "f1_per_class": _per_class("f1", _f1),
    "precision": _per_class("precision", _f1),
    "recall": _per_class("recall", _f1),
    "mean_f1_per_image": lambda counts, *, num_classes: _f1(
        counts, num_classes, "per_image_macro"
    )["mean_f1"],
}


metric_registry = MetricRegistry()
_LOWER_IS_BETTER = {"mse", "rmse", "mae"}
for _name in dict.fromkeys(name for names in VALID_METRICS.values() for name in sorted(names)):
    _families = {family for family, names in VALID_METRICS.items() if _name in names}
    _array_fn = next(
        (_METRIC_FUNS[f][_name] for f in sorted(_families) if _name in _METRIC_FUNS.get(f, {})),
        None,
    )
    if _name == "mean_pearson":
        _fn = _mean_pearson
    elif _array_fn is not None:
        _fn = _frame_metric(_array_fn)
    elif _name == "c_index":
        _fn = _c_index
    else:
        _fn = _GRID_REDUCERS.get(_name)
    metric_registry.register(
        _name, _fn, families=_families, higher_is_better=_name not in _LOWER_IS_BETTER
    )
# The representation-quality path's CRoMa score (soma.benchmark.croma.robustness): direction only.
metric_registry.register("croma_f0", families={"representation"}, higher_is_better=False)
