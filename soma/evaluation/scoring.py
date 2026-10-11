"""``evaluate``: score exactly the predictions it is given against the cohort's targets."""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soma.evaluation.metrics import DEFAULT_METRICS, metric_registry
from soma.evaluation.predictions import (
    PLACEHOLDER_COLUMN,
    Predictions,
    encode_labels,
    family_target_keys,
)
from soma.evaluation.subgroups import subgroup_breakdown

__all__ = ["Evaluation", "compare", "evaluate", "resolve_frame_metrics"]

logger = logging.getLogger(__name__)

MetricLike = str | Callable[[pd.DataFrame, pd.DataFrame], float]
_LABEL_FAMILIES = frozenset(
    {"binary_classification", "multiclass_classification", "ordinal_classification"}
)


@dataclass(frozen=True)
class Evaluation:
    """The scores of one set of predictions.

    ``metrics`` maps each metric to its value (plus ``coverage`` and the sample counts
    when the predictions carry placeholders); ``subgroups`` holds the per-column,
    per-group metrics and permutation p-values (``{"metrics": ..., "stats": ...}``, empty
    without subgroup columns).
    """

    metrics: dict[str, float]
    task_family: str
    num_samples: int
    subgroups: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_family": self.task_family,
            "num_samples": self.num_samples,
            "metrics": dict(self.metrics),
            "subgroups": dict(self.subgroups),
        }

    def to_json(self, path: str | Path | None = None) -> str:
        """The evaluation as JSON text; also written to ``path`` when given."""
        text = json.dumps(self.to_dict(), indent=2) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def resolve_frame_metrics(
    metrics: str | Sequence[MetricLike] | None, task_family: str
) -> dict[str, Callable[[pd.DataFrame, pd.DataFrame], float]]:
    """Name → frame metric for ``metrics`` under ``task_family``.

    ``"default"`` (or ``None`` / an empty list) is the family's catalogue defaults. Each
    entry is a registry key, a dotted import path to a callable, or a callable; a
    registry key outside ``task_family`` is refused.
    """
    if metrics is None or (isinstance(metrics, str) and metrics == "default") or (
        not isinstance(metrics, str) and len(metrics) == 0
    ):
        if task_family not in DEFAULT_METRICS:
            raise ValueError(f"no default metrics for task family {task_family!r}")
        metrics = DEFAULT_METRICS[task_family]
    elif isinstance(metrics, str):
        metrics = [metrics]
    resolved: dict[str, Callable[[pd.DataFrame, pd.DataFrame], float]] = {}
    for entry in metrics:
        if callable(entry):
            resolved[getattr(entry, "__name__", repr(entry))] = entry
            continue
        name = str(entry)
        if name in metric_registry:
            spec = metric_registry.get(name)
            if task_family not in spec.families:
                valid = sorted(n for n in metric_registry.names(task_family) if metric_registry.get(n).fn)
                raise ValueError(
                    f"metric '{name}' does not apply to {task_family!r} predictions; "
                    f"valid metrics: {valid}"
                )
            if spec.fn is None:
                raise ValueError(f"metric '{name}' has no frame implementation to evaluate")
            resolved[name] = spec.fn
            continue
        if "." in name:
            module_name, _, attr = name.rpartition(".")
            try:
                fn = getattr(importlib.import_module(module_name), attr)
            except (ImportError, AttributeError) as error:
                raise ValueError(f"cannot import metric '{name}': {error}") from error
            if not callable(fn):
                raise ValueError(f"'{name}' does not name a callable metric")
            resolved[attr] = fn
            continue
        valid = sorted(n for n in metric_registry.names(task_family) if metric_registry.get(n).fn)
        raise ValueError(
            f"unknown metric '{name}': not a registry key, a dotted path or a callable; "
            f"valid metrics for {task_family!r}: {valid}"
        )
    return resolved


def _targets_by_id(targets: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" in targets.columns:
        targets = targets.set_index("sample_id")
    targets = targets.copy()
    targets.index = targets.index.astype(str)
    targets.index.name = "sample_id"
    return targets


def _ids(values, limit: int = 10) -> str:
    values = sorted(values)
    return f"{values[:limit]}{' ...' if len(values) > limit else ''}"


def _align(predictions: Predictions, targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join predictions and targets by ``sample_id`` under strict set equality.

    Returns both frames indexed by ``sample_id`` in the targets' order, with ``label``
    mapped onto class indices for the label families.
    """
    frame = predictions.frame
    targets = _targets_by_id(targets)
    duplicated = frame["sample_id"][frame["sample_id"].duplicated()]
    if len(duplicated):
        raise ValueError(
            f"{duplicated.nunique()} sample id(s) appear more than once in the predictions: "
            f"{_ids(set(duplicated))}"
        )
    duplicated = targets.index[targets.index.duplicated()]
    if len(duplicated):
        raise ValueError(
            f"{len(set(duplicated))} sample id(s) appear more than once in the targets: "
            f"{_ids(set(duplicated))}"
        )
    predicted, expected = set(frame["sample_id"]), set(targets.index)
    unknown = predicted - expected
    if unknown:
        raise ValueError(
            f"{len(unknown)} prediction id(s) have no target: {_ids(unknown)}. evaluate scores "
            "exactly the ids it is given; filter the predictions or pass the targets of "
            "every predicted sample."
        )
    missing = expected - predicted
    if missing:
        raise ValueError(
            f"{len(missing)} target(s) have no prediction: {_ids(missing)}. Fill them with "
            "soma.evaluation.Baseline (placeholder predictions) or filter the targets to the "
            "predicted ids."
        )
    try:
        keys = family_target_keys(predictions.task_family)
    except ValueError:  # a family soma does not know: its metrics read what they need
        keys = ()
    if len(targets):
        required = [key for key in keys if key not in targets.columns]
        if required:
            raise ValueError(
                f"{predictions.task_family} predictions need target column(s) {required}; "
                f"the targets have {list(targets.columns)}"
            )
    frame = frame.set_index("sample_id").loc[targets.index]
    if predictions.task_family in _LABEL_FAMILIES and "label" in targets.columns and len(targets):
        targets = targets.assign(label=encode_labels(targets["label"], predictions.classes))
    return frame, targets


def evaluate(
    predictions: Predictions,
    targets: pd.DataFrame,
    metrics: str | Sequence[MetricLike] | None = "default",
    subgroups: Sequence[str] = (),
) -> "Evaluation":
    """Score ``predictions`` against ``targets`` (e.g. :meth:`soma.data.Cohort.targets`).

    The two are joined by ``sample_id`` and must cover the same ids: an unknown
    prediction id, a target without a prediction, or a duplicate id is refused. Fill
    samples the model could not score with :class:`~soma.evaluation.Baseline` first.

    Args:
        predictions: What the model said.
        targets: One row per sample, indexed by ``sample_id`` (or with a ``sample_id``
            column), holding the target columns of the predictions' family and any
            metadata columns ``subgroups`` names.
        metrics: ``"default"`` (the family's catalogue defaults) or a list of registry
            keys, dotted paths and ``(predictions, targets) -> float`` callables.
        subgroups: Target columns to break the metrics down by.
    """
    family = predictions.task_family
    resolved = resolve_frame_metrics(metrics, family)
    missing = [column for column in subgroups if column not in _targets_by_id(targets).columns]
    if missing:
        raise ValueError(f"subgroup column(s) {missing} are not target columns")
    frame, aligned = _align(predictions, targets)
    values: dict[str, float | int] = {}
    if len(frame):
        values = {name: float(fn(frame, aligned)) for name, fn in resolved.items()}
    if predictions.has_placeholders:
        placeholder = int(frame[PLACEHOLDER_COLUMN].sum())
        total = len(frame)
        values["coverage"] = float((total - placeholder) / total) if total else 0.0
        values["num_samples"] = total
        values["num_real_samples"] = total - placeholder
        values["num_placeholder_samples"] = placeholder
    breakdown = subgroup_breakdown(frame, aligned, resolved, list(subgroups)) if subgroups else {}
    return Evaluation(metrics=values, task_family=family, num_samples=len(frame), subgroups=breakdown)


def compare(
    predictions_list: Sequence[Predictions],
    targets: pd.DataFrame,
    metric: MetricLike,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
) -> list[float | None]:
    """Paired permutation test of each prediction set against the best one.

    Every set is aligned to ``targets`` as in :func:`evaluate`; sets declaring different
    ``classes`` (including a set without ``classes`` next to one with) are refused. For each permutation the prediction columns of the best set
    and another are swapped on a random half of the samples and the metric delta
    recomputed. The best set gets ``1.0``; a set whose metric cannot be computed, or that
    shares no prediction column with the best, gets ``None``.
    """
    sets = list(predictions_list)
    if len(sets) < 2:
        return [None] * len(sets)
    families = {p.task_family for p in sets}
    if len(families) != 1:
        raise ValueError(f"compare needs predictions of one task family, got {sorted(families)}")
    vocabularies = list(dict.fromkeys(p.classes for p in sets))
    if len(vocabularies) > 1:
        listed = [None if v is None else list(v) for v in vocabularies]
        raise ValueError(f"compare needs prediction sets with the same classes, got {listed}")
    (name, fn), = resolve_frame_metrics([metric], sets[0].task_family).items()
    higher = metric_registry.higher_is_better(name)
    aligned = [_align(p, targets) for p in sets]

    def score(frame: pd.DataFrame, truth: pd.DataFrame) -> float | None:
        if len(frame) < 2:
            return None
        try:
            return float(fn(frame, truth))
        except Exception:
            logger.debug("Metric %s could not be computed for a prediction set", name, exc_info=True)
            return None

    scores = [score(frame, truth) for frame, truth in aligned]
    valid = [(i, s) for i, s in enumerate(scores) if s is not None and np.isfinite(s)]
    if not valid:
        return [None] * len(sets)
    best_index, best_score = (max if higher else min)(valid, key=lambda item: item[1])
    best_frame, truth = aligned[best_index]
    columns = list(sets[best_index].prediction_columns)
    rng = np.random.default_rng(seed)
    p_values: list[float | None] = []
    for i, (frame, _) in enumerate(aligned):
        if i == best_index:
            p_values.append(1.0)
            continue
        if scores[i] is None or not np.isfinite(scores[i]):
            p_values.append(None)
            continue
        shared = [c for c in columns if c in frame.columns]
        if shared != columns:
            p_values.append(None)
            continue
        a, b = best_frame[shared].to_numpy(), frame[shared].to_numpy()
        observed = abs(best_score - scores[i])
        null = np.empty(n_permutations)
        for k in range(n_permutations):
            swap = rng.random(len(truth)) < 0.5
            left = pd.DataFrame(np.where(swap[:, None], b, a), columns=shared, index=truth.index)
            right = pd.DataFrame(np.where(swap[:, None], a, b), columns=shared, index=truth.index)
            null[k] = abs(float(fn(left, truth)) - float(fn(right, truth)))
        p_values.append(float(np.mean(null >= observed)))
    return p_values
