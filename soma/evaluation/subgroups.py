"""Subgroup breakdown: the same metrics over each group of a target metadata column.

:func:`soma.evaluate` joins predictions with the cohort's targets frame, which carries
every metadata column, so a subgroup is just a ``groupby`` over that joined frame.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["is_subgroup_key", "subgroup_breakdown", "subgroup_metrics", "subgroup_stats"]

logger = logging.getLogger(__name__)

Metric = Callable[[pd.DataFrame, pd.DataFrame], float]


def subgroup_metrics(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    metrics: Mapping[str, Metric],
    columns: Sequence[str],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """``{column: {group: {metric: value, "n": count}}}``; groups under 2 samples are
    skipped, and so is a group whose metrics cannot be computed."""
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for column in columns:
        groups: dict[str, dict[str, float | int]] = {}
        for value, index in targets.groupby(column, sort=True).groups.items():
            if len(index) < 2:
                continue
            try:
                values: dict[str, float | int] = {
                    name: float(fn(predictions.loc[index], targets.loc[index]))
                    for name, fn in metrics.items()
                }
            except Exception:
                logger.debug("Subgroup metrics failed for %s=%s", column, value, exc_info=True)
                continue
            values["n"] = int(len(index))
            groups[str(value)] = values
        result[column] = groups
    return result


def subgroup_stats(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    metrics: Mapping[str, Metric],
    columns: Sequence[str],
    *,
    n_permutations: int = 1000,
    seed: int = 42,
    min_group_size: int = 10,
) -> dict[str, dict[str, dict[str, float]]]:
    """Two-sided permutation p-value of each group's metric against the rest.

    ``{column: {group: {metric: p_value}}}``: the observed ``|group - rest|`` delta
    against the deltas of ``n_permutations`` random memberships of the same size. Groups
    under ``min_group_size`` samples, or whose rest has under 2, are skipped.
    """
    rng = np.random.default_rng(seed)
    total = len(targets)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for column in columns:
        column_stats: dict[str, dict[str, float]] = {}
        for value, index in targets.groupby(column, sort=True).groups.items():
            if len(index) < min_group_size:
                continue
            in_group = (targets[column] == value).to_numpy()
            if total - int(in_group.sum()) < 2:
                continue
            p_values: dict[str, float] = {}
            for name, fn in metrics.items():
                try:
                    observed = abs(_delta(fn, predictions, targets, in_group))
                except Exception:
                    logger.debug("Permutation test: could not compute %s for %s", name, value, exc_info=True)
                    continue
                null: list[float] = []
                for _ in range(n_permutations):
                    permuted = np.zeros(total, dtype=bool)
                    permuted[rng.choice(total, size=int(in_group.sum()), replace=False)] = True
                    try:
                        null.append(abs(_delta(fn, predictions, targets, permuted)))
                    except Exception:
                        continue
                if null:
                    p_values[name] = float(np.mean(np.asarray(null) >= observed))
            if p_values:
                column_stats[str(value)] = p_values
        result[column] = column_stats
    return result


def _delta(fn: Metric, predictions: pd.DataFrame, targets: pd.DataFrame, mask: np.ndarray) -> float:
    group = float(fn(predictions[mask], targets[mask]))
    rest = float(fn(predictions[~mask], targets[~mask]))
    return group - rest


def _mean_over(
    breakdown: Mapping[str, Mapping[str, Mapping[str, float | int]]],
    metrics: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """``{column: {metric: mean, "<metric>_groups": count}}`` over a
    :func:`subgroup_metrics` breakdown: the unweighted mean of each metric over the
    column's groups.

    A group whose value is NaN (a single class, no comparable pairs) is skipped, and
    ``<metric>_groups`` counts the groups that contributed. With no finite group the
    mean is NaN.
    """
    result: dict[str, dict[str, float | int]] = {}
    for column, groups in breakdown.items():
        means: dict[str, float | int] = {}
        for name in metrics:
            values = [float(group[name]) for group in groups.values() if name in group]
            finite = [value for value in values if np.isfinite(value)]
            means[name] = float(np.mean(finite)) if finite else float("nan")
            means[f"{name}_groups"] = len(finite)
        result[column] = means
    return result


def subgroup_breakdown(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    metrics: Mapping[str, Metric],
    columns: Sequence[str],
) -> dict[str, dict]:
    """``{"metrics": subgroup_metrics(...), "mean_over": ..., "stats": subgroup_stats(...)}``.

    ``mean_over`` holds ``{column: {metric: mean, "<metric>_groups": count}}``: each
    metric's unweighted mean over the column's groups, NaN groups skipped.
    """
    per_group = subgroup_metrics(predictions, targets, metrics, columns)
    return {
        "metrics": per_group,
        "mean_over": _mean_over(per_group, list(metrics)),
        "stats": subgroup_stats(predictions, targets, metrics, columns),
    }


def is_subgroup_key(name: str) -> bool:
    """Whether a split-relative metric key (the part after ``<split>/``) is a subgroup key:
    ``<column>=<group>/<metric>`` or ``mean_over_<column>/<metric>``.

    A ``/`` alone does not make one, since a split name may hold a ``/`` (the split
    ``test/external`` gives ``external/mean_dice`` under ``test``). Plan time refuses
    ``/`` and ``=`` in subgroup columns and values, so the first segment decides.
    """
    head, slash, _ = name.partition("/")
    return bool(slash) and ("=" in head or head.startswith("mean_over_"))
