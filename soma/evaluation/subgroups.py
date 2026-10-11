"""Subgroup breakdown: the same metrics over each group of a target metadata column.

:func:`soma.evaluate` joins predictions with the cohort's targets frame, which carries
every metadata column, so a subgroup is just a ``groupby`` over that joined frame.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = ["subgroup_breakdown", "subgroup_metrics", "subgroup_stats"]

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


def subgroup_breakdown(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    metrics: Mapping[str, Metric],
    columns: Sequence[str],
) -> dict[str, dict]:
    """``{"metrics": subgroup_metrics(...), "stats": subgroup_stats(...)}``."""
    return {
        "metrics": subgroup_metrics(predictions, targets, metrics, columns),
        "stats": subgroup_stats(predictions, targets, metrics, columns),
    }
