"""Fold results on disk: the per-fold writers and the run summary built from them.

The summary is always aggregated from every fold's persisted ``metrics.json``, never
from this launch's in-memory results, so a resumed run (some folds trained by an earlier
launch) summarizes exactly as a single-shot run does (issue #244). Public so that a
benchmark composing the verbs itself (HEST) writes the same files a ``Pipeline`` run does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from soma._internal.atomic_io import atomic_write_json
from soma.evaluation import Evaluation
from soma.evaluation.metrics import metric_registry
from soma.evaluation.subgroups import is_subgroup_key
from soma.evaluation.summary import summarize_values
from soma.pipeline.layout import reject_undeclared_fold_dirs

__all__ = [
    "aggregate_fold_metrics_from_disk",
    "history_rows",
    "peak_per_metric",
    "save_metrics",
    "save_summary",
    "save_training_history",
    "summarize_fold_metric_dicts",
]


def save_metrics(
    tune_evaluation: Evaluation,
    test_evaluations: dict[str, Evaluation],
    path: Path,
) -> None:
    """Write a fold's ``metrics.json``: ``{"tune": {...}, <test split>: {...}, ...}``."""
    data: dict[str, dict[str, float]] = {"tune": tune_evaluation.metrics}
    for split_name, split_evaluation in test_evaluations.items():
        data[split_name] = split_evaluation.metrics
    atomic_write_json(path, data)


def save_summary(summary: dict[str, float], path: Path) -> None:
    """Write a run's ``summary.json``."""
    atomic_write_json(path, summary)


#: ``History`` columns that are not (split-prefixed) validation-summary keys.
_HISTORY_LOOP_COLUMNS = ("epoch", "train_loss", "lr", "elapsed_seconds")


def history_rows(history: pd.DataFrame) -> list[dict[str, object]]:
    """The ``History`` frame as JSON-ready rows: ``epoch`` an int, every other column a
    float, and ``None`` where an epoch was not evaluated."""
    rows: list[dict[str, object]] = []
    for record in history.to_dict("records"):
        row: dict[str, object] = {}
        for key, value in record.items():
            if key == "epoch":
                row[key] = int(value)
            elif value is None or (isinstance(value, float) and math.isnan(value)):
                row[key] = None
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def peak_per_metric(history: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Per-metric best value across all epochs, with the (1-based) epoch it occurred at.

    DIAGNOSTIC ONLY. This describes a model that never existed at any single epoch (each
    metric peaks at its own epoch), so it is positively biased and must never feed
    checkpoint selection, ``summary.json`` fold aggregates, or any results table. It is
    recorded in ``training_history.json`` purely to judge whether the configured
    ``monitor`` is a good selection proxy. Entries keep the history's split-prefixed
    names (``tune/auroc``, ``tune/mean_over_centre/auroc``); direction follows
    :meth:`soma.evaluation.metric_registry.higher_is_better` of the bare metric. Loss
    keys, one group's keys (``tune/centre=A/auroc``) and the group counts of a mean over
    groups (``tune/mean_over_centre/auroc_groups``) are excluded.
    """
    peaks: dict[str, dict[str, float | int]] = {}
    columns = set(history.columns)
    for name in history.columns:
        if name in _HISTORY_LOOP_COLUMNS or name.rsplit("/", 1)[-1] == "loss":
            continue
        within = name.partition("/")[2]
        if is_subgroup_key(within):
            one_group = "=" in within.partition("/")[0]
            if one_group or (name.endswith("_groups") and name.removesuffix("_groups") in columns):
                continue
        for epoch, raw in zip(history["epoch"], history[name]):
            value = float(raw)
            if not np.isfinite(value):
                continue
            current = peaks.get(name)
            better = current is None or (
                value > current["value"]
                if metric_registry.higher_is_better(name)
                else value < current["value"]
            )
            if better:
                peaks[name] = {"epoch": int(epoch) + 1, "value": value}
    return peaks


def save_training_history(history: pd.DataFrame, path: Path) -> None:
    """Write a fold's ``training_history.json`` from the ``History`` frame."""
    data = {
        "epochs": history_rows(history),
        # Diagnostic only (see peak_per_metric): it never reaches summaries or results
        # tables.
        "peak_per_metric": peak_per_metric(history),
    }
    atomic_write_json(path, data)


def summarize_fold_metric_dicts(
    per_fold: list[dict[str, dict[str, float]]], *, include_tune: bool
) -> dict[str, float]:
    """Aggregate per-fold ``{split_name: {metric: value}}`` dicts into a run summary.

    For a single fold, emit values directly; for multiple folds, emit mean and std.
    Normally the summary covers the test splits only. ``include_tune`` (set for
    ``evaluation.holdout_test`` runs, which have no test splits) additionally emits
    the tune metrics under a ``tune/`` prefix so a tune-only selection sweep produces
    a non-empty, rankable summary.

    Every key any fold reports for a split is summarized, in first-seen order. A fold
    that has the split but lacks the key (a subgroup with under 2 samples there) counts
    as a NaN fold: it is left out of the mean and std and counted in ``_nan_folds``.
    """
    if not per_fold:
        return {}

    single_fold = len(per_fold) == 1
    summary: dict[str, float] = {}
    # Map each reported split name to its per-fold metric dicts. Tune leads (when
    # requested) so it heads the summary; test splits follow in sorted order.
    metrics_by_split: dict[str, list[dict[str, float]]] = {}
    if include_tune:
        metrics_by_split["tune"] = [f["tune"] for f in per_fold if "tune" in f]
    for split_name in sorted({s for f in per_fold for s in f if s != "tune"}):
        metrics_by_split[split_name] = [f[split_name] for f in per_fold if split_name in f]

    for split_name, split_metrics in metrics_by_split.items():
        if not split_metrics:
            continue
        metric_keys = list(dict.fromkeys(key for metrics in split_metrics for key in metrics))
        for key in metric_keys:
            values = [metrics.get(key, math.nan) for metrics in split_metrics]
            if single_fold:
                summary[f"{split_name}/{key}"] = float(values[0])
            else:
                stats = summarize_values(values)
                summary[f"{split_name}/{key}_mean"] = stats.mean
                summary[f"{split_name}/{key}_std"] = stats.std
                if stats.n_nan:
                    # Folds whose split could not support the metric (nan) are
                    # excluded from mean/std; surface how many so the reader knows.
                    summary[f"{split_name}/{key}_nan_folds"] = stats.n_nan

    return summary


def aggregate_fold_metrics_from_disk(
    run_dir: Path, *, single_fold: bool, include_tune: bool, num_folds: int
) -> dict[str, float]:
    """Aggregate every fold's persisted ``metrics.json`` under ``run_dir``.

    Reading from disk rather than in-memory results is what makes a resumed run's
    ``summary.json`` correct (issue #244): folds skipped this session (already complete
    on disk) re-enter the summary, so a resumed 5-fold summary equals the single-shot one.

    A split's ``subgroup_metrics_<split>.json`` beside ``metrics.json`` adds its
    breakdown to the split's metrics: ``<column>=<group>/<metric>`` per group and
    ``mean_over_<column>/<metric>``, so the summary carries
    ``<split>/<column>=<group>/<metric>`` (``_mean``, ``_std``, ``_nan_folds`` over
    several folds). The counts (a group's ``n``, ``<metric>_groups``) stay in the file.
    """
    if single_fold:
        metrics_paths = [Path(run_dir) / "metrics.json"]
    else:
        reject_undeclared_fold_dirs(run_dir, num_folds)
        metrics_paths = [Path(run_dir) / f"fold_{i}" / "metrics.json" for i in range(num_folds)]
    per_fold = []
    for path in metrics_paths:
        if not path.is_file():
            continue
        splits = json.loads(path.read_text())
        for split_name, metrics in splits.items():
            breakdown = path.parent / f"subgroup_metrics_{split_name}.json"
            if breakdown.is_file():
                metrics.update(_subgroup_keys(json.loads(breakdown.read_text())))
        per_fold.append(splits)
    return summarize_fold_metric_dicts(per_fold, include_tune=include_tune)


def _subgroup_keys(breakdown: dict) -> dict[str, float]:
    """A subgroup breakdown's metrics as split-relative keys, column by column: each
    group's ``<column>=<group>/<metric>``, then ``mean_over_<column>/<metric>``."""
    keys: dict[str, float] = {}
    mean_over = breakdown.get("mean_over", {})
    for column, groups in breakdown.get("metrics", {}).items():
        for group, values in groups.items():
            for name, value in values.items():
                if name != "n":
                    keys[f"{column}={group}/{name}"] = value
        means = mean_over.get(column, {})
        for name, value in means.items():
            if f"{name}_groups" in means:  # a metric, not its count
                keys[f"mean_over_{column}/{name}"] = value
    return keys
