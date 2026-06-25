"""Data loading layer for experiment reports.

Provides two paths for building RunData:
- load_run_data(run_dir)              — reads from a completed run directory on disk
- run_data_from_result(result, config) — converts in-memory PipelineResult objects

And cross-run comparison utilities:
- load_comparison_data(run_dirs)      — load multiple runs and compute config diffs
- diff_configs(configs)               — partition config fields into shared and varying
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from soma.evaluation.metrics import probability_columns, resolve_metrics
from soma.reporting.subgroups import enrich_predictions_with_subgroups, subgroup_report_for_predictions
from soma.training.trainer import epoch_log_to_dict

if TYPE_CHECKING:
    from soma.config import PipelineConfig
    from soma.pipeline import PipelineResult


@dataclass
class FoldData:
    """All report data for one fold."""

    fold: int
    training_history: list[dict]  # [{epoch, train_loss, tune_loss, tune_metrics, lr}, ...]
    tune_metrics: dict[str, float]
    test_metrics: dict[str, dict[str, float]]   # split_name → {metric: value}
    predictions: dict[str, pd.DataFrame]         # split_name → predictions df
    subgroup_metrics: dict[str, dict] | None = None  # split_name → subgroup data


@dataclass
class FoldSlice:
    """Single-split view of FoldData — used by chart rendering functions.

    Exposes ``predictions`` as a plain DataFrame (for a specific test split)
    so that existing chart helpers can operate on one split at a time without
    knowing about the multi-split structure.
    """

    fold: int
    training_history: list[dict]
    tune_metrics: dict[str, float]
    test_metrics: dict[str, float]   # flat metrics dict for one split
    predictions: pd.DataFrame
    subgroup_metrics: dict | None = None


def fold_slices_for_split(folds: list[FoldData], split_name: str) -> list[FoldSlice]:
    """Project a list of FoldData to single-split FoldSlice objects for chart rendering."""
    slices = []
    for fd in folds:
        sg = fd.subgroup_metrics.get(split_name) if fd.subgroup_metrics else None
        slices.append(FoldSlice(
            fold=fd.fold,
            training_history=fd.training_history,
            tune_metrics=fd.tune_metrics,
            test_metrics=fd.test_metrics.get(split_name, {}),
            predictions=fd.predictions.get(split_name, pd.DataFrame()),
            subgroup_metrics=sg,
        ))
    return slices


@dataclass
class RunData:
    """All report data for a full run."""

    config: dict
    run_metadata: dict
    summary: dict[str, float]
    folds: list[FoldData]
    task_family: str
    metrics: list[str]            # resolved user-requested metrics
    subgroup_columns: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.subgroup_columns is None:
            self.subgroup_columns = []


def _aggregate_dataframes(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate prediction DataFrames, averaging per sample when duplicates exist.

    When the same sample_id appears in multiple DataFrames (fixed-holdout setup),
    numeric prediction columns are averaged and predicted_label is recomputed:
    - prob_* (classification) → mean, predicted_label = argmax
    - raw_score (ordinal) → mean, predicted_label = round(mean)
    - predicted_value (regression) → mean
    Metadata columns (true_label, subgroup columns) are taken from the first fold.
    """
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if not df["sample_id"].duplicated().any():
        return df

    has_prob_signal = any(c.startswith("prob_") for c in df.columns)
    has_raw_signal = "raw_score" in df.columns
    if "predicted_label" in df.columns and not has_prob_signal and not has_raw_signal:
        label_conflicts = (
            df.groupby("sample_id", sort=False)["predicted_label"]
            .nunique(dropna=False)
        )
        conflicting = label_conflicts[label_conflicts > 1].index.tolist()
        if conflicting:
            preview = ", ".join(map(str, conflicting[:5]))
            suffix = "" if len(conflicting) <= 5 else f" (+{len(conflicting) - 5} more)"
            raise ValueError(
                "Cannot aggregate duplicate samples with conflicting predicted_label "
                f"values and no probability/raw-score signal. Conflicting sample_ids: "
                f"{preview}{suffix}."
            )

    # Classify columns in one pass
    prob_cols: list[str] = []
    mean_cols: list[str] = []
    fixed_cols: list[str] = []
    for c in df.columns:
        if c in {"sample_id", "predicted_label"}:
            continue
        elif c.startswith("prob_"):
            prob_cols.append(c)
            mean_cols.append(c)
        elif c in {"predicted_value", "raw_score"}:
            mean_cols.append(c)
        else:
            fixed_cols.append(c)

    agg: dict[str, str] = {c: "first" for c in fixed_cols}
    if "predicted_label" in df.columns:
        agg["predicted_label"] = "first"
    agg.update({c: "mean" for c in mean_cols})
    result = df.groupby("sample_id", sort=False).agg(agg).reset_index()

    prob_cols_sorted = probability_columns(prob_cols)
    if prob_cols_sorted:
        class_indices = [int(col.removeprefix("prob_")) for col in prob_cols_sorted]
        result["predicted_label"] = [
            class_indices[idx]
            for idx in result[prob_cols_sorted].to_numpy().argmax(axis=1)
        ]
    elif "raw_score" in result.columns:
        result["predicted_label"] = result["raw_score"].round().astype(int)

    return result


def aggregate_fold_predictions(folds: list[FoldData], split_name: str) -> pd.DataFrame:
    """Concatenate test predictions for a given split across all folds."""
    dfs = [
        fd.predictions[split_name]
        for fd in folds
        if split_name in fd.predictions and not fd.predictions[split_name].empty
    ]
    return _aggregate_dataframes(dfs)


def aggregate_slice_predictions(slices: list[FoldSlice]) -> pd.DataFrame:
    """Concatenate predictions from FoldSlice objects (used by chart rendering)."""
    dfs = [s.predictions for s in slices if not s.predictions.empty]
    return _aggregate_dataframes(dfs)


def load_run_data(run_dir: str | Path) -> RunData:
    """Load all report data from a saved run directory on disk.

    Args:
        run_dir: Path to a completed run directory (contains config.yaml, run.yaml,
            summary.json, and either artifacts directly in run_dir for single-fold
            runs or fold_N/ subdirectories for cross-validation runs).

    Returns:
        RunData populated from disk artifacts.
    """
    run_dir = Path(run_dir)

    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise TypeError("config.yaml must contain a mapping")

    run_metadata_path = run_dir / "run.yaml"
    run_metadata = yaml.safe_load(run_metadata_path.read_text()) if run_metadata_path.exists() else {}

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    task_family = config.get("task", {}).get("name", "binary_classification")
    evaluation = config.get("evaluation", {}) or {}
    metrics = resolve_metrics(task_family, evaluation.get("metrics") or [])
    subgroup_columns = list((evaluation.get("subgroups", {}) or {}).get("columns", []) or [])

    # Detect layout: single-fold artifacts in run_dir vs nested fold_N/ subdirs
    flat_layout = (run_dir / "metrics.json").exists()
    if flat_layout:
        fold_dirs_indexed = [(run_dir, 0)]
    else:
        fold_dirs_indexed = [
            (p, int(p.name.split("_")[1]))
            for p in sorted(run_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
        ]

    folds = []
    for fold_dir, fold_idx in fold_dirs_indexed:
        history_path = fold_dir / "training_history.json"
        training_history = _load_training_history(history_path)

        metrics_data = json.loads((fold_dir / "metrics.json").read_text())
        test_split_names = [k for k in metrics_data if k != "tune"]

        predictions: dict[str, pd.DataFrame] = {}
        for split_name in test_split_names:
            p = fold_dir / f"predictions_{split_name}.csv"
            predictions[split_name] = pd.read_csv(p) if p.exists() else pd.DataFrame()

        subgroup_metrics: dict[str, dict] | None = None
        sg_data = {}
        for split_name in test_split_names:
            sg_path = fold_dir / f"subgroup_metrics_{split_name}.json"
            if sg_path.exists():
                sg_data[split_name] = json.loads(sg_path.read_text())
        if sg_data:
            subgroup_metrics = sg_data

        folds.append(FoldData(
            fold=fold_idx,
            training_history=training_history,
            tune_metrics=metrics_data.get("tune", {}),
            test_metrics={k: v for k, v in metrics_data.items() if k != "tune"},
            predictions=predictions,
            subgroup_metrics=subgroup_metrics,
        ))

    return RunData(
        config=config,
        run_metadata=run_metadata,
        summary=summary,
        folds=folds,
        task_family=task_family,
        metrics=metrics,
        subgroup_columns=subgroup_columns,
    )


def _load_training_history(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    payload = json.loads(history_path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        epochs = payload.get("epochs", [])
        return epochs if isinstance(epochs, list) else []
    return []


def run_data_from_result(
    result: PipelineResult,
    config: PipelineConfig,
    *,
    dataset: object = None,
) -> RunData:
    """Build RunData from an in-memory PipelineResult (no heavy disk reads).

    Args:
        result: PipelineResult returned by Pipeline.run() or train().
        config: The PipelineConfig used for the run.
        dataset: Optional Dataset object. Required to compute subgroup metrics
            for the in-memory path when subgroup columns are configured.

    Returns:
        RunData ready for report rendering.
    """
    task_family = config.task.name
    metrics = resolve_metrics(task_family, config.evaluation.metrics)
    subgroup_columns = list(config.evaluation.subgroups.columns)

    config_dict = _config_to_dict(config)

    run_metadata_path = result.run_dir / "run.yaml"
    run_metadata = yaml.safe_load(run_metadata_path.read_text()) if run_metadata_path.exists() else {}

    folds = []
    for fold_result in result.fold_results:
        training_history = (
            [epoch_log_to_dict(log) for log in fold_result.train_result.history]
            if fold_result.train_result is not None
            else []
        )

        predictions: dict[str, pd.DataFrame] = {
            split_name: _predictions_to_dataframe(report.predictions)
            for split_name, report in fold_result.test_reports.items()
        }

        fold_subgroup_metrics: dict[str, dict] | None = None
        if subgroup_columns and dataset is not None:
            sg_data = {}
            for split_name, preds_df in predictions.items():
                if not preds_df.empty:
                    enriched = enrich_predictions_with_subgroups(preds_df, dataset, subgroup_columns)
                    sg_data[split_name] = subgroup_report_for_predictions(
                        task_family=task_family,
                        metrics=metrics,
                        predictions_df=enriched,
                        subgroup_columns=subgroup_columns,
                    )
            if sg_data:
                fold_subgroup_metrics = sg_data

        folds.append(FoldData(
            fold=fold_result.fold,
            training_history=training_history,
            tune_metrics=fold_result.tune_report.metrics,
            test_metrics={k: v.metrics for k, v in fold_result.test_reports.items()},
            predictions=predictions,
            subgroup_metrics=fold_subgroup_metrics,
        ))

    return RunData(
        config=config_dict,
        run_metadata=run_metadata,
        summary=result.summary,
        folds=folds,
        task_family=task_family,
        metrics=metrics,
        subgroup_columns=subgroup_columns,
    )


def _config_to_dict(config: PipelineConfig) -> dict:
    """Serialize PipelineConfig to a plain dict (Paths converted to strings)."""
    from soma.config import _config_to_layout_dict

    return _config_to_layout_dict(config)


def _predictions_to_dataframe(predictions: list) -> pd.DataFrame:
    """Convert a list of SamplePrediction objects to a DataFrame."""
    if not predictions:
        return pd.DataFrame()

    rows = []
    for p in predictions:
        row: dict = {"sample_id": p.sample_id, "true_label": p.true_label}
        if p.predicted_label is not None:
            row["predicted_label"] = p.predicted_label
        if p.probabilities is not None:
            for i, prob in enumerate(p.probabilities):
                row[f"prob_{i}"] = prob
        if p.predicted_value is not None:
            row["predicted_value"] = p.predicted_value
        if p.raw_score is not None:
            row["raw_score"] = p.raw_score
        if p.is_placeholder:
            row["is_placeholder"] = True
        if p.missing_reason is not None:
            row["missing_reason"] = p.missing_reason
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonData:
    """All data needed to render a cross-run comparison report."""

    runs: list[RunData]
    run_dirs: list[Path]
    labels: list[str]           # one short label per run
    shared_config: dict         # flat key → value for fields identical across all runs
    config_diffs: list[dict]    # one flat dict per run, containing only varying fields
    metric_names: list[str]     # union of all runs' requested metrics


def load_comparison_data(
    run_dirs: list[Path],
    *,
    labels: list[str] | None = None,
) -> ComparisonData:
    """Load multiple runs and compute config diffs.

    Args:
        run_dirs: Paths to completed run directories.
        labels: Optional short labels for each run. When None, labels are
            auto-derived: if runs differ in exactly one config field, that
            field's value is used; otherwise the run_id is used.

    Returns:
        ComparisonData ready for report rendering.
    """
    runs = [load_run_data(d) for d in run_dirs]

    configs = [r.config for r in runs]
    shared_config, config_diffs = diff_configs(configs)

    if labels is not None:
        resolved_labels = list(labels)
    elif len(config_diffs) > 0 and all(len(d) == 1 for d in config_diffs):
        # All runs differ in exactly the same single field — use its value as label
        resolved_labels = [next(iter(d.values())) for d in config_diffs]
        resolved_labels = [str(v) for v in resolved_labels]
    else:
        resolved_labels = [r.run_metadata.get("run_id") or f"run_{i}" for i, r in enumerate(runs)]

    # Union of all requested metrics, preserving order
    metric_names: list[str] = []
    for run in runs:
        for m in run.metrics:
            if m not in metric_names:
                metric_names.append(m)

    return ComparisonData(
        runs=runs,
        run_dirs=run_dirs,
        labels=resolved_labels,
        shared_config=shared_config,
        config_diffs=config_diffs,
        metric_names=metric_names,
    )


def diff_configs(configs: list[dict]) -> tuple[dict, list[dict]]:
    """Partition config fields into shared values and per-run diffs.

    Recursively flattens nested dicts using dot notation (e.g.,
    ``training.learning_rate``). Fields whose values are identical across
    all configs go into ``shared``; fields that differ go into per-run
    ``diffs`` dicts.

    Args:
        configs: List of config dicts (one per run).

    Returns:
        ``(shared, diffs)`` where ``shared`` maps flat key → common value and
        ``diffs`` is a list of dicts (one per run) mapping flat key → value
        for every key that varies.
    """
    if not configs:
        return {}, []

    flat_configs = [_flatten(c) for c in configs]
    all_keys: list[str] = []
    for fc in flat_configs:
        for k in fc:
            if k not in all_keys:
                all_keys.append(k)

    shared: dict = {}
    diffs: list[dict] = [{} for _ in configs]

    for key in all_keys:
        values = [fc.get(key) for fc in flat_configs]
        if all(v == values[0] for v in values[1:]):
            shared[key] = values[0]
        else:
            for i, v in enumerate(values):
                diffs[i][key] = v

    return shared, diffs


def _flatten(obj: object, prefix: str = "") -> dict:
    """Recursively flatten a nested dict to dot-separated keys."""
    result: dict = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                result.update(_flatten(value, full_key))
            else:
                result[full_key] = value
    else:
        result[prefix] = obj
    return result
