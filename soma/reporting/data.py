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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import yaml

from soma.evaluation.metrics import compute_subgroup_metrics, resolve_metrics

if TYPE_CHECKING:
    from soma.config import PipelineConfig
    from soma.pipeline import PipelineResult


@dataclass
class FoldData:
    """All report data for one fold."""

    fold: int
    training_history: list[dict]  # [{epoch, train_loss, tune_loss, tune_metrics, lr}, ...]
    tune_metrics: dict[str, float]
    test_metrics: dict[str, float]
    predictions: pd.DataFrame     # columns depend on task family
    subgroup_metrics: dict | None = None  # {column → {group → {metric: value, n: int}}}


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


def aggregate_fold_predictions(folds: list[FoldData]) -> pd.DataFrame:
    """Concatenate test predictions across all folds, averaging per sample when needed.

    In fixed-holdout setups the same test set appears in every fold, producing
    duplicate sample_ids after concatenation. When duplicates are detected,
    numeric prediction columns are averaged per sample and predicted_label is
    recomputed from the averaged values:
    - prob_* (classification) → mean, predicted_label = argmax
    - raw_score (ordinal) → mean, predicted_label = round(mean)
    - predicted_value (regression) → mean
    Metadata columns (true_label, subgroup columns) are taken from the first fold.
    """
    dfs = [fd.predictions for fd in folds if not fd.predictions.empty]
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    if not df["sample_id"].duplicated().any():
        return df

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

    agg = {c: "first" for c in fixed_cols}
    agg.update({c: "mean" for c in mean_cols})
    result = df.groupby("sample_id", sort=False).agg(agg).reset_index()

    prob_cols_sorted = sorted(prob_cols)
    if prob_cols_sorted:
        result["predicted_label"] = result[prob_cols_sorted].to_numpy().argmax(axis=1)
    elif "raw_score" in result.columns:
        result["predicted_label"] = result["raw_score"].round().astype(int)

    return result


def load_run_data(run_dir: str | Path) -> RunData:
    """Load all report data from a saved run directory on disk.

    Args:
        run_dir: Path to a completed run directory (contains config.yaml, run.yaml,
            summary.json, fold_N/ subdirectories).

    Returns:
        RunData populated from disk artifacts.
    """
    run_dir = Path(run_dir)

    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    run_metadata_path = run_dir / "run.yaml"
    run_metadata = yaml.safe_load(run_metadata_path.read_text()) if run_metadata_path.exists() else {}

    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    task_family = config["task"]["name"]
    metrics = resolve_metrics(task_family, config.get("eval", {}).get("metrics") or [])
    subgroup_columns = list(config.get("eval", {}).get("subgroups", {}).get("columns") or [])

    folds = []
    fold_dirs = sorted(
        run_dir.glob("fold_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.split("_")[1])

        history_path = fold_dir / "training_history.json"
        training_history = json.loads(history_path.read_text()) if history_path.exists() else []

        metrics_data = json.loads((fold_dir / "metrics.json").read_text())

        predictions_path = fold_dir / "predictions.csv"
        predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()

        sg_path = fold_dir / "subgroup_metrics.json"
        subgroup_metrics = json.loads(sg_path.read_text()) if sg_path.exists() else None

        folds.append(FoldData(
            fold=fold_idx,
            training_history=training_history,
            tune_metrics=metrics_data.get("tune", {}),
            test_metrics=metrics_data.get("test", {}),
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
    metrics = resolve_metrics(task_family, config.eval.metrics)
    subgroup_columns = list(config.eval.subgroups.columns)

    config_dict = _config_to_dict(config)

    run_metadata_path = result.run_dir / "run.yaml"
    run_metadata = yaml.safe_load(run_metadata_path.read_text()) if run_metadata_path.exists() else {}

    folds = []
    for fold_result in result.fold_results:
        training_history = [asdict(log) for log in fold_result.train_result.history]
        predictions = _predictions_to_dataframe(fold_result.test_report.predictions)

        # Compute subgroup metrics for the in-memory path by joining predictions
        # with dataset metadata on sample_id.
        fold_subgroup_metrics = None
        if subgroup_columns and dataset is not None:
            enriched = _enrich_predictions_df(predictions, dataset, subgroup_columns)
            fold_subgroup_metrics = {"metrics": compute_subgroup_metrics(
                task_family, metrics, enriched, subgroup_columns
            )}

        folds.append(FoldData(
            fold=fold_result.fold,
            training_history=training_history,
            tune_metrics=fold_result.tune_report.metrics,
            test_metrics=fold_result.test_report.metrics,
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
    data = asdict(config)
    _stringify_paths(data)
    return data


def _stringify_paths(obj: object) -> None:
    """Recursively convert Path objects to strings in-place."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, Path):
                obj[key] = str(value)
            else:
                _stringify_paths(value)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            if isinstance(value, Path):
                obj[i] = str(value)
            else:
                _stringify_paths(value)


def _enrich_predictions_df(
    predictions_df: pd.DataFrame,
    dataset: object,
    subgroup_columns: list[str],
) -> pd.DataFrame:
    """Add subgroup columns to predictions DataFrame by joining on sample_id."""
    df = predictions_df.copy()
    samples = dataset.samples  # type: ignore[attr-defined]
    for col in subgroup_columns:
        df[col] = df["sample_id"].map(
            lambda sid, c=col: samples[sid].metadata.get(c) if sid in samples else None
        )
    return df


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
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-run comparison
# ---------------------------------------------------------------------------


@dataclass
class ComparisonData:
    """All data needed to render a cross-run comparison report."""

    runs: list[RunData]
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
