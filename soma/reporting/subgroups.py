"""Shared subgroup enrichment helpers for saved and in-memory reports."""

from __future__ import annotations

from typing import Iterable, Protocol

import pandas as pd

from soma.evaluation.metrics import compute_subgroup_metrics, compute_subgroup_stats


class _PredictionLike(Protocol):
    sample_id: str


def subgroup_data_for_predictions(
    dataset: object,
    predictions: Iterable[_PredictionLike],
    subgroup_columns: list[str],
) -> dict[str, dict[str, object]]:
    """Build sample/patient ID -> subgroup metadata for report predictions."""
    if not subgroup_columns:
        return {}

    samples = dataset.samples  # type: ignore[attr-defined]
    has_patient_ids = dataset.has_patient_ids  # type: ignore[attr-defined]
    patient_groups = dataset.patient_groups if has_patient_ids else {}  # type: ignore[attr-defined]
    subgroup_data: dict[str, dict[str, object]] = {}

    for pred in predictions:
        if pred.sample_id in samples:
            subgroup_data[pred.sample_id] = {
                col: samples[pred.sample_id].metadata.get(col)
                for col in subgroup_columns
            }
            continue

        patient_records = patient_groups.get(pred.sample_id)
        if not patient_records:
            subgroup_data[pred.sample_id] = {}
            continue

        values: dict[str, object] = {}
        for col in subgroup_columns:
            observed_values = {record.metadata.get(col) for record in patient_records}
            if len(observed_values) > 1:
                raise ValueError(
                    f"Patient '{pred.sample_id}' has inconsistent subgroup metadata for column '{col}'."
                )
            values[col] = patient_records[0].metadata.get(col)
        subgroup_data[pred.sample_id] = values

    return subgroup_data


def enrich_predictions_with_subgroups(
    predictions_df: pd.DataFrame,
    dataset: object,
    subgroup_columns: list[str],
) -> pd.DataFrame:
    """Return a predictions DataFrame enriched with slide/patient subgroup metadata."""
    if predictions_df.empty or not subgroup_columns:
        return predictions_df.copy()

    ids = [
        _DataFramePrediction(sample_id=str(sample_id))
        for sample_id in predictions_df["sample_id"].tolist()
    ]
    subgroup_data = subgroup_data_for_predictions(dataset, ids, subgroup_columns)
    df = predictions_df.copy()
    for col in subgroup_columns:
        df[col] = df["sample_id"].map(
            lambda sample_id, c=col: subgroup_data.get(str(sample_id), {}).get(c)
        )
    return df


def subgroup_report_for_predictions(
    *,
    task_family: str,
    metrics: list[str],
    predictions_df: pd.DataFrame,
    subgroup_columns: list[str],
) -> dict[str, dict]:
    """Compute subgroup metrics and counts from an already-enriched predictions frame."""
    return {
        "metrics": compute_subgroup_metrics(task_family, metrics, predictions_df, subgroup_columns),
        "stats": compute_subgroup_stats(task_family, metrics, predictions_df, subgroup_columns),
    }


class _DataFramePrediction:
    def __init__(self, *, sample_id: str) -> None:
        self.sample_id = sample_id
