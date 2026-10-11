"""``check_task_head``: the conformance suite for a set-shape :class:`soma.tasks.TaskHead`."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from soma.evaluation import Baseline, Predictions, evaluate
from soma.evaluation.predictions import PLACEHOLDER_COLUMN, family_target_keys
from soma.testing._errors import ConformanceError

__all__ = ["check_task_head"]


def _targets_frame(head, targets: dict[str, torch.Tensor], sample_ids: list[str]) -> pd.DataFrame:
    """The cohort-side targets of ``batch``: raw labels (decoded through ``classes``)."""
    try:
        keys = family_target_keys(head.task_family)
    except ValueError as error:
        raise ConformanceError("task_family", str(error)) from error
    frame = pd.DataFrame(
        {key: targets[key].detach().cpu().numpy() for key in keys},
        index=pd.Index(sample_ids, name="sample_id", dtype=object),
    )
    classes = head.classes
    if "label" in frame.columns and classes is not None:
        frame["label"] = [classes[int(i)] for i in frame["label"]]
    return frame


def _agree(left: float, right: float, tolerance: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return abs(left - right) <= tolerance


def check_task_head(head, batch: Any, *, tolerance: float = 1e-6) -> None:
    """Check a set-shape task head on one batch on CPU.

    ``batch`` carries ``features`` (the head's input), ``targets`` (the encoded target
    tensors ``compute_metrics`` reads) and ``sample_ids`` — a
    :class:`~soma.training.sample_dataset.SampleBatch` does. Asserts: ``to_columns``
    emits exactly ``prediction_columns``, one numeric value per sample; a
    :class:`~soma.evaluation.Baseline` fitted on the batch's targets yields the same
    columns; and :func:`~soma.evaluate` on the predictions read back from parquet agrees
    with ``compute_metrics`` within ``tolerance`` on every metric the head reports.
    """
    head = head.to(torch.device("cpu")).eval()
    sample_ids = [str(sid) for sid in batch.sample_ids]
    with torch.inference_mode():
        try:
            logits = head(batch.features)
        except Exception as error:  # noqa: BLE001 - any failure is a conformance failure
            raise ConformanceError("forward", f"raised {error!r}") from error
        try:
            declared = tuple(head.prediction_columns)
        except Exception as error:  # noqa: BLE001
            raise ConformanceError("prediction_columns", f"raised {error!r}") from error
        try:
            columns = head.to_columns(head.postprocess(logits))
        except Exception as error:  # noqa: BLE001
            raise ConformanceError("to_columns", f"raised {error!r}") from error
        expected_metrics = head.compute_metrics(logits, batch.targets)

    if set(columns) != set(declared):
        raise ConformanceError(
            "to_columns",
            f"emits {sorted(columns)} but prediction_columns declares {sorted(declared)}",
        )
    for name in declared:
        values = np.asarray(columns[name])
        if values.shape != (len(sample_ids),):
            raise ConformanceError(
                "to_columns", f"'{name}' has shape {values.shape}, want ({len(sample_ids)},)"
            )
        if not (np.issubdtype(values.dtype, np.number) or values.dtype == np.bool_):
            raise ConformanceError("to_columns", f"'{name}' is not numeric (dtype {values.dtype})")

    targets = _targets_frame(head, batch.targets, sample_ids)
    try:
        baseline = Baseline.fit(targets, task_family=head.task_family, classes=head.classes)
        placeholders = baseline.predict(sample_ids)
    except Exception as error:  # noqa: BLE001
        raise ConformanceError("Baseline", f"raised {error!r}") from error
    baseline_columns = [c for c in placeholders.frame.columns if c not in {"sample_id", PLACEHOLDER_COLUMN}]
    if tuple(baseline_columns) != declared:
        raise ConformanceError(
            "prediction_columns",
            f"the family Baseline yields {baseline_columns}, the head declares {list(declared)}",
        )

    frame = pd.DataFrame({"sample_id": sample_ids, **{name: np.asarray(columns[name]) for name in declared}})
    predictions = Predictions(
        frame, task_family=head.task_family, prediction_columns=declared, classes=head.classes
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = predictions.to_parquet(Path(tmp) / "predictions.parquet")
        reread = Predictions.from_parquet(path)
    metric_names = list(getattr(head, "metrics", None) or expected_metrics)
    try:
        evaluation = evaluate(reread, targets, metrics=metric_names)
    except Exception as error:  # noqa: BLE001
        raise ConformanceError("evaluate", f"raised {error!r} on the parquet round-trip") from error
    for name in metric_names:
        if name not in expected_metrics:
            continue
        got, want = float(evaluation.metrics[name]), float(expected_metrics[name])
        if not _agree(got, want, tolerance):
            raise ConformanceError(
                "compute_metrics",
                f"'{name}' is {want} from compute_metrics but {got} from evaluate on the "
                "parquet round-trip",
            )
