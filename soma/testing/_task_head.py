"""``check_task_head``: the conformance suite for a set-shape :class:`soma.models.heads.TaskHead`."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from soma.evaluation import Baseline, Predictions, evaluate, sweep_threshold
from soma.evaluation.predictions import PLACEHOLDER_COLUMN, family_target_keys
from soma.testing._errors import ConformanceError

__all__ = ["check_task_head"]


def _targets_frame(head, targets: dict[str, torch.Tensor], sample_ids: list[str]) -> pd.DataFrame:
    """The cohort-side targets of ``batch``: raw labels (decoded through ``classes``); a
    ``(B, G)`` target spreads into ``<key>_0 .. <key>_{G-1}`` (a multi-output regression's
    ``value_<k>``)."""
    try:
        keys = family_target_keys(head.task_family)
    except ValueError as error:
        raise ConformanceError("task_family", str(error)) from error
    columns: dict[str, np.ndarray] = {}
    for key in keys:
        values = targets[key].detach().cpu().numpy()
        if values.ndim == 2:
            columns.update({f"{key}_{k}": values[:, k] for k in range(values.shape[1])})
        else:
            columns[key] = values
    frame = pd.DataFrame(columns, index=pd.Index(sample_ids, name="sample_id", dtype=object))
    classes = head.classes
    if "label" in frame.columns and classes is not None:
        frame["label"] = [classes[int(i)] for i in frame["label"]]
    return frame


def _agree(left: float, right: float, tolerance: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    return abs(left - right) <= tolerance


class _LabelMaps:
    """An in-memory label-map target source over a batch's ``mask``."""

    target_kind = "label_map"

    def __init__(self, maps: dict[str, torch.Tensor]) -> None:
        self._maps = maps

    @property
    def sample_ids(self) -> list[str]:
        return list(self._maps)

    def load(self, sample_id: str) -> torch.Tensor:
        return self._maps[sample_id]


class _PointTargets:
    """An in-memory point target source over a batch's level-0 ``points`` (+ ``ignore``)."""

    target_kind = "points"

    def __init__(self, points: dict[str, Any]) -> None:
        self._points = points

    @property
    def sample_ids(self) -> list[str]:
        return list(self._points)

    def load(self, sample_id: str):
        return self._points[sample_id]


def _grid_targets(head, batch, sample_ids: list[str]):
    from soma.data import Points

    if head.target_kind == "label_map":
        return _LabelMaps({sid: batch.targets["mask"][i].cpu() for i, sid in enumerate(sample_ids)})
    geometry = head.geometry
    identity = (
        tuple(geometry.origin_level0) == (0.0, 0.0)
        and float(geometry.level0_px_per_token_px) == 1.0
        and tuple(geometry.crop_box[:2]) == (0, 0)
    )
    ignore = batch.targets.get("ignore")
    if ignore is not None and not identity:
        raise ConformanceError(
            "geometry",
            "check_task_head maps a batch's ignore map back to level-0 only for an identity "
            "level-0 anchor (origin 0, scale 1, no crop offset)",
        )
    points = {}
    for i, sid in enumerate(sample_ids):
        rows = batch.targets["points"][i].cpu()
        rows = rows[~torch.isnan(rows).any(dim=1)]
        raster = None
        if ignore is not None:
            raster = (ignore[i].cpu().numpy().astype(np.uint8) * 255)
        points[sid] = Points(xy=rows[:, :2].to(torch.float64), classes=rows[:, 2].round().long(), ignore_mask=raster)
    return _PointTargets(points)


def _check_grid_head(head, batch: Any, *, tolerance: float) -> None:
    """The grid branch of :func:`check_task_head` (see there)."""
    from soma.evaluation.grid import LabelMapPredictions, PointPredictions

    head = head.to(torch.device("cpu")).eval()
    sample_ids = [str(sid) for sid in batch.sample_ids]
    if head.target_kind not in ("label_map", "points"):
        raise ConformanceError("target_kind", f"must be 'label_map' or 'points', got {head.target_kind!r}")
    with torch.inference_mode():
        try:
            logits = head(batch.features)
        except Exception as error:  # noqa: BLE001
            raise ConformanceError("forward", f"raised {error!r}") from error
        size = tuple(int(v) for v in head.geometry.target_size)
        if tuple(int(v) for v in logits.shape[-2:]) != size:
            raise ConformanceError("forward", f"logits are {tuple(logits.shape[-2:])}, the target size is {size}")
        try:
            columns = head.to_columns(head.postprocess(logits))
        except Exception as error:  # noqa: BLE001
            raise ConformanceError("to_columns", f"raised {error!r}") from error
        expected = head.compute_metrics(logits, batch.targets)
    for name, values in columns.items():
        if len(values) != len(sample_ids):
            raise ConformanceError("to_columns", f"'{name}' holds {len(values)} values for {len(sample_ids)} samples")
    try:
        predictions = head.assemble_predictions(sample_ids, {k: list(v) for k, v in columns.items()})
    except Exception as error:  # noqa: BLE001
        raise ConformanceError("assemble_predictions", f"raised {error!r}") from error
    noun = LabelMapPredictions if head.target_kind == "label_map" else PointPredictions
    if not isinstance(predictions, noun):
        raise ConformanceError("assemble_predictions", f"must return {noun.__name__}, got {type(predictions).__name__}")
    if sorted(predictions.sample_ids) != sorted(sample_ids):
        raise ConformanceError("assemble_predictions", "the predictions must carry every sample id of the batch")
    targets = _grid_targets(head, batch, sample_ids)
    swept: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        if noun is LabelMapPredictions:
            reread = LabelMapPredictions.load(predictions.save(Path(tmp) / "label_maps"))
            metrics = [{"name": name} for name in head.metrics]
        else:
            reread = PointPredictions.from_parquet(predictions.to_parquet(Path(tmp) / "points.parquet"))
            # Training-time validation scores the operating point the post-training sweep
            # picks: per-class thresholds swept with greedy matching.
            try:
                threshold, _ = sweep_threshold(reread, targets, per_class=True, matching="greedy")
            except Exception as error:  # noqa: BLE001
                raise ConformanceError("sweep_threshold", f"raised {error!r} on the round-trip") from error
            swept = {f"threshold_class_{c}": float(t) for c, t in threshold.items()}
            reread = reread.with_threshold(threshold)
            metrics = [{"name": name, "params": {"matching": "greedy"}} for name in head.metrics]
    try:
        evaluation = evaluate(reread, targets, metrics=metrics)
    except ConformanceError:
        raise
    except Exception as error:  # noqa: BLE001
        raise ConformanceError("evaluate", f"raised {error!r} on the round-trip") from error
    for name, want in swept.items():
        got = expected.get(name)
        if got is None or not _agree(float(got), want, tolerance):
            raise ConformanceError(
                "compute_metrics",
                f"'{name}' is {got} from compute_metrics but sweep_threshold picks {want} on the round-trip",
            )
    for name, want in expected.items():
        if name in swept:
            continue
        if name not in evaluation.metrics:
            raise ConformanceError("compute_metrics", f"'{name}' has no evaluate counterpart")
        got = float(evaluation.metrics[name])
        if not _agree(got, float(want), tolerance):
            raise ConformanceError(
                "compute_metrics",
                f"'{name}' is {want} from compute_metrics but {got} from evaluate on the round-trip",
            )


def check_task_head(head, batch: Any, *, tolerance: float = 1e-6) -> None:
    """Check a task head on one batch on CPU.

    Set heads: ``batch`` carries ``features`` (the head's input), ``targets`` (the
    encoded target tensors ``compute_metrics`` reads) and ``sample_ids`` — a
    :class:`~soma.learn.sample_dataset.SampleBatch` does. Asserts: ``to_columns``
    emits exactly ``prediction_columns``, one numeric value per sample; a
    :class:`~soma.evaluation.Baseline` fitted on the batch's targets and the head's
    ``target_keys`` (as the pipeline fits it) yields the same columns; and
    :func:`~soma.evaluate` on the predictions read back from parquet agrees with
    ``compute_metrics`` within ``tolerance`` on every metric the head reports.

    Grid heads (``target_kind``): ``batch`` is a :class:`~soma.learn.GridBatch` whose
    ``features`` are decoder logits and whose ``targets`` hold the raw supervision
    (points inside the frame). Asserts: logits land on the target size; ``to_columns``
    holds one value per sample; ``assemble_predictions`` builds the head's noun over
    every id; and :func:`~soma.evaluate` on its disk round-trip agrees with
    ``compute_metrics``. Points are scored with greedy matching at the per-class thresholds
    :func:`~soma.evaluation.sweep_threshold` picks on the round-trip, which
    ``compute_metrics`` must report as ``threshold_class_<c>``.
    """
    if getattr(head, "target_kind", None) is not None:
        _check_grid_head(head, batch, tolerance=tolerance)
        return
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
        baseline = Baseline.fit(
            targets, task_family=head.task_family, classes=head.classes, target_keys=head.target_keys
        )
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
