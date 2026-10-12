"""The grid shape's prediction nouns and their scoring (design §7.2).

Two frozen nouns, each carrying the full id set it was predicted over (ADR 0014: a
sample with no detected point is still a scored sample):

* :class:`PointPredictions` — a frame ``sample_id, x, y, class, score`` in level-0
  pixels, every peak above the head's ``min_score``. Thresholds are frame operations
  (:meth:`PointPredictions.with_threshold`, :func:`sweep_threshold`).
* :class:`LabelMapPredictions` — one ``(H, W)`` uint8 label map per sample, written as
  PNGs plus an ``index.parquet``.

:func:`soma.evaluate` dispatches on the noun: points are matched against a
:class:`~soma.data.PointSource` (Hungarian by default, greedy through a metric leaf's
params), label maps are counted against a :class:`~soma.data.LabelMapSource`. Both reduce
per-image ``(N, C, 3)`` counts with the metric registry's grid reducers.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

__all__ = [
    "LabelMapPredictions",
    "PointPredictions",
    "evaluate_grid",
    "label_map_counts",
    "point_counts",
    "sweep_threshold",
]

_METADATA_KEY = b"soma"
_POINT_COLUMNS = ("sample_id", "x", "y", "class", "score")
_IGNORED = 255


def _ids(values, limit: int = 10) -> str:
    values = sorted(values)
    return f"{values[:limit]}{' ...' if len(values) > limit else ''}"


# --------------------------------------------------------------------------------------
# PointPredictions
# --------------------------------------------------------------------------------------


class PointPredictions:
    """Detected points of a set of samples, in level-0 pixels.

    Args:
        frame: One row per detection: ``sample_id, x, y, class, score``. ``None`` or an
            empty frame when nothing was detected.
        classes: The name of each class index.
        sample_ids: Every sample the predictions cover, including those without a
            detection; ``evaluate`` holds them to the targets' id set.
        match_distance_px: The matching tolerance δ in level-0 pixels (the producing
            head's ``match_distance`` at its spacing).
        threshold: The score threshold already applied (informational), or ``None``.
    """

    task_family = "detection"

    def __init__(
        self,
        frame: pd.DataFrame | None,
        *,
        classes: Sequence[Any],
        sample_ids: Sequence[str],
        match_distance_px: float,
        threshold: float | Mapping[int, float] | None = None,
    ) -> None:
        if frame is None:
            frame = pd.DataFrame({column: [] for column in _POINT_COLUMNS})
        missing = [c for c in _POINT_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"point predictions need column(s) {missing}, got {list(frame.columns)}")
        frame = frame.loc[:, list(_POINT_COLUMNS)].reset_index(drop=True).copy()
        frame["sample_id"] = frame["sample_id"].astype(str)
        for column in ("x", "y", "score"):
            frame[column] = frame[column].astype(np.float64)
        frame["class"] = frame["class"].astype(np.int64)
        self._sample_ids = [str(sid) for sid in sample_ids]
        if len(set(self._sample_ids)) != len(self._sample_ids):
            raise ValueError("point predictions list a sample id more than once")
        unknown = set(frame["sample_id"]) - set(self._sample_ids)
        if unknown:
            raise ValueError(f"{len(unknown)} detection row(s) name samples outside sample_ids: {_ids(unknown)}")
        self._classes = tuple(classes)
        bad = frame["class"][(frame["class"] < 0) | (frame["class"] >= len(self._classes))]
        if len(bad):
            raise ValueError(f"class index(es) {sorted(set(bad))} outside the {len(self._classes)} classes")
        if not float(match_distance_px) > 0:
            raise ValueError(f"match_distance_px must be > 0, got {match_distance_px}")
        self._frame = frame
        self._match_distance_px = float(match_distance_px)
        self._threshold = threshold

    @property
    def frame(self) -> pd.DataFrame:
        return self._frame.copy()

    @property
    def classes(self) -> tuple[Any, ...]:
        return self._classes

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def sample_ids(self) -> list[str]:
        return list(self._sample_ids)

    @property
    def match_distance_px(self) -> float:
        return self._match_distance_px

    @property
    def threshold(self):
        return self._threshold

    def __len__(self) -> int:
        return len(self._frame)

    def __repr__(self) -> str:
        return (
            f"PointPredictions(samples={len(self._sample_ids)}, points={len(self._frame)}, "
            f"classes={list(self._classes)})"
        )

    def points(self, sample_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(xy (k, 2), class (k,), score (k,))`` of one sample."""
        rows = self._frame[self._frame["sample_id"] == str(sample_id)]
        return (
            rows[["x", "y"]].to_numpy(dtype=np.float64).reshape(-1, 2),
            rows["class"].to_numpy(dtype=np.int64),
            rows["score"].to_numpy(dtype=np.float64),
        )

    def _class_thresholds(self, threshold) -> np.ndarray:
        if isinstance(threshold, Mapping):
            names = {str(name): index for index, name in enumerate(self._classes)}
            values = np.full(self.num_classes, -np.inf)
            for key, value in threshold.items():
                index = key if isinstance(key, (int, np.integer)) else names.get(key)
                if index is None or not 0 <= int(index) < self.num_classes:
                    raise ValueError(f"threshold names unknown class {key!r}; classes are {list(self._classes)}")
                values[int(index)] = float(value)
            unset = [self._classes[i] for i in range(self.num_classes) if values[i] == -np.inf]
            if unset:
                raise ValueError(f"a per-class threshold needs every class; missing {unset}")
            return values
        return np.full(self.num_classes, float(threshold))

    def with_threshold(self, threshold: float | Mapping[Any, float]) -> "PointPredictions":
        """Keep the points scoring at least ``threshold`` (a scalar, or ``{class: t}`` for
        every class, keyed by index or name). The id set is unchanged."""
        cut = self._class_thresholds(threshold)
        keep = self._frame["score"].to_numpy() >= cut[self._frame["class"].to_numpy()]
        if isinstance(threshold, Mapping):
            recorded: Any = {int(i): float(cut[i]) for i in range(self.num_classes)}
        else:
            recorded = float(threshold)
        return PointPredictions(
            self._frame[keep],
            classes=self._classes,
            sample_ids=self._sample_ids,
            match_distance_px=self._match_distance_px,
            threshold=recorded,
        )

    def to_parquet(self, path: str | Path) -> Path:
        """Write the frame with the classes, id set, δ and threshold in the schema metadata."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        from soma import __version__

        path = Path(path)
        table = pa.Table.from_pandas(self._frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        threshold = self._threshold
        if isinstance(threshold, Mapping):
            threshold = {str(k): float(v) for k, v in threshold.items()}
        metadata[_METADATA_KEY] = json.dumps(
            {
                "kind": "points",
                "task_family": self.task_family,
                "soma_version": __version__,
                "classes": list(self._classes),
                "sample_ids": self._sample_ids,
                "match_distance_px": self._match_distance_px,
                "threshold": threshold,
            }
        ).encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), path)
        return path

    @classmethod
    def from_parquet(cls, path: str | Path) -> "PointPredictions":
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        raw = (table.schema.metadata or {}).get(_METADATA_KEY)
        meta = json.loads(raw) if raw is not None else {}
        if meta.get("kind") != "points":
            raise ValueError(f"{path} holds no soma point predictions (no 'points' metadata)")
        threshold = meta.get("threshold")
        if isinstance(threshold, dict):
            threshold = {int(k): float(v) for k, v in threshold.items()}
        return cls(
            table.to_pandas(),
            classes=meta["classes"],
            sample_ids=meta["sample_ids"],
            match_distance_px=meta["match_distance_px"],
            threshold=threshold,
        )


# --------------------------------------------------------------------------------------
# LabelMapPredictions
# --------------------------------------------------------------------------------------


def _geometry_dict(geometry) -> dict[str, Any] | None:
    if geometry is None:
        return None
    return {key: list(value) if isinstance(value, tuple) else value for key, value in dataclasses.asdict(geometry).items()}


def _geometry_from_dict(data: dict[str, Any] | None):
    if data is None:
        return None
    from soma.data.geometry import GridGeometry

    return GridGeometry(**{key: tuple(value) if isinstance(value, list) else value for key, value in data.items()})


class LabelMapPredictions:
    """One predicted ``(H, W)`` uint8 label map per sample, held in memory.

    Args:
        label_maps: ``{sample_id: (H, W) uint8}``.
        classes: The name of each class index.
        ignore_index: The value targets use for unsupervised pixels.
        geometry: The producing head's :class:`~soma.data.GridGeometry`, if any.
    """

    task_family = "segmentation"

    def __init__(
        self,
        label_maps: Mapping[str, Any],
        *,
        classes: Sequence[Any],
        ignore_index: int = 255,
        geometry=None,
    ) -> None:
        maps: dict[str, np.ndarray] = {}
        for sample_id, value in label_maps.items():
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.ndim != 2:
                raise ValueError(f"label map for '{sample_id}' must be (H, W), got {array.shape}")
            if array.size and (int(array.min()) < 0 or int(array.max()) > 255):
                raise ValueError(f"label map for '{sample_id}' holds values outside uint8")
            maps[str(sample_id)] = np.ascontiguousarray(array.astype(np.uint8))
        self._maps = maps
        self._classes = tuple(classes)
        self._ignore_index = int(ignore_index)
        self._geometry = geometry

    @property
    def sample_ids(self) -> list[str]:
        return list(self._maps)

    @property
    def label_maps(self) -> dict[str, np.ndarray]:
        return dict(self._maps)

    @property
    def classes(self) -> tuple[Any, ...]:
        return self._classes

    @property
    def num_classes(self) -> int:
        return len(self._classes)

    @property
    def ignore_index(self) -> int:
        return self._ignore_index

    @property
    def geometry(self):
        return self._geometry

    def __getitem__(self, sample_id: str) -> np.ndarray:
        return self._maps[str(sample_id)]

    def __len__(self) -> int:
        return len(self._maps)

    def __repr__(self) -> str:
        return f"LabelMapPredictions(samples={len(self._maps)}, classes={list(self._classes)})"

    def save(self, directory: str | Path) -> Path:
        """Write ``<sample_id>.png`` per sample and ``index.parquet`` (ids, files, sizes,
        with the classes, ignore index and geometry in the schema metadata)."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        from PIL import Image

        from soma import __version__

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for sample_id, array in self._maps.items():
            name = f"{sample_id}.png"
            Image.fromarray(array, mode="L").save(directory / name)
            rows.append({"sample_id": sample_id, "file": name, "height": array.shape[0], "width": array.shape[1]})
        frame = pd.DataFrame(rows, columns=["sample_id", "file", "height", "width"])
        table = pa.Table.from_pandas(frame, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata[_METADATA_KEY] = json.dumps(
            {
                "kind": "label_maps",
                "task_family": self.task_family,
                "soma_version": __version__,
                "classes": list(self._classes),
                "ignore_index": self._ignore_index,
                "geometry": _geometry_dict(self._geometry),
            }
        ).encode("utf-8")
        pq.write_table(table.replace_schema_metadata(metadata), directory / "index.parquet")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> "LabelMapPredictions":
        import pyarrow.parquet as pq
        from PIL import Image

        directory = Path(directory)
        table = pq.read_table(directory / "index.parquet")
        raw = (table.schema.metadata or {}).get(_METADATA_KEY)
        meta = json.loads(raw) if raw is not None else {}
        if meta.get("kind") != "label_maps":
            raise ValueError(f"{directory} holds no soma label-map predictions")
        maps = {}
        for row in table.to_pandas().itertuples(index=False):
            with Image.open(directory / row.file) as image:
                maps[str(row.sample_id)] = np.array(image)
        return cls(
            maps,
            classes=meta["classes"],
            ignore_index=meta["ignore_index"],
            geometry=_geometry_from_dict(meta.get("geometry")),
        )


# --------------------------------------------------------------------------------------
# Counts
# --------------------------------------------------------------------------------------


def _check_label_range(
    pred: np.ndarray, true: np.ndarray, *, num_classes: int, sample_id: str | None = None
) -> None:
    """Refuse a predicted or target label on a supervised pixel outside
    ``[0, num_classes)``: dropping or binning it would score a wrong class mapping as
    perfect. ``sample_id``, when given, is named in the message."""
    undeclared = [
        f"{side} {sorted(np.unique(values[outside]).tolist())}"
        for side, values in (("predicted", pred), ("target", true))
        if (outside := (values < 0) | (values >= num_classes)).any()
    ]
    if undeclared:
        where = "" if sample_id is None else f"label map for '{sample_id}': "
        raise ValueError(
            f"{where}{' and '.join(undeclared)} label value(s) on supervised pixels lie "
            f"outside [0, num_classes={num_classes})"
        )


def label_map_counts(prediction: np.ndarray, target: Any, *, num_classes: int, ignore_index: int) -> np.ndarray:
    """``(C, 3)`` ``(intersection, predicted, target)`` pixel counts of one label map,
    ``ignore_index`` target pixels excluded.

    Raises ``ValueError`` when a predicted or target label on a supervised pixel lies
    outside ``[0, num_classes)``: dropping it would score a wrong class mapping as perfect.
    """
    target = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
    prediction = np.asarray(prediction).astype(np.int64)
    target = target.astype(np.int64)
    valid = target != int(ignore_index)
    pred = prediction[valid]
    true = target[valid]
    _check_label_range(pred, true, num_classes=num_classes)
    counts = np.zeros((int(num_classes), 3), dtype=np.int64)
    counts[:, 1] = np.bincount(pred, minlength=num_classes)
    counts[:, 2] = np.bincount(true, minlength=num_classes)
    counts[:, 0] = np.bincount(pred[pred == true], minlength=num_classes)
    return counts


def _loaded_points(targets, sample_ids: Sequence[str]) -> dict[str, tuple]:
    """``{sid: (gt_xy, gt_class, ignored raster | None)}`` read once from ``targets``."""
    loaded = {}
    for sample_id in sample_ids:
        points = targets.load(sample_id)
        if not hasattr(points, "xy") or not hasattr(points, "classes"):
            raise ValueError(
                f"target source returned {type(points).__name__} for '{sample_id}'; point "
                "predictions are scored against a PointSource"
            )
        xy = np.asarray(points.xy.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 2)
        classes = np.asarray(points.classes.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
        ignored = None if points.ignore_mask is None else np.asarray(points.ignore_mask) == _IGNORED
        loaded[str(sample_id)] = (xy, classes, ignored)
    return loaded


def _scored_points(predictions: "PointPredictions", loaded: Mapping[str, tuple]):
    """Per sample: predictions and ground truth after dropping both on ignored pixels."""
    from soma.detection.encode import points_on_valid

    grouped = {sid: rows for sid, rows in predictions.frame.groupby("sample_id", sort=False)}
    out = []
    for sample_id in predictions.sample_ids:
        gt_xy, gt_cls, ignored = loaded[sample_id]
        rows = grouped.get(sample_id)
        if rows is None:
            pred_xy = np.zeros((0, 2)); pred_cls = np.zeros(0, dtype=np.int64); pred_score = np.zeros(0)
        else:
            pred_xy = rows[["x", "y"]].to_numpy(dtype=np.float64)
            pred_cls = rows["class"].to_numpy(dtype=np.int64)
            pred_score = rows["score"].to_numpy(dtype=np.float64)
        if ignored is not None:
            if gt_xy.shape[0]:
                keep = ~points_on_valid(gt_xy, ignored)
                gt_xy, gt_cls = gt_xy[keep], gt_cls[keep]
            if pred_xy.shape[0]:
                keep = ~points_on_valid(pred_xy, ignored)
                pred_xy, pred_cls, pred_score = pred_xy[keep], pred_cls[keep], pred_score[keep]
        bad = sorted({int(c) for c in gt_cls if not 0 <= int(c) < predictions.num_classes})
        if bad:
            raise ValueError(
                f"points of '{sample_id}' carry class id(s) {bad} outside the predictions' "
                f"{predictions.num_classes} classes; remap them through PointSource(class_remap=...)"
            )
        out.append((pred_xy, pred_cls, pred_score, gt_xy, gt_cls))
    return out


def point_counts(
    predictions: "PointPredictions",
    targets,
    *,
    matching: str = "hungarian",
    match_distance: float | None = None,
    _loaded: Mapping[str, tuple] | None = None,
) -> torch.Tensor:
    """Per-image ``(N, C, 3)`` TP/FP/FN of class-aware point matching.

    ``match_distance`` overrides the predictions' ``match_distance_px`` (level-0 pixels).
    Points (predicted or annotated) on an ignored pixel of the target's ignore mask are
    left out.
    """
    from soma.detection.matching import match_points

    delta = predictions.match_distance_px if match_distance is None else float(match_distance)
    loaded = _loaded if _loaded is not None else _loaded_points(targets, predictions.sample_ids)
    rows = [
        match_points(
            pxy, pcls, pscore, gxy, gcls,
            num_classes=predictions.num_classes, delta=delta, method=matching,
        )
        for pxy, pcls, pscore, gxy, gcls in _scored_points(predictions, loaded)
    ]
    if not rows:
        return torch.zeros(0, predictions.num_classes, 3, dtype=torch.long)
    return torch.from_numpy(np.stack(rows)).to(torch.long)


# --------------------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------------------


def _check_ids(predictions, targets) -> None:
    predicted, expected = set(predictions.sample_ids), set(map(str, targets.sample_ids))
    unknown = predicted - expected
    if unknown:
        raise ValueError(
            f"{len(unknown)} prediction id(s) have no target: {_ids(unknown)}. evaluate scores "
            "exactly the ids it is given; pass the targets of every predicted sample."
        )
    missing = expected - predicted
    if missing:
        raise ValueError(
            f"{len(missing)} target(s) have no prediction: {_ids(missing)}. Restrict the "
            "target source to the predicted ids first (targets.subset(ids))."
        )


def _resolve_grid_metrics(metrics, family: str) -> list[tuple[str, Callable, dict[str, Any]]]:
    """``(name, reducer, params)`` per requested metric of a grid ``family``."""
    from soma.evaluation.metrics import DEFAULT_METRICS, metric_registry

    if metrics is None or (isinstance(metrics, str) and metrics == "default") or (
        not isinstance(metrics, str) and len(metrics) == 0
    ):
        metrics = DEFAULT_METRICS[family]
    elif isinstance(metrics, (str, Mapping)) or callable(metrics):
        metrics = [metrics]
    resolved = []
    for entry in metrics:
        params: dict[str, Any] = {}
        if isinstance(entry, Mapping):
            if "name" not in entry:
                raise ValueError(f"a metric leaf is {{name, params}}, got {entry!r}")
            params = dict(entry.get("params") or {})
            entry = entry["name"]
        if callable(entry):
            resolved.append((getattr(entry, "__name__", repr(entry)), entry, params))
            continue
        name = str(entry)
        if name in metric_registry:
            spec = metric_registry.get(name)
            if family not in spec.families or spec.fn is None:
                valid = sorted(n for n in metric_registry.names(family) if metric_registry.get(n).fn)
                raise ValueError(f"metric '{name}' does not apply to {family} predictions; valid metrics: {valid}")
            resolved.append((name, spec.fn, params))
            continue
        if "." in name:
            module_name, _, attr = name.rpartition(".")
            try:
                fn = getattr(importlib.import_module(module_name), attr)
            except (ImportError, AttributeError) as error:
                raise ValueError(f"cannot import metric '{name}': {error}") from error
            resolved.append((attr, fn, params))
            continue
        valid = sorted(n for n in metric_registry.names(family) if metric_registry.get(n).fn)
        raise ValueError(f"unknown metric '{name}'; valid metrics for {family}: {valid}")
    return resolved


_MATCH_PARAMS = ("matching", "match_distance")


def _reduce(resolved, counts_for: Callable[[dict[str, Any]], torch.Tensor], num_classes: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, fn, params in resolved:
        match = {key: params[key] for key in _MATCH_PARAMS if key in params}
        rest = {key: value for key, value in params.items() if key not in _MATCH_PARAMS}
        result = fn(counts_for(match), num_classes=num_classes, **rest)
        if isinstance(result, Mapping):
            values.update({str(k): float(v) for k, v in result.items()})
        else:
            values[name] = float(result)
    return values


def evaluate_grid(predictions, targets, metrics="default", subgroups: Sequence[str] = ()):
    """Score :class:`PointPredictions` or :class:`LabelMapPredictions` (see :func:`soma.evaluate`)."""
    from soma.evaluation.scoring import Evaluation

    if subgroups:
        raise ValueError(
            "subgroups break a targets frame down by metadata columns; grid predictions are "
            "scored against a target source, so score each group's subset instead"
        )
    _check_ids(predictions, targets)
    family = predictions.task_family
    resolved = _resolve_grid_metrics(metrics, family)
    if isinstance(predictions, PointPredictions):
        loaded = _loaded_points(targets, predictions.sample_ids)
        cache: dict[tuple, torch.Tensor] = {}

        def counts_for(match: dict[str, Any]) -> torch.Tensor:
            key = (match.get("matching", "hungarian"), match.get("match_distance"))
            if key not in cache:
                cache[key] = point_counts(
                    predictions, targets, matching=key[0], match_distance=key[1], _loaded=loaded
                )
            return cache[key]
    elif isinstance(predictions, LabelMapPredictions):
        if any(params for _, _, params in resolved if any(k in params for k in _MATCH_PARAMS)):
            raise ValueError("matching / match_distance apply to point predictions only")
        rows = []
        for sample_id in predictions.sample_ids:
            target = targets.load(sample_id)
            prediction = predictions[sample_id]
            if tuple(prediction.shape) != tuple(target.shape):
                raise ValueError(
                    f"label map for '{sample_id}' is {tuple(prediction.shape)} but its target is "
                    f"{tuple(target.shape)}"
                )
            try:
                rows.append(
                    label_map_counts(
                        prediction, target, num_classes=predictions.num_classes,
                        ignore_index=predictions.ignore_index,
                    )
                )
            except ValueError as error:
                raise ValueError(f"label map for '{sample_id}': {error}") from error
        counts = (
            torch.from_numpy(np.stack(rows)).to(torch.long)
            if rows
            else torch.zeros(0, predictions.num_classes, 3, dtype=torch.long)
        )

        def counts_for(match: dict[str, Any]) -> torch.Tensor:
            return counts
    else:  # pragma: no cover - dispatch guards the type
        raise TypeError(f"not a grid prediction noun: {type(predictions).__name__}")
    values = _reduce(resolved, counts_for, predictions.num_classes) if predictions.sample_ids else {}
    return Evaluation(metrics=values, task_family=family, num_samples=len(predictions.sample_ids))


# --------------------------------------------------------------------------------------
# sweep_threshold
# --------------------------------------------------------------------------------------


class _LoadedPoints:
    """A target source over already loaded points, so a sweep reads each file once."""

    target_kind = "points"

    def __init__(self, targets, sample_ids: Sequence[str]) -> None:
        from soma.data import Points

        self._points = {sid: targets.load(sid) for sid in sample_ids}
        self._type = Points

    @property
    def sample_ids(self) -> list[str]:
        return list(self._points)

    def load(self, sample_id: str):
        return self._points[sample_id]


def _candidate_grid(scores: np.ndarray, num: int = 50) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([0.0])
    quantiles = np.linspace(0.0, 1.0, num)
    lo, hi = float(scores.min()), float(scores.max())
    grid = np.unique(np.concatenate([np.quantile(scores, quantiles), np.linspace(lo, hi, num)]))
    return np.concatenate([[lo - 1e-6], grid, [hi + 1e-6]])


def sweep_threshold(
    predictions: PointPredictions,
    targets,
    metric: str = "mean_f1",
    grid: Sequence[float] | None = None,
    *,
    per_class: bool = False,
    matching: str = "hungarian",
    match_distance: float | None = None,
):
    """Pick the score threshold that maximises ``metric`` on ``targets``.

    Returns ``(threshold, Evaluation)`` with the evaluation of the thresholded
    predictions. ``per_class=True`` picks one threshold per class (returned as
    ``{class_index: t}``) maximising that class's F1, which also maximises ``mean_f1``
    (class-aware matching makes each class's F1 depend on its own points only); it
    supports ``metric="mean_f1"``. ``grid`` lists the candidate thresholds; by default
    the candidates are score quantiles and a uniform grid over the observed scores,
    plus (per class) quantiles of the scores of points within δ of a same-class
    annotation, bracketed by keep-all and drop-all. Ties keep the higher threshold.
    """
    from soma.detection.matching import sweep_score_thresholds
    from soma.evaluation.metrics import metric_registry

    if not isinstance(predictions, PointPredictions):
        raise TypeError("sweep_threshold applies to PointPredictions")
    _check_ids(predictions, targets)
    cached = _LoadedPoints(targets, predictions.sample_ids)
    leaf = {"name": metric, "params": {"matching": matching, **({} if match_distance is None else {"match_distance": match_distance})}}
    if per_class:
        if metric != "mean_f1":
            raise ValueError("per_class=True sweeps per-class F1, which maximises metric='mean_f1' only")
        loaded = _loaded_points(cached, predictions.sample_ids)
        scored = _scored_points(predictions, loaded)
        delta = predictions.match_distance_px if match_distance is None else float(match_distance)
        if grid is None:
            values = sweep_score_thresholds(
                [s[0] for s in scored], [s[1] for s in scored], [s[2] for s in scored],
                [s[3] for s in scored], [s[4] for s in scored],
                num_classes=predictions.num_classes, delta=delta, method=matching,
            )
            best: Any = {index: float(value) for index, value in enumerate(values)}
        else:
            best = {}
            for c in range(predictions.num_classes):
                scores = []
                for t in grid:
                    ev = evaluate_grid(
                        predictions.with_threshold({i: (t if i == c else 0.0) for i in range(predictions.num_classes)}),
                        cached,
                        metrics=[{"name": "f1_per_class", "params": leaf["params"]}],
                    )
                    scores.append((ev.metrics[f"f1_class_{c}"], float(t)))
                best[c] = max(scores)[1]
        return best, evaluate_grid(predictions.with_threshold(best), cached, metrics=[leaf])
    candidates = (
        np.asarray(sorted(float(t) for t in grid))
        if grid is not None
        else _candidate_grid(predictions.frame["score"].to_numpy(dtype=np.float64))
    )
    higher = metric_registry.higher_is_better(metric)
    best_value, best_threshold, best_evaluation = None, None, None
    for t in candidates:
        evaluation = evaluate_grid(predictions.with_threshold(float(t)), cached, metrics=[leaf])
        value = evaluation.metrics[metric]
        better = best_value is None or (value >= best_value if higher else value <= best_value)
        if better:
            best_value, best_threshold, best_evaluation = value, float(t), evaluation
    return best_threshold, best_evaluation
