"""Scoring whole-slide detection points against the slides' annotated points.

``soma.predict_slides`` writes one points parquet per slide. :class:`SlidePoints` is a
handle on one bundle's directory of them, and :func:`soma.evaluate` scores it against a
:class:`~soma.data.PointSource` of the slides' annotated points (level-0 pixels).

The class scheme is the one each points file records, which is its bundle's: ``classes``
lists the annotated ids of each class, and ``drop`` the annotated ids left out. The rules:

- an annotated point whose id is in ``drop`` is removed: a prediction there is a false
  positive;
- an annotated id in neither ``classes`` nor ``drop`` is refused;
- a prediction scoring below the threshold is removed: by default the bundle's tune
  threshold, which the points file records;
- with an ignore mask, a point (annotated or predicted) on a pixel of value 255 is not
  scored. The mask is read through hs2p's aligned mask reader, at any resolution that
  covers the slide; its pixels must be 0 (annotated) or 255 (not annotated);
- predictions match annotations of their class within the bundle's matching distance,
  in µm, at each slide's own spacing.

Each slide gives one row of ``(tp, fp, fn)`` counts per class to the grid reducers, as for
:class:`~soma.evaluation.PointPredictions`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: The pixel values of an ignore mask: 0 is annotated, 255 is not.
_MASK_VALUES = {"annotated": 0, "not annotated": 255}
_IGNORED = 255


class SlidePoints:
    """The points parquets ``soma.predict_slides`` wrote for one bundle, or for one ensemble.

    ``directory`` is a prediction output directory with the points of one bundle, or one of
    its per-bundle directories; either is recognised by the output's ``index.parquet``. A
    slide is predicted when its ``<sample_id>.json`` record exists and parses.

    Pass the handle to :func:`soma.evaluate` with a :class:`~soma.data.PointSource` of the
    slides' annotated points. Only the source's points files and ignore masks are used:
    the class scheme comes from the points files. ``mask_backend`` is the hs2p reader of
    the ignore masks; by default the one the prediction's ``plan.yaml`` records.

    A bundle's points are scored at the threshold the bundle recorded on its tune split.
    An ensemble's points record none: pick one with :func:`~soma.evaluation.sweep_threshold`
    or set one with :meth:`with_threshold`.
    """

    task_family = "detection"

    def __init__(self, directory: str | Path, *, mask_backend: str | None = None) -> None:
        from soma.evaluation.slide_label_maps import _map_directory, _plan_backends, _records

        self._output_dir, self._directory = _map_directory(Path(directory), "points")
        self._records = _records(self._directory)
        self._mask_backend = str(mask_backend or _plan_backends(self._output_dir)[1])
        self._recorded: dict[str, Any] | None = None
        for sample_id in self._records:
            recorded = _recorded(self._points_path(sample_id))
            if self._recorded is None:
                self._recorded = recorded
            elif _scheme(recorded) != _scheme(self._recorded):
                raise ValueError(
                    f"slide '{sample_id}': its points record the class scheme and matching "
                    f"distance {_scheme(recorded)}, not the {_scheme(self._recorded)} of the "
                    "other slides"
                )
        self._threshold = None if self._recorded is None else self._recorded["threshold"]

    @property
    def directory(self) -> Path:
        """The directory of the points files."""
        return self._directory

    @property
    def sample_ids(self) -> list[str]:
        """The predicted slides, in sorted order."""
        return list(self._records)

    @property
    def classes(self) -> tuple[str, ...]:
        """The class names, in class order (empty without a predicted slide)."""
        return () if self._recorded is None else tuple(self._recorded["classes"])

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_scheme(self) -> dict[str, Any]:
        """``classes`` (class name → the annotated ids that form it) and ``drop`` (the
        annotated ids left out), as the points files record them."""
        if self._recorded is None:
            return {"classes": {}, "drop": []}
        return {"classes": dict(self._recorded["classes"]), "drop": list(self._recorded["drop"])}

    @property
    def match_distance_um(self) -> float | None:
        """The bundle's matching distance in µm."""
        return None if self._recorded is None else float(self._recorded["match_distance_um"])

    @property
    def threshold(self) -> float | dict[str, float] | None:
        """The score threshold the points are scored at: one value, or one per class name.
        ``None`` for an ensemble's points, until :meth:`with_threshold` sets one."""
        return self._threshold

    @property
    def mask_backend(self) -> str:
        """The hs2p backend that reads the ignore masks."""
        return self._mask_backend

    def with_threshold(self, threshold: float | Mapping[Any, float]) -> "SlidePoints":
        """The same points scored at ``threshold``: one value, or ``{class: t}`` for every
        class, keyed by index or name."""
        from soma.evaluation.grid import _class_thresholds

        cut = _class_thresholds(threshold, self.classes)
        copy = object.__new__(SlidePoints)
        copy.__dict__.update(self.__dict__)
        copy._threshold = (
            {name: float(cut[k]) for k, name in enumerate(self.classes)}
            if isinstance(threshold, Mapping)
            else float(threshold)
        )
        return copy

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"SlidePoints({str(self._directory)!r}, {len(self)} slide(s), classes={list(self.classes)})"

    def _points_path(self, sample_id: str) -> Path:
        return self._directory / str(self._records[sample_id]["file"])


def scored_slide_points(points: SlidePoints, targets, *, thresholded: bool = True):
    """``points`` and ``targets`` as :class:`~soma.evaluation.PointPredictions` and a target
    source in µm, every point on an ignored pixel and every dropped annotation removed,
    with each slide's report row (its annotated area).

    In µm, one matching distance holds for slides of any spacing. ``thresholded`` applies
    :attr:`SlidePoints.threshold`, which must then be set.
    """
    from soma.data import PointSource
    from soma.evaluation.grid import PointPredictions

    if not isinstance(targets, PointSource):
        raise TypeError(
            "whole-slide points are scored against a PointSource of the slides' annotated "
            f"points, got {type(targets).__name__}"
        )
    if thresholded and points.threshold is None:
        raise ValueError(
            f"the points in {points.directory} record no score threshold: an ensemble's mean "
            "scores were tuned by no bundle. Set one with SlidePoints.with_threshold "
            "(soma evaluate --threshold), or pick one on a tune split with sweep_threshold."
        )
    frames, annotated, rows = [], {}, []
    entries = targets.entries
    for sample_id in points.sample_ids:
        try:
            slide = _slide(points, sample_id, entries[sample_id])
        except ValueError as error:
            raise ValueError(f"slide '{sample_id}': {error}") from error
        frames.append(slide["predicted"])
        annotated[sample_id] = slide["annotated"]
        rows.append({"sample_id": sample_id, "annotated_area_mm2": slide["area_mm2"]})
    frame = pd.concat(frames, ignore_index=True) if frames else None
    predictions = PointPredictions(
        frame,
        classes=points.classes,
        sample_ids=points.sample_ids,
        match_distance_px=points.match_distance_um or 1.0,
    )
    if thresholded:
        predictions = predictions.with_threshold(points.threshold)
    return predictions, _AnnotatedPoints(annotated), rows


class _AnnotatedPoints:
    """A target source over the slides' scored annotations, in µm."""

    target_kind = "points"

    def __init__(self, annotated: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self._annotated = dict(annotated)

    @property
    def sample_ids(self) -> list[str]:
        return list(self._annotated)

    def subset(self, sample_ids: Sequence[str]) -> "_AnnotatedPoints":
        return _AnnotatedPoints({sid: self._annotated[str(sid)] for sid in sample_ids})

    def load(self, sample_id: str):
        import torch

        from soma.data import Points

        xy, classes = self._annotated[sample_id]
        return Points(xy=torch.from_numpy(xy), classes=torch.from_numpy(classes))


# --- the points files --------------------------------------------------------------------- #

_SCHEME_KEYS = ("classes", "drop")


def _recorded(path: Path) -> dict[str, Any]:
    """What a slide's points file records for scoring."""
    import pyarrow.parquet as pq

    from soma.evaluation.whole_slide_points import SLIDE_POINTS_KIND, _METADATA_KEY

    metadata = pq.read_schema(path).metadata or {}
    try:
        recorded = json.loads(metadata[_METADATA_KEY])
    except (KeyError, ValueError):
        recorded = None
    if not isinstance(recorded, dict) or recorded.get("kind") != SLIDE_POINTS_KIND:
        raise ValueError(f"{path} is not a points file of soma predict (kind {SLIDE_POINTS_KIND})")
    if not all(key in recorded for key in _SCHEME_KEYS):
        raise ValueError(
            f"the points file {path} records no class scheme ({', '.join(_SCHEME_KEYS)}): it "
            "was written by an older soma predict. Predict the slides again."
        )
    return {
        "classes": {str(name): [int(v) for v in ids] for name, ids in recorded["classes"].items()},
        "drop": [int(v) for v in recorded["drop"]],
        "match_distance_um": float(recorded["match_distance_um"]),
        "threshold": recorded.get("threshold"),
    }


def _scheme(recorded: Mapping[str, Any]) -> dict[str, Any]:
    return {key: recorded[key] for key in ("classes", "drop", "match_distance_um")}


# --- one slide ------------------------------------------------------------------------------ #


def _slide(points: SlidePoints, sample_id: str, entry) -> dict[str, Any]:
    """One slide's predictions and annotations in µm, the ignored ones removed, and its
    annotated area in mm²."""
    from soma.evaluation.detection.io import read_points

    record = points._records[sample_id]
    spacing = float(record["spacing_um"])
    size = (int(record["width"]), int(record["height"]))
    predicted = pd.read_parquet(points._points_path(sample_id), columns=["x", "y", "class", "score"])
    pred_xy = predicted[["x", "y"]].to_numpy(dtype=np.float64).reshape(-1, 2)
    gt_xy, ids = read_points(entry.points_path)
    area_mm2 = size[0] * size[1] * spacing**2 / 1e6
    if entry.ignore_mask_path is not None:
        ignored = _ignore_mask(entry.ignore_mask_path, points.mask_backend, spacing, size)
        area_mm2 *= int((~ignored).sum()) / ignored.size
        keep = ~_on_ignored(pred_xy, ignored, size)
        predicted, pred_xy = predicted[keep], pred_xy[keep]
        # Ignored annotations carry no supervision, so their ids are never checked.
        keep = ~_on_ignored(gt_xy, ignored, size)
        gt_xy, ids = gt_xy[keep], ids[keep]
    gt_xy, gt_class = _classes_of(gt_xy, ids, points.class_scheme, entry.points_path)
    frame = pd.DataFrame(
        {
            "sample_id": sample_id,
            "x": pred_xy[:, 0] * spacing,
            "y": pred_xy[:, 1] * spacing,
            "class": predicted["class"].to_numpy(dtype=np.int64),
            "score": predicted["score"].to_numpy(dtype=np.float64),
        }
    )
    return {"predicted": frame, "annotated": (gt_xy * spacing, gt_class), "area_mm2": area_mm2}


def _classes_of(xy: np.ndarray, ids: np.ndarray, scheme: Mapping[str, Any], path) -> tuple[np.ndarray, np.ndarray]:
    """The annotated points as class indices, the dropped ones removed."""
    class_of = {int(v): k for k, values in enumerate(scheme["classes"].values()) for v in values}
    dropped = set(scheme["drop"])
    undeclared = sorted({int(v) for v in ids} - class_of.keys() - dropped)
    if undeclared:
        raise ValueError(
            f"its points file {path} has annotated id(s) {undeclared}, which the points' class "
            f"scheme neither gives a class nor drops: {dict(scheme)}"
        )
    kept = np.asarray([int(v) not in dropped for v in ids], dtype=bool)
    classes = np.asarray([class_of[int(v)] for v in ids[kept]], dtype=np.int64)
    return xy[kept].reshape(-1, 2), classes


def _ignore_mask(path, backend: str, spacing: float, size: tuple[int, int]) -> np.ndarray:
    """The ignore mask over the whole slide as a boolean raster (True: not annotated).

    hs2p aligns the mask to the slide, and one mask level is read whole, every native
    pixel: the finest level within hs2p's read cap.
    """
    from hs2p import AnnotationLabels, Mask
    from hs2p.mask import MAX_MASK_READ_PX
    from hs2p.wsi.reader import open_slide

    try:
        with Mask(path=Path(path), labels=AnnotationLabels(pixel_mapping=dict(_MASK_VALUES)), backend=backend) as mask:
            aligned = mask.align_to(reference_spacing_um=spacing, reference_dimensions=size)
            with open_slide(Path(path), mask.backend, require_spacing=False) as reader:
                dimensions = [(int(w), int(h)) for w, h in reader.level_dimensions]
            level = next(
                (k for k, (w, h) in enumerate(dimensions) if w * h <= MAX_MASK_READ_PX), len(dimensions) - 1
            )
            labels = aligned.read_full(
                target_spacing_um=aligned.level_spacings_um[level], target_dimensions=dimensions[level]
            ).labels
    except ValueError as error:
        if "undeclared label IDs" in str(error):
            raise ValueError(
                f"its ignore mask {path} holds values other than 0 (annotated) and 255 (not "
                f"annotated): {error}"
            ) from None
        raise ValueError(f"its ignore mask {path} could not be read: {error}") from None
    return np.asarray(labels) == _IGNORED


def _on_ignored(xy: np.ndarray, ignored: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Which level-0 points lie on an ignored mask pixel. A point on the slide takes its
    nearest level-0 pixel, clamped to the slide (as on a tile's ignore mask), then on each
    axis the mask pixel holding it, by hs2p's rule ``pixel * mask_size // slide_size``. A
    point off the slide is on no mask pixel."""
    width, height = size
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    px = np.clip(np.rint(xy[inside, 0]).astype(np.int64), 0, width - 1)
    py = np.clip(np.rint(xy[inside, 1]).astype(np.int64), 0, height - 1)
    mask_height, mask_width = ignored.shape
    on = np.zeros(xy.shape[0], dtype=bool)
    on[inside] = ignored[py * mask_height // height, px * mask_width // width]
    return on


__all__ = ["SlidePoints", "scored_slide_points"]
