"""Scoring whole-slide label maps against the slides' annotation rasters.

``soma.predict_slides`` writes one label TIFF per slide. :class:`SlideLabelMaps` is a
handle on one bundle's directory of them, and :func:`soma.evaluate` scores it against a
:class:`~soma.data.LabelMapSource` of the slides' annotation rasters.

Each slide is read one window at a time on the map's pixel grid: the map through hs2p's
``tifffile`` reader, and the annotation raster through hs2p's aligned mask reader at the
map's spacing, so a raster at another resolution lands on the map's pixels. Neither raster
is held whole. Each window adds to the slide's confusion counts, and one row of counts per
slide goes to the grid reducers, as for :class:`~soma.evaluation.LabelMapPredictions`.

The class scheme is the one each map records, which is its head's: ``values`` names each
stored value, ``classes`` lists the raw annotation values of each class, ``ignore`` the
raw values left out, and ``pixel_mapping`` is the vocabulary hs2p reads the raster
against. The rules:

- a target pixel whose raw value is in ``ignore`` is not scored;
- a target pixel of class ``c`` that the map did not predict (value 0) is a false negative
  for ``c`` and a false positive for no class;
- a map pixel that lies partly off the slide has no target pixel and is not scored.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: The default side, in map pixels, of the window both rasters are read in.
DEFAULT_WINDOW_PX = 1024
#: The name a map gives the value of the pixels no kept tile covered.
NOT_PREDICTED = "not predicted"


class SlideLabelMaps:
    """The label TIFFs ``soma.predict_slides`` wrote for one bundle, or for one ensemble.

    ``directory`` is a prediction output directory with the maps of one bundle, or one of
    its per-bundle directories; either is recognised by the output's ``index.parquet``. A
    slide is predicted when its ``<sample_id>.json`` record exists and parses.

    Pass the handle to :func:`soma.evaluate` with a :class:`~soma.data.LabelMapSource` of
    the slides' annotation rasters, aligned to the slides (built with an
    :class:`~soma.data.ImageManifest`). Only the source's rasters, slides and backends are
    used: the class scheme comes from the maps. ``window_px`` is the side of the window
    both rasters are read in, rounded down to a whole number of tile strides; it bounds
    the memory a slide takes, not the scores.
    """

    task_family = "segmentation"

    def __init__(self, directory: str | Path, *, window_px: int = DEFAULT_WINDOW_PX) -> None:
        if isinstance(window_px, bool) or not isinstance(window_px, int) or window_px < 1:
            raise ValueError(f"window_px must be a positive integer, got {window_px!r}")
        self._output_dir, self._directory = _map_directory(Path(directory))
        self._window_px = window_px
        self._records = _records(self._directory)
        self._backend, self._mask_backend = _plan_backends(self._output_dir)
        self._scheme = (
            _Scheme.of(self._map_path(next(iter(self._records)))) if self._records else None
        )

    @property
    def directory(self) -> Path:
        """The directory of the maps."""
        return self._directory

    @property
    def sample_ids(self) -> list[str]:
        """The predicted slides, in sorted order."""
        return list(self._records)

    @property
    def classes(self) -> tuple[str, ...]:
        """The class names, in class order (empty without a predicted slide)."""
        return () if self._scheme is None else self._scheme.classes

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def backend(self) -> str:
        """The hs2p backend the prediction read the slides with."""
        return self._backend

    @property
    def mask_backend(self) -> str:
        """The hs2p backend the prediction read the masks with."""
        return self._mask_backend

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"SlideLabelMaps({str(self._directory)!r}, {len(self)} slide(s), classes={list(self.classes)})"

    def _map_path(self, sample_id: str) -> Path:
        return self._directory / str(self._records[sample_id]["file"])


def slide_label_map_counts(maps: SlideLabelMaps, targets) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Each predicted slide's ``(C, 3)`` (intersection, predicted, target) counts, stacked
    in ``maps.sample_ids`` order, and its row of counts for the report."""
    from soma.data import LabelMapSource

    if not isinstance(targets, LabelMapSource):
        raise TypeError(
            "whole-slide label maps are scored against a LabelMapSource of the slides' "
            f"annotation rasters, got {type(targets).__name__}"
        )
    counts, rows = [], []
    entries = targets.entries
    for sample_id in maps.sample_ids:
        try:
            slide = _slide_counts(maps, sample_id, entries[sample_id], targets)
        except ValueError as error:
            raise ValueError(f"slide '{sample_id}': {error}") from error
        counts.append(slide.counts)
        rows.append(slide.row())
    stacked = np.stack(counts) if counts else np.zeros((0, maps.num_classes, 3), dtype=np.int64)
    return stacked, rows


# --- the directory ------------------------------------------------------------------ #


def _map_directory(path: Path) -> tuple[Path, Path]:
    """The prediction output directory and the directory of the maps ``path`` names."""
    import pandas as pd

    from soma.pipeline.slide_prediction import INDEX_FILENAME

    path = path.expanduser().resolve()
    if (path / INDEX_FILENAME).is_file():
        bundles = sorted(set(pd.read_parquet(path / INDEX_FILENAME, columns=["bundle"])["bundle"]))
        if len(bundles) != 1:
            raise ValueError(
                f"{path} holds the maps of {len(bundles)} bundles {bundles}; score one "
                "bundle's directory, such as "
                f"{path / (bundles[0] if bundles else '<bundle>')}"
            )
        return path, path / bundles[0]
    if (path.parent / INDEX_FILENAME).is_file():
        return path.parent, path
    raise ValueError(
        f"{path} is not a soma predict output: neither it nor its parent has {INDEX_FILENAME}"
    )


def _records(directory: Path) -> dict[str, dict[str, Any]]:
    """The records of the slides predicted in ``directory``, by sorted sample id."""
    from soma.pipeline.slide_prediction import _read_record

    records: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        record = _read_record(path)
        if record is not None:
            records[str(record["sample_id"])] = record
    return dict(sorted(records.items()))


def _plan_backends(output_dir: Path) -> tuple[str, str]:
    """The slide and mask backends of the prediction's ``plan.yaml``."""
    import yaml

    from soma.pipeline.slide_prediction import PLAN_FILENAME

    path = output_dir / PLAN_FILENAME
    plan = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    tiling = ((plan or {}).get("extraction") or {}).get("tiling") or {}
    return str(tiling.get("backend", "auto")), str(tiling.get("mask_backend", "auto"))


# --- the class scheme ----------------------------------------------------------------- #


@dataclass(frozen=True)
class _Scheme:
    """A map's recorded class scheme, as lookup tables, and the tiling it was cut with.

    ``target_lut`` takes a raw annotation value to its class index, ``C`` for an ignored
    value. ``map_lut`` takes a stored map value to its code: 0 for not predicted, ``k + 1``
    for class ``k``, and -1 for a value the map does not name. ``tiling`` is the tile
    size and overlap.
    """

    classes: tuple[str, ...]
    ignore: tuple[int, ...]
    pixel_mapping: Mapping[str, Any]
    target_lut: np.ndarray
    map_lut: np.ndarray
    tiling: tuple[int, float]

    @classmethod
    def of(cls, path: Path) -> "_Scheme":
        import tifffile

        from soma.data.targets import build_label_remap

        with tifffile.TiffFile(path) as tiff:
            description = tiff.pages[0].description
        try:
            metadata = json.loads(description)
        except ValueError:
            metadata = None
        if not isinstance(metadata, dict) or not all(key in metadata for key in _SCHEME_KEYS):
            raise ValueError(
                f"the label map {path} records no class scheme ({', '.join(_SCHEME_KEYS)}): "
                "it was written by an older soma predict. Predict the slides again."
            )
        classes = {str(name): value for name, value in metadata["classes"].items()}
        names = tuple(classes)
        codes = {NOT_PREDICTED: 0, **{name: k + 1 for k, name in enumerate(names)}}
        if len(codes) != len(names) + 1:
            raise ValueError(f"the label map {path} has a class named '{NOT_PREDICTED}'")
        map_lut = np.full(256, -1, dtype=np.int64)
        named = []
        for value, name in metadata["values"].items():
            if name not in codes or name in named or not 0 <= int(value) <= 255:
                raise ValueError(
                    f"the label map {path} names its values {metadata['values']}, which do not "
                    f"match its classes {list(names)}"
                )
            map_lut[int(value)] = codes[name]
            named.append(name)
        if set(named) != set(codes):
            raise ValueError(
                f"the label map {path} names no value for {sorted(set(codes) - set(named))}"
            )
        ignore = tuple(int(value) for value in metadata["ignore"])
        return cls(
            classes=names,
            ignore=ignore,
            pixel_mapping=dict(metadata["pixel_mapping"]),
            target_lut=build_label_remap(classes, ignore=ignore, ignore_index=len(names)),
            map_lut=map_lut,
            tiling=(int(metadata["tile_size_px"]), float(metadata["overlap"])),
        )

    def same_as(self, other: "_Scheme") -> bool:
        return (
            self.classes == other.classes
            and self.ignore == other.ignore
            and dict(self.pixel_mapping) == dict(other.pixel_mapping)
            and np.array_equal(self.target_lut, other.target_lut)
            and np.array_equal(self.map_lut, other.map_lut)
        )


_SCHEME_KEYS = ("values", "classes", "ignore", "pixel_mapping")


# --- one slide ------------------------------------------------------------------------ #


@dataclass(frozen=True)
class _SlideCounts:
    sample_id: str
    confusion: np.ndarray  # (C + 1, C + 1): rows target class (C = not scored), columns map code

    @property
    def counts(self) -> np.ndarray:
        """``(C, 3)`` (intersection, predicted, target) over the scored pixels."""
        num_classes = self.confusion.shape[0] - 1
        scored = self.confusion[:num_classes]
        intersection = scored[np.arange(num_classes), np.arange(1, num_classes + 1)]
        return np.stack([intersection, scored[:, 1:].sum(0), scored.sum(1)], axis=1)

    def row(self) -> dict[str, Any]:
        """The slide's report row: its annotated pixels, the share of them not predicted,
        and its per-class true positives, false positives and false negatives."""
        scored = self.confusion[:-1]
        annotated, not_predicted = int(scored.sum()), int(scored[:, 0].sum())
        row: dict[str, Any] = {
            "sample_id": self.sample_id,
            "annotated_px": annotated,
            "not_predicted_px": not_predicted,
            "not_predicted_fraction": not_predicted / annotated if annotated else 0.0,
        }
        for c, (intersection, predicted, target) in enumerate(self.counts.tolist()):
            row[f"tp_class_{c}"] = intersection
            row[f"fp_class_{c}"] = predicted - intersection
            row[f"fn_class_{c}"] = target - intersection
        return row


def _slide_counts(maps: SlideLabelMaps, sample_id: str, entry, targets) -> _SlideCounts:
    from hs2p.wsi.backends.tifffile import TifffileReader

    from soma.extract.dense.reader import _aligned_mask, _reference_geometry

    path = maps._map_path(sample_id)
    scheme = _Scheme.of(path)
    if not scheme.same_as(maps._scheme):
        raise ValueError(
            f"its map records the classes {list(scheme.classes)} (ignore {list(scheme.ignore)}), "
            f"not the classes {list(maps.classes)} (ignore {list(maps._scheme.ignore)}) of the "
            "other maps"
        )
    if entry.reference_path is None:
        raise ValueError(
            "its annotation raster has no slide to align to; build the LabelMapSource with "
            "the cohort's ImageManifest"
        )
    record = maps._records[sample_id]
    level0_spacing, slide_size = _reference_geometry(
        str(entry.reference_path),
        targets.image_backend,
        None if entry.spacing_at_level_0 is None else float(entry.spacing_at_level_0),
    )
    frame = _frame(scheme, record, level0_spacing, slide_size)
    num_classes = len(scheme.classes)
    confusion = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)
    side = frame.stride_px * max(1, maps._window_px // frame.stride_px)
    with TifffileReader(path, require_spacing=False) as reader, _aligned_mask(
        entry.label_mask_path,
        reference_path=entry.reference_path,
        reference_backend=targets.image_backend,
        spacing_at_level_0=entry.spacing_at_level_0,
        pixel_mapping=scheme.pixel_mapping,
        backend=targets.backend,
    ) as raster:
        if tuple(reader.level_dimensions[0]) != (frame.width, frame.height):
            raise ValueError(
                f"its map is {reader.level_dimensions[0]} px, not the {frame.width} x "
                f"{frame.height} px its record and slide give"
            )
        for top in range(0, frame.height, side):
            for left in range(0, frame.width, side):
                size = (min(side, frame.width - left), min(side, frame.height - top))
                codes = _map_codes(reader, scheme, (left, top), size)
                classes = _target_classes(raster, scheme, frame, (left, top), size, sample_id)
                confusion += np.bincount(
                    (classes * (num_classes + 1) + codes).ravel(), minlength=(num_classes + 1) ** 2
                ).reshape(num_classes + 1, num_classes + 1)
    return _SlideCounts(sample_id, confusion)


def _frame(scheme: _Scheme, record: Mapping[str, Any], level0_spacing: float, slide_size):
    """The slide's map frame, rebuilt from the slide and the map's tiling."""
    from soma.evaluation.whole_slide import SlideFrame, stitch_margin_px

    tile_px, overlap = scheme.tiling
    stride = tile_px - 2 * stitch_margin_px(tile_px, overlap)
    step = int(round(float(record["spacing_um"]) * stride / level0_spacing))
    frame = SlideFrame.from_grid(
        tile_px=tile_px,
        overlap=overlap,
        step_lv0=max(step, 1),
        slide_size=slide_size,
        level0_spacing_um=level0_spacing,
    )
    if (frame.width, frame.height) != (int(record["width"]), int(record["height"])):
        raise ValueError(
            f"its map is {record['width']} x {record['height']} px but its slide, at "
            f"{level0_spacing:g} um/px and {slide_size[0]} x {slide_size[1]} px, gives a "
            f"{frame.width} x {frame.height} px map: the map was predicted from another slide"
        )
    return frame


def _map_codes(reader, scheme: _Scheme, origin: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    """A window of the map as codes: 0 not predicted, ``k + 1`` class ``k``."""
    values = np.asarray(reader.read_region(origin, 0, size))
    if values.ndim == 3:
        values = values[..., 0]
    codes = scheme.map_lut[values]
    if (codes < 0).any():
        unknown = sorted(int(v) for v in np.unique(values[codes < 0]))
        raise ValueError(f"its map holds value(s) {unknown}, which its values mapping does not name")
    return codes


def _target_classes(raster, scheme: _Scheme, frame, origin, size, sample_id: str) -> np.ndarray:
    """A window of the annotation raster on the map's grid as class indices; ``C`` marks
    an ignored pixel and a pixel partly off the slide."""
    from soma.data.targets import apply_label_remap

    left, top = origin
    location = (left // frame.stride_px * frame.step_lv0, top // frame.stride_px * frame.step_lv0)
    width, height = size
    classes = np.full((height, width), len(scheme.classes), dtype=np.int64)
    region = {"location": location, "target_spacing_um": frame.spacing_um}
    inside_width, inside_height = raster.dimensions_within_canvas(**region, target_dimensions=size)
    if inside_width > 0 and inside_height > 0:
        raw = raster.read_region(**region, target_dimensions=(inside_width, inside_height)).labels
        classes[:inside_height, :inside_width] = apply_label_remap(raw, scheme.target_lut, sample_id=sample_id)
    return classes
