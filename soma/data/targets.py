"""Target sources: dense supervision in physical units (design §4.2).

A :class:`LabelMapSource` serves one ``(H, W)`` long label map per sample with the class
remap (``task.params.classes`` / ``ignore``) and the ignore index already folded in; a
:class:`CachedLabelMapSource` does the same from the ROI mask crops that annotation
sampling stored; a :class:`PointSource` serves level-0 :class:`~soma.data.Points`. None
of them knows about token grids: the learner maps targets onto the grid with the
feature source's geometry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from soma.class_scheme import assign_raw_values, resolve_classes
from soma.data.manifests import AnnotationManifest, ImageManifest
from soma.data.sources import Points

__all__ = [
    "CLASS_SCHEME_KEYS",
    "CachedLabelMapSource",
    "LabelMapEntry",
    "LabelMapSource",
    "PointEntry",
    "PointSource",
    "UNDECLARED_LABEL",
    "apply_label_remap",
    "build_label_remap",
    "resolve_class_scheme",
]

#: LUT entry for a raw pixel value that is in neither ``classes`` nor ``ignore``. Outside
#: every valid class index and every ``ignore_index``, so the segmentation head can fail
#: loud on it instead of silently dropping (or aliasing) a class's supervision.
UNDECLARED_LABEL = int(np.iinfo(np.int64).min)

#: ``task.params`` keys that define the class scheme rather than configure the head.
CLASS_SCHEME_KEYS = ("num_classes", "classes", "ignore")


def _label_lut(class_of: Mapping[int, int], ignore: Sequence[int], ignore_index: int) -> np.ndarray:
    lut = np.full(256, UNDECLARED_LABEL, dtype=np.int64)
    for value, class_index in class_of.items():
        lut[value] = class_index
    lut[list(ignore)] = int(ignore_index)
    return lut


def build_label_remap(
    classes: Mapping[str, Any],
    *,
    ignore: Sequence[int] = (),
    ignore_index: int = 255,
) -> np.ndarray:
    """Build the raw-pixel → class-index lookup table from ``task.params.classes``/``ignore``.

    Annotation rasters carry the *dataset's own* pixel vocabulary (e.g. BEETLE's
    ``{0 unannotated, 1 other, 2 non-invasive, 3 invasive, 4 necrosis}``); the segmentation
    head needs contiguous class indices ``[0, num_classes)``. ``classes`` maps each class
    name to the raw value(s) that form it — several values merge into one class — and the
    class index is the declaration order. ``ignore`` lists the raw values excluded from
    loss and metrics (they map to ``ignore_index``). The rules are those of
    :mod:`soma.class_scheme`, with raw values capped at a single byte.

    Every value declared nowhere maps to :data:`UNDECLARED_LABEL`.

    Returns the 256-entry LUT (indexable by a raw uint8/int mask).
    """
    class_of, ignored = assign_raw_values(
        classes, excluded=ignore, excluded_name="ignore", max_value=255
    )
    return _label_lut(class_of, ignored, ignore_index)


def resolve_class_scheme(
    task_params: Mapping[str, Any], *, annotation_rasters: bool
) -> tuple[int, tuple[str, ...], np.ndarray | None]:
    """Resolve ``(num_classes, class names, raw-pixel → class-index LUT)`` for a run.

    ``task.params.classes`` and ``task.params.ignore`` define the training targets (see
    :func:`build_label_remap`). They are independent of
    ``preprocessing.masks.pixel_mapping``, which only governs ROI sampling. Without
    ``classes`` the masks must already hold contiguous class indices (LUT ``None``) — only
    possible for pre-cropped tiles, since ``annotation_rasters`` carry the dataset's own
    values.
    """
    if annotation_rasters and task_params.get("classes") is None:
        raise ValueError(
            "segmentation from annotation rasters (preprocessing.masks) requires "
            "task.params.classes: name each class and the raw mask value(s) that form "
            "it, e.g. classes: {tumor: [1, 2], stroma: [3]} with ignore: [0] for the "
            "values to exclude from loss and metrics."
        )
    num_classes, names, class_of, ignored = resolve_classes(
        task_params, excluded_key="ignore", subject="segmentation", max_value=255
    )
    if class_of is None:
        return num_classes, names, None
    return num_classes, names, _label_lut(
        class_of, ignored, int(task_params.get("ignore_index", 255))
    )


def apply_label_remap(array: np.ndarray, label_remap: np.ndarray, *, sample_id: str) -> np.ndarray:
    """Map a raw annotation raster onto class indices (+ ``ignore_index``) through the LUT.

    ``label_remap`` comes from :func:`resolve_class_scheme`. Fails, naming the sample, on
    a value outside the single byte the LUT covers and on a raw value declared in neither
    ``task.params.classes`` nor ``ignore``. Every path that turns a mask into targets
    goes through here, so they cannot diverge.
    """
    array = np.asarray(array, dtype=np.int64)
    if int(array.max(initial=0)) > 255 or int(array.min(initial=0)) < 0:
        raise ValueError(
            f"mask for '{sample_id}' has raw pixel value(s) outside [0, 255]; "
            "the label remap LUT only covers single-byte annotation rasters."
        )
    remapped = label_remap[array]
    undeclared = remapped == UNDECLARED_LABEL
    if undeclared.any():
        raise ValueError(
            f"mask for '{sample_id}' has raw value(s) "
            f"{sorted(int(v) for v in np.unique(array[undeclared]))} declared in "
            "neither task.params.classes nor task.params.ignore."
        )
    return remapped


def _check_remap(label_remap) -> np.ndarray | None:
    if label_remap is None:
        return None
    label_remap = np.asarray(label_remap)
    if label_remap.shape != (256,):
        raise ValueError(f"label_remap must be a length-256 LUT, got shape {label_remap.shape}.")
    return label_remap


def _unknown(sample_id: str, known) -> KeyError:
    return KeyError(f"Sample '{sample_id}' not found in target source. Available: {sorted(known)}")


@dataclass(frozen=True)
class LabelMapEntry:
    """One sample's annotation raster and the image it registers against."""

    sample_id: str
    label_mask_path: Path
    reference_path: Path | None = None
    spacing_at_level_0: float | None = None


class LabelMapSource:
    """Label maps read from annotation rasters at a requested spacing.

    The reader routes by format: flat PNG/JPEG (or no spacing) through PIL with spacing
    ignored; pyramidal, spacing-bearing masks through hs2p at ``spacing_um``, aligned to
    ``reference_path`` and read on ``size``. ``label_remap`` (from
    :func:`resolve_class_scheme`) folds the dataset's raw vocabulary onto class indices.
    """

    def __init__(
        self,
        entries: Mapping[str, LabelMapEntry] | Sequence[LabelMapEntry],
        *,
        size: tuple[int, int],
        mask_vocabulary: Mapping[str, int | list[int]],
        label_remap: np.ndarray | None = None,
        spacing_um: float | None = None,
        spacing_policy: str = "strict",
        tolerance: float = 0.05,
        backend: str = "auto",
        image_backend: str = "auto",
    ) -> None:
        if not isinstance(entries, Mapping):
            entries = {entry.sample_id: entry for entry in entries}
        self._entries: dict[str, LabelMapEntry] = dict(entries)
        self._height, self._width = (int(v) for v in size)
        self._mask_vocabulary = dict(mask_vocabulary)
        self._label_remap = _check_remap(label_remap)
        self._spacing_um = None if spacing_um is None else float(spacing_um)
        self._spacing_policy = spacing_policy
        self._tolerance = float(tolerance)
        self._backend = backend
        self._image_backend = image_backend

    @classmethod
    def from_manifests(
        cls,
        annotations: AnnotationManifest,
        images: ImageManifest | None = None,
        **params: Any,
    ) -> "LabelMapSource":
        entries = []
        for entry in annotations:
            if entry.label_mask_path is None:
                raise ValueError(f"sample '{entry.sample_id}' has no label_mask_path")
            image = images[entry.sample_id] if images is not None else None
            entries.append(
                LabelMapEntry(
                    sample_id=entry.sample_id,
                    label_mask_path=entry.label_mask_path,
                    reference_path=None if image is None else image.image_path,
                    spacing_at_level_0=None if image is None else image.spacing_at_level_0,
                )
            )
        return cls(entries, **params)

    @property
    def sample_ids(self) -> list[str]:
        return list(self._entries)

    @property
    def entries(self) -> dict[str, LabelMapEntry]:
        return dict(self._entries)

    def load(self, sample_id: str) -> Tensor:
        from soma.dense.reader import read_mask_at_spacing
        from soma.spacing import resolve_effective_spacing_um

        entry = self._entries.get(sample_id)
        if entry is None:
            raise _unknown(sample_id, self._entries)
        try:
            array = read_mask_at_spacing(
                entry.label_mask_path,
                spacing_um=(
                    resolve_effective_spacing_um(
                        requested_spacing_um=self._spacing_um,
                        spacing_at_level_0=entry.spacing_at_level_0,
                        tolerance=self._tolerance,
                        policy=self._spacing_policy,
                    )
                    if self._spacing_um is not None
                    else None
                ),
                size=(self._width, self._height),
                reference_path=entry.reference_path,
                reference_backend=self._image_backend,
                spacing_at_level_0=entry.spacing_at_level_0,
                pixel_mapping=self._mask_vocabulary,
                backend=self._backend,
            )
        except ValueError as error:
            raise ValueError(f"segmentation sample '{sample_id}': {error}") from error
        array = np.ascontiguousarray(array).astype(np.int64)
        if self._label_remap is not None:
            array = apply_label_remap(array, self._label_remap, sample_id=sample_id)
        return torch.from_numpy(np.ascontiguousarray(array).astype(np.int64))

    def __len__(self) -> int:
        return len(self._entries)


class CachedLabelMapSource:
    """Label maps from the ROI mask crops that annotation sampling stored.

    Each crop is the in-slide window of the whole-slide raster, read at the spacing its
    grid was read at. An edge ROI overhanging the slide stores only its in-slide part;
    beyond the slide there is nothing to learn from, so those pixels get ``ignore_index``
    and are never remapped.
    """

    def __init__(
        self,
        crops: Mapping[str, Path | str],
        *,
        size: tuple[int, int],
        label_remap: np.ndarray | None = None,
        ignore_index: int = 255,
    ) -> None:
        self._crops = {str(sample_id): Path(path) for sample_id, path in crops.items()}
        self._height, self._width = (int(v) for v in size)
        self._label_remap = _check_remap(label_remap)
        self._ignore_index = int(ignore_index)

    @property
    def sample_ids(self) -> list[str]:
        return list(self._crops)

    @property
    def crops(self) -> dict[str, Path]:
        return dict(self._crops)

    def load(self, sample_id: str) -> Tensor:
        from soma.dense.reader import read_mask_crop

        path = self._crops.get(sample_id)
        if path is None:
            raise _unknown(sample_id, self._crops)
        try:
            array, inside = read_mask_crop(path, size=(self._width, self._height))
        except ValueError as error:
            raise ValueError(f"segmentation sample '{sample_id}': {error}") from error
        array = np.ascontiguousarray(array).astype(np.int64)
        in_slide = array if inside is None else array[inside]
        if self._label_remap is not None:
            in_slide = apply_label_remap(in_slide, self._label_remap, sample_id=sample_id)
        if inside is None:
            array = in_slide
        else:
            array = np.full(array.shape, self._ignore_index, dtype=np.int64)
            array[inside] = in_slide
        return torch.from_numpy(np.ascontiguousarray(array).astype(np.int64))

    def __len__(self) -> int:
        return len(self._crops)


@dataclass(frozen=True)
class PointEntry:
    sample_id: str
    points_path: Path
    ignore_mask_path: Path | None = None


class PointSource:
    """Level-0 point annotations (``x,y,class`` files) with their optional ignore rasters."""

    def __init__(self, entries: Mapping[str, PointEntry] | Sequence[PointEntry]) -> None:
        if not isinstance(entries, Mapping):
            entries = {entry.sample_id: entry for entry in entries}
        self._entries: dict[str, PointEntry] = dict(entries)

    @classmethod
    def from_manifest(cls, annotations: AnnotationManifest) -> "PointSource":
        entries = []
        for entry in annotations:
            if entry.points_path is None:
                raise ValueError(f"sample '{entry.sample_id}' has no points_path")
            entries.append(
                PointEntry(
                    sample_id=entry.sample_id,
                    points_path=entry.points_path,
                    ignore_mask_path=entry.ignore_mask_path,
                )
            )
        return cls(entries)

    @property
    def sample_ids(self) -> list[str]:
        return list(self._entries)

    @property
    def entries(self) -> dict[str, PointEntry]:
        return dict(self._entries)

    def load(self, sample_id: str) -> Points:
        from soma.detection.io import read_ignore_mask, read_points

        entry = self._entries.get(sample_id)
        if entry is None:
            raise _unknown(sample_id, self._entries)
        try:
            xy, classes = read_points(entry.points_path)
            ignore_mask = (
                None if entry.ignore_mask_path is None else read_ignore_mask(entry.ignore_mask_path)
            )
        except ValueError as error:
            raise ValueError(f"detection sample '{sample_id}': {error}") from error
        return Points(
            xy=torch.as_tensor(np.asarray(xy, dtype=np.float64)).reshape(-1, 2),
            classes=torch.as_tensor(np.asarray(classes, dtype=np.int64)).reshape(-1),
            ignore_mask=ignore_mask,
        )

    def __len__(self) -> int:
        return len(self._entries)
