"""Tile/mask reading for the dense (segmentation) path — flat or spacing-aware.

Two input regimes, routed per file:

* **Flat rasters** (``.png``/``.jpg``/``.jpeg``, or any input when no spacing is
  set) are read with **classic PIL**. The user already rendered them at their chosen
  resolution, so a requested spacing is **ignored** — there is no pyramid to resample
  from and nothing to select.
* **Pyramidal / spacing-bearing** inputs (e.g. multi-resolution TIFF) are read
  **spacing-aware** via hs2p. Images (:meth:`hs2p.wsi.wsi.WSI.read_full_at_spacing`)
  read the finest pyramid level ``<=`` the requested µm/px and downscale it with
  ``area`` interpolation (never upsampling). Masks are :class:`hs2p.Mask` objects
  aligned to their image's level-0 grid and read ``nearest`` onto the image's read
  size, so a mask coarser than its image still registers; values stay integer and
  must be declared by the mask's label vocabulary.

At an exact-level match (e.g. a 0.5 µm/px ROI read at 0.5) hs2p does no resize, so
the result is byte-identical to the plain PIL page-0 read — the parity the cached
dense path is verified against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from soma.class_scheme import assign_raw_values, resolve_classes

# Flat raster formats carry no pyramid/spacing — always read with PIL, spacing N/A.
_FLAT_SUFFIXES = {".png", ".jpg", ".jpeg"}


#: LUT entry for a raw pixel value that is in neither ``classes`` nor ``ignore``. Outside
#: every valid class index and every ``ignore_index``, so the segmentation head can fail
#: loud on it instead of silently dropping (or aliasing) a class's supervision.
UNDECLARED_LABEL = int(np.iinfo(np.int64).min)


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


#: ``task.params`` keys that define the class scheme rather than configure the head.
CLASS_SCHEME_KEYS = ("num_classes", "classes", "ignore")


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
    (the head, the live dataset) goes through here, so they cannot diverge.
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


def accepted_mask_values(
    *, num_classes: int, ignore_index: int, label_remap: np.ndarray | None
) -> dict[str, int]:
    """The label vocabulary a mask is read against when no ``pixel_mapping`` applies.

    Pre-cropped tiles have no sampling vocabulary, so their masks declare exactly the raw
    values the head accepts: those ``label_remap`` maps (``task.params.classes`` and
    ``ignore``), or, without a class scheme, the class indices and ``ignore_index``. Only
    single-byte values can occur in a mask raster.
    """
    if label_remap is not None:
        values = np.flatnonzero(np.asarray(label_remap) != UNDECLARED_LABEL).tolist()
    else:
        values = [*range(int(num_classes)), int(ignore_index)]
    return {f"value_{value}": int(value) for value in sorted(set(values)) if 0 <= value <= 255}


def check_classes_declared(
    task_params: Mapping[str, Any], pixel_mapping: Mapping[str, int | list[int]]
) -> None:
    """Fail unless every raw value in ``task.params.classes`` / ``ignore`` is in ``pixel_mapping``.

    hs2p reads an annotation raster against the closed ``preprocessing.masks.pixel_mapping``
    vocabulary and rejects any other value, so a raw value the training scheme names but
    the sampling vocabulary omits could never be read. This is the one coupling between the
    two layers; call it after :func:`resolve_class_scheme` has validated the scheme.
    """
    declared = {
        int(value)
        for entry in pixel_mapping.values()
        for value in (entry if isinstance(entry, (list, tuple)) else [entry])
    }
    classes = task_params.get("classes") or {}
    owners = [(f"task.params.classes {name!r}", values) for name, values in classes.items()]
    owners.append(("task.params.ignore", task_params.get("ignore") or ()))
    for owner, values in owners:
        for value in values if isinstance(values, (list, tuple)) else [values]:
            if int(value) not in declared:
                raise ValueError(
                    f"{owner} lists raw value {int(value)}, which "
                    "preprocessing.masks.pixel_mapping does not declare. Annotation masks "
                    "are read against that vocabulary, so add the value to pixel_mapping; "
                    "left out of min_coverage, its label is read but never selects a ROI."
                )


def _is_flat(path: str | Path, spacing_um: float | None) -> bool:
    """Read flat (PIL) when the format is flat or no spacing was requested."""
    return spacing_um is None or Path(path).suffix.lower() in _FLAT_SUFFIXES


def _load_flat_mask(path: str | Path) -> np.ndarray:
    """Load a flat single-channel integer class-index mask as a 2-D ``np.ndarray``.

    Fails loud on RGB/palette masks (a classic silent corruption — class indices
    smeared across channels) and on non-integer dtypes.
    """
    with Image.open(path) as image:
        # Palette ("P"/"PA") images load as a 2-D index array and would pass the
        # rank/dtype checks below, but palette indices are NOT class ids.
        if image.mode in ("P", "PA"):
            raise ValueError(
                f"mask '{path}' is a palette image (mode '{image.mode}'); palette indices "
                "are not class ids. Save masks as single-channel integer (e.g. 'L'/'I') rasters."
            )
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(
            f"mask '{path}' must be a 2-D single-channel class-index raster, got shape "
            f"{array.shape}. RGB/palette masks are not supported."
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"mask '{path}' must have an integer dtype (class indices), got {array.dtype}."
        )
    return array


def load_mask(path: str | Path, *, expected_size: tuple[int, int] | None = None) -> Tensor:
    """Load a flat segmentation mask as a 2-D ``long`` tensor of class indices.

    The flat (non-spacing) mask loader: PIL read + validation. v1 loads at native
    resolution (no resize); when resizing arrives it must be nearest-neighbor only so
    class indices and ``ignore_index`` survive.
    """
    array = _load_flat_mask(path)
    tensor = torch.from_numpy(array.astype(np.int64))
    if expected_size is not None and tuple(int(s) for s in tensor.shape) != tuple(expected_size):
        raise ValueError(
            f"mask '{path}' is {tuple(int(s) for s in tensor.shape)}, expected {tuple(expected_size)}."
        )
    return tensor


def read_image_at_spacing(
    path: str | Path,
    *,
    spacing_um: float | None,
    backend: str = "auto",
    tolerance: float = 0.05,
    interpolation: str = "area",
) -> np.ndarray:
    """Read an RGB tile as a ``(H, W, 3)`` uint8 array (flat PIL or spacing-aware hs2p)."""
    if _is_flat(path, spacing_um):
        with Image.open(path) as image:
            return np.ascontiguousarray(np.array(image.convert("RGB")))
    from hs2p.wsi.wsi import WSI

    wsi = WSI(path=Path(path), backend=backend)
    arr = wsi.read_full_at_spacing(
        float(spacing_um), tolerance=float(tolerance), interpolation=interpolation
    )
    return np.ascontiguousarray(arr[..., :3])


def read_image_region_at_spacing(
    path: str | Path,
    *,
    location: tuple[int, int],
    size: tuple[int, int],
    spacing_um: float,
    backend: str = "auto",
    tolerance: float = 0.05,
    interpolation: str = "area",
) -> np.ndarray:
    """Read a ``size=(w, h)`` RGB region at ``(x, y)`` (level-0) and ``spacing_um``.

    The region counterpart of :func:`read_image_at_spacing` for slide-manifest ROIs: the
    image is a whole-slide raster, so each ROI reads only its window (never the whole
    gigapixel slide) at the same spacing/size as its dense grid — mirroring the read the
    slide-manifest extractor uses, so overlays register to the features and the supervision.
    """
    from hs2p.wsi.wsi import WSI

    wsi = WSI(path=Path(path), backend=backend)
    arr = wsi.read_region_at_spacing(
        location=tuple(location),
        requested_spacing_um=float(spacing_um),
        size=tuple(size),
        tolerance=float(tolerance),
        interpolation=interpolation,
    )
    return np.ascontiguousarray(np.asarray(arr)[..., :3])


@lru_cache(maxsize=1024)
def _reference_geometry(
    image_path: str, backend: str, spacing_at_level_0: float | None
) -> tuple[float, tuple[int, int]]:
    """``(level-0 spacing µm/px, level-0 (w, h))`` of the image a mask is aligned to.

    Opened once per image per process: every ROI of a slide shares it. A declared
    ``spacing_at_level_0`` overrides the file's own, as it does for the image reads.
    """
    from hs2p.wsi.reader import open_slide

    with open_slide(
        Path(image_path), backend, spacing_override=spacing_at_level_0
    ) as slide:
        width, height = slide.level_dimensions[0]
        return float(slide.spacing), (int(width), int(height))


@contextmanager
def _aligned_mask(
    path: str | Path,
    *,
    reference_path: str | Path,
    reference_backend: str,
    spacing_at_level_0: float | None,
    pixel_mapping: Mapping[str, int | list[int]],
    backend: str,
):
    """Open ``path`` as an hs2p annotation ``Mask`` aligned to its image's level-0 grid.

    ``reference_backend`` reads the image, ``backend`` the mask.
    """
    from hs2p import AnnotationLabels, Mask

    reference_spacing_um, reference_dimensions = _reference_geometry(
        str(reference_path),
        reference_backend,
        None if spacing_at_level_0 is None else float(spacing_at_level_0),
    )
    with Mask(
        path=Path(path),
        labels=AnnotationLabels(pixel_mapping=dict(pixel_mapping)),
        backend=backend,
    ) as mask:
        yield mask.align_to(
            reference_spacing_um=reference_spacing_um,
            reference_dimensions=reference_dimensions,
        )


def read_mask_at_spacing(
    path: str | Path,
    *,
    spacing_um: float | None,
    size: tuple[int, int],
    reference_path: str | Path,
    pixel_mapping: Mapping[str, int | list[int]],
    reference_backend: str = "auto",
    spacing_at_level_0: float | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Read a label mask as a 2-D integer raster (flat PIL or spacing-aware hs2p).

    A flat mask is read as is. Otherwise the mask is aligned to its image
    (``reference_path``, read with ``reference_backend``; ``backend`` reads the mask) and
    read as ``size=(w, h)`` labels at ``spacing_um``: the image's
    own read size, so mask and image register whatever the mask's resolution. hs2p
    rejects a mask that does not cover the image at one scale, and any value
    ``pixel_mapping`` does not declare.
    """
    if _is_flat(path, spacing_um):
        return _load_flat_mask(path)
    with _aligned_mask(
        path,
        reference_path=reference_path,
        reference_backend=reference_backend,
        spacing_at_level_0=spacing_at_level_0,
        pixel_mapping=pixel_mapping,
        backend=backend,
    ) as aligned:
        return aligned.read_full(
            target_spacing_um=float(spacing_um), target_dimensions=tuple(size)
        ).labels


def read_mask_region_at_spacing(
    path: str | Path,
    *,
    location: tuple[int, int],
    size: tuple[int, int],
    spacing_um: float,
    reference_path: str | Path,
    pixel_mapping: Mapping[str, int | list[int]],
    reference_backend: str = "auto",
    spacing_at_level_0: float | None = None,
    backend: str = "auto",
) -> np.ndarray:
    """Read a ``size=(w, h)`` label-mask region at ``(x, y)`` and ``spacing_um``.

    The region counterpart of :func:`read_mask_at_spacing` for slide-manifest ROIs:
    the mask is a whole-slide annotation raster, so each ROI reads its window at the
    same spacing/size as its dense grid (so the supervision registers to the features).
    ``location`` is in the slide's (``reference_path``) level-0 pixels, not the mask's.
    """
    with _aligned_mask(
        path,
        reference_path=reference_path,
        reference_backend=reference_backend,
        spacing_at_level_0=spacing_at_level_0,
        pixel_mapping=pixel_mapping,
        backend=backend,
    ) as aligned:
        return aligned.read_region(
            location=tuple(location),
            target_spacing_um=float(spacing_um),
            target_dimensions=tuple(size),
        ).labels


def read_mask_region_within_slide(
    path: str | Path,
    *,
    location: tuple[int, int],
    size: tuple[int, int],
    spacing_um: float,
    reference_path: str | Path,
    pixel_mapping: Mapping[str, int | list[int]],
    reference_backend: str = "auto",
    spacing_at_level_0: float | None = None,
    backend: str = "auto",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a ROI's labels like :func:`read_mask_region_at_spacing`, tolerating an overhang.

    Tiling keeps a tile that starts inside the slide and extends past its right or bottom
    edge when its in-slide part meets ``min_coverage``; hs2p never pads a mask read past
    the canvas. The in-slide part, as hs2p measures it, is read at the same location and
    spacing (output pixel ``i`` samples the same slide point as in a full read), and the
    rest is reported as outside. Returns ``(labels, inside)``: ``inside`` is ``None`` when
    the whole ROI lies on the slide, else a boolean array marking the pixels that do;
    ``labels`` holds ``0`` outside, which carries no meaning.
    """
    width, height = (int(v) for v in size)
    with _aligned_mask(
        path,
        reference_path=reference_path,
        reference_backend=reference_backend,
        spacing_at_level_0=spacing_at_level_0,
        pixel_mapping=pixel_mapping,
        backend=backend,
    ) as aligned:
        region = dict(location=tuple(location), target_spacing_um=float(spacing_um))
        inside_width, inside_height = aligned.dimensions_within_canvas(
            **region, target_dimensions=(width, height)
        )
        if (inside_width, inside_height) == (width, height):
            return aligned.read_region(**region, target_dimensions=(width, height)).labels, None
        labels = np.zeros((height, width), dtype=np.uint8)
        inside = np.zeros((height, width), dtype=bool)
        if inside_width > 0 and inside_height > 0:
            labels[:inside_height, :inside_width] = aligned.read_region(
                **region, target_dimensions=(inside_width, inside_height)
            ).labels
            inside[:inside_height, :inside_width] = True
        return labels, inside
