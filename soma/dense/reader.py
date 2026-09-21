"""Tile/mask reading for the dense (segmentation) path — flat or spacing-aware.

Two input regimes, routed per file:

* **Flat rasters** (``.png``/``.jpg``/``.jpeg``, or any input when no spacing is
  set) are read with **classic PIL**. The user already rendered them at their chosen
  resolution, so a requested spacing is **ignored** — there is no pyramid to resample
  from and nothing to select.
* **Pyramidal / spacing-bearing** inputs (e.g. multi-resolution TIFF) are read
  **spacing-aware** via hs2p (:meth:`hs2p.wsi.wsi.WSI.read_full_at_spacing` /
  :func:`hs2p.wsi.masks.read_label_at_spacing`): the finest pyramid level ``<=`` the
  requested µm/px is read and downscaled (never upsampled). Images use ``area``
  interpolation; masks use ``nearest`` (label-preserving) and stay integer.

At an exact-level match (e.g. a 0.5 µm/px ROI read at 0.5) hs2p does no resize, so
the result is byte-identical to the plain PIL page-0 read — the parity the cached
dense path is verified against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor

# Flat raster formats carry no pyramid/spacing — always read with PIL, spacing N/A.
_FLAT_SUFFIXES = {".png", ".jpg", ".jpeg"}


#: LUT entry for a raw pixel value that is in neither ``classes`` nor ``ignore``. Outside
#: every valid class index and every ``ignore_index``, so the segmentation head can fail
#: loud on it instead of silently dropping (or aliasing) a class's supervision.
UNDECLARED_LABEL = int(np.iinfo(np.int64).min)


def _raw_values(owner: str, values: Any) -> list[int]:
    """Normalize one ``classes`` entry / the ``ignore`` list to raw single-byte values."""
    if not isinstance(values, (list, tuple)):
        values = [values]
    for value in values:
        is_integer = isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        if not is_integer or not 0 <= value <= 255:
            raise ValueError(
                f"'{owner}' raw values must be integers in [0, 255], got {value!r}."
            )
    return [int(value) for value in values]


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
    loss and metrics (they map to ``ignore_index``). No name is reserved.

    A raw value may belong to one class or to ``ignore``, never two. Every value declared
    nowhere maps to :data:`UNDECLARED_LABEL`.

    Returns the 256-entry LUT (indexable by a raw uint8/int mask).
    """
    if not classes:
        raise ValueError("classes must name at least one class.")
    lut = np.full(256, UNDECLARED_LABEL, dtype=np.int64)
    owners: dict[int, str] = {}

    def assign(owner: str, values: Any, target: int) -> None:
        for value in _raw_values(owner, values):
            if value in owners:
                raise ValueError(
                    f"'{owners[value]}' and '{owner}' both list raw value {value}; a raw "
                    "value belongs to exactly one class or to ignore."
                )
            owners[value] = owner
            lut[value] = target

    for class_index, (name, values) in enumerate(classes.items()):
        if isinstance(values, (list, tuple)) and not values:
            raise ValueError(f"class '{name}' lists no raw values.")
        assign(str(name), values, class_index)
    assign("ignore", list(ignore), int(ignore_index))
    return lut


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
    classes = task_params.get("classes")
    ignore = task_params.get("ignore")
    num_classes = task_params.get("num_classes")
    if classes is None:
        if annotation_rasters:
            raise ValueError(
                "segmentation from annotation rasters (preprocessing.masks) requires "
                "task.params.classes: name each class and the raw mask value(s) that form "
                "it, e.g. classes: {tumor: [1, 2], stroma: [3]} with ignore: [0] for the "
                "values to exclude from loss and metrics."
            )
        if ignore is not None:
            raise ValueError("task.params.ignore needs task.params.classes.")
        if num_classes is None:
            raise ValueError(
                "dataset_type='segmentation' requires task.params.classes (or "
                "task.params.num_classes when masks already hold class indices)."
            )
        num_classes = int(num_classes)
        return num_classes, tuple(f"class_{index}" for index in range(num_classes)), None

    if not isinstance(classes, Mapping):
        raise ValueError("task.params.classes must map class name → raw mask value(s).")
    lut = build_label_remap(
        classes, ignore=ignore or (), ignore_index=int(task_params.get("ignore_index", 255))
    )
    if num_classes is not None and int(num_classes) != len(classes):
        raise ValueError(
            f"task.params.num_classes={num_classes} disagrees with the {len(classes)} "
            "classes in task.params.classes; drop num_classes (it is derived)."
        )
    return len(classes), tuple(str(name) for name in classes), lut


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

    wsi = WSI(Path(path), backend=backend)
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

    wsi = WSI(Path(path), backend=backend)
    arr = wsi.read_region_at_spacing(
        tuple(location), float(spacing_um), tuple(size),
        tolerance=float(tolerance), interpolation=interpolation,
    )
    return np.ascontiguousarray(np.asarray(arr)[..., :3])


def read_mask_at_spacing(
    path: str | Path,
    *,
    spacing_um: float | None,
    backend: str = "auto",
    tolerance: float = 0.05,
) -> np.ndarray:
    """Read a label mask as a 2-D integer class-index raster (flat PIL or spacing-aware hs2p)."""
    if _is_flat(path, spacing_um):
        return _load_flat_mask(path)
    from hs2p.wsi.masks import read_label_at_spacing
    from hs2p.wsi.wsi import WSI

    wsi = WSI(Path(path), backend=backend)
    return read_label_at_spacing(wsi, float(spacing_um), tolerance=float(tolerance))


def read_mask_region_at_spacing(
    path: str | Path,
    *,
    location: tuple[int, int],
    size: tuple[int, int],
    spacing_um: float,
    backend: str = "auto",
    tolerance: float = 0.05,
) -> np.ndarray:
    """Read a ``size=(w, h)`` label-mask region at ``(x, y)`` (level-0) and ``spacing_um``.

    The region counterpart of :func:`read_mask_at_spacing` for slide-manifest ROIs:
    the mask is a whole-slide annotation raster, so each ROI reads its window at the
    same spacing/size as its dense grid (so the supervision registers to the features).
    """
    from hs2p.wsi.masks import read_label_region_at_spacing
    from hs2p.wsi.wsi import WSI

    wsi = WSI(Path(path), backend=backend)
    return read_label_region_at_spacing(
        wsi, tuple(location), float(spacing_um), tuple(size), tolerance=float(tolerance)
    )
