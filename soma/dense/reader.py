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

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor

# Flat raster formats carry no pyramid/spacing — always read with PIL, spacing N/A.
_FLAT_SUFFIXES = {".png", ".jpg", ".jpeg"}


def build_label_remap(
    pixel_mapping: dict[str, int],
    *,
    num_classes: int | None = None,
    ignore_index: int = 255,
) -> tuple[np.ndarray, int]:
    """Build a raw-pixel → class-index lookup table from a ``masks.pixel_mapping``.

    Slide-manifest annotation rasters carry the *dataset's own* pixel vocabulary (e.g.
    BEETLE's ``{0 unannotated, 1 other, 2 non-invasive, 3 invasive, 4 necrosis}``); the
    segmentation head needs contiguous class indices ``[0, num_classes)`` with unannotated
    pixels collapsed to ``ignore_index``. This derives that mapping from ``pixel_mapping``
    (class name → raw pixel value, mirroring hs2p's ``masks`` config). No reserved label
    name is required; ``background`` is an opt-in name that only selects the ignore-label
    mode:

    * ``background`` **present** — its role is decided by ``num_classes`` (the task's
      class count):

      * ``num_classes is None`` or ``== len(pixel_mapping) - 1`` — ``background`` is the
        **unannotated/ignore** label: it maps to ``ignore_index`` and the non-background
        classes take class index = their order in ``pixel_mapping`` (``other`` → 0,
        ``non_invasive`` → 1, …). This is the default.
      * ``num_classes == len(pixel_mapping)`` — **every** label (including ``background``)
        is a real class, taking class index = its order (``background`` → 0).

    * ``background`` **absent** — there is no ignore-label name, so **every** named label
      is a real class (index = order) and ``num_classes`` (when given) must equal the
      mapping size. A background-free vocabulary like ``{tumor: 2}`` therefore maps the
      single named value to class 0; every unlisted raw value (the unannotated regions,
      expressed without naming them) collapses to ``ignore_index``.

    Any raw pixel value not present in ``pixel_mapping`` always maps to ``ignore_index`` —
    a stray value never silently aliases an in-range class.

    Returns the 256-entry LUT (indexable by a raw uint8/int mask) and the resolved
    ``num_classes`` it implies.
    """
    names = list(pixel_mapping)
    has_background = "background" in pixel_mapping
    # ``background`` is the ignore label only in the (size - 1) mode; in every other case
    # (no background, or num_classes == size) every named label is a real class.
    background_is_class = (not has_background) or (
        num_classes is not None and int(num_classes) == len(names)
    )
    if background_is_class:
        classes = names  # every label is a real class, background (if any) takes its order
    else:
        classes = [name for name in names if name != "background"]
    if num_classes is not None and len(classes) != int(num_classes):
        hint = (
            "Provide a pixel_mapping whose class count equals num_classes."
            if not has_background
            else (
                "Provide a pixel_mapping whose class count is either num_classes "
                "(background is a class) or num_classes + 1 (background is the "
                "unannotated/ignore label)."
            )
        )
        raise ValueError(
            f"masks.pixel_mapping implies {len(classes)} class(es) but num_classes="
            f"{num_classes}. {hint}"
        )
    lut = np.full(256, int(ignore_index), dtype=np.int64)
    for class_index, name in enumerate(classes):
        lut[int(pixel_mapping[name])] = class_index
    if has_background and not background_is_class:
        lut[int(pixel_mapping["background"])] = int(ignore_index)
    return lut, len(classes)


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
