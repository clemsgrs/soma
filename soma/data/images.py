"""``TileImageSource``: the :class:`~soma.data.ImageSource` over an image manifest.

The live grid path re-encodes pixels every step, so the pixels are a source of their
own, under the same id / geometry / spacing contract as a cached
:class:`~soma.data.GridSource`. Flat rasters (PNG/JPEG, or any file without a requested
spacing) are read as stored, unless the manifest declares a ``spacing_at_level_0``
outside ``tolerance`` of ``spacing_um``: then they are area-resampled to ``spacing_um``,
as slide2vec's cached extraction reads them. Pyramidal, spacing-bearing images are read
through hs2p at ``spacing_um``. Under ``spacing_policy="native_if_coarser"`` an image
whose declared spacing is coarser than ``spacing_um`` is read at its own spacing instead,
as the label maps and cached extraction resolve it. :class:`RegionImageSource` serves
windows of whole slides (slide-manifest ROIs) under the same contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from soma.data.geometry import GridGeometry, normalize_hw
from soma.data.manifests import ImageEntry, ImageManifest

__all__ = ["TileImageSource"]


class TileImageSource:
    """Pixels of pre-cropped tiles (or whole small images), one ``(3, H, W)`` uint8 per id.

    Args:
        entries: The images, keyed by sample id (an :class:`~soma.data.ImageManifest`'s
            entries).
        size: The declared ``(H, W)`` (or side) the encoder stage expects; images larger
            than it reach that size only through an augmentation crop.
        spacing_um: µm/px to read pyramidal images at, and flat ones whose declared
            ``spacing_at_level_0`` lies outside ``tolerance`` of it; ``None`` reads every
            image as stored.
        spacing_policy: ``"strict"`` reads every image at ``spacing_um``;
            ``"native_if_coarser"`` reads one whose declared ``spacing_at_level_0`` is
            coarser than it at that native spacing (no upsampling).
        backend: hs2p backend for spacing-aware reads.
        tolerance: hs2p spacing tolerance: a level within it is read without resampling.
    """

    def __init__(
        self,
        entries: Mapping[str, ImageEntry] | Sequence[ImageEntry],
        *,
        size: int | tuple[int, int],
        spacing_um: float | None = None,
        spacing_policy: str = "strict",
        backend: str = "auto",
        tolerance: float = 0.05,
    ) -> None:
        if not isinstance(entries, Mapping):
            entries = {entry.sample_id: entry for entry in entries}
        self._entries: dict[str, ImageEntry] = {str(k): v for k, v in entries.items()}
        self._size = normalize_hw(size, name="size")
        self._spacing_um = None if spacing_um is None else float(spacing_um)
        self._spacing_policy = spacing_policy
        self._backend = backend
        self._tolerance = float(tolerance)

    @classmethod
    def from_manifest(
        cls,
        image_manifest: ImageManifest,
        *,
        size: int | tuple[int, int],
        spacing_um: float | None = None,
        spacing_policy: str = "strict",
        backend: str = "auto",
        tolerance: float = 0.05,
    ) -> "TileImageSource":
        return cls(
            image_manifest.entries,
            size=size,
            spacing_um=spacing_um,
            spacing_policy=spacing_policy,
            backend=backend,
            tolerance=tolerance,
        )

    @property
    def sample_ids(self) -> list[str]:
        return list(self._entries)

    @property
    def size(self) -> tuple[int, int]:
        return self._size

    def subset(self, sample_ids: Sequence[str]) -> "TileImageSource":
        """The same source restricted to ``sample_ids``."""
        return TileImageSource(
            {sid: self._entry(sid) for sid in map(str, sample_ids)},
            size=self._size,
            spacing_um=self._spacing_um,
            spacing_policy=self._spacing_policy,
            backend=self._backend,
            tolerance=self._tolerance,
        )

    def _entry(self, sample_id: str) -> ImageEntry:
        try:
            return self._entries[str(sample_id)]
        except KeyError:
            raise KeyError(
                f"Sample '{sample_id}' not found in image source. Available: {sorted(self._entries)}"
            ) from None

    def _read_spacing(self, entry: ImageEntry) -> float | None:
        """The µm/px ``entry`` is read at: ``spacing_um``, resolved per entry through
        ``spacing_policy`` against its declared ``spacing_at_level_0``."""
        if self._spacing_um is None:
            return None
        from soma.spacing import resolve_effective_spacing_um

        return resolve_effective_spacing_um(
            requested_spacing_um=self._spacing_um,
            spacing_at_level_0=entry.spacing_at_level_0,
            tolerance=self._tolerance,
            policy=self._spacing_policy,
        )

    def _flat(self, entry: ImageEntry) -> bool:
        from soma.dense.reader import _is_flat

        return _is_flat(entry.image_path, self._spacing_um)

    def _flat_resample(self, entry: ImageEntry) -> float | None:
        """The declared level-0 µm/px of a flat raster ``load`` resamples to its read
        spacing, or ``None`` when it is served as stored (no declared or requested spacing,
        or the two within ``tolerance``). hs2p's image policy refuses upsampling beyond it."""
        read = self._read_spacing(entry)
        if read is None or entry.spacing_at_level_0 is None:
            return None
        from hs2p.wsi.geometry import select_level_for_spacing_read

        declared = float(entry.spacing_at_level_0)
        selection = select_level_for_spacing_read(
            requested_spacing_um=read,
            level0_spacing_um=declared,
            level_downsamples=[(1.0, 1.0)],
            tolerance=self._tolerance,
            content_kind="image",
        )
        return None if selection.is_within_tolerance else declared

    @staticmethod
    def _level0_override(entry: ImageEntry) -> float | None:
        return None if entry.spacing_at_level_0 is None else float(entry.spacing_at_level_0)

    def _spacings(self, entry: ImageEntry) -> tuple[float | None, float | None]:
        """``(level-0 µm/px, served µm/px)`` of one image; ``None`` where undeclared.

        The served spacing is the one hs2p's level selection reads at (a pyramid level
        within tolerance is served unresized), so the geometry matches what ``load``
        returns.
        """
        if self._flat(entry):
            declared = self._flat_resample(entry)
            if declared is not None:
                return declared, self._read_spacing(entry)
            level0 = entry.spacing_at_level_0 if entry.spacing_at_level_0 is not None else self._spacing_um
            return level0, level0
        from soma.dense.reader import _served_spacing

        return _served_spacing(
            str(entry.image_path),
            self._backend,
            self._level0_override(entry),
            float(self._read_spacing(entry)),
            self._tolerance,
        )

    def load(self, sample_id: str) -> Tensor:
        from soma.dense.reader import read_image_at_spacing

        entry = self._entry(sample_id)
        array = read_image_at_spacing(
            entry.image_path,
            spacing_um=self._read_spacing(entry),
            backend=self._backend,
            tolerance=self._tolerance,
            spacing_at_level_0=self._level0_override(entry),
        )
        declared = self._flat_resample(entry) if self._flat(entry) else None
        if declared is not None:
            from hs2p.wsi.wsi import resize_array

            scale = declared / float(self._read_spacing(entry))
            height, width = array.shape[:2]
            output = (int(round(width * scale)), int(round(height * scale)))  # (w, h)
            array = resize_array(array, output, interpolation="area")
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()

    def _origin(self, sample_id: str) -> tuple[float, float]:
        """The level-0 origin of the served pixels (a whole tile starts at 0, 0)."""
        return (0.0, 0.0)

    def geometry(self, sample_id: str) -> GridGeometry:
        """The pixel frame: ``target_size`` = the declared size, one token per pixel,
        anchored at the level-0 origin with ``read / level-0`` level-0 pixels per pixel."""
        entry = self._entry(sample_id)
        level0, read = self._spacings(entry)
        scale = 1.0 if level0 is None or read is None else read / level0
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        return GridGeometry.from_sizes(
            target_size=self._size,
            patch_size=1,
            origin_level0=self._origin(sample_id),
            level0_px_per_token_px=scale,
        )

    def spacing(self, sample_id: str) -> float:
        """µm/px of the served pixels (the read spacing; a flat raster's declared one)."""
        entry = self._entry(sample_id)
        _, read = self._spacings(entry)
        if read is None:
            raise ValueError(
                f"image '{sample_id}' declares no spacing: set spacing_um or the manifest's "
                "spacing_at_level_0"
            )
        return float(read)

    def __len__(self) -> int:
        return len(self._entries)


class RegionImageSource(TileImageSource):
    """Windows of whole-slide images, one ``(3, H, W)`` uint8 per id (slide-manifest ROIs).

    Each id's pixels are the ``size`` window whose level-0 top-left is its ``regions``
    entry, read at ``spacing_um`` without opening the whole slide, as the slide-manifest
    extractor reads it, so the pixels register to the ROI's grid and supervision.

    Args:
        entries: The slides, keyed by sample (ROI) id.
        regions: Level-0 ``(x, y)`` origin of each id's window.
        size: The window ``(H, W)`` (or side) at ``spacing_um``.
        spacing_um: µm/px to read the windows at.
        spacing_policy: As :class:`TileImageSource`'s, resolved per slide.
        backend: hs2p backend.
        tolerance: hs2p spacing tolerance: a level within it is read without resampling.
    """

    def __init__(
        self,
        entries: Mapping[str, ImageEntry] | Sequence[ImageEntry],
        *,
        regions: Mapping[str, tuple[int, int]],
        size: int | tuple[int, int],
        spacing_um: float,
        spacing_policy: str = "strict",
        backend: str = "auto",
        tolerance: float = 0.05,
    ) -> None:
        super().__init__(
            entries,
            size=size,
            spacing_um=spacing_um,
            spacing_policy=spacing_policy,
            backend=backend,
            tolerance=tolerance,
        )
        self._regions = {str(k): (int(v[0]), int(v[1])) for k, v in regions.items()}
        missing = sorted(set(self._entries) - set(self._regions))
        if missing:
            raise ValueError(f"no region for {len(missing)} image(s): {missing[:10]}")

    def subset(self, sample_ids: Sequence[str]) -> "RegionImageSource":
        ids = [str(sid) for sid in sample_ids]
        return RegionImageSource(
            {sid: self._entry(sid) for sid in ids},
            regions={sid: self._regions[sid] for sid in ids},
            size=self._size,
            spacing_um=self._spacing_um,
            spacing_policy=self._spacing_policy,
            backend=self._backend,
            tolerance=self._tolerance,
        )

    def _origin(self, sample_id: str) -> tuple[float, float]:
        x, y = self._regions[str(sample_id)]
        return (float(x), float(y))

    def load(self, sample_id: str) -> Tensor:
        from soma.dense.reader import read_image_region_at_spacing

        entry = self._entry(sample_id)
        height, width = self._size
        array = read_image_region_at_spacing(
            entry.image_path,
            location=self._regions[str(sample_id)],
            size=(width, height),  # hs2p sizes are (w, h)
            spacing_um=self._read_spacing(entry),
            backend=self._backend,
            tolerance=self._tolerance,
            spacing_at_level_0=self._level0_override(entry),
        )
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()
