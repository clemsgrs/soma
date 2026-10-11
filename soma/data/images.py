"""``TileImageSource``: the :class:`~soma.data.ImageSource` over an image manifest.

The live grid path re-encodes pixels every step, so the pixels are a source of their
own, under the same id / geometry / spacing contract as a cached
:class:`~soma.data.GridSource`. Flat rasters (PNG/JPEG, or any file without a requested
spacing) are read as stored; pyramidal, spacing-bearing images are read through hs2p at
``spacing_um``.
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
        spacing_um: µm/px to read pyramidal images at; ``None`` reads every image as
            stored.
        backend: hs2p backend for spacing-aware reads.
        tolerance: hs2p spacing tolerance: a level within it is read without resampling.
    """

    def __init__(
        self,
        entries: Mapping[str, ImageEntry] | Sequence[ImageEntry],
        *,
        size: int | tuple[int, int],
        spacing_um: float | None = None,
        backend: str = "auto",
        tolerance: float = 0.05,
    ) -> None:
        if not isinstance(entries, Mapping):
            entries = {entry.sample_id: entry for entry in entries}
        self._entries: dict[str, ImageEntry] = {str(k): v for k, v in entries.items()}
        self._size = normalize_hw(size, name="size")
        self._spacing_um = None if spacing_um is None else float(spacing_um)
        self._backend = backend
        self._tolerance = float(tolerance)

    @classmethod
    def from_manifest(
        cls,
        image_manifest: ImageManifest,
        *,
        size: int | tuple[int, int],
        spacing_um: float | None = None,
        backend: str = "auto",
        tolerance: float = 0.05,
    ) -> "TileImageSource":
        return cls(
            image_manifest.entries,
            size=size,
            spacing_um=spacing_um,
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

    def _flat(self, entry: ImageEntry) -> bool:
        from soma.dense.reader import _is_flat

        return _is_flat(entry.image_path, self._spacing_um)

    def _spacings(self, entry: ImageEntry) -> tuple[float | None, float | None]:
        """``(level-0 µm/px, read µm/px)`` of one image; ``None`` where undeclared."""
        if self._flat(entry):
            level0 = entry.spacing_at_level_0 if entry.spacing_at_level_0 is not None else self._spacing_um
            return level0, level0
        from soma.dense.reader import _reference_geometry

        level0, _ = _reference_geometry(
            str(entry.image_path),
            self._backend,
            None if entry.spacing_at_level_0 is None else float(entry.spacing_at_level_0),
        )
        requested = float(self._spacing_um)
        within = abs(level0 - requested) <= requested * self._tolerance
        return float(level0), float(level0) if within else requested

    def load(self, sample_id: str) -> Tensor:
        from soma.dense.reader import read_image_at_spacing

        entry = self._entry(sample_id)
        array = read_image_at_spacing(
            entry.image_path,
            spacing_um=self._spacing_um,
            backend=self._backend,
            tolerance=self._tolerance,
        )
        return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1).contiguous()

    def geometry(self, sample_id: str) -> GridGeometry:
        """The pixel frame: ``target_size`` = the declared size, one token per pixel,
        anchored at the level-0 origin with ``read / level-0`` level-0 pixels per pixel."""
        entry = self._entry(sample_id)
        level0, read = self._spacings(entry)
        scale = 1.0 if level0 is None or read is None else read / level0
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        return GridGeometry.from_sizes(
            target_size=self._size, patch_size=1, level0_px_per_token_px=scale
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
