"""cuCIM-backed whole-slide reader with optional batched region reads."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np


def _as_rgb_uint8(region: Any) -> np.ndarray:
    arr = np.asarray(region)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] >= 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_value = float(np.nanmax(arr)) if arr.size else 0.0
            if max_value <= 1.0:
                arr = np.clip(arr, 0.0, 1.0) * 255.0
            else:
                arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    return arr


def _read_bounds_with_padding(
    *,
    location: tuple[int, int],
    size: tuple[int, int],
    level_dimensions: tuple[int, int],
    downsample: float,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    width, height = int(size[0]), int(size[1])
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    if width <= 0 or height <= 0:
        return canvas, (0, 0), (0, 0), (0, 0)

    x_level = int(np.floor(location[0] / downsample))
    y_level = int(np.floor(location[1] / downsample))
    x1 = max(x_level, 0)
    y1 = max(y_level, 0)
    x2 = min(x_level + width, int(level_dimensions[0]))
    y2 = min(y_level + height, int(level_dimensions[1]))
    if x2 <= x1 or y2 <= y1:
        return canvas, (0, 0), (0, 0), (0, 0)

    read_width = x2 - x1
    read_height = y2 - y1
    read_location = (
        int(round(x1 * downsample)),
        int(round(y1 * downsample)),
    )
    paste_offset = (x1 - x_level, y1 - y_level)
    return canvas, read_location, (read_width, read_height), paste_offset


def _extract_spacing_from_metadata(metadata: dict[str, Any]) -> float:
    for entry in metadata.values():
        if not isinstance(entry, dict):
            continue
        if "MPP" in entry:
            return float(entry["MPP"])
        if "DICOM_PIXEL_SPACING" in entry:
            spacing = entry["DICOM_PIXEL_SPACING"]
            if isinstance(spacing, (list, tuple)):
                spacing = spacing[0]
            return float(spacing) * 1000.0
        if "spacing" not in entry:
            continue
        spacing = entry["spacing"]
        if isinstance(spacing, (list, tuple)):
            spacing = spacing[0]
        units = entry.get("spacing_units")
        if isinstance(units, (list, tuple)):
            units = units[0]
        factor = {
            "mm": 1000.0,
            "millimeters": 1000.0,
            "millimeter": 1000.0,
            "cm": 10000.0,
            "centimeters": 10000.0,
            "centimeter": 10000.0,
            "um": 1.0,
            "microns": 1.0,
            "micrometers": 1.0,
            "micrometer": 1.0,
        }.get(str(units).lower() if units is not None else "")
        if factor is not None:
            return float(spacing) * factor
    raise ValueError(
        "Unable to infer slide spacing from cuCIM metadata. "
        "Provide spacing_override or use a slide with valid spacing metadata."
    )


class CuCIMReader:
    """SlideReader backed by cucim.CuImage."""

    def __init__(
        self, path: str | Path, *, spacing_override: float | None = None
    ) -> None:
        try:
            cucim = importlib.import_module("cucim")
        except ImportError as exc:
            msg = "cucim is required for the cucim backend. Install it with: pip install cucim"
            raise ImportError(msg) from exc

        self._slide = cucim.CuImage(str(path))
        self._metadata = getattr(self._slide, "metadata", {}) or {}
        self._resolutions = (
            self._metadata.get("cucim", {}).get("resolutions", {}) if isinstance(self._metadata, dict) else {}
        )
        self._level_dimensions = self._resolve_level_dimensions()
        self._level_downsamples = self._resolve_level_downsamples()
        self._dimensions = self._resolve_dimensions()
        self._spacing = (
            float(spacing_override)
            if spacing_override is not None
            else _extract_spacing_from_metadata(self._metadata)
        )

    @property
    def backend_name(self) -> str:
        return "cucim"

    def _resolve_dimensions(self) -> tuple[int, int]:
        if self._level_dimensions:
            return self._level_dimensions[0]
        shape = self._metadata.get("cucim", {}).get("shape")
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            return int(shape[1]), int(shape[0])
        return 0, 0

    def _resolve_level_dimensions(self) -> list[tuple[int, int]]:
        dims = self._resolutions.get("level_dimensions")
        if isinstance(dims, (list, tuple)):
            return [tuple(int(v) for v in dim[:2]) for dim in dims]
        return []

    def _resolve_level_downsamples(self) -> list[float]:
        downsamples = self._resolutions.get("level_downsamples")
        if isinstance(downsamples, (list, tuple)):
            return [float(value) for value in downsamples]
        return [1.0]

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._dimensions

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def level_count(self) -> int:
        count = self._resolutions.get("level_count")
        if count is not None:
            return int(count)
        return max(1, len(self._level_dimensions))

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self._level_dimensions)

    @property
    def level_downsamples(self) -> list[float]:
        return list(self._level_downsamples)

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        pad_missing: bool = False,
    ) -> np.ndarray:
        if pad_missing:
            return self._read_region_with_padding(location, level, size)
        region = self._slide.read_region(
            (int(location[0]), int(location[1])),
            (int(size[0]), int(size[1])),
            level=int(level),
        )
        return _as_rgb_uint8(region)

    def read_regions(
        self,
        locations: list[tuple[int, int]],
        level: int,
        size: tuple[int, int],
        *,
        num_workers: int | None = None,
        pad_missing: bool = False,
    ) -> Iterable[np.ndarray]:
        if pad_missing:
            for location in locations:
                yield self.read_region(location, level, size, pad_missing=True)
            return
        normalized_locations = [
            (int(location[0]), int(location[1])) for location in locations
        ]
        size = (int(size[0]), int(size[1]))
        regions = self._slide.read_region(
            normalized_locations,
            size,
            level=int(level),
            num_workers=max(1, int(num_workers or 1)),
        )

        for region in regions:
            yield _as_rgb_uint8(region)

    def _read_region_with_padding(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        level_dimensions = self.level_dimensions[level]
        downsample = float(self.level_downsamples[level])
        padded = _read_bounds_with_padding(
            location=location,
            size=size,
            level_dimensions=level_dimensions,
            downsample=downsample,
        )
        assert padded is not None
        canvas, read_location, read_size, paste_offset = padded
        read_width, read_height = read_size
        if read_width <= 0 or read_height <= 0:
            return canvas

        region = self._slide.read_region(
            (int(read_location[0]), int(read_location[1])),
            (int(read_width), int(read_height)),
            level=int(level),
        )
        region_rgb = _as_rgb_uint8(region)
        paste_x, paste_y = paste_offset
        canvas[paste_y : paste_y + read_height, paste_x : paste_x + read_width] = region_rgb
        return canvas

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        if not self._level_dimensions:
            return self.read_region((0, 0), 0, size)

        level = self.level_count - 1
        level_dims = self._level_dimensions[level]
        region = self.read_region((0, 0), level, level_dims)
        target_w, target_h = int(size[0]), int(size[1])
        if target_w <= 0 or target_h <= 0:
            return region
        scale = min(target_w / max(region.shape[1], 1), target_h / max(region.shape[0], 1))
        width = max(1, int(round(region.shape[1] * scale)))
        height = max(1, int(round(region.shape[0] * scale)))
        if (width, height) == (region.shape[1], region.shape[0]):
            return region
        return cv2.resize(region, (width, height), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        close = getattr(self._slide, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> CuCIMReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
