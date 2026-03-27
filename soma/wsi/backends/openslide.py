"""OpenSlide-backed whole-slide reader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class OpenSlideReader:
    """SlideReader backed by openslide-python."""

    def __init__(
        self, path: str | Path, *, spacing_override: float | None = None
    ) -> None:
        try:
            import openslide
        except ImportError as exc:
            msg = (
                "openslide-python is required for OpenSlideReader. "
                "Install it with: pip install openslide-python"
            )
            raise ImportError(msg) from exc

        self._slide = openslide.OpenSlide(str(path))
        self._spacing = (
            spacing_override
            if spacing_override is not None
            else self._extract_spacing()
        )

    def _extract_spacing(self) -> float:
        props = self._slide.properties
        mpp_x = props.get("openslide.mpp-x")
        if mpp_x is not None:
            return float(mpp_x)
        objective = props.get("openslide.objective-power")
        if objective is not None:
            objective_value = float(objective)
            if objective_value > 0:
                return 10.0 / objective_value
        raise ValueError(
            "Unable to infer slide spacing from OpenSlide metadata. "
            "Provide spacing_override or use a slide with valid spacing metadata."
        )

    @property
    def backend_name(self) -> str:
        return "openslide"

    @property
    def dimensions(self) -> tuple[int, int]:
        return self._slide.dimensions

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def level_count(self) -> int:
        return self._slide.level_count

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        return list(self._slide.level_dimensions)

    @property
    def level_downsamples(self) -> list[float]:
        return list(self._slide.level_downsamples)

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
        pil_image = self._slide.read_region(location, level, size)
        return np.array(pil_image.convert("RGB"))

    def _read_region_with_padding(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> np.ndarray:
        width, height = int(size[0]), int(size[1])
        canvas = np.full((height, width, 3), 255, dtype=np.uint8)
        if width <= 0 or height <= 0:
            return canvas

        level_width, level_height = self.level_dimensions[level]
        downsample = float(self.level_downsamples[level])

        x_level = int(np.floor(location[0] / downsample))
        y_level = int(np.floor(location[1] / downsample))
        x1 = max(x_level, 0)
        y1 = max(y_level, 0)
        x2 = min(x_level + width, level_width)
        y2 = min(y_level + height, level_height)
        if x2 <= x1 or y2 <= y1:
            return canvas

        read_width = x2 - x1
        read_height = y2 - y1
        read_location = (
            int(round(x1 * downsample)),
            int(round(y1 * downsample)),
        )
        region = self._slide.read_region(read_location, level, (read_width, read_height))
        region_rgb = np.array(region.convert("RGB"))

        paste_x = x1 - x_level
        paste_y = y1 - y_level
        canvas[paste_y : paste_y + read_height, paste_x : paste_x + read_width] = region_rgb
        return canvas

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        return np.array(self._slide.get_thumbnail(size).convert("RGB"))

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> OpenSlideReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
