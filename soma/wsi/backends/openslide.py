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
            return 10.0 / float(objective)
        return 0.5

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
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        pil_image = self._slide.read_region(location, level, size)
        return np.array(pil_image.convert("RGB"))

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        return np.array(self._slide.get_thumbnail(size).convert("RGB"))

    def close(self) -> None:
        self._slide.close()

    def __enter__(self) -> OpenSlideReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
