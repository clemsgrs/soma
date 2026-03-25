"""SlideReader protocol and concrete backends for whole-slide image I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SlideReader(Protocol):
    """Protocol for reading whole-slide images.

    All backends must conform to this interface.
    """

    @property
    def dimensions(self) -> tuple[int, int]:
        """(width, height) at level 0."""
        ...

    @property
    def spacing(self) -> float:
        """Pixel spacing in µm/px at level 0."""
        ...

    @property
    def level_count(self) -> int:
        """Number of pyramid levels."""
        ...

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        """(width, height) at each pyramid level."""
        ...

    @property
    def level_downsamples(self) -> list[float]:
        """Downsample factor at each pyramid level."""
        ...

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        """Read a region from the slide.

        Args:
            location: (x, y) top-left corner in level-0 coordinates.
            level: Pyramid level to read from.
            size: (width, height) in pixels at the target level.

        Returns:
            RGB array of shape (height, width, 3), dtype uint8.
        """
        ...

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        """Get a thumbnail of the slide.

        Args:
            size: Maximum (width, height) of the thumbnail.

        Returns:
            RGB array of shape (h, w, 3), dtype uint8.
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...

    def __enter__(self) -> SlideReader: ...
    def __exit__(self, *args: Any) -> None: ...


class OpenSlideReader:
    """SlideReader backed by openslide-python."""

    def __init__(
        self, path: str | Path, *, spacing_override: float | None = None
    ) -> None:
        try:
            import openslide
        except ImportError:
            msg = (
                "openslide-python is required for OpenSlideReader. "
                "Install it with: pip install openslide-python"
            )
            raise ImportError(msg)

        self._slide = openslide.OpenSlide(str(path))
        self._spacing = (
            spacing_override
            if spacing_override is not None
            else self._extract_spacing()
        )

    def _extract_spacing(self) -> float:
        """Extract µm/px from slide properties (MPP or objective power)."""
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


def open_slide(
    path: str | Path,
    backend: str = "auto",
    *,
    spacing_override: float | None = None,
) -> SlideReader:
    """Open a whole-slide image with the specified backend.

    Args:
        path: Path to the WSI file.
        backend: "openslide", "cucim", or "auto" (tries cucim first, then openslide).
        spacing_override: If set, use this spacing (µm/px) instead of reading from metadata.
    """
    if backend == "auto":
        try:
            return _open_cucim(path, spacing_override=spacing_override)
        except ImportError:
            return OpenSlideReader(path, spacing_override=spacing_override)
    elif backend == "openslide":
        return OpenSlideReader(path, spacing_override=spacing_override)
    elif backend == "cucim":
        return _open_cucim(path, spacing_override=spacing_override)
    else:
        msg = f"Unknown backend: '{backend}'. Available: openslide, cucim, auto"
        raise ValueError(msg)


def _open_cucim(
    path: str | Path, *, spacing_override: float | None = None
) -> SlideReader:
    """Open a slide with cucim. Raises ImportError if cucim is not available."""
    try:
        import cucim  # noqa: F401
    except ImportError:
        msg = "cucim is required for the cucim backend. Install it with: pip install cucim"
        raise ImportError(msg)
    raise NotImplementedError("cucim backend not yet implemented")


# --- Level selection ---


@dataclass(frozen=True)
class LevelSelection:
    """Result of selecting a pyramid level for a requested spacing."""

    level: int
    effective_spacing_um: float
    is_within_tolerance: bool


def select_level(
    requested_spacing_um: float,
    level_downsamples: list[float],
    base_spacing_um: float,
    *,
    tolerance: float = 0.05,
) -> LevelSelection:
    """Select the best pyramid level for a requested spacing.

    Finds the level whose effective spacing is closest to but does not exceed
    the requested spacing (never upsamples). If no level has spacing ≤ requested,
    falls back to level 0.

    Args:
        requested_spacing_um: Target spacing in µm/px.
        level_downsamples: Downsample factor at each pyramid level.
        base_spacing_um: Native spacing at level 0 in µm/px.
        tolerance: Fraction within which effective ≈ requested (default 5%).

    Returns:
        LevelSelection with the chosen level, its effective spacing,
        and whether it is within tolerance of the requested spacing.
    """
    best_level = 0
    best_spacing = base_spacing_um

    for level, downsample in enumerate(level_downsamples):
        effective = base_spacing_um * downsample
        if effective <= requested_spacing_um and effective >= best_spacing:
            best_level = level
            best_spacing = effective

    relative_error = abs(best_spacing - requested_spacing_um) / requested_spacing_um
    is_within_tolerance = relative_error <= tolerance

    return LevelSelection(
        level=best_level,
        effective_spacing_um=best_spacing,
        is_within_tolerance=is_within_tolerance,
    )
