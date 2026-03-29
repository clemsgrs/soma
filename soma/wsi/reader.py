"""Public WSI reader protocols, backend factory, and level selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class SlideReader(Protocol):
    """Protocol for reading whole-slide images."""

    @property
    def backend_name(self) -> str:
        """Concrete backend name used to open the slide."""
        ...

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
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        pad_missing: bool = False,
    ) -> np.ndarray:
        """Read a single region from the slide."""
        ...

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        """Get a thumbnail of the slide."""
        ...

    def close(self) -> None:
        """Release resources."""
        ...

    def __enter__(self) -> SlideReader: ...
    def __exit__(self, *args: Any) -> None: ...


@runtime_checkable
class BatchRegionReader(SlideReader, Protocol):
    """Optional capability for batch reading multiple regions in one call."""

    def read_regions(
        self,
        locations: list[tuple[int, int]],
        level: int,
        size: tuple[int, int],
        *,
        num_workers: int | None = None,
        pad_missing: bool = False,
    ) -> Iterable[np.ndarray]:
        """Read multiple same-sized regions from one pyramid level."""
        ...


@dataclass(frozen=True)
class _BackendSpec:
    name: str
    opener: Callable[[str | Path], SlideReader]
    supports_path: Callable[[str | Path], bool]


_CUCIM_SUPPORTED_SUFFIXES = {".svs", ".tif", ".tiff"}


def _supports_all_paths(path: str | Path) -> bool:
    return True


def _supports_cucim_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _CUCIM_SUPPORTED_SUFFIXES


def _open_openslide(
    path: str | Path, *, spacing_override: float | None = None
) -> SlideReader:
    from soma.wsi.backends.openslide import OpenSlideReader

    return OpenSlideReader(path, spacing_override=spacing_override)


def _open_cucim(
    path: str | Path, *, spacing_override: float | None = None
) -> SlideReader:
    from soma.wsi.backends.cucim import CuCIMReader

    return CuCIMReader(path, spacing_override=spacing_override)


_BACKENDS: dict[str, _BackendSpec] = {
    "openslide": _BackendSpec(
        name="openslide",
        opener=_open_openslide,
        supports_path=_supports_all_paths,
    ),
    "cucim": _BackendSpec(
        name="cucim",
        opener=_open_cucim,
        supports_path=_supports_cucim_path,
    ),
}

_AUTO_BACKEND_ORDER = ("cucim", "openslide")


def open_slide(
    path: str | Path,
    backend: str = "auto",
    *,
    spacing_override: float | None = None,
) -> SlideReader:
    """Open a whole-slide image with the requested backend."""
    backend = (backend or "auto").strip().lower()
    if backend == "auto":
        return _open_slide_auto(path, spacing_override=spacing_override)

    spec = _BACKENDS.get(backend)
    if spec is None:
        available = ", ".join(["auto", *_BACKENDS])
        raise ValueError(f"Unknown backend: '{backend}'. Available: {available}")
    return spec.opener(path, spacing_override=spacing_override)


def _open_slide_auto(
    path: str | Path, *, spacing_override: float | None = None
) -> SlideReader:
    errors: list[ImportError] = []
    attempted = False

    for backend_name in _AUTO_BACKEND_ORDER:
        spec = _BACKENDS.get(backend_name)
        if spec is None or not spec.supports_path(path):
            continue
        attempted = True
        try:
            return spec.opener(path, spacing_override=spacing_override)
        except ImportError as exc:
            errors.append(exc)

    if errors:
        raise errors[-1]
    if attempted:
        raise RuntimeError(f"Unable to open slide with auto backend: {path}")
    raise RuntimeError(f"No registered backend can open slide path: {path}")


@dataclass(frozen=True)
class LevelSelection:
    """Result of selecting a pyramid level for a requested spacing."""

    level: int
    effective_spacing_um: float
    is_within_tolerance: bool


def select_level_for_downsample(
    requested_downsample: float,
    level_downsamples: list[float],
) -> int:
    """Select the stored pyramid level closest to a requested downsample."""
    if len(level_downsamples) == 0:
        raise ValueError("level_downsamples must not be empty")
    return int(
        np.argmin([abs(float(downsample) - requested_downsample) for downsample in level_downsamples])
    )


def select_level(
    requested_spacing_um: float,
    level_downsamples: list[float],
    base_spacing_um: float,
    *,
    tolerance: float = 0.05,
) -> LevelSelection:
    """Select the best pyramid level for a requested spacing."""
    effective_spacings = [base_spacing_um * downsample for downsample in level_downsamples]
    level = int(
        np.argmin(
            [abs(effective_spacing - requested_spacing_um) for effective_spacing in effective_spacings]
        )
    )
    best_spacing = effective_spacings[level]

    relative_error = abs(best_spacing - requested_spacing_um) / requested_spacing_um
    is_within_tolerance = relative_error <= tolerance
    if not is_within_tolerance:
        while level > 0 and best_spacing > requested_spacing_um:
            level -= 1
            best_spacing = effective_spacings[level]
            relative_error = abs(best_spacing - requested_spacing_um) / requested_spacing_um
            is_within_tolerance = relative_error <= tolerance

    return LevelSelection(
        level=level,
        effective_spacing_um=best_spacing,
        is_within_tolerance=is_within_tolerance,
    )
