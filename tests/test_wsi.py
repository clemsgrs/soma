"""Tests for soma.wsi — SlideReader protocol and backends."""

import numpy as np
import pytest

from soma.wsi.reader import OpenSlideReader, SlideReader


# --- Protocol conformance via a synthetic test backend ---


class SyntheticSlideReader:
    """In-memory slide reader for testing. Conforms to SlideReader protocol."""

    def __init__(
        self,
        width: int = 1000,
        height: int = 800,
        spacing: float = 0.5,
        n_levels: int = 3,
    ) -> None:
        self._width = width
        self._height = height
        self._spacing = spacing
        self._n_levels = n_levels
        # Generate a simple synthetic image with tissue-like regions
        rng = np.random.RandomState(42)
        self._image = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)

    @property
    def dimensions(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def level_count(self) -> int:
        return self._n_levels

    @property
    def level_dimensions(self) -> list[tuple[int, int]]:
        dims = []
        for i in range(self._n_levels):
            factor = 2**i
            dims.append((self._width // factor, self._height // factor))
        return dims

    @property
    def level_downsamples(self) -> list[float]:
        return [float(2**i) for i in range(self._n_levels)]

    def read_region(
        self, location: tuple[int, int], level: int, size: tuple[int, int]
    ) -> np.ndarray:
        x, y = location
        w, h = size
        downsample = int(self.level_downsamples[level])
        # Read from the base image at the corresponding coordinates
        x0 = x
        y0 = y
        x1 = min(x0 + w * downsample, self._width)
        y1 = min(y0 + h * downsample, self._height)
        region = self._image[y0:y1:downsample, x0:x1:downsample]
        # Pad if region is smaller than requested
        if region.shape[0] < h or region.shape[1] < w:
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            padded[: region.shape[0], : region.shape[1]] = region
            return padded
        return region[:h, :w]

    def get_thumbnail(self, size: tuple[int, int]) -> np.ndarray:
        from PIL import Image

        img = Image.fromarray(self._image)
        img.thumbnail(size)
        return np.array(img)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# --- Tests ---


def test_synthetic_reader_dimensions():
    reader = SyntheticSlideReader(width=1000, height=800)
    assert reader.dimensions == (1000, 800)


def test_synthetic_reader_spacing():
    reader = SyntheticSlideReader(spacing=0.25)
    assert reader.spacing == 0.25


def test_synthetic_reader_level_dimensions():
    reader = SyntheticSlideReader(width=1000, height=800, n_levels=3)
    dims = reader.level_dimensions
    assert len(dims) == 3
    assert dims[0] == (1000, 800)
    assert dims[1] == (500, 400)
    assert dims[2] == (250, 200)


def test_synthetic_reader_read_region():
    reader = SyntheticSlideReader(width=1000, height=800)
    region = reader.read_region(location=(0, 0), level=0, size=(256, 256))
    assert region.shape == (256, 256, 3)
    assert region.dtype == np.uint8


def test_synthetic_reader_read_region_at_level1():
    reader = SyntheticSlideReader(width=1000, height=800)
    region = reader.read_region(location=(0, 0), level=1, size=(256, 256))
    assert region.shape == (256, 256, 3)


def test_synthetic_reader_context_manager():
    with SyntheticSlideReader() as reader:
        assert reader.dimensions == (1000, 800)


def test_synthetic_reader_level_downsamples():
    reader = SyntheticSlideReader(n_levels=3)
    ds = reader.level_downsamples
    assert ds == [1.0, 2.0, 4.0]


def test_synthetic_reader_level_count():
    reader = SyntheticSlideReader(n_levels=4)
    assert reader.level_count == 4


# --- OpenSlideReader import guard ---


def test_openslide_reader_raises_without_library(tmp_path):
    """OpenSlideReader should raise ImportError if openslide is not installed."""
    try:
        import openslide  # noqa: F401

        pytest.skip("openslide is installed")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="openslide"):
        OpenSlideReader(tmp_path / "fake.svs")


# --- open_slide factory ---


def test_open_slide_with_unknown_backend():
    from soma.wsi.reader import open_slide

    with pytest.raises(ValueError, match="backend"):
        open_slide("fake.svs", backend="nonexistent")
