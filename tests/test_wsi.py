"""Tests for soma.wsi — SlideReader protocol and backends."""

import sys
import types
import numpy as np
import pytest
from PIL import Image

from soma.wsi.reader import BatchRegionReader, SlideReader


# --- Protocol conformance via a synthetic test backend ---


class SyntheticSlideReader:
    """In-memory slide reader for testing. Conforms to SlideReader protocol."""

    def __init__(
        self,
        width: int = 1000,
        height: int = 800,
        spacing: float = 0.5,
        n_levels: int = 3,
        backend_name: str = "synthetic",
    ) -> None:
        self._width = width
        self._height = height
        self._spacing = spacing
        self._n_levels = n_levels
        self._backend_name = backend_name
        # Generate a simple synthetic image with tissue-like regions
        rng = np.random.RandomState(42)
        self._image = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)

    @property
    def backend_name(self) -> str:
        return self._backend_name

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
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
        *,
        pad_missing: bool = False,
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


class SyntheticBatchSlideReader(SyntheticSlideReader):
    def read_regions(
        self,
        locations: list[tuple[int, int]],
        level: int,
        size: tuple[int, int],
        *,
        num_workers: int | None = None,
        pad_missing: bool = False,
    ):
        return [
            self.read_region(location, level, size, pad_missing=pad_missing)
            for location in locations
        ]


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


def test_synthetic_batch_reader_conforms_to_optional_batch_protocol():
    reader = SyntheticBatchSlideReader()
    assert isinstance(reader, BatchRegionReader)
    regions = list(reader.read_regions([(0, 0), (32, 32)], 0, (16, 16), num_workers=2))
    assert len(regions) == 2
    assert all(region.shape == (16, 16, 3) for region in regions)


# --- OpenSlideReader import guard ---


def test_openslide_reader_raises_without_library(tmp_path):
    """OpenSlideReader should raise ImportError if openslide is not installed."""
    from soma.wsi.backends.openslide import OpenSlideReader

    try:
        import openslide  # noqa: F401

        pytest.skip("openslide is installed")
    except ImportError:
        pass

    with pytest.raises(ImportError, match="openslide"):
        OpenSlideReader(tmp_path / "fake.svs")


def test_openslide_reader_raises_when_spacing_metadata_is_missing(monkeypatch, tmp_path):
    from soma.wsi.backends.openslide import OpenSlideReader

    class FakeOpenSlide:
        def __init__(self, path):
            self.path = path
            self.properties = {}

    fake_module = types.SimpleNamespace(OpenSlide=FakeOpenSlide)
    monkeypatch.setitem(sys.modules, "openslide", fake_module)

    with pytest.raises(ValueError, match="spacing"):
        OpenSlideReader(tmp_path / "fake.svs")


def test_openslide_reader_pads_out_of_bounds_regions_white(monkeypatch, tmp_path):
    from soma.wsi.backends.openslide import OpenSlideReader

    class FakeOpenSlide:
        def __init__(self, path):
            self.path = path
            self.properties = {"openslide.mpp-x": "0.5"}
            self.dimensions = (10, 10)
            self.level_count = 1
            self.level_dimensions = [(10, 10)]
            self.level_downsamples = [1.0]

        def read_region(self, location, level, size):
            width, height = size
            arr = np.full((height, width, 4), 64, dtype=np.uint8)
            arr[..., 3] = 255
            return Image.fromarray(arr, mode="RGBA")

        def close(self):
            pass

    fake_module = types.SimpleNamespace(OpenSlide=FakeOpenSlide)
    monkeypatch.setitem(sys.modules, "openslide", fake_module)

    reader = OpenSlideReader(tmp_path / "fake.svs")
    region = reader.read_region((8, 7), 0, (4, 5), pad_missing=True)

    assert region.shape == (5, 4, 3)
    assert np.all(region[:3, :2] == 64)
    assert np.all(region[:, 2:] == 255)
    assert np.all(region[3:, :] == 255)


# --- open_slide factory ---


def test_open_slide_with_unknown_backend():
    from soma.wsi.reader import open_slide

    with pytest.raises(ValueError, match="backend"):
        open_slide("fake.svs", backend="nonexistent")


def test_soma_wsi_exports_protocols_and_factory_only():
    import soma.wsi as wsi

    assert wsi.__all__ == [
        "SlideReader",
        "BatchRegionReader",
        "open_slide",
        "select_level",
        "select_level_for_downsample",
        "LevelSelection",
    ]


def test_open_slide_with_explicit_backend_uses_registered_opener(monkeypatch):
    import soma.wsi.reader as reader_mod

    sentinel = SyntheticSlideReader(backend_name="openslide")

    monkeypatch.setattr(
        reader_mod,
        "_BACKENDS",
        {
            "openslide": reader_mod._BackendSpec(
                name="openslide",
                opener=lambda path, *, spacing_override=None: sentinel,
                supports_path=lambda path: True,
            )
        },
    )
    opened = reader_mod.open_slide("fake.svs", backend="openslide")
    assert opened is sentinel


def test_open_slide_with_auto_prefers_first_supported_backend(monkeypatch):
    import soma.wsi.reader as reader_mod

    seen: list[str] = []
    cucim_reader = SyntheticSlideReader(backend_name="cucim")

    monkeypatch.setattr(
        reader_mod,
        "_BACKENDS",
        {
            "cucim": reader_mod._BackendSpec(
                name="cucim",
                opener=lambda path, *, spacing_override=None: seen.append("cucim") or cucim_reader,
                supports_path=lambda path: True,
            ),
            "openslide": reader_mod._BackendSpec(
                name="openslide",
                opener=lambda path, *, spacing_override=None: seen.append("openslide") or SyntheticSlideReader(),
                supports_path=lambda path: True,
            ),
        },
    )
    monkeypatch.setattr(reader_mod, "_AUTO_BACKEND_ORDER", ("cucim", "openslide"))

    opened = reader_mod.open_slide("fake.svs", backend="auto")
    assert opened is cucim_reader
    assert opened.backend_name == "cucim"
    assert seen == ["cucim"]


def test_open_slide_with_auto_skips_unsupported_backend(monkeypatch):
    import soma.wsi.reader as reader_mod

    seen: list[str] = []
    openslide_reader = SyntheticSlideReader(backend_name="openslide")

    monkeypatch.setattr(
        reader_mod,
        "_BACKENDS",
        {
            "cucim": reader_mod._BackendSpec(
                name="cucim",
                opener=lambda path, *, spacing_override=None: seen.append("cucim") or SyntheticSlideReader(),
                supports_path=lambda path: False,
            ),
            "openslide": reader_mod._BackendSpec(
                name="openslide",
                opener=lambda path, *, spacing_override=None: seen.append("openslide") or openslide_reader,
                supports_path=lambda path: True,
            ),
        },
    )
    monkeypatch.setattr(reader_mod, "_AUTO_BACKEND_ORDER", ("cucim", "openslide"))

    opened = reader_mod.open_slide("fake.svs", backend="auto")
    assert opened is openslide_reader
    assert opened.backend_name == "openslide"
    assert seen == ["openslide"]


def test_cucim_reader_raises_without_library(tmp_path):
    from soma.wsi.backends.cucim import CuCIMReader

    with pytest.raises(ImportError, match="cucim"):
        CuCIMReader(tmp_path / "fake.svs")


def test_cucim_reader_normalizes_arrays_and_supports_batch_reads(monkeypatch, tmp_path):
    from soma.wsi.backends.cucim import CuCIMReader

    metadata = {
        "cucim": {
            "shape": [512, 1024, 4],
            "resolutions": {
                "level_count": 2,
                "level_dimensions": [(1024, 512), (512, 256)],
                "level_downsamples": [1.0, 2.0],
            },
        },
        "aperio": {"MPP": "0.25"},
    }

    class FakeCuImage:
        def __init__(self, path):
            self.path = path
            self.metadata = metadata
            self.calls = []

        def read_region(self, location=None, size=None, level=0, num_workers=None):
            self.calls.append(
                {
                    "location": location,
                    "size": size,
                    "level": level,
                    "num_workers": num_workers,
                }
            )
            if isinstance(location, list):
                return iter(
                    [
                        np.full((size[1], size[0], 4), fill_value=10 + idx, dtype=np.uint8)
                        for idx, _ in enumerate(location)
                    ]
                )
            return np.full((size[1], size[0], 4), fill_value=7, dtype=np.uint8)

    fake_module = types.SimpleNamespace(CuImage=FakeCuImage)
    monkeypatch.setattr(
        "soma.wsi.backends.cucim.importlib.import_module",
        lambda name: fake_module if name == "cucim" else __import__(name),
    )

    reader = CuCIMReader(tmp_path / "slide.svs")
    assert isinstance(reader, BatchRegionReader)
    assert reader.dimensions == (1024, 512)
    assert reader.spacing == pytest.approx(0.25)
    assert reader.level_count == 2
    assert reader.level_dimensions == [(1024, 512), (512, 256)]
    assert reader.level_downsamples == [1.0, 2.0]

    region = reader.read_region((0, 0), 0, (8, 6))
    assert region.shape == (6, 8, 3)
    assert np.all(region[..., 0] == 7)

    regions = list(reader.read_regions([(0, 0), (16, 16)], 1, (4, 5), num_workers=3))
    assert len(regions) == 2
    assert regions[0].shape == (5, 4, 3)
    assert regions[1].shape == (5, 4, 3)

    thumbnail = reader.get_thumbnail((32, 32))
    assert thumbnail.ndim == 3
    assert thumbnail.shape[-1] == 3
    assert reader._slide.path == str(tmp_path / "slide.svs")


def test_cucim_reader_pads_out_of_bounds_regions_white(monkeypatch, tmp_path):
    from soma.wsi.backends.cucim import CuCIMReader

    metadata = {
        "cucim": {
            "shape": [10, 10, 3],
            "resolutions": {
                "level_count": 1,
                "level_dimensions": [(10, 10)],
                "level_downsamples": [1.0],
            },
        },
        "aperio": {"MPP": "0.25"},
    }

    class FakeCuImage:
        def __init__(self, path):
            self.path = path
            self.metadata = metadata

        def read_region(self, location=None, size=None, level=0, num_workers=None):
            width, height = size
            return np.full((height, width, 3), 70, dtype=np.uint8)

    fake_module = types.SimpleNamespace(CuImage=FakeCuImage)
    monkeypatch.setattr(
        "soma.wsi.backends.cucim.importlib.import_module",
        lambda name: fake_module if name == "cucim" else __import__(name),
    )

    reader = CuCIMReader(tmp_path / "slide.svs")
    region = reader.read_region((8, 7), 0, (4, 5), pad_missing=True)
    regions = list(reader.read_regions([(8, 7)], 0, (4, 5), pad_missing=True))

    assert region.shape == (5, 4, 3)
    assert np.all(region[:3, :2] == 70)
    assert np.all(region[:, 2:] == 255)
    assert np.all(region[3:, :] == 255)
    assert len(regions) == 1
    np.testing.assert_array_equal(regions[0], region)


def test_cucim_reader_raises_when_spacing_metadata_is_missing(monkeypatch, tmp_path):
    from soma.wsi.backends.cucim import CuCIMReader

    metadata = {
        "cucim": {
            "shape": [512, 1024, 4],
            "resolutions": {
                "level_count": 1,
                "level_dimensions": [(1024, 512)],
                "level_downsamples": [1.0],
            },
        },
        "vendor": {"unexpected": "value"},
    }

    class FakeCuImage:
        def __init__(self, path):
            self.path = path
            self.metadata = metadata

    fake_module = types.SimpleNamespace(CuImage=FakeCuImage)
    monkeypatch.setattr(
        "soma.wsi.backends.cucim.importlib.import_module",
        lambda name: fake_module if name == "cucim" else __import__(name),
    )

    with pytest.raises(ValueError, match="spacing"):
        CuCIMReader(tmp_path / "slide.svs")
