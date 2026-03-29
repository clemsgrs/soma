"""Tests for soma.encoders.extraction."""

from __future__ import annotations

import ctypes
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import Tensor

from soma.encoders.base import SlideEncoder, TileEncoder
from soma.encoders.extraction import (
    SlideExtractionResult,
    SuperTileBatchSampler,
    TileBatchCollator,
    TileIndexDataset,
    _build_loader,
    extract_slide_features,
    extract_tile_features,
    save_features,
)
from soma.encoders.tile_reader import build_supertile_index
from soma.preprocessing.tiling import TilingResult
from soma.wsi.reader import BatchRegionReader


def _pixel_value_transform(img: np.ndarray) -> Tensor:
    return torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)


class MockTileEncoder(TileEncoder):
    def __init__(self, dim: int = 8):
        self._dim = dim
        self._device = torch.device("cpu")

    def get_transform(self):
        def _t(img: np.ndarray) -> Tensor:
            return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        return _t

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return torch.ones(batch.shape[0], self._dim)

    @property
    def encode_dim(self) -> int:
        return self._dim

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


class MockSlideEncoder(SlideEncoder):
    def __init__(self, dim: int = 4):
        self._dim = dim
        self._device = torch.device("cpu")

    def encode_slide(
        self,
        tile_features: Tensor,
        coordinates: Tensor | None = None,
        *,
        tile_size_lv0: int | None = None,
    ) -> Tensor:
        assert coordinates is not None
        xy_mean = coordinates.float().mean(dim=0)
        return torch.cat([tile_features.mean(dim=0)[:2], xy_mean])[: self._dim]

    @property
    def encode_dim(self) -> int:
        return self._dim

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


class RecordingSlideEncoder(MockSlideEncoder):
    def __init__(self, dim: int = 4):
        super().__init__(dim=dim)
        self.prepare_calls: list[dict[str, object]] = []

    def prepare_coordinates(
        self,
        coordinates: Tensor,
        *,
        base_spacing_um: float,
        target_spacing_um: float,
    ) -> Tensor:
        self.prepare_calls.append(
            {
                "coordinates": coordinates.clone(),
                "base_spacing_um": base_spacing_um,
                "target_spacing_um": target_spacing_um,
            }
        )
        return coordinates


class PixelValueTileEncoder(TileEncoder):
    def __init__(self):
        self._device = torch.device("cpu")

    def get_transform(self):
        return _pixel_value_transform

    def encode_tiles(self, batch: Tensor) -> Tensor:
        values = batch[:, 0, 0, 0].unsqueeze(1)
        return values

    @property
    def encode_dim(self) -> int:
        return 1

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


def _make_grid_tiling(
    nx: int,
    ny: int,
    tile_size_lv0: int = 512,
    *,
    base_spacing_um: float = 0.5,
) -> TilingResult:
    coords = []
    for y in range(ny):
        for x in range(nx):
            coords.append([x * tile_size_lv0, y * tile_size_lv0])
    return TilingResult(
        coordinates=np.array(coords, dtype=np.int64),
        tissue_fractions=np.ones(len(coords), dtype=np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=0,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=tile_size_lv0,
        is_within_tolerance=True,
        base_spacing_um=base_spacing_um,
    )


def _mock_reader() -> MagicMock:
    reader = MagicMock()

    def _read_region(location, level, size, *, pad_missing=False):
        w, h = size
        return np.full((h, w, 3), 128, dtype=np.uint8)

    reader.read_region.side_effect = _read_region
    reader.level_downsamples = [1.0]
    reader.spacing = 0.5
    return reader


class RecordingBatchReader:
    def __init__(self):
        self.backend_name = "recording"
        self.dimensions = (1024, 1024)
        self.level_downsamples = [1.0]
        self.level_count = 1
        self.level_dimensions = [(1024, 1024)]
        self.spacing = 0.5
        self.read_region_calls: list[dict[str, object]] = []
        self.read_regions_calls: list[dict[str, object]] = []

    def read_region(self, location, level, size, *, pad_missing=False):
        self.read_region_calls.append(
            {
                "location": location,
                "level": level,
                "size": size,
                "pad_missing": pad_missing,
            }
        )
        w, h = size
        value = int(location[0] // 512) + 10 * int(location[1] // 512)
        return np.full((h, w, 3), value, dtype=np.uint8)

    def read_regions(self, locations, level, size, *, num_workers=None, pad_missing=False):
        self.read_regions_calls.append(
            {
                "locations": list(locations),
                "level": level,
                "size": size,
                "num_workers": num_workers,
                "pad_missing": pad_missing,
            }
        )
        w, h = size
        for location in locations:
            value = int(location[0] // 512) + 10 * int(location[1] // 512)
            yield np.full((h, w, 3), value, dtype=np.uint8)

    def get_thumbnail(self, size):
        w, h = size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


assert isinstance(RecordingBatchReader(), BatchRegionReader)


class PatternBatchReader:
    def __init__(self, tile_span: int = 256):
        self.backend_name = "pattern"
        self.dimensions = (4096, 4096)
        self.level_downsamples = [1.0]
        self.level_count = 1
        self.level_dimensions = [self.dimensions]
        self.spacing = 0.5
        self.tile_span = tile_span
        self.read_region_calls: list[dict[str, object]] = []
        self.read_regions_calls: list[dict[str, object]] = []

    def _region(self, location, size):
        w, h = size
        xs = location[0] + np.arange(w, dtype=np.int64)
        ys = location[1] + np.arange(h, dtype=np.int64)
        grid_x = (xs // self.tile_span).astype(np.uint8)
        grid_y = (ys // self.tile_span).astype(np.uint8)
        values = grid_y[:, None] * 10 + grid_x[None, :]
        return np.repeat(values[:, :, None], 3, axis=2)

    def read_region(self, location, level, size, *, pad_missing=False):
        self.read_region_calls.append(
            {
                "location": location,
                "level": level,
                "size": size,
                "pad_missing": pad_missing,
            }
        )
        return self._region(location, size)

    def read_regions(self, locations, level, size, *, num_workers=None, pad_missing=False):
        self.read_regions_calls.append(
            {
                "locations": list(locations),
                "level": level,
                "size": size,
                "num_workers": num_workers,
                "pad_missing": pad_missing,
            }
        )
        for location in locations:
            yield self._region(location, size)

    def get_thumbnail(self, size):
        w, h = size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


assert isinstance(PatternBatchReader(), BatchRegionReader)


class CtypesReader:
    def __init__(self):
        self.backend_name = "ctypes"
        self.dimensions = (1024, 1024)
        self.level_downsamples = [1.0]
        self.level_count = 1
        self.level_dimensions = [self.dimensions]
        self.spacing = 0.5
        self._pointer = ctypes.pointer(ctypes.c_int(1))

    def read_region(self, location, level, size, *, pad_missing=False):
        w, h = size
        return np.full((h, w, 3), 7, dtype=np.uint8)

    def get_thumbnail(self, size):
        w, h = size
        return np.zeros((h, w, 3), dtype=np.uint8)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class TestTileIndexDataset:
    def test_len(self):
        assert len(TileIndexDataset(np.arange(100, dtype=np.int64))) == 100

    def test_getitem(self):
        ds = TileIndexDataset(np.array([7, 3, 11], dtype=np.int64))
        assert ds[0] == 7
        assert ds[2] == 11


class TestSuperTileBatchSampler:
    def test_all_indices_covered(self):
        sampler = SuperTileBatchSampler([np.arange(0, 16), np.arange(16, 20)], 32)
        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)
        assert sorted(all_indices) == list(range(20))

    def test_single_large_group(self):
        batches = list(SuperTileBatchSampler([np.arange(0, 64)], 32))
        assert len(batches) == 1
        assert len(batches[0]) == 64


class TestExtractTileFeatures:
    def test_supertile_loader_reorders_dataset_indices(self):
        tiling = _make_grid_tiling(9, 9, tile_size_lv0=256)
        st_index = build_supertile_index(tiling)

        loader = _build_loader(
            PixelValueTileEncoder(),
            RecordingBatchReader(),
            tiling,
            batch_size=64,
            adaptive_batching=False,
            num_workers=0,
            use_supertiles=True,
        )

        batch_indices, _ = next(iter(loader))
        assert batch_indices.tolist() == st_index.ordered_indices[:64].tolist()
        assert batch_indices.tolist() != list(range(64))

    def test_supertile_batching_defaults_to_fixed_batch_size(self):
        reader = RecordingBatchReader()
        features = extract_tile_features(
            PixelValueTileEncoder(),
            reader,
            _make_grid_tiling(4, 4, tile_size_lv0=256),
            batch_size=8,
            num_workers=0,
            use_supertiles=True,
        )

        assert features.shape == (16, 1)
        assert reader.read_region_calls == []
        assert len(reader.read_regions_calls) == 2
        assert reader.read_regions_calls == [
            {
                "locations": [(0, 0)],
                "level": 0,
                "size": (1024, 1024),
                "num_workers": 0,
                "pad_missing": True,
            },
            {
                "locations": [(0, 0)],
                "level": 0,
                "size": (1024, 1024),
                "num_workers": 0,
                "pad_missing": True,
            },
        ]

    def test_adaptive_batching_keeps_large_group_intact(self):
        reader = RecordingBatchReader()
        features = extract_tile_features(
            PixelValueTileEncoder(),
            reader,
            _make_grid_tiling(4, 4, tile_size_lv0=256),
            batch_size=8,
            adaptive_batching=True,
            num_workers=0,
            use_supertiles=True,
        )

        assert features.shape == (16, 1)
        assert reader.read_region_calls == []
        assert len(reader.read_regions_calls) == 1
        assert reader.read_regions_calls[0] == {
            "locations": [(0, 0)],
            "level": 0,
            "size": (1024, 1024),
            "num_workers": 0,
            "pad_missing": True,
        }

    def test_supertile_reordering_preserves_feature_positions_for_mixed_sizes(self):
        reader = PatternBatchReader(tile_span=256)
        features = extract_tile_features(
            PixelValueTileEncoder(),
            reader,
            _make_grid_tiling(10, 10, tile_size_lv0=256),
            batch_size=32,
            num_workers=0,
            use_supertiles=True,
        )

        expected = [
            float(x + 10 * y)
            for y in range(10)
            for x in range(10)
        ]
        assert features.squeeze(1).tolist() == expected

    def test_output_shape(self):
        features = extract_tile_features(
            MockTileEncoder(dim=8),
            _mock_reader(),
            _make_grid_tiling(4, 4),
            batch_size=8,
            num_workers=0,
        )
        assert features.shape == (16, 8)

    def test_dtype_float32(self):
        features = extract_tile_features(
            MockTileEncoder(dim=4),
            _mock_reader(),
            _make_grid_tiling(2, 2),
            batch_size=4,
            num_workers=0,
        )
        assert features.dtype == torch.float32

    def test_empty_tiling(self):
        tiling = TilingResult(
            coordinates=np.empty((0, 2), dtype=np.int64),
            tissue_fractions=np.empty((0,), dtype=np.float32),
            requested_tile_size_px=256,
            requested_spacing_um=0.5,
            read_level=0,
            effective_tile_size_px=256,
            effective_spacing_um=0.5,
            tile_size_lv0=512,
            is_within_tolerance=True,
        )
        features = extract_tile_features(
            MockTileEncoder(dim=8),
            _mock_reader(),
            tiling,
            batch_size=4,
            num_workers=0,
        )
        assert features.shape == (0, 8)

    def test_batch_capable_reader_uses_read_regions_and_preserves_order(self):
        reader = RecordingBatchReader()
        features = extract_tile_features(
            PixelValueTileEncoder(),
            reader,
            _make_grid_tiling(2, 2, tile_size_lv0=512),
            batch_size=4,
            num_workers=0,
            use_supertiles=False,
        )
        assert reader.read_region_calls == []
        assert len(reader.read_regions_calls) == 1
        assert reader.read_regions_calls[0] == {
            "locations": [(0, 0), (512, 0), (0, 512), (512, 512)],
            "level": 0,
            "size": (256, 256),
            "num_workers": 0,
            "pad_missing": True,
        }
        assert features.squeeze(1).tolist() == [0.0, 1.0, 10.0, 11.0]

    def test_batch_capable_reader_batches_supertile_reads(self):
        reader = RecordingBatchReader()
        features = extract_tile_features(
            PixelValueTileEncoder(),
            reader,
            _make_grid_tiling(2, 2, tile_size_lv0=256),
            batch_size=4,
            adaptive_batching=True,
            num_workers=0,
            use_supertiles=True,
        )
        assert features.shape == (4, 1)
        assert reader.read_region_calls == []
        assert len(reader.read_regions_calls) == 1
        assert reader.read_regions_calls[0] == {
            "locations": [(0, 0)],
            "level": 0,
            "size": (512, 512),
            "num_workers": 0,
            "pad_missing": True,
        }

    def test_sequential_reader_forwards_use_padding_to_read_region(self):
        reader = _mock_reader()
        tiling = _make_grid_tiling(1, 1, tile_size_lv0=256)
        collator = TileBatchCollator(
            reader,
            tiling,
            MockTileEncoder().get_transform(),
            num_workers=0,
        )

        collator([0])

        reader.read_region.assert_called_once_with(
            (0, 0),
            0,
            (256, 256),
            pad_missing=True,
        )

    @patch("soma.encoders.extraction.open_slide")
    def test_collator_pickle_reopens_reader_without_serializing_ctypes_handles(
        self,
        mock_open,
    ):
        reopened_reader = RecordingBatchReader()
        mock_open.return_value = reopened_reader
        tiling = _make_grid_tiling(1, 1, tile_size_lv0=256)
        collator = TileBatchCollator(
            CtypesReader(),
            tiling,
            _pixel_value_transform,
            num_workers=1,
            slide_path="/tmp/test-wsi.svs",
            backend="openslide",
        )

        restored = pickle.loads(pickle.dumps(collator))
        indices, images = restored([0])

        mock_open.assert_called_once_with("/tmp/test-wsi.svs", backend="openslide")
        assert indices.tolist() == [0]
        assert images.shape == (1, 3, 256, 256)
        assert reopened_reader.read_region_calls == []
        assert reopened_reader.read_regions_calls == [
            {
                "locations": [(0, 0)],
                "level": 0,
                "size": (256, 256),
                "num_workers": 1,
                "pad_missing": True,
            }
        ]


class TestExtractSlideFeatures:
    def test_returns_slide_result(self):
        result = extract_slide_features(
            MockSlideEncoder(dim=4),
            MockTileEncoder(dim=8),
            _mock_reader(),
            _make_grid_tiling(2, 2),
            batch_size=4,
            num_workers=0,
        )
        assert isinstance(result, SlideExtractionResult)
        assert result.slide_features.shape == (4,)
        assert result.tile_features is None

    def test_optionally_returns_tile_features(self):
        result = extract_slide_features(
            MockSlideEncoder(dim=4),
            MockTileEncoder(dim=8),
            _mock_reader(),
            _make_grid_tiling(2, 2),
            batch_size=4,
            num_workers=0,
            return_tile_features=True,
        )
        assert result.slide_features.shape == (4,)
        assert result.tile_features is not None
        assert result.tile_features.shape == (4, 8)

    def test_prepare_coordinates_uses_requested_spacing_not_effective_read_spacing(self):
        reader = _mock_reader()
        reader.spacing = 0.25
        tiling = _make_grid_tiling(2, 2, base_spacing_um=0.2)
        tiling = TilingResult(
            coordinates=tiling.coordinates,
            tissue_fractions=tiling.tissue_fractions,
            requested_tile_size_px=tiling.requested_tile_size_px,
            requested_spacing_um=0.75,
            read_level=tiling.read_level,
            effective_tile_size_px=384,
            effective_spacing_um=0.5,
            tile_size_lv0=tiling.tile_size_lv0,
            is_within_tolerance=False,
            use_padding=tiling.use_padding,
            tissue_mask=tiling.tissue_mask,
            base_spacing_um=tiling.base_spacing_um,
        )
        slide_encoder = RecordingSlideEncoder(dim=4)

        extract_slide_features(
            slide_encoder,
            MockTileEncoder(dim=8),
            reader,
            tiling,
            batch_size=4,
            num_workers=0,
        )

        assert len(slide_encoder.prepare_calls) == 1
        prepare_call = slide_encoder.prepare_calls[0]
        assert torch.equal(
            prepare_call["coordinates"],
            torch.as_tensor(tiling.coordinates, dtype=torch.long),
        )
        assert prepare_call["base_spacing_um"] == 0.2
        assert prepare_call["target_spacing_um"] == 0.75


class TestSaveFeatures:
    def test_save_and_load(self, tmp_path: Path):
        features = torch.randn(10, 64)
        path = save_features(features, tmp_path, "slide_001")
        assert path.exists()
        assert torch.equal(features, torch.load(path, weights_only=True))

    def test_atomic_save(self, tmp_path: Path):
        save_features(torch.randn(5, 32), tmp_path, "slide_002")
        assert list(tmp_path.glob("*.tmp")) == []

    def test_save_reorders_tile_features_by_tile_index(self, tmp_path: Path):
        features = torch.tensor(
            [[30.0, 300.0], [10.0, 100.0], [20.0, 200.0]],
            dtype=torch.float32,
        )
        tile_index = torch.tensor([2, 0, 1], dtype=torch.long)

        path = save_features(
            features,
            tmp_path,
            "slide_003",
            tile_index=tile_index,
        )

        loaded = torch.load(path, weights_only=True)
        expected = torch.tensor(
            [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]],
            dtype=torch.float32,
        )
        assert torch.equal(loaded, expected)
