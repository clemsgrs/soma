"""Tests for soma.encoders.extraction."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from torch import Tensor

from soma.encoders.base import SlideEncoder, TileEncoder
from soma.encoders.extraction import (
    SlideExtractionResult,
    SuperTileBatchSampler,
    TileBatchCollator,
    TileIndexDataset,
    extract_slide_features,
    extract_tile_features,
    save_features,
)
from soma.preprocessing.tiling import TilingResult


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


def _make_grid_tiling(nx: int, ny: int, tile_size_lv0: int = 512) -> TilingResult:
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
    )


def _mock_reader() -> MagicMock:
    reader = MagicMock()

    def _read_region(location, level, size):
        w, h = size
        return np.full((h, w, 3), 128, dtype=np.uint8)

    reader.read_region.side_effect = _read_region
    reader.level_downsamples = [1.0]
    reader.spacing = 0.5
    return reader


class TestTileIndexDataset:
    def test_len(self):
        assert len(TileIndexDataset(100)) == 100

    def test_getitem(self):
        ds = TileIndexDataset(10)
        assert ds[0] == 0
        assert ds[9] == 9


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


class TestSaveFeatures:
    def test_save_and_load(self, tmp_path: Path):
        features = torch.randn(10, 64)
        path = save_features(features, tmp_path, "slide_001")
        assert path.exists()
        assert torch.equal(features, torch.load(path, weights_only=True))

    def test_atomic_save(self, tmp_path: Path):
        save_features(torch.randn(5, 32), tmp_path, "slide_002")
        assert list(tmp_path.glob("*.tmp")) == []
