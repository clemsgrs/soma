"""Tests for soma.encoders.distributed — multi-GPU feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from soma.encoders.base import SlideEncoder, TileEncoder
from soma.encoders.distributed import (
    SlideTask,
    _assign_slides_to_ranks,
    extract_dataset,
)
from soma.encoders.registry import encoder_registry
from soma.preprocessing.tiling import TilingResult

_TEST_TILE_ENCODER_NAME = "_test_distributed_tile"
_TEST_SLIDE_ENCODER_NAME = "_test_distributed_slide"
_TEST_DIM = 8
_TEST_SLIDE_DIM = 4


class _TestTileEncoder(TileEncoder):
    def __init__(self, **kwargs):
        self._device = torch.device("cpu")
        self._output_variant = kwargs.get("output_variant") or "default"

    def get_transform(self):
        def _t(img):
            return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        return _t

    def encode_tiles(self, batch):
        return torch.ones(batch.shape[0], _TEST_DIM)

    @property
    def encode_dim(self):
        return _TEST_DIM

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


class _TestSlideEncoder(SlideEncoder):
    def __init__(self, **kwargs):
        self._device = torch.device("cpu")
        self._output_variant = kwargs.get("output_variant") or "default"

    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0: int | None = None):
        return torch.arange(_TEST_SLIDE_DIM, dtype=torch.float32, device=tile_features.device)

    @property
    def encode_dim(self):
        return _TEST_SLIDE_DIM

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


if _TEST_TILE_ENCODER_NAME not in encoder_registry:
    encoder_registry.register(
        _TEST_TILE_ENCODER_NAME,
        _TestTileEncoder,
        metadata={
            "level": "tile",
            "input_size": 224,
            "output_variants": {"default": {"encode_dim": _TEST_DIM}},
            "default_output_variant": "default",
            "supported_spacing_um": 0.5,
            "precision": "fp16",
        },
    )

if _TEST_SLIDE_ENCODER_NAME not in encoder_registry:
    encoder_registry.register(
        _TEST_SLIDE_ENCODER_NAME,
        _TestSlideEncoder,
        metadata={
            "level": "slide",
            "tile_encoder": _TEST_TILE_ENCODER_NAME,
            "tile_encoder_output_variant": "default",
            "output_variants": {"default": {"encode_dim": _TEST_SLIDE_DIM}},
            "default_output_variant": "default",
            "supported_spacing_um": 0.5,
            "precision": "fp16",
        },
    )


def _make_slide_task(
    slide_id: str, nx: int = 4, ny: int = 4, tile_size_lv0: int = 512
) -> SlideTask:
    coords = []
    for y in range(ny):
        for x in range(nx):
            coords.append([x * tile_size_lv0, y * tile_size_lv0])
    tiling = TilingResult(
        coordinates=np.array(coords, dtype=np.int64),
        tissue_fractions=np.ones(len(coords), dtype=np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=0,
        effective_tile_size_px=256,
        effective_spacing_um=0.5,
        tile_size_lv0=tile_size_lv0,
        is_within_tolerance=True,
        base_spacing_um=0.5,
    )
    return SlideTask(
        slide_path=f"/fake/path/{slide_id}.svs",
        tiling_result=tiling,
        slide_id=slide_id,
    )


def _mock_open_slide(*args, **kwargs):
    reader = MagicMock()

    def _read_region(location, level, size, *, pad_missing=False):
        w, h = size
        return np.full((h, w, 3), 128, dtype=np.uint8)

    reader.read_region.side_effect = _read_region
    reader.level_downsamples = [1.0]
    reader.spacing = 0.5
    reader.close.return_value = None
    return reader


class TestAssignSlidesToRanks:
    def test_balanced(self):
        slides = [
            _make_slide_task("a", 10, 10),
            _make_slide_task("b", 10, 8),
            _make_slide_task("c", 10, 6),
            _make_slide_task("d", 10, 4),
            _make_slide_task("e", 10, 2),
        ]
        assignments = _assign_slides_to_ranks(slides, 2)
        assert [s.slide_id for s in assignments[0]] == ["a", "d", "e"]
        assert [s.slide_id for s in assignments[1]] == ["b", "c"]


class TestExtractDataset:
    def test_requires_level_metadata(self, tmp_path: Path):
        encoder_name = "_test_distributed_missing_level"
        if encoder_name not in encoder_registry:
            encoder_registry.register(
                encoder_name,
                _TestTileEncoder,
                metadata={
                    "input_size": 224,
                    "output_variants": {"default": {"encode_dim": _TEST_DIM}},
                    "default_output_variant": "default",
                    "supported_spacing_um": 0.5,
                    "precision": "fp16",
                },
            )

        with pytest.raises(ValueError, match="level metadata"):
            extract_dataset(
                encoder_name,
                [_make_slide_task("missing_level", 2, 2)],
                tmp_path,
                num_gpus=1,
                batch_size=4,
                num_workers=0,
                progress=False,
            )

    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_creates_tile_files(self, mock_open, tmp_path: Path):
        slides = [_make_slide_task(f"slide_{i}", 2, 2) for i in range(3)]
        summary = extract_dataset(
            _TEST_TILE_ENCODER_NAME,
            slides,
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            progress=False,
        )
        assert len(summary.completed) == 3
        for i in range(3):
            feats = torch.load(tmp_path / f"slide_{i}.pt", weights_only=True)
            assert feats.shape == (4, _TEST_DIM)

    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_creates_slide_files(self, mock_open, tmp_path: Path):
        slides = [_make_slide_task(f"slide_{i}", 2, 2) for i in range(2)]
        summary = extract_dataset(
            _TEST_SLIDE_ENCODER_NAME,
            slides,
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            progress=False,
        )
        assert len(summary.completed) == 2
        for i in range(2):
            feats = torch.load(tmp_path / f"slide_{i}.pt", weights_only=True)
            assert feats.shape == (_TEST_SLIDE_DIM,)

    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_slide_extraction_can_save_intermediate_tile_features(
        self, mock_open, tmp_path: Path
    ):
        slides = [_make_slide_task("slide_0", 2, 2)]
        extract_dataset(
            _TEST_SLIDE_ENCODER_NAME,
            slides,
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            progress=False,
            save_tile_features=True,
        )
        tile_feats = torch.load(
            tmp_path / "tile_features" / "slide_0.pt",
            weights_only=True,
        )
        assert tile_feats.shape == (4, _TEST_DIM)

    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_skip_existing(self, mock_open, tmp_path: Path):
        existing = torch.randn(_TEST_SLIDE_DIM)
        torch.save(existing, tmp_path / "slide_0.pt")
        summary = extract_dataset(
            _TEST_SLIDE_ENCODER_NAME,
            [_make_slide_task(f"slide_{i}", 2, 2) for i in range(3)],
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            progress=False,
        )
        assert "slide_0" in summary.skipped
        assert len(summary.completed) == 2
        assert torch.equal(
            torch.load(tmp_path / "slide_0.pt", weights_only=True),
            existing,
        )

    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_progress_events_emitted(self, mock_open, tmp_path: Path):
        extract_dataset(
            _TEST_SLIDE_ENCODER_NAME,
            [_make_slide_task("prog_slide", 2, 2)],
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            progress=True,
        )
        log_path = tmp_path / ".progress" / "rank_0.jsonl"
        events = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        kinds = [e["kind"] for e in events]
        assert "extraction.started" in kinds
        assert "extraction.slide.started" in kinds
        assert "extraction.slide.completed" in kinds
        assert "extraction.completed" in kinds

    @patch("soma.encoders.distributed.open_slide")
    def test_error_handling_raise(self, mock_open, tmp_path: Path):
        mock_open.side_effect = RuntimeError("Simulated failure")
        with pytest.raises(RuntimeError, match="Simulated failure"):
            extract_dataset(
                _TEST_TILE_ENCODER_NAME,
                [_make_slide_task("fail_slide", 2, 2)],
                tmp_path,
                num_gpus=1,
                num_workers=0,
                error_handling="raise",
                progress=False,
            )
