"""Tests for soma.encoders.distributed — multi-GPU feature extraction."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import Tensor

from soma.encoders.base import Encoder
from soma.encoders.distributed import (
    ExtractionSummary,
    SlideTask,
    _assign_slides_to_ranks,
    extract_dataset,
)
from soma.encoders.registry import encoder_registry
from soma.preprocessing.tiling import TilingResult


# ---------------------------------------------------------------------------
# Test encoder — registered once at module level
# ---------------------------------------------------------------------------

_TEST_ENCODER_NAME = "_test_distributed"
_TEST_DIM = 8


class _TestEncoder(Encoder):
    def __init__(self, **kwargs):
        self._device = torch.device("cpu")

    def get_transform(self):
        def _t(img):
            return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        return _t

    def encode(self, batch: Tensor) -> Tensor:
        return torch.ones(batch.shape[0], _TEST_DIM)

    @property
    def encode_dim(self) -> int:
        return _TEST_DIM

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


# Register if not already registered (tests may re-import)
if _TEST_ENCODER_NAME not in encoder_registry:
    encoder_registry.register(_TEST_ENCODER_NAME, _TestEncoder)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    )
    return SlideTask(
        slide_path=f"/fake/path/{slide_id}.svs",
        tiling_result=tiling,
        slide_id=slide_id,
    )


def _mock_open_slide(*args, **kwargs):
    reader = MagicMock()

    def _read_region(location, level, size):
        w, h = size
        return np.full((h, w, 3), 128, dtype=np.uint8)

    reader.read_region.side_effect = _read_region
    reader.level_downsamples = [1.0]
    reader.close.return_value = None
    return reader


# ---------------------------------------------------------------------------
# Assignment tests
# ---------------------------------------------------------------------------


class TestAssignSlidesToRanks:
    def test_balanced(self):
        """5 slides [100,80,60,40,20] tiles, 2 ranks → load-balanced."""
        slides = [
            _make_slide_task("a", 10, 10),  # 100 tiles
            _make_slide_task("b", 10, 8),  # 80 tiles
            _make_slide_task("c", 10, 6),  # 60 tiles
            _make_slide_task("d", 10, 4),  # 40 tiles
            _make_slide_task("e", 10, 2),  # 20 tiles
        ]
        assignments = _assign_slides_to_ranks(slides, 2)
        rank0_ids = [s.slide_id for s in assignments[0]]
        rank1_ids = [s.slide_id for s in assignments[1]]
        # Heaviest first: a(100)->r0, b(80)->r1, c(60)->r1(140), d(40)->r0(140), e(20)->r0(160)
        assert rank0_ids == ["a", "d", "e"]
        assert rank1_ids == ["b", "c"]

    def test_single_rank(self):
        slides = [_make_slide_task(f"s{i}", 2, 2) for i in range(3)]
        assignments = _assign_slides_to_ranks(slides, 1)
        assert len(assignments) == 1
        assert len(assignments[0]) == 3

    def test_more_ranks_than_slides(self):
        slides = [_make_slide_task("x", 4, 4), _make_slide_task("y", 4, 4)]
        assignments = _assign_slides_to_ranks(slides, 4)
        assert len(assignments) == 4
        non_empty = [a for a in assignments if len(a) > 0]
        assert len(non_empty) == 2

    def test_empty(self):
        assignments = _assign_slides_to_ranks([], 2)
        assert len(assignments) == 2
        assert all(len(a) == 0 for a in assignments)


# ---------------------------------------------------------------------------
# Integration tests (num_gpus=1, no actual spawn)
# ---------------------------------------------------------------------------


class TestExtractDataset:
    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_creates_files(self, mock_open, tmp_path: Path):
        slides = [_make_slide_task(f"slide_{i}", 2, 2) for i in range(3)]
        summary = extract_dataset(
            _TEST_ENCODER_NAME,
            slides,
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            progress=False,
        )
        assert len(summary.completed) == 3
        assert len(summary.skipped) == 0
        assert len(summary.failed) == 0
        for i in range(3):
            pt_path = tmp_path / f"slide_{i}.pt"
            assert pt_path.exists()
            feats = torch.load(pt_path, weights_only=True)
            assert feats.shape == (4, _TEST_DIM)

    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_skip_existing(self, mock_open, tmp_path: Path):
        # Pre-create one file
        existing = torch.randn(4, _TEST_DIM)
        torch.save(existing, tmp_path / "slide_0.pt")

        slides = [_make_slide_task(f"slide_{i}", 2, 2) for i in range(3)]
        summary = extract_dataset(
            _TEST_ENCODER_NAME,
            slides,
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            progress=False,
        )
        assert "slide_0" in summary.skipped
        assert len(summary.completed) == 2
        # Verify original file was NOT overwritten
        loaded = torch.load(tmp_path / "slide_0.pt", weights_only=True)
        assert torch.equal(loaded, existing)

    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_empty_slides(self, mock_open, tmp_path: Path):
        summary = extract_dataset(
            _TEST_ENCODER_NAME,
            [],
            tmp_path,
            num_gpus=1,
            num_workers=0,
            progress=False,
        )
        assert summary.completed == []
        assert summary.skipped == []
        assert summary.failed == []
        assert summary.duration_s >= 0

    @patch("soma.encoders.distributed.open_slide")
    def test_error_handling_skip(self, mock_open, tmp_path: Path):
        call_count = 0

        def _side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Simulated slide open failure")
            return _mock_open_slide()

        mock_open.side_effect = _side_effect

        slides = [_make_slide_task(f"slide_{i}", 2, 2) for i in range(3)]
        summary = extract_dataset(
            _TEST_ENCODER_NAME,
            slides,
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            error_handling="skip",
            progress=False,
        )
        assert len(summary.failed) == 1
        assert len(summary.completed) == 2

    @patch("soma.encoders.distributed.open_slide")
    def test_error_handling_raise(self, mock_open, tmp_path: Path):
        def _side_effect(*args, **kwargs):
            raise RuntimeError("Simulated failure")

        mock_open.side_effect = _side_effect

        slides = [_make_slide_task("fail_slide", 2, 2)]
        with pytest.raises(RuntimeError, match="Simulated failure"):
            extract_dataset(
                _TEST_ENCODER_NAME,
                slides,
                tmp_path,
                num_gpus=1,
                num_workers=0,
                error_handling="raise",
                progress=False,
            )

    @patch("soma.encoders.distributed.open_slide", side_effect=_mock_open_slide)
    def test_progress_events_emitted(self, mock_open, tmp_path: Path):
        slides = [_make_slide_task("prog_slide", 2, 2)]
        extract_dataset(
            _TEST_ENCODER_NAME,
            slides,
            tmp_path,
            num_gpus=1,
            batch_size=4,
            num_workers=0,
            progress=True,
        )
        log_path = tmp_path / ".progress" / "rank_0.jsonl"
        assert log_path.exists()
        events = [json.loads(line) for line in log_path.read_text().strip().split("\n")]
        kinds = [e["kind"] for e in events]
        assert "extraction.started" in kinds
        assert "extraction.slide.started" in kinds
        assert "extraction.slide.completed" in kinds
        assert "extraction.completed" in kinds
