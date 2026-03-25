"""Tests for soma.extraction — FeatureExtractor class."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
import pytest

from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.extraction import FeatureExtractor
from soma.features import FeatureStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

D = 16
NUM_SAMPLES = 3


def _make_dataset(tmp_path: Path) -> Dataset:
    """Create a small dataset with synthetic WSI paths."""
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(NUM_SAMPLES)],
            "image_path": [str(tmp_path / f"s{i}.svs") for i in range(NUM_SAMPLES)],
            "label": ["tumor", "normal", "tumor"],
        }
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _make_synthetic_reader():
    """Create a SyntheticSlideReader-like mock."""
    reader = MagicMock()
    reader.dimensions = (1000, 800)
    reader.spacing = 0.5
    reader.level_count = 3
    reader.level_dimensions = [(1000, 800), (500, 400), (250, 200)]
    reader.level_downsamples = [1.0, 2.0, 4.0]

    # Thumbnail: colored region (tissue) with white border (background)
    thumb = np.full((100, 125, 3), 255, dtype=np.uint8)
    thumb[10:90, 10:115] = np.array([150, 80, 100], dtype=np.uint8)  # tissue-like
    reader.get_thumbnail.return_value = thumb

    reader.close = MagicMock()
    reader.__enter__ = MagicMock(return_value=reader)
    reader.__exit__ = MagicMock(return_value=False)
    return reader


def _mock_extract_dataset(encoder_name, slides, output_dir, **kwargs):
    """Mock extract_dataset that writes fake .pt files."""
    from soma.encoders.distributed import ExtractionSummary

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    skipped = []
    for task in slides:
        pt_path = output_dir / f"{task.slide_id}.pt"
        if kwargs.get("skip_existing", True) and pt_path.exists():
            skipped.append(task.slide_id)
            continue
        n_tiles = len(task.tiling_result.coordinates)
        torch.save(torch.randn(n_tiles, D), pt_path)
        completed.append(task.slide_id)
    return ExtractionSummary(
        completed=completed, skipped=skipped, failed=[], duration_s=0.1
    )


# ---------------------------------------------------------------------------
# FeatureExtractor.preprocess()
# ---------------------------------------------------------------------------


class TestPreprocess:
    @patch("soma.extraction.open_slide")
    def test_saves_tiling_artifacts(self, mock_open, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        extractor.preprocess(tiling_dir)

        for i in range(NUM_SAMPLES):
            assert (tiling_dir / f"s{i}.coordinates.npz").exists()
            assert (tiling_dir / f"s{i}.coordinates.meta.json").exists()

    @patch("soma.extraction.open_slide")
    def test_skip_existing(self, mock_open, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        extractor.preprocess(tiling_dir)

        # Preprocess again — should skip existing
        mock_open.reset_mock()
        extractor.preprocess(tiling_dir, skip_existing=True)

        # open_slide should not have been called again
        mock_open.assert_not_called()

    @patch("soma.extraction.open_slide")
    def test_no_skip_existing(self, mock_open, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        extractor.preprocess(tiling_dir)

        mock_open.reset_mock()
        mock_open.return_value = _make_synthetic_reader()
        extractor.preprocess(tiling_dir, skip_existing=False)

        assert mock_open.call_count == NUM_SAMPLES


# ---------------------------------------------------------------------------
# FeatureExtractor.extract()
# ---------------------------------------------------------------------------


class TestExtract:
    @patch("soma.extraction.extract_dataset", side_effect=_mock_extract_dataset)
    @patch("soma.extraction.open_slide")
    def test_returns_feature_store(self, mock_open, mock_extract, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        tiling_dir = tmp_path / "tiling"
        feature_dir = tmp_path / "features"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        extractor.preprocess(tiling_dir)

        store = extractor.extract(feature_dir, tiling_dir=tiling_dir)

        assert isinstance(store, FeatureStore)
        assert len(store) == NUM_SAMPLES
        for i in range(NUM_SAMPLES):
            features = store.load(f"s{i}")
            assert features.shape[1] == D


# ---------------------------------------------------------------------------
# FeatureExtractor.run()
# ---------------------------------------------------------------------------


class TestRun:
    @patch("soma.extraction.extract_dataset", side_effect=_mock_extract_dataset)
    @patch("soma.extraction.open_slide")
    def test_end_to_end(self, mock_open, mock_extract, tmp_path: Path):
        mock_open.return_value = _make_synthetic_reader()
        dataset = _make_dataset(tmp_path)
        feature_dir = tmp_path / "features"

        extractor = FeatureExtractor(dataset, EncoderConfig())
        store = extractor.run(feature_dir)

        assert isinstance(store, FeatureStore)
        assert len(store) == NUM_SAMPLES
        # Tiling artifacts saved under .tiling/
        tiling_dir = feature_dir / ".tiling"
        assert (tiling_dir / "s0.coordinates.npz").exists()
