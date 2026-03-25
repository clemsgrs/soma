"""Tests for soma.features — FeatureStore for loading precomputed embeddings."""

from pathlib import Path

import pytest
import torch

from soma.features import FeatureStore


@pytest.fixture()
def feature_dir(tmp_path: Path) -> Path:
    """Create a feature directory with 3 slides of varying sizes."""
    d = tmp_path / "features"
    d.mkdir()

    torch.save(torch.randn(100, 512), d / "s1.pt")
    torch.save(torch.randn(200, 512), d / "s2.pt")
    torch.save(torch.randn(50, 512), d / "s3.pt")
    return d


def test_available_samples(feature_dir: Path):
    store = FeatureStore(feature_dir)
    assert sorted(store.available_samples) == ["s1", "s2", "s3"]


def test_feature_dim(feature_dir: Path):
    store = FeatureStore(feature_dir)
    assert store.feature_dim == 512


def test_load_single_slide(feature_dir: Path):
    store = FeatureStore(feature_dir)
    features = store.load("s1")
    assert features.shape == (100, 512)
    assert features.dtype == torch.float32


def test_load_unknown_slide_raises(feature_dir: Path):
    store = FeatureStore(feature_dir)
    with pytest.raises(KeyError, match="s99"):
        store.load("s99")


def test_validate_coverage_passes(feature_dir: Path):
    store = FeatureStore(feature_dir)
    store.validate_coverage(["s1", "s2"])  # Should not raise


def test_validate_coverage_raises_for_missing(feature_dir: Path):
    store = FeatureStore(feature_dir)
    with pytest.raises(ValueError, match="s99"):
        store.validate_coverage(["s1", "s99"])


def test_empty_feature_dir(tmp_path: Path):
    d = tmp_path / "empty"
    d.mkdir()
    store = FeatureStore(d)
    assert store.available_samples == []


def test_len(feature_dir: Path):
    store = FeatureStore(feature_dir)
    assert len(store) == 3
