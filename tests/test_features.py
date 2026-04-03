"""Tests for soma.features — FeatureStore for loading precomputed embeddings."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hs2p import SlideSpec

from soma.features import FeatureStore
from soma.slide2vec_adapter import LoadedTiling, Slide2VecArtifactAdapter


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


def test_is_slide_level_false_for_tile_features(feature_dir: Path):
    store = FeatureStore(feature_dir)
    assert store.is_slide_level is False


# --- Slide-level features ---


@pytest.fixture()
def slide_feature_dir(tmp_path: Path) -> Path:
    """Create a feature directory with slide-level (1-D) embeddings."""
    d = tmp_path / "slide_features"
    d.mkdir()
    torch.save(torch.randn(512), d / "s1.pt")
    torch.save(torch.randn(512), d / "s2.pt")
    return d


def test_is_slide_level_true(slide_feature_dir: Path):
    store = FeatureStore(slide_feature_dir)
    assert store.is_slide_level is True


def test_slide_feature_dim(slide_feature_dir: Path):
    store = FeatureStore(slide_feature_dir)
    assert store.feature_dim == 512


def test_load_slide_features(slide_feature_dir: Path):
    store = FeatureStore(slide_feature_dir)
    features = store.load("s1")
    assert features.shape == (512,)


def test_cache_directory_resolves_to_features_payload(tmp_path: Path):
    cache_dir = tmp_path / "feature_cache" / "tile" / "abc123"
    features_dir = cache_dir / "features"
    features_dir.mkdir(parents=True)
    (cache_dir / "cache_metadata.json").write_text("{}")
    torch.save(torch.randn(10, 32), features_dir / "s1.pt")

    store = FeatureStore(cache_dir)
    assert store.available_samples == ["s1"]
    assert store.feature_dim == 32


def test_slide2vec_artifact_root_prefers_slide_embeddings(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    tile_dir = artifact_root / "tile_embeddings"
    slide_dir = artifact_root / "slide_embeddings"
    tile_dir.mkdir(parents=True)
    slide_dir.mkdir(parents=True)
    torch.save(torch.randn(5, 8), tile_dir / "s1.pt")
    torch.save(torch.randn(8), slide_dir / "s1.pt")

    store = FeatureStore(artifact_root)
    assert store.is_slide_level is True
    assert store.feature_dim == 8


def test_artifact_adapter_rebuilds_tile_artifacts_from_cache_payload(tmp_path: Path):
    features_dir = tmp_path / "cache" / "features"
    features_dir.mkdir(parents=True)
    torch.save(torch.ones(2, 8), features_dir / "s1.pt")

    loaded = [
        LoadedTiling(
            slide=SlideSpec(
                sample_id="s1",
                image_path=Path("/slides/s1.svs"),
                mask_path=Path("/masks/s1.tif"),
                spacing_at_level_0=None,
            ),
            tiling_result=SimpleNamespace(
                coordinates_npz_path=Path("/coords/s1.npz"),
                coordinates_meta_path=Path("/coords/s1.meta.json"),
            ),
        )
    ]

    adapter = Slide2VecArtifactAdapter()
    artifacts = adapter.build_tile_artifacts_from_cache_payload(
        features_dir=features_dir,
        loaded_tilings=loaded,
        work_dir=tmp_path / "tile_metadata",
    )

    assert len(artifacts) == 1
    assert artifacts[0].path == features_dir / "s1.pt"
    metadata = json.loads(artifacts[0].metadata_path.read_text(encoding="utf-8"))
    assert metadata["coordinates_npz_path"] == "/coords/s1.npz"
    assert metadata["mask_path"] == "/masks/s1.tif"
