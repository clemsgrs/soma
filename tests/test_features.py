"""Tests for soma.features — FeatureStore for loading precomputed embeddings."""

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from hs2p import SlideSpec

from soma.features import FeatureStore
from soma.cache import build_tile_artifacts_from_cache_payload
from soma.dataset import Dataset
from soma.slide2vec_adapter import LoadedTiling, load_tilings


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
    assert store.is_hierarchical is False
    assert store.feature_rank == 2


@pytest.fixture()
def hierarchical_feature_dir(tmp_path: Path) -> Path:
    """Create a feature directory with hierarchical (3-D) embeddings."""
    d = tmp_path / "hier_features"
    d.mkdir()
    torch.save(torch.randn(4, 9, 512), d / "s1.pt")
    torch.save(torch.randn(2, 9, 512), d / "s2.pt")
    return d


def test_is_hierarchical_true(hierarchical_feature_dir: Path):
    store = FeatureStore(hierarchical_feature_dir)
    assert store.is_slide_level is False
    assert store.is_hierarchical is True
    assert store.feature_rank == 3
    assert store.feature_dim == 512


def test_load_hierarchical_features(hierarchical_feature_dir: Path):
    store = FeatureStore(hierarchical_feature_dir)
    features = store.load("s1")
    assert features.shape == (4, 9, 512)


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


def test_feature_manifest_tracks_success_and_empty_samples(tmp_path: Path):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    torch.save(torch.randn(10, 32), feature_dir / "s1.pt")
    torch.save(torch.randn(10, 32), feature_dir / "s2.pt")
    pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "feature_status": "success",
                "feature_path": str((feature_dir / "s1.pt").resolve()),
                "num_tiles": 10,
                "feature_rank": 2,
                "feature_dim": 32,
            },
            {
                "sample_id": "s2",
                "feature_status": "empty",
                "feature_path": "",
                "num_tiles": 0,
                "feature_rank": 2,
                "feature_dim": 32,
            },
        ]
    ).to_csv(feature_dir / "process_list.csv", index=False)

    store = FeatureStore(feature_dir)
    assert store.has_feature_manifest is True
    assert store.feature_statuses == {"s1": "success", "s2": "empty"}
    assert store.expected_feature_samples == ["s1"]
    assert store.empty_feature_samples == ["s2"]


def test_slide2vec_artifact_root_prefers_slide_embeddings(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    tile_dir = artifact_root / "tile_embeddings"
    hier_dir = artifact_root / "hierarchical_embeddings"
    slide_dir = artifact_root / "slide_embeddings"
    tile_dir.mkdir(parents=True)
    hier_dir.mkdir(parents=True)
    slide_dir.mkdir(parents=True)
    torch.save(torch.randn(5, 8), tile_dir / "s1.pt")
    torch.save(torch.randn(4, 9, 8), hier_dir / "s1.pt")
    torch.save(torch.randn(8), slide_dir / "s1.pt")

    store = FeatureStore(artifact_root)
    assert store.is_slide_level is True
    assert store.feature_dim == 8


def test_slide2vec_artifact_root_prefers_hierarchical_embeddings_when_no_slide_dir(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    tile_dir = artifact_root / "tile_embeddings"
    hier_dir = artifact_root / "hierarchical_embeddings"
    tile_dir.mkdir(parents=True)
    hier_dir.mkdir(parents=True)
    torch.save(torch.randn(5, 8), tile_dir / "s1.pt")
    torch.save(torch.randn(4, 9, 8), hier_dir / "s1.pt")

    store = FeatureStore(artifact_root)
    assert store.is_hierarchical is True
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

    artifacts = build_tile_artifacts_from_cache_payload(
        features_dir=features_dir,
        loaded_tilings=loaded,
        work_dir=tmp_path / "tile_metadata",
    )

    assert len(artifacts) == 1
    assert artifacts[0].path == features_dir / "s1.pt"
    metadata = json.loads(artifacts[0].metadata_path.read_text(encoding="utf-8"))
    assert metadata["coordinates_npz_path"] == "/coords/s1.npz"
    assert metadata["mask_path"] == "/masks/s1.tif"


def test_load_tilings_uses_slide2vec_process_list_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s1"],
            "image_path": [str(tmp_path / "slides" / "s1.svs")],
            "label": ["tumor"],
            "mask_path": [str(tmp_path / "masks" / "s1.tif")],
        }
    ).to_csv(dataset_csv, index=False)
    dataset = Dataset(dataset_csv)

    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    process_list_path = tiling_dir / "process_list.csv"
    process_list_path.write_text("sample_id\n", encoding="utf-8")

    row = {
        "sample_id": "s1",
        "tiling_status": "success",
        "error": None,
    }
    loader_calls: list[Path] = []

    def _fake_process_loader(path: Path):
        loader_calls.append(Path(path))
        return pd.DataFrame([row])

    tiling_result = SimpleNamespace(sample_id="s1")

    monkeypatch.setattr("soma.slide2vec_adapter.load_tiling_process_df", _fake_process_loader)
    monkeypatch.setattr("soma.slide2vec_adapter.load_tiling_result_from_row", lambda loaded_row: tiling_result)
    monkeypatch.setattr(
        "soma.slide2vec_adapter.validate_tiling_result_provenance",
        lambda *args, **kwargs: None,
    )

    loaded = load_tilings(
        dataset=dataset,
        tiling_dir=tiling_dir,
        tissue_mask_tissue_value=1,
    )

    assert loader_calls == [process_list_path]
    assert len(loaded) == 1
    assert loaded[0].slide.sample_id == "s1"
    assert loaded[0].tiling_result is tiling_result


def test_load_tilings_passes_mask_path_to_hs2p_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset_csv = tmp_path / "dataset.csv"
    mask_path = tmp_path / "masks" / "s1.tif"
    pd.DataFrame(
        {
            "sample_id": ["s1"],
            "image_path": [str(tmp_path / "slides" / "s1.svs")],
            "label": ["tumor"],
            "mask_path": [str(mask_path)],
        }
    ).to_csv(dataset_csv, index=False)
    dataset = Dataset(dataset_csv)

    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    (tiling_dir / "process_list.csv").write_text("sample_id\n", encoding="utf-8")

    row = {
        "sample_id": "s1",
        "tiling_status": "success",
        "error": None,
    }
    monkeypatch.setattr("soma.slide2vec_adapter.load_tiling_process_df", lambda path: pd.DataFrame([row]))
    monkeypatch.setattr("soma.slide2vec_adapter.load_tiling_result_from_row", lambda loaded_row: SimpleNamespace())

    captured: dict[str, object] = {}

    def _fake_validate(result, *, sample_id, image_path, mask_path, tissue_mask_tissue_value):
        captured.update(
            sample_id=sample_id,
            image_path=image_path,
            mask_path=mask_path,
            tissue_mask_tissue_value=tissue_mask_tissue_value,
        )

    monkeypatch.setattr("soma.slide2vec_adapter.validate_tiling_result_provenance", _fake_validate)

    load_tilings(
        dataset=dataset,
        tiling_dir=tiling_dir,
        tissue_mask_tissue_value=1,
    )

    assert captured["sample_id"] == "s1"
    assert captured["image_path"] == dataset.samples["s1"].image_path
    assert captured["mask_path"] == mask_path
    assert captured["tissue_mask_tissue_value"] == 1
