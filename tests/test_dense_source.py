from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.dense import (
    CacheBackedDenseSource,
    DenseFeatureStore,
    DenseSourceProvenance,
    compute_dense_geometry,
    dense_grid_metadata,
    write_dense_grid,
)


def _write_source_grid(root: Path, sample_id: str) -> torch.Tensor:
    geometry = compute_dense_geometry(target_size=32, patch_size=16)
    grid = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
    metadata = dense_grid_metadata(
        geometry,
        feature_dim=3,
        pad_mode="reflect",
        spacing_um=0.5,
    )
    write_dense_grid(root / "dense_embeddings", sample_id, grid, metadata)
    return grid


def test_cache_backed_dense_sources_share_training_contract(tmp_path: Path):
    sample_id = "slide_a__x0_y0"
    cached_grid = _write_source_grid(tmp_path / "cached", sample_id)
    slide_grid = _write_source_grid(tmp_path / "slide_manifest", sample_id)

    cached = CacheBackedDenseSource(
        DenseFeatureStore(tmp_path / "cached"),
        provenance=DenseSourceProvenance(
            kind="dense_cache",
            feature_dir=tmp_path / "cached" / "dense_embeddings",
            dataset_csv=tmp_path / "tiles.csv",
            splits_csv=tmp_path / "splits.csv",
        ),
    )
    slide_manifest = CacheBackedDenseSource(
        DenseFeatureStore(tmp_path / "slide_manifest"),
        provenance=DenseSourceProvenance(
            kind="slide_manifest_dense_cache",
            feature_dir=tmp_path / "slide_manifest" / "dense_embeddings",
            dataset_csv=tmp_path / "roi_manifest.csv",
            splits_csv=tmp_path / "roi_splits.csv",
            parent_dataset_csv=tmp_path / "slides.csv",
            parent_splits_csv=tmp_path / "slide_splits.csv",
        ),
    )

    for source, expected_grid in [(cached, cached_grid), (slide_manifest, slide_grid)]:
        assert source.available_samples == [sample_id]
        assert source.feature_dim == 3
        assert torch.equal(source.load(sample_id), expected_grid)
        assert source.geometry(sample_id) == compute_dense_geometry(target_size=32, patch_size=16)
        assert source.spacing_um(sample_id) == 0.5
        assert source.metadata(sample_id)["target_size"] == [32, 32]
        source.validate_coverage([sample_id])
        with pytest.raises(ValueError, match="Missing dense features"):
            source.validate_coverage([sample_id, "missing"])

    assert cached.provenance.to_dict() == {
        "kind": "dense_cache",
        "feature_dir": str(tmp_path / "cached" / "dense_embeddings"),
        "dataset_csv": str(tmp_path / "tiles.csv"),
        "splits_csv": str(tmp_path / "splits.csv"),
    }
    assert slide_manifest.provenance.to_dict() == {
        "kind": "slide_manifest_dense_cache",
        "feature_dir": str(tmp_path / "slide_manifest" / "dense_embeddings"),
        "dataset_csv": str(tmp_path / "roi_manifest.csv"),
        "splits_csv": str(tmp_path / "roi_splits.csv"),
        "parent_dataset_csv": str(tmp_path / "slides.csv"),
        "parent_splits_csv": str(tmp_path / "slide_splits.csv"),
    }
