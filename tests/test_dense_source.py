from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from soma.dense import (
    CacheBackedDenseSource,
    DenseFeatureStore,
    DenseSampleSpacing,
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
    metadata["source_spacing_um"] = 0.25
    metadata["effective_spacing_um"] = 0.5
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
        ),
    )
    slide_manifest = CacheBackedDenseSource(
        DenseFeatureStore(tmp_path / "slide_manifest"),
        provenance=DenseSourceProvenance(
            kind="slide_manifest_dense_cache",
            feature_dir=tmp_path / "slide_manifest" / "dense_embeddings",
            dataset_csv=tmp_path / "roi_manifest.csv",
            parent_dataset_csv=tmp_path / "slides.csv",
        ),
    )

    for source, expected_grid in [(cached, cached_grid), (slide_manifest, slide_grid)]:
        assert source.available_samples == [sample_id]
        assert source.feature_dim == 3
        assert torch.equal(source.load(sample_id), expected_grid)
        assert source.geometry(sample_id) == compute_dense_geometry(target_size=32, patch_size=16)
        assert source.spacing(sample_id) == DenseSampleSpacing(
            source_spacing_um=0.25,
            effective_spacing_um=0.5,
        )
        assert source.spacing_um(sample_id) == 0.5
        assert source.metadata(sample_id)["target_size"] == [32, 32]
        source.validate_coverage([sample_id])
        with pytest.raises(ValueError, match="Missing dense features"):
            source.validate_coverage([sample_id, "missing"])

    assert cached.provenance.to_dict() == {
        "kind": "dense_cache",
        "feature_dir": str(tmp_path / "cached" / "dense_embeddings"),
        "dataset_csv": str(tmp_path / "tiles.csv"),
    }
    assert slide_manifest.provenance.to_dict() == {
        "kind": "slide_manifest_dense_cache",
        "feature_dir": str(tmp_path / "slide_manifest" / "dense_embeddings"),
        "dataset_csv": str(tmp_path / "roi_manifest.csv"),
        "parent_dataset_csv": str(tmp_path / "slides.csv"),
    }


def test_cache_backed_source_preserves_store_spacing_field_compatibility(tmp_path: Path):
    """The adapter must preserve slide2vec's ROI-sidecar spacing vocabulary."""
    sample_id = "slide_a__x0_y0"
    _write_source_grid(tmp_path, sample_id)
    sidecar = tmp_path / "dense_embeddings" / f"{sample_id}.meta.json"
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata.pop("spacing_um")
    metadata["declared_spacing_um"] = 0.5
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")

    source = CacheBackedDenseSource(
        DenseFeatureStore(tmp_path),
        provenance=DenseSourceProvenance(kind="slide_manifest_dense_cache"),
    )

    assert source.spacing_um(sample_id) == 0.5


def test_dense_store_reads_resolved_spacing_not_the_requested_declaration(tmp_path: Path):
    geometry = compute_dense_geometry(target_size=32, patch_size=16)
    metadata = dense_grid_metadata(
        geometry,
        feature_dim=3,
        pad_mode="reflect",
        spacing_um=0.5,
    )
    metadata.update(
        declared_spacing_um=0.5,
        source_spacing_um=0.252,
        effective_spacing_um=0.504,
    )
    write_dense_grid(
        tmp_path,
        "s0",
        torch.zeros(3, 2, 2),
        metadata,
    )

    assert DenseFeatureStore(tmp_path).spacing("s0") == DenseSampleSpacing(
        source_spacing_um=0.252,
        effective_spacing_um=0.504,
    )


@pytest.mark.parametrize("field", ["source_spacing_um", "effective_spacing_um"])
@pytest.mark.parametrize("value", [None, 0.0, float("inf"), True, "invalid"])
def test_dense_store_rejects_missing_or_invalid_resolved_spacing(
    tmp_path: Path, field: str, value
):
    geometry = compute_dense_geometry(target_size=32, patch_size=16)
    metadata = dense_grid_metadata(geometry, feature_dim=3, pad_mode="reflect")
    metadata.update(source_spacing_um=0.25, effective_spacing_um=0.5)
    metadata[field] = value
    write_dense_grid(tmp_path, "s0", torch.zeros(3, 2, 2), metadata)

    with pytest.raises(ValueError, match=rf"s0.*{field}"):
        DenseFeatureStore(tmp_path).spacing("s0")
