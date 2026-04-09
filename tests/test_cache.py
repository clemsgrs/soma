"""Tests for soma.cache — shared feature-cache utilities."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
import torch

from soma.cache import (
    CACHE_METADATA_NAME,
    build_slide_cache_key,
    build_hierarchical_cache_key,
    build_tile_cache_key,
    manifest_digest,
    resolve_feature_payload_dir,
    resolve_hierarchical_cache,
    resolve_slide_cache,
    resolve_tile_cache,
    write_cache_payload,
)
from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset


def _make_dataset(tmp_path: Path, rows: list[dict[str, object]] | None = None) -> Dataset:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        rows
        or [
            {"sample_id": "s2", "image_path": "/slides/s2.svs", "label": "normal"},
            {"sample_id": "s1", "image_path": "/slides/s1.svs", "label": "tumor"},
        ]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def test_manifest_digest_stable_under_row_order(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    rows = [
        {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "mask_path": None,
        }
        for sample in dataset.samples.values()
    ]
    assert manifest_digest(rows) == manifest_digest(list(reversed(rows)))


def test_tile_cache_key_changes_with_preprocessing(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(target_tile_size_px=224),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(target_tile_size_px=256),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_tissue_mask_value(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(tissue_mask_tissue_value=1),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(tissue_mask_tissue_value=2),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_precision(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp32"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_output_variant(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="h0-mini",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="h0-mini", output_variant="cls"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="h0-mini",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="h0-mini", output_variant="cls_patch_mean"),
    )
    assert key_a != key_b


def test_tile_cache_key_changes_with_backend(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(backend="auto"),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(backend="openslide"),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_resolve_tile_cache_records_backend_provenance(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    provenance = {
        "requested_backend": "auto",
        "backend": "openslide",
        "backend_by_sample_id": {
            "s1": "openslide",
            "s2": "openslide",
        },
    }
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(backend="auto"),
        execution=EncoderConfig(name="virchow", precision="fp16"),
        backend_provenance=provenance,
    )

    metadata = json.loads(resolution.metadata_path.read_text())
    assert metadata["requested_backend"] == "auto"
    assert metadata["backend"] == "openslide"
    assert metadata["backend_by_sample_id"] == {"s1": "openslide", "s2": "openslide"}


def test_hierarchical_cache_key_changes_with_region_geometry(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_hierarchical_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            target_tile_size_px=224,
            target_region_size_px=1344,
            region_tile_multiple=6,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_hierarchical_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            target_tile_size_px=224,
            target_region_size_px=896,
            region_tile_multiple=4,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert key_a != key_b


def test_slide_cache_key_changes_with_upstream_tile_cache_key(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_slide_cache_key(
        dataset=dataset,
        slide_encoder_name="prism",
        tile_cache_key="aaa111",
        execution=EncoderConfig(name="prism", precision="fp16"),
    )
    key_b = build_slide_cache_key(
        dataset=dataset,
        slide_encoder_name="prism",
        tile_cache_key="bbb222",
        execution=EncoderConfig(name="prism", precision="fp16"),
    )
    assert key_a != key_b


def test_resolve_tile_cache_reuses_complete_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    for sample_id in dataset.sample_ids:
        torch.save(torch.randn(4, 16), resolution.features_dir / f"{sample_id}.pt")

    reused = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert reused.complete is True
    assert reused.reused is True


def test_resolve_tile_cache_marks_incomplete_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save(torch.randn(4, 16), resolution.features_dir / "s1.pt")

    resumed = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert resumed.complete is False
    assert resumed.missing_sample_ids() == ["s2"]


def test_resolve_cache_fails_on_metadata_mismatch(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["encoder_name"] = "other-encoder"
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="metadata mismatch"):
        resolve_tile_cache(
            cache_root=cache_root,
            dataset=dataset,
            tile_encoder_name="virchow",
            preprocessing=PreprocessingConfig(),
            execution=EncoderConfig(name="virchow", precision="fp16"),
        )


def test_resolve_feature_payload_dir_understands_cache_dir(tmp_path: Path):
    cache_dir = tmp_path / "feature_cache" / "tile" / "abc123"
    features_dir = cache_dir / "features"
    features_dir.mkdir(parents=True)
    (cache_dir / CACHE_METADATA_NAME).write_text("{}")
    assert resolve_feature_payload_dir(cache_dir) == features_dir
    plain_dir = tmp_path / "plain_features"
    plain_dir.mkdir()
    assert resolve_feature_payload_dir(plain_dir) == plain_dir


def test_resolve_feature_payload_dir_understands_slide2vec_root(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    slide_dir = artifact_root / "slide_embeddings"
    hier_dir = artifact_root / "hierarchical_embeddings"
    slide_dir.mkdir(parents=True)
    hier_dir.mkdir(parents=True)
    assert resolve_feature_payload_dir(artifact_root) == slide_dir


def test_resolve_feature_payload_dir_prefers_hierarchical_embeddings(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    hier_dir = artifact_root / "hierarchical_embeddings"
    tile_dir = artifact_root / "tile_embeddings"
    hier_dir.mkdir(parents=True)
    tile_dir.mkdir(parents=True)
    assert resolve_feature_payload_dir(artifact_root) == hier_dir


def test_write_cache_payload_reuses_pt_artifacts_without_reserializing(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts" / "tile_embeddings"
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "s1.pt"
    torch.save(torch.ones(3, 7), artifact_path)
    cache_dir = tmp_path / "cache" / "features"

    artifact = SimpleNamespace(sample_id="s1", path=artifact_path)

    with patch(
        "soma.cache.torch.save",
        side_effect=AssertionError("torch.save should not be used"),
    ):
        feature_dim = write_cache_payload([artifact], output_dir=cache_dir)

    cached_path = cache_dir / "s1.pt"
    assert feature_dim == 7
    assert cached_path.is_file()
    assert torch.load(cached_path, weights_only=True, map_location="cpu").shape == (3, 7)


def test_resolve_hierarchical_cache_reuses_complete_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "feature_cache"
    resolution = resolve_hierarchical_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            target_tile_size_px=224,
            target_spacing_um=0.5,
            target_region_size_px=1344,
            region_tile_multiple=6,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    metadata = json.loads(resolution.metadata_path.read_text())
    metadata["feature_dim"] = 16
    resolution.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    for sample_id in dataset.sample_ids:
        torch.save(torch.randn(4, 9, 16), resolution.features_dir / f"{sample_id}.pt")

    reused = resolve_hierarchical_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(
            target_tile_size_px=224,
            target_spacing_um=0.5,
            target_region_size_px=1344,
            region_tile_multiple=6,
        ),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    assert reused.complete is True
    assert reused.reused is True
