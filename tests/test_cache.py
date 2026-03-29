"""Tests for soma.cache — shared feature-cache utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from soma.cache import (
    CACHE_METADATA_NAME,
    build_slide_cache_key,
    build_tile_cache_key,
    manifest_digest,
    resolve_feature_payload_dir,
    resolve_slide_cache,
    resolve_tile_cache,
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
            "tissue_mask_path": None,
        }
        for sample in dataset.samples.values()
    ]
    assert manifest_digest(rows) == manifest_digest(list(reversed(rows)))


def test_tile_cache_key_changes_with_preprocessing(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    key_a = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(requested_tile_size_px=224),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )
    key_b = build_tile_cache_key(
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=PreprocessingConfig(requested_tile_size_px=256),
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
