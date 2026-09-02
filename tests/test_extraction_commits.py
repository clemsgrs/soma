"""Chunked cache commits, resolved tile-image cache key, shared dtype resolver."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from soma.cache import resolve_cache_dtype
from soma.cache.keys import build_tile_cache_key
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig
from soma.dataset import Dataset
from soma.extraction.commit import (
    DEFAULT_IMAGE_COMMIT_EVERY,
    commit_chunks,
    resolve_commit_every,
)
from soma.tile_extraction import _TileFeatureExtractor
from tests.test_extraction import _TEST_TILE, _RecordingModel, _register_test_encoders


@pytest.fixture(autouse=True)
def _encoders():
    _register_test_encoders()


@pytest.fixture
def recording_model(monkeypatch: pytest.MonkeyPatch):
    _RecordingModel.calls = []
    monkeypatch.setattr("soma.tile_extraction.Model", _RecordingModel)
    return _RecordingModel


def _tile_dataset(tmp_path: Path, n: int) -> Dataset:
    from PIL import Image

    rows = []
    for i in range(n):
        image = tmp_path / f"s{i}.png"
        Image.new("RGB", (8, 8), color="white").save(image)
        rows.append({"sample_id": f"s{i}", "image_path": str(image), "label": "tumor"})
    csv = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return Dataset(csv)


class TestCommitHelpers:
    def test_chunks_cover_items_in_order(self):
        assert list(commit_chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
        assert list(commit_chunks([], 3)) == []

    def test_resolve_commit_every_default_and_validation(self):
        assert resolve_commit_every(None, default=7) == 7
        assert resolve_commit_every(3, default=7) == 3
        with pytest.raises(ValueError, match="commit_every"):
            resolve_commit_every(0, default=7)

    def test_cache_config_rejects_non_positive_commit_every(self):
        with pytest.raises(ValueError, match="commit_every"):
            CacheConfig(commit_every=0)
        assert CacheConfig(commit_every=16).commit_every == 16


class TestTileImageChunkedCommits:
    def test_signatures_are_committed_per_chunk(self, tmp_path: Path, recording_model):
        dataset = _tile_dataset(tmp_path, 5)
        cache = CacheConfig(enabled=True, root_dir=tmp_path / "cache", commit_every=2)

        _TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE),
            execution=ExecutionConfig(num_workers_per_gpu=0),
            cache=cache,
        ).run(feature_dir=tmp_path / "features")

        assert [c["sample_ids"] for c in recording_model.calls] == [
            ["s0", "s1"], ["s2", "s3"], ["s4"]
        ]

    def test_default_chunk_is_1024_images(self, tmp_path: Path, recording_model):
        dataset = _tile_dataset(tmp_path, 3)
        _TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE),
            execution=ExecutionConfig(num_workers_per_gpu=0),
            cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
        ).run(feature_dir=tmp_path / "features")
        assert DEFAULT_IMAGE_COMMIT_EVERY == 1024
        assert len(recording_model.calls) == 1

    def test_interrupted_run_resumes_after_the_last_committed_chunk(
        self, tmp_path: Path, recording_model, monkeypatch
    ):
        dataset = _tile_dataset(tmp_path, 4)
        cache = CacheConfig(enabled=True, root_dir=tmp_path / "cache", commit_every=2)
        original = _RecordingModel.embed_images

        def crash_on_second_chunk(self, images, *, execution):
            if len(type(self).calls) == 1:
                raise RuntimeError("simulated crash")
            return original(self, images, execution=execution)

        monkeypatch.setattr(_RecordingModel, "embed_images", crash_on_second_chunk)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _TileFeatureExtractor(
                dataset,
                EncoderConfig(name=_TEST_TILE),
                execution=ExecutionConfig(num_workers_per_gpu=0),
                cache=cache,
            ).run(feature_dir=tmp_path / "features")
        assert [c["sample_ids"] for c in recording_model.calls] == [["s0", "s1"]]

        monkeypatch.setattr(_RecordingModel, "embed_images", original)
        store = _TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE),
            execution=ExecutionConfig(num_workers_per_gpu=0),
            cache=cache,
        ).run(feature_dir=tmp_path / "features")

        # Only the un-committed chunk is re-encoded; s0/s1 survive from the first run.
        assert [c["sample_ids"] for c in recording_model.calls][1:] == [["s2", "s3"]]
        assert store.available_samples == ["s0", "s1", "s2", "s3"]


class TestTileImageCacheKey:
    def test_null_variant_shares_cache_with_explicit_default(self, tmp_path: Path, recording_model):
        dataset = _tile_dataset(tmp_path, 1)
        implicit = _TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE),
            execution=ExecutionConfig(num_workers_per_gpu=0),
            cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
        ).run(feature_dir=tmp_path / "f1")
        explicit = _TileFeatureExtractor(
            dataset,
            EncoderConfig(name=_TEST_TILE, output_variant="default"),
            execution=ExecutionConfig(num_workers_per_gpu=0),
            cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
        ).run(feature_dir=tmp_path / "f2")

        assert implicit.feature_dir == explicit.feature_dir
        assert len(recording_model.calls) == 1  # second run is a cache hit
        meta = json.loads((implicit.feature_dir.parent / "cache_metadata.json").read_text())
        assert meta["cache_key"] == build_tile_cache_key(
            tile_encoder_name=_TEST_TILE,
            preprocessing=None,
            execution=EncoderConfig(name=_TEST_TILE),
            output_variant="default",
            feature_type="tile",
            dtype="fp16",
        )

    def test_legacy_key_is_logged_for_null_variant_configs(
        self, tmp_path: Path, recording_model, caplog
    ):
        dataset = _tile_dataset(tmp_path, 1)
        with caplog.at_level("INFO", logger="soma.tile_extraction"):
            _TileFeatureExtractor(
                dataset,
                EncoderConfig(name=_TEST_TILE),
                execution=ExecutionConfig(num_workers_per_gpu=0),
                cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
            ).run(feature_dir=tmp_path / "features")
        legacy = build_tile_cache_key(
            tile_encoder_name=_TEST_TILE,
            preprocessing=None,
            execution=EncoderConfig(name=_TEST_TILE),
            output_variant=None,
            feature_type="tile",
            dtype="fp16",
        )
        messages = [r.getMessage() for r in caplog.records]
        assert any(f"pre-1.13 key with output_variant=null: image/{legacy}" in m for m in messages)


class TestSharedDtypeResolver:
    def test_follows_registry_precision_when_nothing_is_set(self):
        # _TEST_TILE recommends fp16 in the registry; the old dense resolution ignored
        # the registry and keyed fp32 while slide2vec computed in fp16.
        assert resolve_cache_dtype(None, EncoderConfig(name=_TEST_TILE)) == "fp16"

    def test_encoder_override_and_explicit_dtype(self):
        assert resolve_cache_dtype(None, EncoderConfig(name=_TEST_TILE, precision="fp32")) == "fp32"
        assert resolve_cache_dtype("fp32", EncoderConfig(name=_TEST_TILE)) == "fp32"
        assert resolve_cache_dtype("fp16", EncoderConfig(name=_TEST_TILE, precision="fp32")) == "fp16"
