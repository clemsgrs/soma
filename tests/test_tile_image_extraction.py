"""Pre-cropped tile images are extracted in one ``Model.embed_images`` call.

soma resolves the image cache, invalidates the signatures of the samples it hands to
slide2vec, makes one upstream call, and commits feature dimension, signatures and the
first feature identity in one final metadata update. slide2vec validates each image's
source and feature provenance (``on_image_mismatch="reencode"``) and decides what can be
reused. These scenarios run real slide2vec extraction with the weight-free stub encoder of
``tests/test_feature_identity.py``, so resume decisions are slide2vec's own and encodes are
counted, not inferred from the ids soma requested.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import slide2vec
import torch
from PIL import Image
from slide2vec.artifacts import write_image_embedding

import soma.cache.io as cache_io
import soma.tile_extraction as tile_extraction
from soma.cache import MissingFeatureIdentity, resolve_cache_dtype
from soma.cache.keys import build_tile_cache_key
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig
from soma.data._legacy import legacy_samples_from_csv
from soma.extraction import FeatureExtractor
from soma.tile_extraction import _TileFeatureExtractor
from tests.test_extraction import (
    _TEST_TILE,
    _make_tile_dataset,
    _RecordingModel,
    _register_test_encoders,
)
from tests.test_feature_identity import (  # noqa: F401  (``encoder`` is a fixture)
    _ORIGINAL_MEAN,
    _STD,
    STUB_ENCODER,
    _cache_dir,
    _cache_metadata,
    _extract_images,
    _interrupt_before_first_commit,
    encoder,
)


@pytest.fixture
def embed_calls(monkeypatch) -> list[dict]:
    """Every ``Model.embed_images`` call: the requested ids and the execution options."""
    calls: list[dict] = []
    original = slide2vec.Model.embed_images

    def recording(self, images, *, execution=None):
        images = list(images)
        calls.append({"sample_ids": [spec.sample_id for spec in images], "execution": execution})
        return original(self, images, execution=execution)

    monkeypatch.setattr(slide2vec.Model, "embed_images", recording)
    return calls


def _colour(index: int) -> tuple[int, int, int]:
    return (index % 256, 90, 160)


def _expected(colour: tuple[int, int, int]) -> torch.Tensor:
    """Mean normalized RGB of a flat image, worked out from its colour."""
    return torch.tensor(
        [(value / 255 - m) / s for value, m, s in zip(colour, _ORIGINAL_MEAN, _STD)]
    )


def _write_dataset(root: Path, colours: dict[str, tuple[int, int, int]], *, name: str) -> TileDataset:
    """A dataset named ``name`` with one flat 8px image per sample.

    An image's path depends on its colour only, so datasets that give a sample the same
    colour give it the same source path.
    """
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id, colour in colours.items():
        image_path = image_dir / ("rgb_" + "_".join(str(value) for value in colour) + ".png")
        if not image_path.is_file():
            Image.new("RGB", (8, 8), color=colour).save(image_path)
        rows.append({"sample_id": sample_id, "image_path": str(image_path), "label": 0})
    dataset_csv = root / f"{name}.csv"
    pd.DataFrame(rows).to_csv(dataset_csv, index=False)
    return legacy_samples_from_csv(dataset_csv)


def _assert_vectors(result, expected: dict[str, torch.Tensor]) -> None:
    """The requested samples load the expected vectors (the cache may hold others)."""
    for sample_id, vector in expected.items():
        assert torch.allclose(result.source.load(sample_id).float(), vector, atol=1e-5), sample_id


# --- One upstream call ------------------------------------------------------------------


def test_extraction_larger_than_the_old_chunk_makes_one_embed_images_call(
    tmp_path, encoder, embed_calls
):
    colours = {f"s{index:04d}": _colour(index) for index in range(1100)}
    dataset = _write_dataset(tmp_path, colours, name="images")

    _extract_images(dataset, tmp_path)

    (call,) = embed_calls
    assert sorted(call["sample_ids"]) == sorted(colours)
    assert encoder.encoded_images == 1100

    reused = _extract_images(dataset, tmp_path)

    # A full hit asks slide2vec nothing and encodes nothing.
    assert len(embed_calls) == 1
    assert encoder.encoded_images == 1100
    _assert_vectors(reused, {sample_id: _expected(colour) for sample_id, colour in colours.items()})


def test_slide2vec_replaces_an_image_recorded_for_another_source(tmp_path, encoder, embed_calls):
    dataset = _write_dataset(tmp_path, {"s0": _colour(1)}, name="images")

    _extract_images(dataset, tmp_path)

    (call,) = embed_calls
    assert call["execution"].on_image_mismatch == "reencode"


# --- Resume inside the one call ---------------------------------------------------------


def _features_dir(root: Path) -> Path:
    return _cache_dir(root, "image") / "image_embeddings"


def _signatures(root: Path) -> dict[str, str]:
    return _cache_metadata(root, "image")["sample_identity_signature_by_id"]


def _six_images(root: Path) -> tuple[TileDataset, dict[str, torch.Tensor]]:
    colours = {f"s{index}": _colour(30 * index) for index in range(6)}
    dataset = _write_dataset(root, colours, name="images")
    return dataset, {sample_id: _expected(colour) for sample_id, colour in colours.items()}


def test_interrupted_extraction_resumes_without_reencoding_completed_images(
    tmp_path, encoder, embed_calls
):
    dataset, expected = _six_images(tmp_path)
    # Batches of two: the call fails after slide2vec published four images.
    _interrupt_before_first_commit(lambda: _extract_images(dataset, tmp_path), written_batches=2)
    assert _signatures(tmp_path) == {}

    resumed = _extract_images(dataset, tmp_path)

    # The interrupted call published s0..s3; only s4 and s5 are encoded again.
    assert encoder.encoded_images == 4 + 2
    assert len(embed_calls) == 2
    _assert_vectors(resumed, expected)
    assert sorted(_signatures(tmp_path)) == sorted(expected)


def test_payload_without_its_sidecar_is_repaired_upstream_and_keeps_the_pending_record(
    tmp_path, encoder
):
    dataset, expected = _six_images(tmp_path)
    _interrupt_before_first_commit(lambda: _extract_images(dataset, tmp_path), written_batches=2)
    # The process died before s0's and s1's sidecars were published.
    for sample_id in ("s0", "s1"):
        (_features_dir(tmp_path) / f"{sample_id}.meta.json").unlink()
    assert _cache_metadata(tmp_path, "image")["feature_identity"]["identity"] is None

    resumed = _extract_images(dataset, tmp_path)

    # The cache is kept: s2 and s3 are reused, s0 and s1 repaired, s4 and s5 encoded.
    assert encoder.encoded_images == 4 + 4
    _assert_vectors(resumed, expected)
    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _ORIGINAL_MEAN


@pytest.mark.parametrize("lost", [".meta.json", ".pt"], ids=["sidecar", "payload"])
def test_committed_image_missing_half_of_its_pair_is_not_a_full_hit(
    tmp_path, encoder, embed_calls, lost
):
    dataset, expected = _six_images(tmp_path)
    _extract_images(dataset, tmp_path)
    (_features_dir(tmp_path) / f"s1{lost}").unlink()

    repaired = _extract_images(dataset, tmp_path)

    assert embed_calls[-1]["sample_ids"] == ["s1"]
    assert encoder.encoded_images == 6 + 1
    _assert_vectors(repaired, expected)


# --- Re-pointed samples -----------------------------------------------------------------


_A = _colour(10)
_B = _colour(200)


def test_repointed_sample_yields_the_new_source_vector(tmp_path, encoder):
    _extract_images(_write_dataset(tmp_path, {"s0": _A, "s1": _colour(50)}, name="a"), tmp_path)
    signature = _signatures(tmp_path)["s1"]

    repointed = _extract_images(
        _write_dataset(tmp_path, {"s0": _B, "s1": _colour(50)}, name="b"), tmp_path
    )

    assert encoder.encoded_images == 2 + 1
    _assert_vectors(repointed, {"s0": _expected(_B), "s1": _expected(_colour(50))})
    # The unaffected sample keeps its signature.
    assert _signatures(tmp_path)["s1"] == signature


def test_repointed_sample_is_not_served_from_a_pack_of_the_old_features(tmp_path, encoder):
    first = _extract_images(_write_dataset(tmp_path, {"s0": _A, "s1": _colour(50)}, name="a"), tmp_path)
    # Loading packs every 1-D feature of the cache into one file next to them.
    _assert_vectors(first, {"s0": _expected(_A)})

    repointed = _extract_images(
        _write_dataset(tmp_path, {"s0": _B, "s1": _colour(50)}, name="b"), tmp_path
    )

    _assert_vectors(repointed, {"s0": _expected(_B), "s1": _expected(_colour(50))})


def _extract_images_uncached(dataset, root: Path):
    return FeatureExtractor(
        dataset,
        EncoderConfig(name=STUB_ENCODER, precision="fp32", batch_size=2),
        execution=ExecutionConfig(num_gpus=1, num_workers_per_gpu=0),
        cache=CacheConfig(enabled=False),
        output_root=root / "run",
        unit="tile",
    ).extract()


def test_uncached_repointed_sample_is_not_served_from_a_pack_of_the_old_features(
    tmp_path, encoder
):
    first = _extract_images_uncached(
        _write_dataset(tmp_path, {"s0": _A, "s1": _colour(50)}, name="a"), tmp_path
    )
    # Loading packs every 1-D feature of the output directory into one file next to them.
    _assert_vectors(first, {"s0": _expected(_A)})

    repointed = _extract_images_uncached(
        _write_dataset(tmp_path, {"s0": _B, "s1": _colour(50)}, name="b"), tmp_path
    )

    assert encoder.encoded_images == 2 + 1
    _assert_vectors(repointed, {"s0": _expected(_B), "s1": _expected(_colour(50))})


@pytest.mark.parametrize("field", ["image_path", "compatibility"])
def test_unsigned_image_whose_sidecar_lacks_provenance_is_reencoded(
    tmp_path, encoder, field
):
    dataset, expected = _six_images(tmp_path)
    _interrupt_before_first_commit(lambda: _extract_images(dataset, tmp_path), written_batches=1)
    sidecar_path = _features_dir(tmp_path) / "s0.meta.json"
    sidecar = json.loads(sidecar_path.read_text())
    del sidecar[field]
    sidecar_path.write_text(json.dumps(sidecar))

    resumed = _extract_images(dataset, tmp_path)

    # s1 is reused; s0 is re-encoded, never accepted, along with s2..s5.
    assert encoder.encoded_images == 2 + 5
    _assert_vectors(resumed, expected)


def _crash_at_final_commit(monkeypatch) -> None:
    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("soma.tile_extraction.commit_extracted_samples", crash)


def test_switching_back_after_a_crash_before_the_commit_returns_the_original_vector(
    tmp_path, encoder
):
    dataset_a = _write_dataset(tmp_path, {"s0": _A}, name="a")
    _extract_images(dataset_a, tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        _crash_at_final_commit(patch)
        with pytest.raises(KeyboardInterrupt):
            _extract_images(_write_dataset(tmp_path, {"s0": _B}, name="b"), tmp_path)
    # B's features are on disk under s0, and nothing vouches for them.
    assert _signatures(tmp_path) == {}

    switched_back = _extract_images(dataset_a, tmp_path)

    assert encoder.encoded_images == 3
    _assert_vectors(switched_back, {"s0": _expected(_A)})


def test_switching_back_after_a_failure_before_encoding_returns_the_original_vector(
    tmp_path, encoder, embed_calls, monkeypatch
):
    dataset_a = _write_dataset(tmp_path, {"s0": _A}, name="a")
    _extract_images(dataset_a, tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            slide2vec.Model,
            "embed_images",
            lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("launch failed")),
        )
        with pytest.raises(RuntimeError, match="launch failed"):
            _extract_images(_write_dataset(tmp_path, {"s0": _B}, name="b"), tmp_path)
    assert _signatures(tmp_path) == {}

    switched_back = _extract_images(dataset_a, tmp_path)

    # Unsigned, so slide2vec is asked; its sidecar still records A, so nothing is encoded.
    assert embed_calls[-1]["sample_ids"] == ["s0"]
    assert encoder.encoded_images == 1
    _assert_vectors(switched_back, {"s0": _expected(_A)})


def test_restarting_with_a_subset_after_a_crash_keeps_the_other_samples_unsigned(
    tmp_path, encoder
):
    colours_a = {"s0": _A, "s1": _colour(60)}
    dataset_a = _write_dataset(tmp_path, colours_a, name="a")
    _extract_images(dataset_a, tmp_path)
    with pytest.MonkeyPatch.context() as patch:
        _crash_at_final_commit(patch)
        with pytest.raises(KeyboardInterrupt):
            _extract_images(
                _write_dataset(tmp_path, {"s0": _B, "s1": _colour(220)}, name="b"), tmp_path
            )

    subset = _extract_images(_write_dataset(tmp_path, {"s0": _A}, name="a_subset"), tmp_path)

    _assert_vectors(subset, {"s0": _expected(_A)})
    # s1 was not requested: it stays unsigned rather than inheriting a signature.
    assert sorted(_signatures(tmp_path)) == ["s0"]

    full = _extract_images(dataset_a, tmp_path)

    _assert_vectors(full, {sample_id: _expected(colour) for sample_id, colour in colours_a.items()})
    assert encoder.encoded_images == 2 + 2 + 1 + 1


# --- Metadata commits -------------------------------------------------------------------


@pytest.fixture
def metadata_writes(monkeypatch) -> list[dict]:
    """Every ``cache_metadata.json`` write, in order."""
    writes: list[dict] = []
    original = cache_io.atomic_write_json

    def recording(path, data, **kwargs):
        if Path(path).name == "cache_metadata.json":
            writes.append(json.loads(json.dumps(data)))
        return original(path, data, **kwargs)

    monkeypatch.setattr(cache_io, "atomic_write_json", recording)
    return writes


def _is_final_commit(write: dict, sample_ids) -> bool:
    return (
        set(sample_ids) <= set(write["sample_identity_signature_by_id"])
        and write["feature_dim"] == 3
        and write["feature_identity"]["identity"] is not None
    )


def test_a_new_cache_is_initialized_then_committed_once(tmp_path, encoder, metadata_writes):
    colours = {f"s{index:04d}": _colour(index) for index in range(1030)}

    _extract_images(_write_dataset(tmp_path, colours, name="images"), tmp_path)

    initialization, final = metadata_writes
    assert initialization["sample_identity_signature_by_id"] == {}
    assert initialization["feature_dim"] is None
    assert initialization["feature_identity"]["identity"] is None
    assert _is_final_commit(final, colours)


def test_replacing_signed_samples_invalidates_them_then_commits_once(
    tmp_path, encoder, metadata_writes
):
    _extract_images(_write_dataset(tmp_path, {"s0": _A, "s1": _colour(50)}, name="a"), tmp_path)
    metadata_writes.clear()

    _extract_images(_write_dataset(tmp_path, {"s0": _B, "s1": _colour(50)}, name="b"), tmp_path)

    invalidation, final = metadata_writes
    assert sorted(invalidation["sample_identity_signature_by_id"]) == ["s1"]
    assert _is_final_commit(final, ["s0", "s1"])


def test_resuming_unsigned_samples_writes_only_the_final_commit(
    tmp_path, encoder, metadata_writes
):
    dataset, expected = _six_images(tmp_path)
    _interrupt_before_first_commit(lambda: _extract_images(dataset, tmp_path), written_batches=1)
    metadata_writes.clear()

    _extract_images(dataset, tmp_path)

    (final,) = metadata_writes
    assert _is_final_commit(final, expected)


def test_final_commit_failing_before_replacement_leaves_a_safe_resume_state(
    tmp_path, encoder, monkeypatch
):
    dataset_a = _write_dataset(tmp_path, {"s0": _A, "s1": _colour(50)}, name="a")
    _extract_images(dataset_a, tmp_path)
    before = _cache_metadata(tmp_path, "image")
    commit = tile_extraction.commit_extracted_samples

    def commit_interrupted_before_replacement(*args, **kwargs):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError("full")))
            return commit(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tile_extraction, "commit_extracted_samples", commit_interrupted_before_replacement)
        with pytest.raises(OSError, match="full"):
            _extract_images(
                _write_dataset(tmp_path, {"s0": _B, "s1": _colour(50)}, name="b"), tmp_path
            )

    # The previous record, with s0 invalidated, is what a reader finds.
    after = _cache_metadata(tmp_path, "image")
    assert after["sample_identity_signature_by_id"] == {
        "s1": before["sample_identity_signature_by_id"]["s1"]
    }
    _assert_vectors(_extract_images(dataset_a, tmp_path), {"s0": _expected(_A), "s1": _expected(_colour(50))})


# --- Recording fake: cache key, dtype and the writer identity contract ---------------


@pytest.fixture
def recording_model(monkeypatch: pytest.MonkeyPatch):
    _register_test_encoders()
    _RecordingModel.calls = []
    monkeypatch.setattr("soma.tile_extraction.Model", _RecordingModel)
    return _RecordingModel


def test_null_output_variant_shares_the_cache_of_the_explicit_default(
    tmp_path: Path, recording_model
):
    dataset = _make_tile_dataset(tmp_path, ("s0",))
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


class TestSharedDtypeResolver:
    def test_follows_registry_precision_when_nothing_is_set(self):
        _register_test_encoders()
        # _TEST_TILE recommends fp16 in the registry; the old dense resolution ignored
        # the registry and keyed fp32 while slide2vec computed in fp16.
        assert resolve_cache_dtype(None, EncoderConfig(name=_TEST_TILE)) == "fp16"

    def test_encoder_override_and_explicit_dtype(self):
        _register_test_encoders()
        assert resolve_cache_dtype(None, EncoderConfig(name=_TEST_TILE, precision="fp32")) == "fp32"
        assert resolve_cache_dtype("fp32", EncoderConfig(name=_TEST_TILE)) == "fp32"
        assert resolve_cache_dtype("fp16", EncoderConfig(name=_TEST_TILE, precision="fp32")) == "fp16"


def test_features_written_without_a_feature_identity_are_not_committed(
    tmp_path: Path, recording_model, monkeypatch
):
    def embed_without_identity(self, images, *, execution):
        return [
            write_image_embedding(
                torch.ones(4),
                output_dir=execution.output_dir,
                sample_id=spec.sample_id,
                output_format=execution.output_format,
                metadata={"artifact_type": "image_embeddings", "image_path": str(spec.image_path)},
            )
            for spec in images
        ]

    monkeypatch.setattr(recording_model, "embed_images", embed_without_identity)
    extractor = _TileFeatureExtractor(
        _make_tile_dataset(tmp_path, ("s0", "s1")),
        EncoderConfig(name=_TEST_TILE),
        execution=ExecutionConfig(num_workers_per_gpu=0),
        cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
    )

    with pytest.raises(MissingFeatureIdentity, match="no feature identity"):
        extractor.run(feature_dir=tmp_path / "features")

    (metadata_path,) = (tmp_path / "cache" / "image").glob("*/cache_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["sample_identity_signature_by_id"] == {}
    assert metadata["feature_identity"]["identity"] is None
