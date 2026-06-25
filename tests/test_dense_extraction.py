"""Offline tests for the dense extraction loop (soma.dense_extraction).

Uses a random-weights ``vit_tiny_patch16_224`` so the whole loop — dense transform
→ pad → encode_tiles_dense → write_dense_grid → DenseFeatureStore round-trip — runs
on CPU with no weight downloads. The GPU/weights-dependent ``DenseTileFeatureExtractor.run``
construction step is intentionally not exercised here (the loop is injectable).
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from PIL import Image  # noqa: E402

from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402

from soma.config import EncoderConfig  # noqa: E402
from soma.dataset import SampleRecord  # noqa: E402
from soma.dense import DenseFeatureStore, compute_dense_geometry  # noqa: E402
from soma.dense_extraction import _pad_image_to_encoded, extract_dense_grids  # noqa: E402


def _encoder() -> TimmTileEncoder:
    return TimmTileEncoder("vit_tiny_patch16_224", pretrained=False, dynamic_img_size=True)


def _make_tiles(tmp_path: Path, n: int, size: int) -> list[SampleRecord]:
    records = []
    for i in range(n):
        path = tmp_path / f"tile{i}.png"
        Image.fromarray(
            (torch.rand(size, size, 3) * 255).to(torch.uint8).numpy()
        ).save(path)
        records.append(SampleRecord(sample_id=f"s{i}", image_path=path, label="x"))
    return records


def test_pad_image_to_encoded_reflect_and_noop():
    g_clean = compute_dense_geometry(target_size=32, patch_size=16)  # no pad
    x = torch.randn(3, 32, 32)
    assert _pad_image_to_encoded(x, g_clean, pad_mode="reflect", image_pad_value=None) is x

    g_pad = compute_dense_geometry(target_size=40, patch_size=16)  # -> 48, pad (8, 8)
    padded = _pad_image_to_encoded(x.new_zeros(3, 40, 40), g_pad, pad_mode="reflect", image_pad_value=None)
    assert tuple(padded.shape) == (3, 48, 48)


def test_extract_dense_grids_roundtrip(tmp_path: Path):
    enc = _encoder()
    geometry = compute_dense_geometry(target_size=32, patch_size=16)  # 2x2 grid
    records = _make_tiles(tmp_path, n=3, size=32)
    out_dir = tmp_path / "dense_embeddings"

    feature_dim = extract_dense_grids(
        encoder=enc,
        device="cpu",
        dense_transform=enc.get_dense_transform(),
        geometry=geometry,
        records=records,
        out_dir=out_dir,
        window_size=None,
        overlap=0.0,
        batch_size=2,
    )
    assert feature_dim == enc.encode_dim  # 192 for vit_tiny

    store = DenseFeatureStore(out_dir)
    assert sorted(store.available_samples) == ["s0", "s1", "s2"]
    assert store.feature_dim == enc.encode_dim
    assert store.grid_shape == (2, 2)
    assert tuple(store.load("s0").shape) == (enc.encode_dim, 2, 2)
    # Sidecar records the geometry, and (this slice) leaves mask_pad_value unset.
    meta = store.metadata("s0")
    assert meta["target_size"] == [32, 32] and meta["grid_shape"] == [2, 2]
    assert meta["mask_pad_value"] is None


def test_extract_dense_grids_padded_patch_multiple(tmp_path: Path):
    # target 40 is not a patch-16 multiple -> encoded 48 -> 3x3 grid (pad path).
    enc = _encoder()
    geometry = compute_dense_geometry(target_size=40, patch_size=16)
    assert geometry.grid_shape == (3, 3) and geometry.pad == (8, 8)
    records = _make_tiles(tmp_path, n=1, size=40)
    out_dir = tmp_path / "dense_embeddings"

    extract_dense_grids(
        encoder=enc,
        device="cpu",
        dense_transform=enc.get_dense_transform(),
        geometry=geometry,
        records=records,
        out_dir=out_dir,
        window_size=None,
        overlap=0.0,
        batch_size=1,
    )
    assert tuple(DenseFeatureStore(out_dir).load("s0").shape) == (enc.encode_dim, 3, 3)


def test_extract_attention_grids_roundtrip_and_sidecar(tmp_path: Path):
    """feature_kind='cls_attention' writes a (K, gh, gw) grid (K = nh per CLS row)
    and records the channel-order contract in the sidecar."""
    enc = _encoder()
    nh = enc._model.blocks[-1].attn.num_heads
    geometry = compute_dense_geometry(target_size=32, patch_size=16)  # 2x2 grid
    records = _make_tiles(tmp_path, n=2, size=32)
    out_dir = tmp_path / "dense_embeddings"

    feature_dim = extract_dense_grids(
        encoder=enc,
        device="cpu",
        dense_transform=enc.get_dense_transform(),
        geometry=geometry,
        records=records,
        out_dir=out_dir,
        window_size=None,
        overlap=0.0,
        feature_kind="cls_attention",
        attention_blocks=(-1,),
        attention_include_registers=False,
        batch_size=2,
    )
    assert feature_dim == nh  # 1 CLS row * nh heads (vit_tiny: nh=3)

    store = DenseFeatureStore(out_dir)
    assert store.feature_dim == nh
    assert tuple(store.load("s0").shape) == (nh, 2, 2)
    meta = store.metadata("s0")
    assert meta["feature_kind"] == "cls_attention"
    assert meta["attention_blocks"] == [-1]
    assert meta["attention_include_registers"] is False
    assert meta["channel_order"] == "[block][cls, reg…][head]"
    # attention maps are non-negative softmax slices.
    assert (store.load("s0") >= 0).all()


def test_extract_attention_multiblock_channel_count(tmp_path: Path):
    enc = _encoder()
    nh = enc._model.blocks[-1].attn.num_heads
    geometry = compute_dense_geometry(target_size=32, patch_size=16)
    records = _make_tiles(tmp_path, n=1, size=32)
    out_dir = tmp_path / "dense_embeddings"
    feature_dim = extract_dense_grids(
        encoder=enc,
        device="cpu",
        dense_transform=enc.get_dense_transform(),
        geometry=geometry,
        records=records,
        out_dir=out_dir,
        window_size=None,
        overlap=0.0,
        feature_kind="cls_attention",
        attention_blocks=(-1, -2),
        attention_include_registers=False,
        batch_size=1,
    )
    assert feature_dim == 2 * nh  # 2 blocks * 1 CLS * nh


def test_extract_dense_grids_rejects_wrong_tile_size(tmp_path: Path):
    enc = _encoder()
    geometry = compute_dense_geometry(target_size=32, patch_size=16)
    records = _make_tiles(tmp_path, n=1, size=48)  # 48 != target_size 32
    with pytest.raises(ValueError, match="target_size"):
        extract_dense_grids(
            encoder=enc,
            device="cpu",
            dense_transform=enc.get_dense_transform(),
            geometry=geometry,
            records=records,
            out_dir=tmp_path / "dense_embeddings",
            window_size=None,
            overlap=0.0,
            batch_size=1,
        )


def test_extract_then_cache_resolve_complete_and_store_read(tmp_path: Path):
    """The composition run() depends on: loop writes through write_dense_grid into
    the dense cache's features_dir, record metadata, re-resolve reports complete,
    and DenseFeatureStore reads it. Couples the injectable loop to the real cache
    functions offline (no GPU), covering filename<->feature_path_for_id agreement
    and validator/store co-satisfaction (complete ⇒ readable)."""
    import pandas as pd

    from soma.cache import (
        record_feature_dim,
        record_sample_identity_signatures,
        resolve_dense_cache,
    )
    from soma.dataset import Dataset

    enc = _encoder()
    records = _make_tiles(tmp_path, n=3, size=32)
    csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"sample_id": r.sample_id, "image_path": str(r.image_path), "label": "x"} for r in records]
    ).to_csv(csv, index=False)
    dataset = Dataset(csv)

    kw = dict(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        tile_encoder_name="uni",  # name only feeds the key here
        target_size=(32, 32),
        patch_size=(16, 16),
        pad_mode="reflect",
        execution=EncoderConfig(name="uni", precision="fp32"),
        window_size=None,
        overlap=0.0,
    )
    res = resolve_dense_cache(**kw)
    assert res.complete is False

    geometry = compute_dense_geometry(target_size=32, patch_size=16)
    feature_dim = extract_dense_grids(
        encoder=enc,
        device="cpu",
        dense_transform=enc.get_dense_transform(),
        geometry=geometry,
        records=[dataset.samples[i] for i in dataset.sample_ids],
        out_dir=res.features_dir,  # write into the cache payload dir
        window_size=None,
        overlap=0.0,
        batch_size=2,
    )
    record_feature_dim(res, feature_dim)
    record_sample_identity_signatures(res, list(dataset.sample_ids))

    resumed = resolve_dense_cache(**kw, validate_payloads=True)
    assert resumed.complete is True
    store = DenseFeatureStore(resumed.cache_dir)  # cache dir, descends into dense_embeddings/
    assert tuple(store.load(dataset.sample_ids[0]).shape) == (enc.encode_dim, 2, 2)


def test_extract_dense_grids_sliding_window_writes_full_grid_and_metadata(tmp_path: Path):
    # target 64 -> 4x4 grid; a 32px window (+0.5 overlap) slides over it and stitches
    # back to the same 4x4 grid. The sidecar records the derived sliding mode + knobs.
    enc = _encoder()
    geometry = compute_dense_geometry(target_size=64, patch_size=16)
    records = _make_tiles(tmp_path, n=2, size=64)
    out_dir = tmp_path / "dense_embeddings"
    feature_dim = extract_dense_grids(
        encoder=enc,
        device="cpu",
        dense_transform=enc.get_dense_transform(),
        geometry=geometry,
        records=records,
        out_dir=out_dir,
        window_size=32,
        overlap=0.5,
        batch_size=2,
    )
    assert feature_dim == enc.encode_dim
    store = DenseFeatureStore(out_dir)
    assert tuple(store.load("s0").shape) == (enc.encode_dim, 4, 4)
    meta = store.metadata("s0")
    assert meta["dense_input_mode"] == "sliding_window"
    assert meta["window_size"] == 32 and meta["overlap"] == 0.5


def test_extract_dense_grids_window_none_records_whole(tmp_path: Path):
    enc = _encoder()
    geometry = compute_dense_geometry(target_size=32, patch_size=16)
    extract_dense_grids(
        encoder=enc,
        device="cpu",
        dense_transform=enc.get_dense_transform(),
        geometry=geometry,
        records=_make_tiles(tmp_path, n=1, size=32),
        out_dir=tmp_path / "dense_embeddings",
        window_size=None,
        overlap=0.0,
    )
    meta = DenseFeatureStore(tmp_path / "dense_embeddings").metadata("s0")
    assert meta["dense_input_mode"] == "whole" and meta["window_size"] is None


def test_dense_tile_extractor_resume_encodes_only_missing(tmp_path: Path, monkeypatch):
    """DenseTileFeatureExtractor.run resumes: only the absent tiles are re-encoded,
    and the already-materialized grids are left untouched (#140)."""
    from types import SimpleNamespace

    import pandas as pd

    import soma.dense_extraction as de
    from soma.config import CacheConfig, EncoderConfig
    from soma.dataset import Dataset
    from soma.dense.store import DENSE_SIDECAR_SUFFIX
    from soma.dense_extraction import DenseTileFeatureExtractor

    enc = _encoder()  # vit_tiny, random weights, offline
    records = _make_tiles(tmp_path, n=3, size=32)
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"sample_id": r.sample_id, "image_path": str(r.image_path), "label": "x"} for r in records]
    ).to_csv(csv_path, index=False)
    dataset = Dataset(csv_path)

    monkeypatch.setattr(de, "load_model", lambda **kw: SimpleNamespace(model=enc, device="cpu"))

    def _make_extractor():
        return DenseTileFeatureExtractor(
            dataset,
            EncoderConfig(name="uni", precision="fp32"),
            target_size=32,
            spacing_um=0.5,
            cache=CacheConfig(enabled=True),
        )

    feature_dir = tmp_path / "features"
    store = _make_extractor().run(feature_dir)
    assert sorted(store.available_samples) == ["s0", "s1", "s2"]
    features_dir = store.feature_dir

    # Crash window: s1's grid + sidecar never landed.
    (features_dir / "s1.pt").unlink()
    (features_dir / f"s1{DENSE_SIDECAR_SUFFIX}").unlink()
    survivor_mtimes = {
        sid: (features_dir / f"{sid}.pt").stat().st_mtime_ns for sid in ("s0", "s2")
    }

    # Spy on the encode core to capture exactly which records are encoded on resume.
    real_extract = de.extract_dense_grids
    captured: dict = {}

    def _spy(**kw):
        captured["records"] = [r.sample_id for r in kw["records"]]
        return real_extract(**kw)

    monkeypatch.setattr(de, "extract_dense_grids", _spy)

    resumed = _make_extractor().run(feature_dir)
    assert captured["records"] == ["s1"]  # only the absent tile re-encoded
    for sid, mtime in survivor_mtimes.items():
        assert (features_dir / f"{sid}.pt").stat().st_mtime_ns == mtime  # untouched
    assert sorted(resumed.available_samples) == ["s0", "s1", "s2"]
