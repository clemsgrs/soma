"""Tests for the dense (segmentation) feature cache: geometry, key, store, validator.

Fully offline — no encoder weights. The headline guard is that a dense ``(d, h, w)``
grid reports ``feature_dim = d`` (channel axis), not ``w`` (grid width), which is
exactly the rank-collision the dedicated DenseFeatureStore + dense_grid metadata
exist to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from soma.cache import (
    build_dense_cache_key,
    build_tile_cache_key,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_dense_cache,
)
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.dense import (
    DenseFeatureStore,
    compute_dense_geometry,
    dense_grid_metadata,
    write_dense_grid,
)


def _make_dataset(tmp_path: Path) -> Dataset:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            # label is unused by the cache layer (the seg loader makes it optional);
            # the base Dataset still requires the column, so supply a placeholder.
            {"sample_id": "s1", "image_path": "/tiles/s1.png", "mask_path": "/masks/s1.png", "label": "a"},
            {"sample_id": "s2", "image_path": "/tiles/s2.png", "mask_path": "/masks/s2.png", "label": "b"},
        ]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _enc() -> EncoderConfig:
    return EncoderConfig(name="uni", precision="fp16")


# --------------------------------------------------------------------------- #
# Geometry — the highest-bug-density code: pad-to-patch-multiple + crop box.
# --------------------------------------------------------------------------- #


def test_geometry_patch16_512_is_clean():
    g = compute_dense_geometry(target_size=512, patch_size=16)
    assert g.encoded_size == (512, 512)
    assert g.grid_shape == (32, 32)
    assert g.pad == (0, 0)
    assert g.crop_box == (0, 0, 512, 512)
    assert g.is_padded is False


def test_geometry_patch14_512_pads_up_to_518():
    # 512 is NOT a patch-14 multiple: ceil(512/14)*14 = 518 -> 37x37, pad 6.
    g = compute_dense_geometry(target_size=512, patch_size=14)
    assert g.encoded_size == (518, 518)
    assert g.grid_shape == (37, 37)
    assert g.pad == (6, 6)
    assert g.crop_box == (0, 0, 512, 512)  # crop logits back to the 512 mask
    assert g.is_padded is True


def test_geometry_non_square_target():
    g = compute_dense_geometry(target_size=(300, 512), patch_size=16)
    assert g.encoded_size == (304, 512)  # 300 -> 304 (19*16), 512 clean
    assert g.grid_shape == (19, 32)
    assert g.pad == (4, 0)
    assert g.crop_box == (0, 0, 300, 512)


def test_geometry_rejects_nonpositive():
    with pytest.raises(ValueError, match="must be positive"):
        compute_dense_geometry(target_size=0, patch_size=16)
    with pytest.raises(ValueError, match="must be positive"):
        compute_dense_geometry(target_size=512, patch_size=(16, -1))


# --------------------------------------------------------------------------- #
# Cache key — distinctness, and independence from output_variant.
# --------------------------------------------------------------------------- #


def _key(**overrides) -> str:
    base = dict(
        tile_encoder_name="uni",
        target_size=(512, 512),
        patch_size=(16, 16),
        pad_mode="reflect",
        execution=_enc(),
    )
    base.update(overrides)
    return build_dense_cache_key(**base)


def test_dense_key_changes_with_target_size():
    assert _key(target_size=(512, 512)) != _key(target_size=(224, 224))


def test_dense_key_changes_with_patch_and_pad_and_mode():
    assert _key(patch_size=(16, 16)) != _key(patch_size=(14, 14))
    assert _key(pad_mode="reflect") != _key(pad_mode="zero")
    assert _key(dense_input_mode="whole") != _key(dense_input_mode="sliding_window")


def test_dense_key_independent_of_output_variant():
    # Dense grid is pre-pooling, so the variant must not split the cache.
    k_cls = _key(execution=EncoderConfig(name="uni", precision="fp16", output_variant="cls"))
    k_mean = _key(
        execution=EncoderConfig(name="uni", precision="fp16", output_variant="cls_patch_mean")
    )
    assert k_cls == k_mean


def test_dense_key_distinct_from_pooled_tile_key():
    dense = _key()
    pooled = build_tile_cache_key(
        tile_encoder_name="uni",
        preprocessing=PreprocessingConfig(),
        execution=_enc(),
        feature_type="bag",
    )
    assert dense != pooled


# --------------------------------------------------------------------------- #
# DenseFeatureStore — reads shape from the sidecar, never from rank.
# --------------------------------------------------------------------------- #


def _write_grid(out_dir: Path, sample_id: str, d: int, gh: int, gw: int) -> dict:
    g = compute_dense_geometry(target_size=(gh * 16, gw * 16), patch_size=16)
    meta = dense_grid_metadata(
        g, feature_dim=d, pad_mode="reflect", image_pad_value=0.0, mask_pad_value=255
    )
    write_dense_grid(out_dir, sample_id, torch.randn(d, gh, gw), meta)
    return meta


def test_store_reports_feature_dim_d_not_grid_width(tmp_path: Path):
    # d=1536 channels, 32x32 grid. A rank-based store would wrongly read 32 (w).
    _write_grid(tmp_path, "s1", d=1536, gh=32, gw=32)
    store = DenseFeatureStore(tmp_path)
    assert store.feature_dim == 1536
    assert store.grid_shape == (32, 32)
    assert tuple(store.load("s1").shape) == (1536, 32, 32)


def test_store_descends_into_payload_subdir(tmp_path: Path):
    # Mirrors FeatureStore: given a cache dir, descend into dense_embeddings/.
    payload_dir = tmp_path / "dense_embeddings"
    _write_grid(payload_dir, "s1", d=1536, gh=32, gw=32)
    store = DenseFeatureStore(tmp_path)  # cache dir, NOT the payload subdir
    assert store.available_samples == ["s1"]
    assert store.feature_dim == 1536


def test_store_prefers_dense_subdir_over_pooled_sibling(tmp_path: Path):
    # A root holding BOTH tile_embeddings/ and dense_embeddings/ must resolve to
    # the dense grids, not the pooled tile tensors (generic resolver checks tile
    # first; the dense resolver must not fall through to it).
    (tmp_path / "tile_embeddings").mkdir()
    torch.save(torch.randn(196, 768), tmp_path / "tile_embeddings" / "s1.pt")  # pooled bag
    _write_grid(tmp_path / "dense_embeddings", "s1", d=1536, gh=32, gw=32)
    store = DenseFeatureStore(tmp_path)
    assert store.feature_dir.name == "dense_embeddings"
    assert store.feature_dim == 1536
    assert tuple(store.load("s1").shape) == (1536, 32, 32)


@pytest.mark.parametrize("bad_id", ["../escaped", "a/b", "/abs", "..", ".", r"a\b"])
def test_write_dense_grid_rejects_path_traversal(tmp_path: Path, bad_id: str):
    g = compute_dense_geometry(target_size=512, patch_size=16)
    meta = dense_grid_metadata(
        g, feature_dim=8, pad_mode="reflect", image_pad_value=0.0, mask_pad_value=255
    )
    out_dir = tmp_path / "dense_embeddings"
    with pytest.raises(ValueError, match="Unsafe sample_id"):
        write_dense_grid(out_dir, bad_id, torch.randn(8, 32, 32), meta)
    # Nothing escaped to the parent root.
    assert not (tmp_path / "escaped.pt").exists()
    assert list(tmp_path.glob("*.pt")) == []


def test_store_load_normalizes_to_float32(tmp_path: Path):
    g = compute_dense_geometry(target_size=512, patch_size=16)
    meta = dense_grid_metadata(
        g, feature_dim=8, pad_mode="reflect", image_pad_value=0.0, mask_pad_value=255
    )
    write_dense_grid(tmp_path, "s1", torch.randn(8, 32, 32, dtype=torch.float16), meta)
    loaded = DenseFeatureStore(tmp_path).load("s1")
    assert loaded.dtype == torch.float32


def test_store_missing_sidecar_fails_loud(tmp_path: Path):
    torch.save(torch.randn(8, 32, 32), tmp_path / "s1.pt")  # no .meta.json
    store = DenseFeatureStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing its required sidecar"):
        store.metadata("s1")


def test_write_dense_grid_rejects_shape_metadata_mismatch(tmp_path: Path):
    g = compute_dense_geometry(target_size=512, patch_size=16)
    meta = dense_grid_metadata(
        g, feature_dim=1536, pad_mode="reflect", image_pad_value=0.0, mask_pad_value=255
    )
    with pytest.raises(ValueError, match="metadata declares"):
        write_dense_grid(tmp_path, "s1", torch.randn(1536, 16, 16), meta)  # grid 16x16 != 32x32


def test_store_load_detects_on_disk_shape_mismatch(tmp_path: Path):
    _write_grid(tmp_path, "s1", d=8, gh=32, gw=32)
    torch.save(torch.randn(8, 31, 32), tmp_path / "s1.pt")  # corrupt: overwrite with wrong grid
    store = DenseFeatureStore(tmp_path)
    with pytest.raises(ValueError, match="sidecar declares"):
        store.load("s1")


# --------------------------------------------------------------------------- #
# Validator — dense_grid reads the channel axis, requires the sidecar.
# --------------------------------------------------------------------------- #


def _dense_kw(tmp_path: Path, dataset) -> dict:
    return dict(
        cache_root=tmp_path / "dense_cache",
        dataset=dataset,
        tile_encoder_name="uni",
        target_size=(512, 512),
        patch_size=(16, 16),
        pad_mode="reflect",
        execution=_enc(),
    )


def _populate(res, dataset, *, d: int, gh: int, gw: int) -> None:
    """Write proper grid + sidecar for every sample via the real writer."""
    geom = compute_dense_geometry(target_size=(gh * 16, gw * 16), patch_size=16)
    for sid in dataset.sample_ids:
        meta = dense_grid_metadata(geom, feature_dim=d, pad_mode="reflect")
        write_dense_grid(res.features_dir, sid, torch.randn(d, gh, gw), meta)
    record_feature_dim(res, d)
    record_sample_identity_signatures(res, list(dataset.sample_ids))


def test_dense_cache_validator_accepts_real_payloads(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    kw = _dense_kw(tmp_path, dataset)
    _populate(resolve_dense_cache(**kw), dataset, d=1536, gh=32, gw=32)
    assert resolve_dense_cache(**kw, validate_payloads=True).complete is True


def test_dense_cache_validator_requires_sidecar(tmp_path: Path):
    # validator-complete must imply store-readable: a .pt with no sidecar (e.g. if
    # population were ever routed through the hardlink helper) is NOT complete, even
    # on the default path that loads no tensors.
    dataset = _make_dataset(tmp_path)
    kw = _dense_kw(tmp_path, dataset)
    res = resolve_dense_cache(**kw)
    record_feature_dim(res, 1536)
    for sid in dataset.sample_ids:
        torch.save(torch.randn(1536, 32, 32), res.feature_path_for_id(sid))  # NO sidecar
    record_sample_identity_signatures(res, list(dataset.sample_ids))
    assert resolve_dense_cache(**kw).complete is False


def test_dense_cache_validator_flags_wrong_grid_shape(tmp_path: Path):
    # Correct sidecar + channel dim, but the .pt grid is 16x16 != recorded 32x32.
    # Rank-3 alone can't catch this; the grid_shape check must. Only under
    # validate_payloads — the default path never loads the tensor.
    dataset = _make_dataset(tmp_path)
    kw = _dense_kw(tmp_path, dataset)
    res = resolve_dense_cache(**kw)
    _populate(res, dataset, d=1536, gh=32, gw=32)
    for sid in dataset.sample_ids:  # corrupt the .pt, keep the (correct) sidecar
        torch.save(torch.randn(1536, 16, 16), res.feature_path_for_id(sid))
    assert resolve_dense_cache(**kw).complete is True  # default: no tensor load
    assert resolve_dense_cache(**kw, validate_payloads=True).complete is False


def test_dense_cache_validator_flags_wrong_channel_dim(tmp_path: Path):
    # Correct sidecar (declares d=1536) but the .pt channel axis is 768. If the
    # validator read shape[-1] (=32) it would never catch this.
    dataset = _make_dataset(tmp_path)
    kw = _dense_kw(tmp_path, dataset)
    res = resolve_dense_cache(**kw)
    _populate(res, dataset, d=1536, gh=32, gw=32)
    for sid in dataset.sample_ids:
        torch.save(torch.randn(768, 32, 32), res.feature_path_for_id(sid))
    assert resolve_dense_cache(**kw, validate_payloads=True).complete is False
