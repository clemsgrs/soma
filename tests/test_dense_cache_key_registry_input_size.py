"""The dense cache key derives from slide2vec's ``DenseOptions``, not the registry size.

A dense key carries every field of the slide2vec dense options the grids are read and
encoded with (ADR 0019), the encoder's static patch size, its precision and the storage
dtype. A registry edit upstream (slide2vec 6.0.0 moved ``dinov2-vitb14`` from 224 to 518)
must not move it: the dense extraction's real encoder input is the configured
target/window, which the options already carry. Pooled keys keep the registry size.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest
import slide2vec
import torch

from soma.cache import (
    build_dense_cache_key,
    build_hierarchical_cache_key,
    build_patient_cache_key,
    build_slide_cache_key,
    build_tile_cache_key,
    execution_signature,
    preprocessing_signature,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_dense_cache,
)
from soma.data._legacy import legacy_samples_from_csv
from soma.dense import compute_dense_geometry, dense_grid_metadata, write_dense_grid
from tests.extraction_configs import extraction


class _StubRegistry:
    """Registry whose only moving part is the default ``input_size``."""

    def __init__(self, input_size: int) -> None:
        self._input_size = input_size

    def info(self, name: str) -> dict:
        return {
            "input_size": self._input_size,
            "supported_spacing_um": 0.5,
            "precision": "fp16",
        }


def _stub_registry(monkeypatch: pytest.MonkeyPatch, input_size: int) -> None:
    monkeypatch.setattr("soma.cache.keys.encoder_registry", _StubRegistry(input_size))
    monkeypatch.setattr("soma.encoders.validation.encoder_registry", _StubRegistry(input_size))


_FP16 = slide2vec.ExecutionOptions(precision="fp16")
_DENSE = slide2vec.DenseOptions(spacing_um=0.5, target_size=1024, window_size=448, overlap=0.5)


def _dense_key(**overrides) -> str:
    kwargs = dict(
        tile_encoder_name="stub-encoder",
        patch_size=(14, 14),
        dense=_DENSE,
        execution=_FP16,
        dtype="fp16",
    )
    kwargs.update(overrides)
    return build_dense_cache_key(**kwargs)


def test_dense_key_ignores_the_registry_default_input_size(monkeypatch: pytest.MonkeyPatch):
    _stub_registry(monkeypatch, 224)
    before = _dense_key()
    _stub_registry(monkeypatch, 518)
    assert _dense_key() == before


def _dense_variations():
    changed = {
        "spacing_um": 0.25,
        "target_size": 512,
        "tolerance": 0.1,
        "backend": "openslide",
        "pad_mode": "constant",
        "image_pad_value": 0.0,
        "window_size": 224,
        "overlap": 0.25,
        "feature_kind": "cls_attention",
        "attention_blocks": (-2, -1),
        "attention_include_registers": True,
    }
    assert set(changed) == {field.name for field in dataclasses.fields(slide2vec.DenseOptions)}
    for name, value in changed.items():
        yield name, {"dense": dataclasses.replace(_DENSE, **{name: value})}
    yield "precision", {"execution": slide2vec.ExecutionOptions(precision="fp32")}
    yield "dtype", {"dtype": "fp32"}
    yield "patch_size", {"patch_size": (16, 16)}
    yield "sampling", {"sampling_signature": {"spacing_policy": "native_if_coarser"}}


@pytest.mark.parametrize(("field", "override"), list(_dense_variations()), ids=lambda v: v if isinstance(v, str) else "")
def test_the_dense_key_follows_every_field_the_extraction_computes_from(
    monkeypatch: pytest.MonkeyPatch, field: str, override: dict
):
    _stub_registry(monkeypatch, 224)
    assert _dense_key(**override) != _dense_key(), field


def test_dense_resources_leave_the_key_alone(monkeypatch: pytest.MonkeyPatch):
    _stub_registry(monkeypatch, 224)
    resources = slide2vec.ExecutionOptions(precision="fp16", batch_size=3, num_gpus=1, num_workers_per_gpu=1)
    assert _dense_key(execution=resources) == _dense_key()


def test_the_image_and_region_options_of_one_recipe_share_a_key(monkeypatch: pytest.MonkeyPatch):
    """slide2vec's DenseImageOptions holds the same fields; the image route keys the
    same recipe the same way (its cache kind keeps the two apart on disk)."""
    _stub_registry(monkeypatch, 224)
    image = slide2vec.DenseImageOptions(
        **{field.name: getattr(_DENSE, field.name) for field in dataclasses.fields(slide2vec.DenseOptions)}
    )
    assert _dense_key(dense=image) == _dense_key()


def _pooled_keys() -> dict[str, str]:
    preprocessing = extraction(0.5, 224)
    tile_dependency = {
        "tile_encoder_name": "stub-encoder",
        "tile_preprocessing": preprocessing_signature(preprocessing),
        "tile_execution": execution_signature(
            _FP16, encoder_name="stub-encoder", preprocessing=preprocessing, output_variant="cls"
        ),
    }
    return {
        "tile": build_tile_cache_key(
            tile_encoder_name="stub-encoder", preprocessing=preprocessing, execution=_FP16, output_variant="cls"
        ),
        "slide": build_slide_cache_key(
            slide_encoder_name="stub-slide",
            tile_dependency_signature=tile_dependency,
            execution=_FP16,
            output_variant="cls",
        ),
        "patient": build_patient_cache_key(
            patient_encoder_name="stub-patient",
            tile_dependency_signature=tile_dependency,
            execution=_FP16,
            output_variant="cls",
        ),
        "hierarchical": build_hierarchical_cache_key(
            tile_encoder_name="stub-encoder", preprocessing=preprocessing, execution=_FP16, output_variant="cls"
        ),
    }


def test_pooled_keys_keep_the_registry_input_size(monkeypatch: pytest.MonkeyPatch):
    _stub_registry(monkeypatch, 224)
    small = _pooled_keys()
    _stub_registry(monkeypatch, 518)
    large = _pooled_keys()
    assert all(small[kind] != large[kind] for kind in small)


def _dataset(tmp_path: Path):
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"sample_id": "s1", "image_path": "/tiles/s1.png", "mask_path": "/masks/s1.png", "label": "a"},
            {"sample_id": "s2", "image_path": "/tiles/s2.png", "mask_path": "/masks/s2.png", "label": "b"},
        ]
    ).to_csv(csv_path, index=False)
    return legacy_samples_from_csv(csv_path)


def _resolve(tmp_path: Path, dataset):
    return resolve_dense_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        tile_encoder_name="stub-encoder",
        patch_size=(16, 16),
        dense=slide2vec.DenseOptions(spacing_um=0.5, target_size=64, window_size=32, overlap=0.5),
        execution=_FP16,
        dtype="fp16",
        cache_kind="dense_image",
    )


def _write_grids(resolution, dataset) -> None:
    geometry = compute_dense_geometry(target_size=(64, 64), patch_size=16)
    for sample_id in dataset.sample_ids:
        sidecar = dense_grid_metadata(geometry, feature_dim=8, pad_mode="reflect")
        sidecar.update(window_size=32, overlap=0.5)
        write_dense_grid(resolution.features_dir, sample_id, torch.randn(8, 4, 4), sidecar)
    record_feature_dim(resolution, 8)
    record_sample_identity_signatures(resolution, list(dataset.sample_ids))


def test_dense_cache_written_under_one_registry_size_is_a_hit_under_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset = _dataset(tmp_path)
    _stub_registry(monkeypatch, 224)
    written = _resolve(tmp_path, dataset)
    _write_grids(written, dataset)

    _stub_registry(monkeypatch, 518)
    reloaded = _resolve(tmp_path, dataset)

    assert reloaded.cache_dir == written.cache_dir
    assert reloaded.complete is True
    recorded = json.loads(reloaded.metadata_path.read_text())
    assert recorded["precision"] == "fp16"
    assert recorded["dense"]["spacing_um"] == 0.5
