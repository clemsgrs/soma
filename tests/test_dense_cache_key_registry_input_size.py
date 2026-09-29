"""The dense cache key must not depend on the encoder registry's default ``input_size``.

A registry edit upstream (slide2vec 6.0.0 moved ``dinov2-vitb14`` from 224 to 518) must not
move a dense cache key: the dense extraction's real encoder input is the configured
target/window, which the key already carries. Pooled keys keep the registry size.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
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
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.dense import compute_dense_geometry, dense_grid_metadata, write_dense_grid


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


def _encoder() -> EncoderConfig:
    return EncoderConfig(name="stub-encoder", precision="fp16")


def _dense_key(**overrides) -> str:
    kwargs = dict(
        tile_encoder_name="stub-encoder",
        target_size=(1024, 1024),
        patch_size=(14, 14),
        pad_mode="reflect",
        execution=_encoder(),
        preprocessing=PreprocessingConfig(requested_spacing_um=0.5),
        dense_input_mode="sliding_window",
        window_size=448,
        overlap=0.5,
        dtype="fp16",
    )
    kwargs.update(overrides)
    return build_dense_cache_key(**kwargs)


def test_dense_key_ignores_the_registry_default_input_size(monkeypatch: pytest.MonkeyPatch):
    _stub_registry(monkeypatch, 224)
    before = _dense_key()
    _stub_registry(monkeypatch, 518)
    assert _dense_key() == before


@pytest.mark.parametrize(
    "override",
    [
        {"target_size": (512, 512)},
        {"dense_input_mode": "whole"},
        {"window_size": 224},
        {"overlap": 0.25},
        {"preprocessing": PreprocessingConfig(requested_spacing_um=0.25)},
        {"execution": EncoderConfig(name="stub-encoder", precision="fp32")},
        {"dtype": "fp32"},
    ],
    ids=["target_size", "dense_input_mode", "window_size", "overlap", "spacing", "precision", "dtype"],
)
def test_dense_key_still_follows_what_the_extraction_computes(
    monkeypatch: pytest.MonkeyPatch, override: dict
):
    _stub_registry(monkeypatch, 224)
    assert _dense_key(**override) != _dense_key()


def _pooled_keys() -> dict[str, str]:
    encoder = EncoderConfig(name="stub-encoder", precision="fp16", output_variant="cls")
    preprocessing = PreprocessingConfig(requested_spacing_um=0.5)
    tile_dependency = {
        "tile_encoder_name": "stub-encoder",
        "tile_preprocessing": preprocessing_signature(preprocessing),
        "tile_execution": execution_signature(
            encoder, encoder_name="stub-encoder", preprocessing=preprocessing
        ),
    }
    return {
        "tile": build_tile_cache_key(
            tile_encoder_name="stub-encoder", preprocessing=preprocessing, execution=encoder
        ),
        "slide": build_slide_cache_key(
            slide_encoder_name="stub-slide",
            tile_dependency_signature=tile_dependency,
            execution=encoder,
        ),
        "patient": build_patient_cache_key(
            patient_encoder_name="stub-patient",
            tile_dependency_signature=tile_dependency,
            execution=encoder,
        ),
        "hierarchical": build_hierarchical_cache_key(
            tile_encoder_name="stub-encoder", preprocessing=preprocessing, execution=encoder
        ),
    }


# Literals recorded on main (9f4e589), before the dense key dropped the registry size.
@pytest.mark.parametrize(
    ("input_size", "expected"),
    [
        (
            224,
            {
                "tile": "91d6340252dc0a80",
                "slide": "a4bdbe7278d987c7",
                "patient": "fd7e7ee45d7c2d48",
                "hierarchical": "ed4e511cc929fe84",
            },
        ),
        (
            518,
            {
                "tile": "34b11f827c2358eb",
                "slide": "081ac377b3b514f7",
                "patient": "49a3d72b3942aa74",
                "hierarchical": "d8114a02e28b7b75",
            },
        ),
    ],
)
def test_pooled_keys_are_byte_identical_and_keep_the_registry_input_size(
    monkeypatch: pytest.MonkeyPatch, input_size: int, expected: dict[str, str]
):
    _stub_registry(monkeypatch, input_size)
    assert _pooled_keys() == expected


def _dataset(tmp_path: Path) -> Dataset:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"sample_id": "s1", "image_path": "/tiles/s1.png", "mask_path": "/masks/s1.png", "label": "a"},
            {"sample_id": "s2", "image_path": "/tiles/s2.png", "mask_path": "/masks/s2.png", "label": "b"},
        ]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _resolve(tmp_path: Path, dataset: Dataset):
    return resolve_dense_cache(
        cache_root=tmp_path / "cache",
        dataset=dataset,
        tile_encoder_name="stub-encoder",
        target_size=(64, 64),
        patch_size=(16, 16),
        pad_mode="reflect",
        execution=_encoder(),
        preprocessing=PreprocessingConfig(requested_spacing_um=0.5),
        dense_input_mode="sliding_window",
        window_size=32,
        overlap=0.5,
        dtype="fp16",
        cache_kind="dense_image",
    )


def _write_grids(resolution, dataset: Dataset) -> None:
    geometry = compute_dense_geometry(target_size=(64, 64), patch_size=16)
    for sample_id in dataset.sample_ids:
        sidecar = dense_grid_metadata(geometry, feature_dim=8, pad_mode="reflect")
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
    assert recorded["execution"] == {"precision": "fp16", "spacing_um": 0.5}
