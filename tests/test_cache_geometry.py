"""Extraction geometry recorded in the feature cache, and validated on reuse (ADR 0008).

The record is the triple — requested tile size, per-slide read tile size, and the effective
encoder input — and only the last is validated, because it is the one soma can derive from
config plus registry without loading a model, and the one whose change means the cached
features are registered to a different extent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from soma.cache import (
    CacheGeometryMismatch,
    GEOMETRY_METADATA_KEY,
    dense_extraction_geometry,
    pooled_extraction_geometry,
    resolve_dense_cache,
    resolve_tile_cache,
)
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset

# phikon is 224 px native and cannot take a variable encoder input; uni is 224 px native
# and can, so a larger request reaches the encoder at that larger size.
FIXED_ENCODER = "phikon"
VARIABLE_ENCODER = "uni"


def _dataset(tmp_path: Path) -> Dataset:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"sample_id": "s1", "image_path": "/slides/s1.svs", "label": "tumor"},
            {"sample_id": "s2", "image_path": "/slides/s2.svs", "label": "normal"},
        ]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def test_pooled_geometry_records_the_shipped_transform_regime():
    """A request the encoder's own transform resizes: the encoder sees its preset size."""
    geometry = pooled_extraction_geometry(
        encoder_name=FIXED_ENCODER,
        requested_tile_size_px=224,
        read_tile_size_px_by_id={"s1": 448, "s2": 224},
    )
    assert geometry["requested_tile_size_px"] == [224, 224]
    assert geometry["encoder_input_size_px"] == [224, 224]
    # Per slide: the same request reads 448 px off a 0.25 µm slide and 224 off a 0.5 one.
    assert geometry["read_tile_size_px_by_id"] == {"s1": 448, "s2": 224}


def test_pooled_geometry_records_the_normalization_only_regime():
    """The same request on a variable-input encoder reaches the encoder unresized.

    This is the regime shift the record exists to catch: identical request, different
    effective encoder input.
    """
    shipped = pooled_extraction_geometry(
        encoder_name=VARIABLE_ENCODER, requested_tile_size_px=224
    )
    normalization_only = pooled_extraction_geometry(
        encoder_name=VARIABLE_ENCODER,
        requested_tile_size_px=512,
        # A non-preset pooled size deviates from the model card's tiling recipe, so it is
        # opt-in — the same gate the extractor passes through from EncoderConfig.
        allow_non_recommended_settings=True,
    )
    assert shipped["encoder_input_size_px"] == [224, 224]
    assert normalization_only["encoder_input_size_px"] == [512, 512]


def test_pooled_geometry_is_undeclarable_before_the_tile_size_resolves():
    assert (
        pooled_extraction_geometry(encoder_name=FIXED_ENCODER, requested_tile_size_px=None)
        is None
    )


def test_dense_geometry_derives_the_encoder_input_from_the_supervision_size():
    """Whole-tile encodes the padded tile; sliding encodes one patch-aligned window."""
    whole = dense_extraction_geometry(
        encoder_name=FIXED_ENCODER, target_size_px=224, window_size=None
    )
    sliding = dense_extraction_geometry(
        encoder_name=FIXED_ENCODER, target_size_px=512, window_size=224
    )
    assert whole["requested_tile_size_px"] == [224, 224]
    assert whole["encoder_input_size_px"] == [224, 224]
    assert sliding["requested_tile_size_px"] == [512, 512]
    assert sliding["encoder_input_size_px"] == [224, 224]


def test_tile_cache_records_the_geometry(tmp_path: Path):
    dataset = _dataset(tmp_path)
    resolution = resolve_tile_cache(
        cache_root=tmp_path / "feature_cache",
        dataset=dataset,
        tile_encoder_name=VARIABLE_ENCODER,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        execution=EncoderConfig(name=VARIABLE_ENCODER),
        extraction_geometry=pooled_extraction_geometry(
            encoder_name=VARIABLE_ENCODER,
            requested_tile_size_px=224,
            read_tile_size_px_by_id={"s1": 448},
        ),
    )
    recorded = json.loads(resolution.metadata_path.read_text())[GEOMETRY_METADATA_KEY]
    assert recorded["encoder_input_size_px"] == [224, 224]
    assert recorded["read_tile_size_px_by_id"] == {"s1": 448}


def test_reusing_a_cache_under_a_changed_encoder_input_is_a_hard_error(tmp_path: Path):
    """A regime shift raises rather than silently recomputing a whole feature set."""
    dataset = _dataset(tmp_path)
    kwargs = dict(
        cache_root=tmp_path / "feature_cache",
        dataset=dataset,
        tile_encoder_name=VARIABLE_ENCODER,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        execution=EncoderConfig(name=VARIABLE_ENCODER),
    )
    resolve_tile_cache(
        **kwargs,
        extraction_geometry=pooled_extraction_geometry(
            encoder_name=VARIABLE_ENCODER, requested_tile_size_px=224
        ),
    )

    # The same key, resolved by a run whose 224 px request now reaches the encoder at 512:
    # the cached features are registered to a different extent.
    with pytest.raises(CacheGeometryMismatch, match="224x224px.*512x512px"):
        resolve_tile_cache(
            **kwargs,
            extraction_geometry=pooled_extraction_geometry(
                encoder_name=VARIABLE_ENCODER,
                requested_tile_size_px=512,
                allow_non_recommended_settings=True,
            ),
        )


def test_an_unchanged_encoder_input_reuses_the_cache(tmp_path: Path):
    dataset = _dataset(tmp_path)
    kwargs = dict(
        cache_root=tmp_path / "feature_cache",
        dataset=dataset,
        tile_encoder_name=VARIABLE_ENCODER,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        execution=EncoderConfig(name=VARIABLE_ENCODER),
    )
    first = resolve_tile_cache(
        **kwargs,
        extraction_geometry=pooled_extraction_geometry(
            encoder_name=VARIABLE_ENCODER,
            requested_tile_size_px=224,
            read_tile_size_px_by_id={"s1": 448},
        ),
    )
    # A different *read* size is provenance, not a mismatch: it cannot change what the
    # encoder saw, and a genuinely different read already changes the tiling behind the key.
    second = resolve_tile_cache(
        **kwargs,
        extraction_geometry=pooled_extraction_geometry(
            encoder_name=VARIABLE_ENCODER,
            requested_tile_size_px=224,
            read_tile_size_px_by_id={"s1": 224},
        ),
    )
    assert second.cache_dir == first.cache_dir


def test_a_cache_predating_the_record_is_still_reusable(tmp_path: Path):
    """Nothing recorded means nothing to disagree with — validation stays silent."""
    dataset = _dataset(tmp_path)
    kwargs = dict(
        cache_root=tmp_path / "feature_cache",
        dataset=dataset,
        tile_encoder_name=VARIABLE_ENCODER,
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        execution=EncoderConfig(name=VARIABLE_ENCODER),
    )
    resolve_tile_cache(**kwargs)  # written without a geometry record

    resumed = resolve_tile_cache(
        **kwargs,
        extraction_geometry=pooled_extraction_geometry(
            encoder_name=VARIABLE_ENCODER, requested_tile_size_px=224
        ),
    )
    assert resumed.metadata_path.is_file()


def test_dense_cache_records_the_geometry(tmp_path: Path):
    dataset = _dataset(tmp_path)
    resolution = resolve_dense_cache(
        cache_root=tmp_path / "feature_cache",
        dataset=dataset,
        tile_encoder_name=FIXED_ENCODER,
        target_size=(512, 512),
        patch_size=(16, 16),
        pad_mode="reflect",
        execution=EncoderConfig(name=FIXED_ENCODER),
        window_size=224,
        overlap=0.0,
        extraction_geometry=dense_extraction_geometry(
            encoder_name=FIXED_ENCODER, target_size_px=(512, 512), window_size=224
        ),
    )
    recorded = json.loads(resolution.metadata_path.read_text())[GEOMETRY_METADATA_KEY]
    assert recorded["requested_tile_size_px"] == [512, 512]
    assert recorded["encoder_input_size_px"] == [224, 224]
