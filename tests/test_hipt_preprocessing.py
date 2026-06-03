"""Tests for HIPT-specific preprocessing derivation."""

from soma.config import AggregatorConfig, EncoderConfig, PreprocessingConfig
from soma.encoders.validation import resolve_preprocessing_config
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator


def test_hipt_derives_hierarchical_preprocessing_from_tile_multiple():
    preprocessing = derive_preprocessing_for_aggregator(
        PreprocessingConfig(requested_spacing_um=0.5),
        AggregatorConfig(name="hipt", params={"tile_multiple": 6}),
    )

    assert preprocessing.region_tile_multiple == 6
    assert preprocessing.requested_region_size_px is None

    resolved = resolve_preprocessing_config(
        EncoderConfig(name="uni2"),
        preprocessing,
    )

    assert resolved.requested_tile_size_px == 224
    assert resolved.read_tile_size_px == 224
    assert resolved.region_tile_multiple == 6
    assert resolved.requested_region_size_px == 1344
    assert resolved.read_region_size_px == 1344


def test_hipt_rejects_conflicting_requested_region_size():
    preprocessing = derive_preprocessing_for_aggregator(
        PreprocessingConfig(
            requested_spacing_um=0.5,
            requested_region_size_px=1024,
        ),
        AggregatorConfig(name="hipt", params={"tile_multiple": 6}),
    )

    try:
        resolve_preprocessing_config(EncoderConfig(name="uni2"), preprocessing)
    except ValueError as exc:
        assert "requested_region_size_px" in str(exc)
    else:
        raise AssertionError("Expected conflicting HIPT region size to raise ValueError")
