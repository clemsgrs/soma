"""Tests for HIPT-specific preprocessing derivation."""

from soma.config import AggregatorConfig, PreprocessingConfig
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator


def test_hipt_derives_hierarchical_preprocessing_from_aggregator():
    resolved = derive_preprocessing_for_aggregator(
        PreprocessingConfig(requested_spacing_um=0.5),
        AggregatorConfig(
            name="hipt",
            params={"region_size": 4096, "patch_size": 256},
        ),
    )

    assert resolved.hierarchical is True
    assert resolved.npatch == 16
    assert resolved.hierarchical_patch_size_px == 256
    assert resolved.requested_tile_size_px == 4096
    assert resolved.requested_spacing_um == 0.5


def test_hipt_rejects_conflicting_requested_tile_size():
    try:
        derive_preprocessing_for_aggregator(
            PreprocessingConfig(
                requested_tile_size_px=2048,
                requested_spacing_um=0.5,
            ),
            AggregatorConfig(
                name="hipt",
                params={"region_size": 4096, "patch_size": 256},
            ),
        )
    except ValueError as exc:
        assert "requested_tile_size_px" in str(exc)
    else:
        raise AssertionError("Expected conflicting HIPT tile size to raise ValueError")
