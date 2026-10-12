"""Tests for HIPT-specific preprocessing derivation."""

import pytest
import slide2vec

from soma.config import AggregatorConfig
from soma.encoders.validation import resolve_preprocessing_config
from soma.preprocessing.hierarchy import derive_preprocessing_for_aggregator


def test_hipt_derives_hierarchical_preprocessing_from_tile_multiple():
    preprocessing = derive_preprocessing_for_aggregator(
        slide2vec.PreprocessingConfig(),
        AggregatorConfig(name="hipt", params={"tile_multiple": 6}),
    )

    assert preprocessing.region_tile_multiple == 6
    assert preprocessing.requested_region_size_px is None

    resolved = resolve_preprocessing_config(preprocessing, encoder_name="uni2")

    assert resolved.tiling.requested_tile_size_px == 224
    assert resolved.region_tile_multiple == 6
    assert resolved.requested_region_size_px == 1344


def test_hipt_rejects_conflicting_requested_region_size():
    preprocessing = derive_preprocessing_for_aggregator(
        slide2vec.PreprocessingConfig(requested_region_size_px=1024),
        AggregatorConfig(name="hipt", params={"tile_multiple": 6}),
    )

    with pytest.raises(ValueError, match="requested_region_size_px"):
        resolve_preprocessing_config(preprocessing, encoder_name="uni2")
