"""Tests for soma.encoders.validation — config vs recommended settings checks."""

from __future__ import annotations

import numpy as np
import pytest

from soma.config import EncoderConfig
from soma.encoders.validation import validate_encoder_config
from soma.preprocessing.tiling import TilingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _metadata(
    *,
    encode_dim: int = 1536,
    input_size: int = 224,
    recommended_spacing_um: float | list[float] = 0.5,
    precision: str = "fp16",
    source: str = "",
) -> dict:
    return {
        "encode_dim": encode_dim,
        "input_size": input_size,
        "recommended_spacing_um": recommended_spacing_um,
        "precision": precision,
        "source": source,
    }


def _tiling_result(effective_spacing_um: float = 0.5) -> TilingResult:
    return TilingResult(
        coordinates=np.array([[0, 0]], dtype=np.int64),
        tissue_fractions=np.array([1.0], dtype=np.float32),
        requested_tile_size_px=256,
        requested_spacing_um=0.5,
        read_level=0,
        effective_tile_size_px=256,
        effective_spacing_um=effective_spacing_um,
        tile_size_lv0=256,
        is_within_tolerance=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestValidateEncoderConfig:
    def test_no_warnings_when_matching(self):
        config = EncoderConfig(name="uni2", precision="fp16")
        metadata = _metadata(precision="fp16", recommended_spacing_um=0.5)
        warnings = validate_encoder_config(config, metadata)
        assert warnings == []

    def test_precision_mismatch(self):
        config = EncoderConfig(name="conch", precision="fp16")
        metadata = _metadata(precision="fp32")
        warnings = validate_encoder_config(config, metadata)
        assert any("precision" in w.lower() for w in warnings)

    def test_spacing_mismatch(self):
        config = EncoderConfig(name="uni2", spacing_um=1.0)
        metadata = _metadata(recommended_spacing_um=0.5)
        warnings = validate_encoder_config(config, metadata)
        assert any("spacing" in w.lower() for w in warnings)

    def test_spacing_matches_one_of_multiple(self):
        config = EncoderConfig(name="model", spacing_um=1.0)
        metadata = _metadata(recommended_spacing_um=[0.5, 1.0])
        warnings = validate_encoder_config(config, metadata)
        # No spacing warning since 1.0 is in the list
        assert not any("spacing" in w.lower() for w in warnings)

    def test_spacing_none_single_recommended(self):
        """When spacing_um is None and model has one default, no warning."""
        config = EncoderConfig(name="uni2")
        metadata = _metadata(recommended_spacing_um=0.5)
        warnings = validate_encoder_config(config, metadata)
        assert not any("spacing" in w.lower() for w in warnings)

    def test_spacing_none_multiple_recommended_errors(self):
        """When spacing_um is None and model has multiple defaults, error."""
        config = EncoderConfig(name="model")
        metadata = _metadata(recommended_spacing_um=[0.5, 1.0])
        with pytest.raises(ValueError, match="spacing"):
            validate_encoder_config(config, metadata)

    def test_input_size_mismatch(self):
        config = EncoderConfig(name="uni2", input_size=512)
        metadata = _metadata(input_size=224)
        warnings = validate_encoder_config(config, metadata)
        assert any("input_size" in w.lower() for w in warnings)

    def test_input_size_none_uses_default(self):
        config = EncoderConfig(name="uni2")
        metadata = _metadata(input_size=224)
        warnings = validate_encoder_config(config, metadata)
        assert not any("input_size" in w.lower() for w in warnings)

    def test_effective_spacing_mismatch_with_tiling(self):
        config = EncoderConfig(name="uni2", spacing_um=0.5)
        metadata = _metadata(recommended_spacing_um=0.5)
        tiling = _tiling_result(effective_spacing_um=0.75)
        warnings = validate_encoder_config(config, metadata, tiling_result=tiling)
        assert any("effective" in w.lower() for w in warnings)
