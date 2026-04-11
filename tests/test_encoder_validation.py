"""Tests for soma.encoders.validation — config vs recommended settings checks."""

from __future__ import annotations

import numpy as np
import pytest

from soma.config import EncoderConfig, PreprocessingConfig
from slide2vec.encoders.registry import encoder_registry
from soma.encoders.validation import validate_encoder_config
from hs2p.preprocessing import TileGeometry


def _metadata(
    *,
    output_variants: dict | None = None,
    default_output_variant: str = "default",
    level: str = "tile",
    input_size: int | None = 224,
    tile_encoder: str | None = None,
    tile_encoder_output_variant: str | None = None,
    supported_spacing_um: float | list[float] = 0.5,
    precision: str | None = "fp16",
    source: str = "",
) -> dict:
    payload = {
        "output_variants": output_variants or {"default": {"encode_dim": 1536}},
        "default_output_variant": default_output_variant,
        "level": level,
        "input_size": input_size,
        "tile_encoder": tile_encoder,
        "tile_encoder_output_variant": tile_encoder_output_variant,
        "supported_spacing_um": supported_spacing_um,
        "source": source,
    }
    if precision is not None:
        payload["precision"] = precision
    return payload


def _tiling_result(
    read_spacing_um: float = 0.5,
    requested_tile_size_px: int = 256,
    requested_spacing_um: float = 0.5,
) -> TileGeometry:
    return TileGeometry(
        x=np.array([0], dtype=np.int64),
        y=np.array([0], dtype=np.int64),
        tissue_fractions=np.array([1.0], dtype=np.float32),
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        read_level=0,
        read_tile_size_px=requested_tile_size_px,
        read_spacing_um=read_spacing_um,
        tile_size_lv0=requested_tile_size_px,
        is_within_tolerance=True,
        base_spacing_um=0.25,
        slide_dimensions=[1000, 1000],
        level_downsamples=[1.0],
        overlap=0.0,
        min_tissue_fraction=0.5,
    )


class TestValidateEncoderConfig:
    def test_no_warnings_when_matching(self):
        config = EncoderConfig(name="uni2", precision="fp16")
        warnings = validate_encoder_config(config, _metadata())
        assert warnings == []

    def test_missing_level_metadata_errors(self):
        with pytest.raises(ValueError, match="level metadata"):
            validate_encoder_config(
                EncoderConfig(name="uni2", precision="fp16"),
                {
                    "output_variants": {"default": {"encode_dim": 1536}},
                    "default_output_variant": "default",
                    "input_size": 224,
                    "supported_spacing_um": 0.5,
                    "precision": "fp16",
                    "source": "",
                },
            )

    def test_unset_precision_uses_model_recommendation_without_warning(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="uni2", precision=None),
            _metadata(precision="fp16"),
        )
        assert not any("precision" in w.lower() for w in warnings)

    def test_unset_precision_falls_back_to_fp32_when_metadata_has_no_recommendation(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="uni2", precision=None),
            _metadata(precision=None),
        )
        assert not any("precision" in w.lower() for w in warnings)

    def test_precision_mismatch(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="conch", precision="fp16"),
            _metadata(precision="fp32"),
        )
        assert any("precision" in w.lower() for w in warnings)

    def test_spacing_mismatch(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="uni2", spacing_um=1.0),
            _metadata(supported_spacing_um=0.5),
        )
        assert any("spacing" in w.lower() for w in warnings)

    def test_spacing_matches_one_of_multiple(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="model", spacing_um=1.0),
            _metadata(supported_spacing_um=[0.5, 1.0]),
        )
        assert not any("spacing" in w.lower() for w in warnings)

    def test_spacing_none_multiple_recommended_errors(self):
        with pytest.raises(ValueError, match="spacing"):
            validate_encoder_config(
                EncoderConfig(name="model"),
                _metadata(supported_spacing_um=[0.5, 1.0]),
            )

    def test_input_size_mismatch(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="uni2", input_size=512),
            _metadata(input_size=224),
        )
        assert any("input_size" in w.lower() for w in warnings)

    def test_requested_spacing_match_does_not_warn_when_read_spacing_differs(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="uni2", spacing_um=0.5),
            _metadata(supported_spacing_um=0.5),
            tiling_result=_tiling_result(read_spacing_um=0.75),
        )
        assert not any("spacing" in w.lower() for w in warnings)

    def test_requested_spacing_mismatch_with_tiling_warns(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="uni2", spacing_um=0.5),
            _metadata(supported_spacing_um=0.5),
            tiling_result=_tiling_result(
                requested_spacing_um=0.75,
                read_spacing_um=0.5,
            ),
        )
        assert any("requested tiling spacing" in w.lower() for w in warnings)

    def test_preprocessing_tile_size_mismatch(self):
        if "test-validate-slide-base" not in encoder_registry:
            encoder_registry.register(
                "test-validate-slide-base",
                object,
                metadata={
                    "level": "tile",
                    "input_size": 224,
                    "output_variants": {"default": {"encode_dim": 224}},
                    "default_output_variant": "default",
                    "supported_spacing_um": 0.5,
                },
            )
        warnings = validate_encoder_config(
            EncoderConfig(name="prism"),
            _metadata(
                level="slide",
                tile_encoder="test-validate-slide-base",
                tile_encoder_output_variant="default",
                input_size=None,
            ),
            preprocessing_config=PreprocessingConfig(requested_tile_size_px=256),
        )
        assert any("tile size" in w.lower() for w in warnings)

    def test_slide_encoder_rejects_output_variant_override(self):
        with pytest.raises(ValueError, match="output_variant"):
            validate_encoder_config(
                EncoderConfig(name="prism", output_variant="cls"),
                _metadata(
                    level="slide",
                    tile_encoder="test-validate-slide-base",
                    tile_encoder_output_variant="default",
                    input_size=None,
                ),
            )

    def test_slide_encoder_requires_tile_encoder_metadata(self):
        with pytest.raises(ValueError, match="tile_encoder"):
            validate_encoder_config(
                EncoderConfig(name="prism"),
                _metadata(level="slide", tile_encoder=None),
            )

    def test_slide_encoder_dependency_must_exist(self):
        with pytest.raises(KeyError):
            validate_encoder_config(
                EncoderConfig(name="prism"),
                _metadata(level="slide", tile_encoder="missing-tile-encoder"),
            )

    def test_slide_encoder_dependency_must_be_tile_level(self):
        class DummySlide:
            pass

        if "test-slide-validation-dep" not in encoder_registry:
            encoder_registry.register(
                "test-slide-validation-dep",
                DummySlide,
                metadata={"level": "slide"},
            )

        with pytest.raises(ValueError, match="tile encoder"):
            validate_encoder_config(
                EncoderConfig(name="prism"),
                _metadata(level="slide", tile_encoder="test-slide-validation-dep"),
            )

    def test_slide_encoder_dependency_requires_level_metadata(self):
        class DummyMissingLevel:
            pass

        if "test-slide-validation-missing-level" not in encoder_registry:
            encoder_registry.register(
                "test-slide-validation-missing-level",
                DummyMissingLevel,
                metadata={},
            )

        with pytest.raises(ValueError, match="level metadata"):
            validate_encoder_config(
                EncoderConfig(name="prism"),
                _metadata(
                    level="slide",
                    tile_encoder="test-slide-validation-missing-level",
                    tile_encoder_output_variant="default",
                ),
            )

    def test_hoptimus_models_reject_non_default_output_variants(self):
        with pytest.raises(ValueError, match="output_variant"):
            validate_encoder_config(
                EncoderConfig(name="h-optimus-1", output_variant="cls"),
                encoder_registry.info("h-optimus-1"),
            )

    def test_h0_mini_allows_cls_output_variant(self):
        warnings = validate_encoder_config(
            EncoderConfig(name="h0-mini", output_variant="cls"),
            encoder_registry.info("h0-mini"),
        )
        assert warnings == []
