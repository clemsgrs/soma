"""What an encoder decides about extraction: the tiling and precision left to it.

soma holds slide2vec's ``PreprocessingConfig`` and ``ExecutionOptions`` as given (ADR
0019); ``tiling=None`` and ``precision=None`` are resolved from slide2vec's encoder
registry, as slide2vec does when it runs.
"""

from __future__ import annotations

import pytest
import slide2vec
from hs2p import FilterConfig

from soma.models.encoders.validation import (
    resolve_encoder_precision,
    resolve_filter_config,
    resolve_preprocessing_config,
)
from tests.extraction_configs import tiling


@pytest.mark.parametrize(
    ("encoder_name", "expected_tile_size"),
    [("dinov2-vitb14", 224), ("dinov3-vitb16", 256)],
)
def test_an_unset_tiling_takes_the_encoder_preset(encoder_name, expected_tile_size):
    resolved = resolve_preprocessing_config(slide2vec.PreprocessingConfig(), encoder_name=encoder_name)

    assert resolved.tiling == tiling(0.5, expected_tile_size)
    # slide2vec's default filter: hs2p's, with the reference tile at the tile size.
    assert resolved.filter == FilterConfig(ref_tile_size=expected_tile_size)


def test_an_explicit_tiling_and_filter_are_kept():
    held = slide2vec.PreprocessingConfig(
        tiling=tiling(0.25, 256, overlap=0.5),
        filter=FilterConfig(ref_tile_size=64, a_t=2),
    )

    resolved = resolve_preprocessing_config(held, encoder_name="dinov3-vitb16")

    assert (resolved.tiling, resolved.filter) == (held.tiling, held.filter)


def test_an_unset_filter_keeps_every_annotation_component_and_hs2p_s_tissue_threshold():
    explicit = FilterConfig(ref_tile_size=64, a_t=2)

    assert resolve_filter_config(explicit, tile_size_px=224) is explicit
    assert resolve_filter_config(explicit, tile_size_px=224, annotation_sampling=True) is explicit
    # Tissue tiling: hs2p's defaults (a_t=4), the reference tile at the tile size.
    tissue = resolve_filter_config(None, tile_size_px=224)
    assert tissue == FilterConfig(ref_tile_size=224)
    assert tissue.a_t == 4
    # Annotation sampling: no contour-area threshold, so small annotations are sampled.
    assert resolve_filter_config(None, tile_size_px=224, annotation_sampling=True) == FilterConfig(
        ref_tile_size=224, a_t=0
    )


def test_region_geometry_is_completed_from_either_side():
    from_multiple = resolve_preprocessing_config(
        slide2vec.PreprocessingConfig(tiling=tiling(0.5, 224), region_tile_multiple=4), encoder_name="uni2"
    )
    from_size = resolve_preprocessing_config(
        slide2vec.PreprocessingConfig(tiling=tiling(0.5, 224), requested_region_size_px=896), encoder_name="uni2"
    )

    assert (from_multiple.requested_region_size_px, from_multiple.region_tile_multiple) == (896, 4)
    assert (from_size.requested_region_size_px, from_size.region_tile_multiple) == (896, 4)
    with pytest.raises(ValueError, match="divisible"):
        resolve_preprocessing_config(
            slide2vec.PreprocessingConfig(tiling=tiling(0.5, 224), requested_region_size_px=900),
            encoder_name="uni2",
        )


def test_precision_is_explicit_else_the_recommendation_else_fp32():
    assert resolve_encoder_precision("bf16", encoder_name="uni2") == "bf16"
    assert resolve_encoder_precision(None, encoder_name="x", model_metadata={"precision": "fp16"}) == "fp16"
    assert resolve_encoder_precision(None, encoder_name="x", model_metadata={}) == "fp32"
