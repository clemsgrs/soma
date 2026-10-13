"""Concise builders for the slide2vec / hs2p extraction dataclasses soma holds (ADR 0019).

Tests state only the fields they are about; everything else is hs2p's and slide2vec's own
defaults, as a user constructing the dataclasses would get them.
"""

from __future__ import annotations

from typing import Any

import slide2vec
from hs2p import DEFAULT_MASK_SAMPLING, FilterConfig, MaskSamplingConfig, SegmentationConfig, TilingConfig

__all__ = ["annotation_masks", "extraction", "tiling", "tissue_masks"]


def tiling(spacing: float = 0.5, tile_size: int = 224, **fields: Any) -> TilingConfig:
    """hs2p's ``TilingConfig`` at ``spacing`` / ``tile_size`` (tolerance 0.05, no overlap)."""
    values = {"tolerance": 0.05, "overlap": 0.0, **fields}
    return TilingConfig(requested_spacing_um=float(spacing), requested_tile_size_px=int(tile_size), **values)


def tissue_masks(threshold: float = 0.1) -> MaskSamplingConfig:
    """hs2p's binary tissue vocabulary at a tissue-coverage threshold."""
    from dataclasses import replace

    return replace(DEFAULT_MASK_SAMPLING, min_coverage={"tissue": float(threshold)})


def annotation_masks(
    pixel_mapping: dict[str, Any] | None = None,
    min_coverage: dict[str, float] | None = None,
    **fields: Any,
) -> MaskSamplingConfig:
    """An hs2p annotation vocabulary, merged output by default (what soma extracts)."""
    values = {"output_mode": "merged", **fields}
    return MaskSamplingConfig(
        pixel_mapping=pixel_mapping if pixel_mapping is not None else {"background": 0, "tumor": 1},
        min_coverage=min_coverage if min_coverage is not None else {"tumor": 0.5},
        **values,
    )


def extraction(
    spacing: float | None = 0.5,
    tile_size: int | None = 224,
    *,
    method: str = "hsv",
    seg_downsample: int = 64,
    masks: MaskSamplingConfig | None = None,
    tiling_fields: dict[str, Any] | None = None,
    filter: FilterConfig | None = None,
    **fields: Any,
) -> slide2vec.PreprocessingConfig:
    """slide2vec's ``PreprocessingConfig``: a tiling at ``spacing`` / ``tile_size`` (none when
    either is ``None``), hs2p segmentation by ``method``, soma's default 0.1 tissue
    threshold unless ``masks`` is given, and slide2vec's other defaults."""
    held_tiling = None
    if spacing is not None and tile_size is not None:
        held_tiling = tiling(spacing, tile_size, **(tiling_fields or {}))
    return slide2vec.PreprocessingConfig(
        tiling=held_tiling,
        segmentation=SegmentationConfig(method=method, downsample=seg_downsample),
        filter=filter,
        masks=masks if masks is not None else tissue_masks(),
        **fields,
    )
