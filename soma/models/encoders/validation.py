"""Resolve what an encoder decides about extraction: its precision and its tile geometry.

soma holds slide2vec's ``PreprocessingConfig`` and ``ExecutionOptions`` as given (ADR
0019). Where the user left a value to the encoder (``tiling=None``, ``precision=None``),
these helpers fill it from slide2vec's encoder registry, exactly as slide2vec does when it
runs, so the cache key soma derives and the features slide2vec computes cannot disagree.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from hs2p import FilterConfig, TilingConfig
from hs2p.configs.resolvers import is_binary_tissue_tiling
from slide2vec import PreprocessingConfig
from slide2vec.encoders.registry import encoder_registry, resolve_preprocessing_fields

__all__ = ["resolve_encoder_precision", "resolve_filter_config", "resolve_preprocessing_config"]


def resolve_encoder_precision(
    precision: str | None,
    *,
    encoder_name: str,
    model_metadata: dict[str, Any] | None = None,
) -> str:
    """The precision an encoder runs at: ``precision`` when set, else the registry's
    recommendation, else ``fp32``."""
    if precision is not None:
        return str(precision)
    metadata = model_metadata
    if metadata is None:
        try:
            metadata = encoder_registry.info(encoder_name)
        except Exception:
            metadata = None
    recommended = None if metadata is None else metadata.get("precision")
    return "fp32" if recommended is None else str(recommended)


def resolve_filter_config(
    filter_config: FilterConfig | None,
    *,
    tile_size_px: int,
    annotation_sampling: bool = False,
) -> FilterConfig:
    """``filter_config`` when set. Unset, hs2p's filter defaults with ``ref_tile_size`` at
    the tile size (what slide2vec resolves an unset filter to); for annotation sampling
    also ``a_t=0``, so hs2p keeps every annotated component of nonzero area, where its
    default ``a_t=4`` drops those of at most 4 tiles' area."""
    if filter_config is not None:
        return filter_config
    if annotation_sampling:
        return FilterConfig(ref_tile_size=int(tile_size_px), a_t=0)
    return FilterConfig(ref_tile_size=int(tile_size_px))


def resolve_preprocessing_config(
    preprocessing: PreprocessingConfig,
    *,
    encoder_name: str,
    model_metadata: dict[str, Any] | None = None,
) -> PreprocessingConfig:
    """``preprocessing`` with what slide2vec leaves to the encoder resolved.

    ``tiling=None`` takes the spacing and tile size from the encoder preset (tolerance
    0.05, no overlap, ``auto`` backends: slide2vec's own defaults); ``filter=None`` takes
    hs2p's filter defaults with ``ref_tile_size`` at the tile size, and ``a_t=0`` when
    ``masks`` samples annotations (see :func:`resolve_filter_config`); hierarchical region
    geometry is completed from whichever of ``requested_region_size_px`` and
    ``region_tile_multiple`` is set. Explicit values are kept.
    """
    tiling = preprocessing.tiling
    if tiling is None:
        fields = resolve_preprocessing_fields(
            encoder_name,
            requested_spacing_um=None,
            requested_tile_size_px=None,
            metadata=model_metadata,
        )
        tiling = TilingConfig(
            requested_spacing_um=float(fields["spacing_um"]),
            requested_tile_size_px=int(fields["tile_size_px"]),
            tolerance=0.05,
            overlap=0.0,
        )
    tile_size_px = int(tiling.requested_tile_size_px)
    filter_config = resolve_filter_config(
        preprocessing.filter,
        tile_size_px=tile_size_px,
        annotation_sampling=not is_binary_tissue_tiling(preprocessing.masks),
    )

    region_size_px = preprocessing.requested_region_size_px
    multiple = preprocessing.region_tile_multiple
    if multiple is not None:
        multiple = int(multiple)
        if multiple < 2:
            raise ValueError("region_tile_multiple must be >= 2")
        derived = tile_size_px * multiple
        if region_size_px is not None and int(region_size_px) != derived:
            raise ValueError(
                "Hierarchical preprocessing requires requested_region_size_px to match "
                f"requested_tile_size_px × region_tile_multiple ({derived})."
            )
        region_size_px = derived
    elif region_size_px is not None:
        region_size_px = int(region_size_px)
        if region_size_px % tile_size_px != 0:
            raise ValueError("requested_region_size_px must be divisible by requested_tile_size_px")
        multiple = region_size_px // tile_size_px
        if multiple < 2:
            raise ValueError("Hierarchical preprocessing requires region_tile_multiple >= 2")
    return dataclasses.replace(
        preprocessing,
        tiling=tiling,
        filter=filter_config,
        requested_region_size_px=region_size_px,
        region_tile_multiple=multiple,
    )
