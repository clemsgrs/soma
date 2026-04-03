"""Hierarchy-specific preprocessing derivation for aggregators like HIPT."""

from __future__ import annotations

from dataclasses import replace

from soma.config import AggregatorConfig, PreprocessingConfig


def derive_preprocessing_for_aggregator(
    preprocessing: PreprocessingConfig,
    aggregator: AggregatorConfig | None,
) -> PreprocessingConfig:
    if aggregator is None or aggregator.name != "hipt":
        return preprocessing

    params = aggregator.params
    region_size = params.get("region_size")
    patch_size = params.get("patch_size")
    if region_size is None or patch_size is None:
        raise ValueError("HIPT aggregator requires 'region_size' and 'patch_size' in params")

    region_size = int(region_size)
    patch_size = int(patch_size)
    if region_size % patch_size != 0:
        raise ValueError(
            f"region_size ({region_size}) must be divisible by patch_size ({patch_size})"
        )
    if region_size < 2 * patch_size:
        raise ValueError(
            f"region_size ({region_size}) must be >= 2 * patch_size ({patch_size})"
        )
    if (
        preprocessing.requested_tile_size_px is not None
        and int(preprocessing.requested_tile_size_px) != region_size
    ):
        raise ValueError(
            "HIPT derives requested_tile_size_px from aggregator.params.region_size; "
            f"got requested_tile_size_px={preprocessing.requested_tile_size_px} and "
            f"region_size={region_size}."
        )

    return replace(
        preprocessing,
        requested_tile_size_px=region_size,
        hierarchical=True,
        npatch=region_size // patch_size,
        hierarchical_patch_size_px=patch_size,
    )
