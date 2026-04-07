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
    tile_multiple = params.get("tile_multiple")
    region_size = params.get("region_size")
    patch_size = params.get("patch_size")

    if tile_multiple is None and region_size is None and patch_size is None:
        if preprocessing.has_hierarchical_geometry:
            resolved_region_tile_multiple = preprocessing.region_tile_multiple
            if resolved_region_tile_multiple is not None:
                resolved_region_tile_multiple = int(resolved_region_tile_multiple)
                if resolved_region_tile_multiple < 2:
                    raise ValueError("region_tile_multiple must be >= 2")
            return replace(
                preprocessing,
                hierarchical=True,
                npatch=resolved_region_tile_multiple,
            )
        raise ValueError(
            "HIPT aggregator requires 'tile_multiple' or legacy 'region_size' and "
            "'patch_size' params"
        )

    resolved_region_tile_multiple: int | None = None
    resolved_target_region_size_px: int | None = preprocessing.target_region_size_px
    resolved_hierarchical_patch_size_px: int | None = preprocessing.hierarchical_patch_size_px

    if tile_multiple is not None:
        resolved_region_tile_multiple = int(tile_multiple)
        if resolved_region_tile_multiple < 2:
            raise ValueError("HIPT aggregator requires tile_multiple >= 2")
        if patch_size is not None:
            resolved_hierarchical_patch_size_px = int(patch_size)
        if region_size is not None:
            resolved_target_region_size_px = int(region_size)
    elif region_size is not None or patch_size is not None:
        if region_size is None or patch_size is None:
            raise ValueError(
                "HIPT aggregator requires both 'region_size' and 'patch_size' when "
                "'tile_multiple' is not provided"
            )
        resolved_target_region_size_px = int(region_size)
        resolved_hierarchical_patch_size_px = int(patch_size)
        if resolved_target_region_size_px % resolved_hierarchical_patch_size_px != 0:
            raise ValueError(
                f"region_size ({resolved_target_region_size_px}) must be divisible "
                f"by patch_size ({resolved_hierarchical_patch_size_px})"
            )
        if resolved_target_region_size_px < 2 * resolved_hierarchical_patch_size_px:
            raise ValueError(
                f"region_size ({resolved_target_region_size_px}) must be >= 2 * "
                f"patch_size ({resolved_hierarchical_patch_size_px})"
            )
        resolved_region_tile_multiple = (
            resolved_target_region_size_px // resolved_hierarchical_patch_size_px
        )
    if preprocessing.target_region_size_px is not None and resolved_target_region_size_px is not None:
        if int(preprocessing.target_region_size_px) != int(resolved_target_region_size_px):
            raise ValueError(
                "HIPT derives target_region_size_px from aggregator params; "
                f"got target_region_size_px={preprocessing.target_region_size_px} and "
                f"resolved_region_size_px={resolved_target_region_size_px}."
            )

    return replace(
        preprocessing,
        target_region_size_px=resolved_target_region_size_px,
        region_tile_multiple=resolved_region_tile_multiple,
        hierarchical=True,
        npatch=resolved_region_tile_multiple,
        hierarchical_patch_size_px=resolved_hierarchical_patch_size_px,
    )
