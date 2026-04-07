"""Validate encoder config against model recommended settings."""

from __future__ import annotations

from typing import Any
from dataclasses import replace

from soma.config import EncoderConfig, PreprocessingConfig
from slide2vec.encoders.registry import (
    encoder_registry,
    require_encoder_metadata_field,
    resolve_encoder_output,
    resolve_encoder_level,
    resolve_preprocessing_requirements,
    resolve_tile_dependency_output,
)
from hs2p.preprocessing import TileGeometry


def resolve_preprocessing_config(
    encoder_config: EncoderConfig,
    preprocessing_config: PreprocessingConfig,
    *,
    model_metadata: dict[str, Any] | None = None,
) -> PreprocessingConfig:
    """Fill encoder-driven preprocessing defaults without overriding explicit values."""
    requirements = resolve_preprocessing_requirements(
        encoder_config.name,
        metadata=model_metadata,
    )
    target_tile_size_px = preprocessing_config.target_tile_size_px
    if target_tile_size_px is None:
        target_tile_size_px = int(requirements["tile_size_px"])

    target_spacing_um = preprocessing_config.target_spacing_um
    if target_spacing_um is None:
        spacing_override = encoder_config.spacing_um
        if spacing_override is not None:
            target_spacing_um = float(spacing_override)
        else:
            rec_spacing = requirements["spacing_um"]
            if isinstance(rec_spacing, list):
                if len(rec_spacing) != 1:
                    raise ValueError(
                        f"Encoder '{encoder_config.name}' supports multiple spacings "
                        f"{rec_spacing}; please specify EncoderConfig.spacing_um or "
                        "PreprocessingConfig.target_spacing_um."
                    )
                target_spacing_um = float(rec_spacing[0])
            else:
                target_spacing_um = float(rec_spacing)

    ref_tile_size_px = preprocessing_config.ref_tile_size_px
    if ref_tile_size_px is None:
        ref_tile_size_px = int(target_tile_size_px)

    target_region_size_px = preprocessing_config.target_region_size_px
    region_tile_multiple = preprocessing_config.region_tile_multiple
    effective_region_size_px = preprocessing_config.effective_region_size_px
    if region_tile_multiple is not None:
        region_tile_multiple = int(region_tile_multiple)
        if region_tile_multiple < 2:
            raise ValueError("region_tile_multiple must be >= 2")
        derived_region_size_px = int(target_tile_size_px) * region_tile_multiple
        if target_region_size_px is not None and int(target_region_size_px) != derived_region_size_px:
            raise ValueError(
                "Hierarchical preprocessing requires target_region_size_px to match "
                f"target_tile_size_px × region_tile_multiple ({derived_region_size_px})."
            )
        target_region_size_px = derived_region_size_px
    elif target_region_size_px is not None:
        target_region_size_px = int(target_region_size_px)
        if target_region_size_px % int(target_tile_size_px) != 0:
            raise ValueError(
                "target_region_size_px must be divisible by target_tile_size_px"
            )
        region_tile_multiple = target_region_size_px // int(target_tile_size_px)
        if region_tile_multiple < 2:
            raise ValueError("Hierarchical preprocessing requires region_tile_multiple >= 2")

    if effective_region_size_px is None and target_region_size_px is not None:
        effective_region_size_px = target_region_size_px

    return replace(
        preprocessing_config,
        target_tile_size_px=target_tile_size_px,
        target_spacing_um=target_spacing_um,
        ref_tile_size_px=ref_tile_size_px,
        target_region_size_px=target_region_size_px,
        region_tile_multiple=region_tile_multiple,
        effective_tile_size_px=int(target_tile_size_px),
        effective_region_size_px=effective_region_size_px,
    )


def validate_encoder_config(
    config: EncoderConfig,
    model_metadata: dict[str, Any],
    *,
    preprocessing_config: PreprocessingConfig | None = None,
    tiling_result: TileGeometry | None = None,
) -> list[str]:
    """Check config against recommended model settings.

    Returns a list of warning strings. Raises ValueError when the configuration
    is invalid or ambiguous.
    """
    warnings: list[str] = []

    level = resolve_encoder_level(config.name, model_metadata)

    tile_encoder_name = model_metadata.get("tile_encoder")
    if level == "slide":
        if not tile_encoder_name:
            raise ValueError("Slide encoder metadata must declare a tile_encoder")
        tile_metadata = encoder_registry.info(tile_encoder_name)
        tile_level = resolve_encoder_level(tile_encoder_name, tile_metadata)
        if tile_level != "tile":
            raise ValueError(
                f"Slide encoder dependency '{tile_encoder_name}' must resolve to a tile encoder"
            )

    rec_precision = str(
        require_encoder_metadata_field(config.name, model_metadata, "precision")
    )
    if config.precision != rec_precision:
        warnings.append(
            f"Precision mismatch: config uses '{config.precision}', "
            f"model recommends '{rec_precision}'."
        )

    resolve_encoder_output(
        config.name,
        requested_output_variant=config.output_variant,
        metadata=model_metadata,
    )
    if level == "slide":
        resolve_tile_dependency_output(config.name, metadata=model_metadata)

    rec_requirements = resolve_preprocessing_requirements(config.name, metadata=model_metadata)
    rec_spacing = rec_requirements["spacing_um"]
    if config.spacing_um is None:
        if isinstance(rec_spacing, list) and len(rec_spacing) > 1:
            raise ValueError(
                f"Model supports multiple spacings {rec_spacing} but "
                f"config.spacing_um is None. Please specify one."
            )
    else:
        valid_spacings = rec_spacing if isinstance(rec_spacing, list) else [rec_spacing]
        if config.spacing_um not in valid_spacings:
            warnings.append(
                f"Spacing mismatch: config uses {config.spacing_um} µm/px, "
                f"model recommends {rec_spacing}."
            )

    rec_input_size = model_metadata.get("input_size")
    if config.input_size is not None and rec_input_size is not None:
        if config.input_size != rec_input_size:
            warnings.append(
                f"input_size mismatch: config uses {config.input_size}px, "
                f"model recommends {rec_input_size}px. The model's transform "
                f"will resize/crop to {rec_input_size}px unless overridden."
            )

    rec_tile_size = rec_requirements["tile_size_px"]
    rec_spacing = rec_requirements["spacing_um"]
    if preprocessing_config is not None and rec_tile_size is not None:
        preprocessing_tile_size = preprocessing_config.target_tile_size_px
        if preprocessing_tile_size != rec_tile_size:
            warnings.append(
                f"Tile size mismatch: preprocessing uses "
                f"{preprocessing_tile_size}px, model recommends "
                f"{rec_tile_size}px."
            )
        requested_spacing = config.spacing_um or preprocessing_config.target_spacing_um

        valid_spacings = rec_spacing if isinstance(rec_spacing, list) else [rec_spacing]
        if requested_spacing not in valid_spacings:
            warnings.append(
                f"Spacing mismatch: preprocessing uses {requested_spacing} µm/px, "
                f"model recommends {rec_spacing}."
            )

    if tiling_result is not None:
        target_spacing = config.spacing_um or (
            rec_spacing if isinstance(rec_spacing, (int, float)) else None
        )
        if target_spacing is not None:
            requested_spacing = tiling_result.requested_spacing_um
            if abs(requested_spacing - target_spacing) / target_spacing > 0.05:
                warnings.append(
                    f"Requested tiling spacing ({requested_spacing:.4f} µm/px) differs from "
                    f"target ({target_spacing} µm/px) by more than 5%."
                )

        if rec_tile_size is not None and tiling_result.requested_tile_size_px != rec_tile_size:
            warnings.append(
                f"TilingResult uses tile size {tiling_result.requested_tile_size_px}px, "
                f"model recommends {rec_tile_size}px."
            )

    return warnings
