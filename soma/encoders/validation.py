"""Validate encoder config against model recommended settings."""

from __future__ import annotations

from typing import Any
from dataclasses import replace

from soma.config import EncoderConfig, PreprocessingConfig
from soma.encoders.registry import (
    encoder_registry,
    require_encoder_metadata_field,
    resolve_encoder_output,
    resolve_encoder_level,
    resolve_preprocessing_requirements,
    resolve_tile_dependency_output,
)
from soma.preprocessing.tiling import TilingResult


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
    requested_tile_size_px = preprocessing_config.requested_tile_size_px
    if requested_tile_size_px is None:
        requested_tile_size_px = int(requirements["tile_size_px"])

    requested_spacing_um = preprocessing_config.requested_spacing_um
    if requested_spacing_um is None:
        spacing_override = encoder_config.spacing_um
        if spacing_override is not None:
            requested_spacing_um = float(spacing_override)
        else:
            rec_spacing = requirements["spacing_um"]
            if isinstance(rec_spacing, list):
                if len(rec_spacing) != 1:
                    raise ValueError(
                        f"Encoder '{encoder_config.name}' supports multiple spacings "
                        f"{rec_spacing}; please specify EncoderConfig.spacing_um or "
                        "PreprocessingConfig.requested_spacing_um."
                    )
                requested_spacing_um = float(rec_spacing[0])
            else:
                requested_spacing_um = float(rec_spacing)

    ref_tile_size_px = preprocessing_config.ref_tile_size_px
    if ref_tile_size_px is None:
        ref_tile_size_px = int(requested_tile_size_px)

    return replace(
        preprocessing_config,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        ref_tile_size_px=ref_tile_size_px,
    )


def validate_encoder_config(
    config: EncoderConfig,
    model_metadata: dict[str, Any],
    *,
    preprocessing_config: PreprocessingConfig | None = None,
    tiling_result: TilingResult | None = None,
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
        if preprocessing_config.requested_tile_size_px != rec_tile_size:
            if preprocessing_config.hierarchical:
                # In hierarchical mode (HIPT), tile_size is the region size,
                # which is intentionally larger than the encoder's tile_size.
                if preprocessing_config.requested_tile_size_px < 2 * rec_tile_size:
                    warnings.append(
                        f"Hierarchical tile size "
                        f"{preprocessing_config.requested_tile_size_px}px is less than "
                        f"2× the encoder tile size ({rec_tile_size}px)."
                    )
            else:
                warnings.append(
                    f"Tile size mismatch: preprocessing uses "
                    f"{preprocessing_config.requested_tile_size_px}px, model recommends "
                    f"{rec_tile_size}px."
                )
        requested_spacing = config.spacing_um or preprocessing_config.requested_spacing_um
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
