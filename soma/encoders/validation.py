"""Validate encoder config against model recommended settings."""

from __future__ import annotations

from typing import Any
from dataclasses import replace

from soma.config import EncoderConfig, PreprocessingConfig
from slide2vec.encoders.registry import (
    encoder_registry,
    resolve_encoder_output,
    resolve_encoder_level,
    resolve_preprocessing_requirements,
    resolve_tile_dependency_output,
)
from hs2p.preprocessing import TileGeometry


def resolve_encoder_precision(
    config: EncoderConfig,
    *,
    encoder_name: str | None = None,
    model_metadata: dict[str, Any] | None = None,
) -> str:
    """Resolve the precision used at runtime.

    Explicit user overrides win. Otherwise use the model recommendation when
    available and fall back to fp32 when the registry does not provide one.
    """
    if config.precision is not None:
        return str(config.precision)

    metadata = model_metadata
    effective_encoder_name = encoder_name or config.name
    if metadata is None:
        try:
            metadata = encoder_registry.info(effective_encoder_name)
        except Exception:
            metadata = None

    recommended_precision = None
    if metadata is not None:
        recommended_precision = metadata.get("precision")
    if recommended_precision is not None:
        return str(recommended_precision)
    return "fp32"


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

    requested_region_size_px = preprocessing_config.requested_region_size_px
    region_tile_multiple = preprocessing_config.region_tile_multiple
    read_region_size_px = preprocessing_config.read_region_size_px
    if region_tile_multiple is not None:
        region_tile_multiple = int(region_tile_multiple)
        if region_tile_multiple < 2:
            raise ValueError("region_tile_multiple must be >= 2")
        derived_region_size_px = int(requested_tile_size_px) * region_tile_multiple
        if requested_region_size_px is not None and int(requested_region_size_px) != derived_region_size_px:
            raise ValueError(
                "Hierarchical preprocessing requires requested_region_size_px to match "
                f"requested_tile_size_px × region_tile_multiple ({derived_region_size_px})."
            )
        requested_region_size_px = derived_region_size_px
    elif requested_region_size_px is not None:
        requested_region_size_px = int(requested_region_size_px)
        if requested_region_size_px % int(requested_tile_size_px) != 0:
            raise ValueError(
                "requested_region_size_px must be divisible by requested_tile_size_px"
            )
        region_tile_multiple = requested_region_size_px // int(requested_tile_size_px)
        if region_tile_multiple < 2:
            raise ValueError("Hierarchical preprocessing requires region_tile_multiple >= 2")

    if read_region_size_px is None and requested_region_size_px is not None:
        read_region_size_px = requested_region_size_px

    return replace(
        preprocessing_config,
        requested_tile_size_px=requested_tile_size_px,
        requested_spacing_um=requested_spacing_um,
        ref_tile_size_px=ref_tile_size_px,
        requested_region_size_px=requested_region_size_px,
        region_tile_multiple=region_tile_multiple,
        read_tile_size_px=int(requested_tile_size_px),
        read_region_size_px=read_region_size_px,
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

    rec_precision = model_metadata.get("precision")
    resolved_precision = resolve_encoder_precision(config, model_metadata=model_metadata)
    if config.precision is not None and rec_precision is not None and config.precision != rec_precision:
        warnings.append(
            f"Precision mismatch: config uses '{resolved_precision}', "
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
        preprocessing_tile_size = preprocessing_config.requested_tile_size_px
        if preprocessing_tile_size != rec_tile_size:
            warnings.append(
                f"Tile size mismatch: preprocessing uses "
                f"{preprocessing_tile_size}px, model recommends "
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
        recommended_spacing_um = config.spacing_um or (
            rec_spacing if isinstance(rec_spacing, (int, float)) else None
        )
        if recommended_spacing_um is not None:
            requested_spacing = tiling_result.requested_spacing_um
            if abs(requested_spacing - recommended_spacing_um) / recommended_spacing_um > 0.05:
                warnings.append(
                    f"Requested tiling spacing ({requested_spacing:.4f} µm/px) differs from "
                    f"recommended ({recommended_spacing_um} µm/px) by more than 5%."
                )

        if rec_tile_size is not None and tiling_result.requested_tile_size_px != rec_tile_size:
            warnings.append(
                f"TilingResult uses tile size {tiling_result.requested_tile_size_px}px, "
                f"model recommends {rec_tile_size}px."
            )

    return warnings
