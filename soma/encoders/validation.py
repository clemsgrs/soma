"""Validate encoder config against model recommended settings."""

from __future__ import annotations

from typing import Any

from soma.config import EncoderConfig, PreprocessingConfig
from soma.encoders.registry import encoder_registry
from soma.preprocessing.tiling import TilingResult


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

    level = model_metadata.get("level", "tile")
    if level not in {"tile", "slide"}:
        raise ValueError(f"Unsupported encoder level '{level}'")

    tile_encoder_name = model_metadata.get("tile_encoder")
    if level == "slide":
        if not tile_encoder_name:
            raise ValueError("Slide encoder metadata must declare a tile_encoder")
        tile_metadata = encoder_registry.info(tile_encoder_name)
        if tile_metadata.get("level") != "tile":
            raise ValueError(
                f"Slide encoder dependency '{tile_encoder_name}' must resolve to a tile encoder"
            )

    rec_precision = model_metadata.get("precision", "fp16")
    if config.precision != rec_precision:
        warnings.append(
            f"Precision mismatch: config uses '{config.precision}', "
            f"model recommends '{rec_precision}'."
        )

    rec_spacing = model_metadata.get("recommended_spacing_um")
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

    rec_tile_size = model_metadata.get("recommended_tile_size_px")
    if preprocessing_config is not None and rec_tile_size is not None:
        if preprocessing_config.requested_tile_size_px != rec_tile_size:
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
            eff = tiling_result.effective_spacing_um
            if abs(eff - target_spacing) / target_spacing > 0.05:
                warnings.append(
                    f"Effective spacing ({eff:.4f} µm/px) differs from "
                    f"target ({target_spacing} µm/px) by more than 5%."
                )

        if rec_tile_size is not None and tiling_result.requested_tile_size_px != rec_tile_size:
            warnings.append(
                f"TilingResult uses tile size {tiling_result.requested_tile_size_px}px, "
                f"model recommends {rec_tile_size}px."
            )

    return warnings
