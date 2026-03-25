"""Validate encoder config against model recommended settings."""

from __future__ import annotations

from typing import Any

from soma.config import EncoderConfig
from soma.preprocessing.tiling import TilingResult


def validate_encoder_config(
    config: EncoderConfig,
    model_metadata: dict[str, Any],
    tiling_result: TilingResult | None = None,
) -> list[str]:
    """Check config against recommended model settings.

    Returns a list of warning strings. Raises ValueError if the config
    is ambiguous (e.g. no spacing specified when multiple are available).
    """
    warnings: list[str] = []

    # Precision check
    rec_precision = model_metadata.get("precision", "fp16")
    if config.precision != rec_precision:
        warnings.append(
            f"Precision mismatch: config uses '{config.precision}', "
            f"model recommends '{rec_precision}'."
        )

    # Spacing check
    rec_spacing = model_metadata.get("recommended_spacing_um")
    if config.spacing_um is None:
        if isinstance(rec_spacing, list) and len(rec_spacing) > 1:
            msg = (
                f"Model supports multiple spacings {rec_spacing} but "
                f"config.spacing_um is None. Please specify one."
            )
            raise ValueError(msg)
    else:
        valid_spacings = rec_spacing if isinstance(rec_spacing, list) else [rec_spacing]
        if config.spacing_um not in valid_spacings:
            warnings.append(
                f"Spacing mismatch: config uses {config.spacing_um} µm/px, "
                f"model recommends {rec_spacing}."
            )

    # Input size check
    rec_input_size = model_metadata.get("input_size")
    if config.input_size is not None and rec_input_size is not None:
        if config.input_size != rec_input_size:
            warnings.append(
                f"input_size mismatch: config uses {config.input_size}px, "
                f"model recommends {rec_input_size}px. The model's transform "
                f"will resize/crop to {rec_input_size}px unless overridden."
            )

    # Effective spacing from tiling result
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

    return warnings
