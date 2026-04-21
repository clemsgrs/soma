"""Encoder validation and discovery helpers for soma."""

from soma.encoders.validation import resolve_preprocessing_config, validate_encoder_config
from slide2vec.encoders.registry import encoder_registry


def list_models(level: str | None = None) -> list[str]:
    """Return available encoder presets in a stable order."""
    if level is None:
        return sorted(encoder_registry.names())

    normalized_level = str(level).strip().lower()
    if normalized_level not in {"tile", "slide", "patient"}:
        raise ValueError("list_models(level=...) must be one of: tile, slide, patient")

    return sorted(
        name
        for name in encoder_registry.names()
        if encoder_registry.info(name)["level"] == normalized_level
    )

__all__ = [
    "list_models",
    "resolve_preprocessing_config",
    "validate_encoder_config",
]
