"""Encoder validation and discovery helpers for soma."""

from soma.config import AggregatorConfig
from soma.encoders.validation import resolve_preprocessing_config, validate_encoder_config
from slide2vec.encoders.registry import encoder_registry, resolve_encoder_level


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


def resolve_aggregator(
    encoder_name: str,
    tile_recipe: AggregatorConfig,
) -> AggregatorConfig | None:
    """Resolve a slide benchmark's aggregation mechanism from encoder metadata."""
    try:
        metadata = encoder_registry.info(encoder_name)
    except KeyError as exc:
        available = ", ".join(sorted(encoder_registry.names())) or "(none)"
        raise ValueError(
            f"Unknown encoder name '{encoder_name}'. Available encoders: {available}"
        ) from exc
    level = resolve_encoder_level(encoder_name, metadata)
    if level == "tile":
        return tile_recipe
    if level == "slide":
        return None
    if level == "patient":
        raise ValueError(
            "Slide-level benchmarks cannot consume patient-level representations "
            f"from encoder '{encoder_name}'; choose a tile- or slide-level encoder."
        )
    raise ValueError(f"Cannot resolve an aggregator for encoder level '{level}'")


__all__ = [
    "list_models",
    "resolve_aggregator",
    "resolve_preprocessing_config",
    "validate_encoder_config",
]
