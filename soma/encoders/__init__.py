"""Encoder validation and discovery helpers for soma."""

from __future__ import annotations

from typing import Any

from soma.config import AggregatorConfig
from soma.encoders.validation import resolve_preprocessing_config, validate_encoder_config
from soma.registry import Registry
from slide2vec.encoders.registry import encoder_registry as _slide2vec_encoders
from slide2vec.encoders.registry import resolve_encoder_level


class EncoderRegistry(Registry):
    """soma's ``encoders`` domain: a read-only view of slide2vec's encoder registry.

    Encoders belong to slide2vec, so a new one is registered there (an encoder
    provider), never here; their constructor params are slide2vec's, so an entry's
    ``params`` is ``None``. Each entry carries slide2vec's metadata, ``level`` included.
    """

    def __init__(self) -> None:
        super().__init__("encoders")

    def register(self, name: str, cls: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"encoder '{name}' cannot be registered with soma: encoders are slide2vec's; "
            "register it with slide2vec as an encoder provider"
        )

    def get(self, name: str) -> type:
        return _slide2vec_encoders.require(name)

    def __contains__(self, name: str) -> bool:
        return name in _slide2vec_encoders

    def list(self) -> list[str]:
        return list(_slide2vec_encoders.names())

    def info(self, name: str) -> dict[str, Any]:
        metadata = _slide2vec_encoders.info(name)
        return self._describe(name, _slide2vec_encoders.require(name), None, metadata)

    def _lookup(self, name: str) -> tuple[Any, None]:
        return self.get(name), None


#: The ``encoders`` domain of :func:`soma.components`.
encoder_registry = EncoderRegistry()


def resolve_aggregator(
    encoder_name: str,
    tile_recipe: AggregatorConfig,
) -> AggregatorConfig | None:
    """Resolve a slide benchmark's aggregation mechanism from encoder metadata."""
    try:
        metadata = _slide2vec_encoders.info(encoder_name)
    except KeyError as exc:
        available = ", ".join(sorted(_slide2vec_encoders.names())) or "(none)"
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
    "EncoderRegistry",
    "encoder_registry",
    "resolve_aggregator",
    "resolve_preprocessing_config",
    "validate_encoder_config",
]
