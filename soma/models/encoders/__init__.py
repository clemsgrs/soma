"""The ``encoders`` registry and the helpers that resolve what an encoder decides."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from slide2vec import Model
from slide2vec.encoders.registry import encoder_registry as _slide2vec_encoders
from slide2vec.encoders.registry import resolve_encoder_level

from soma.config import Component
from soma.models.encoders.composite import CompositeMember, CompositeParams
from soma.models.encoders.validation import resolve_encoder_precision, resolve_preprocessing_config
from soma.registry import Registry, _Entry, signature_params

#: The params of every slide2vec encoder: ``Model.from_preset``'s keywords. The device is
#: the run's to choose, so it is not a param.
EncoderParams = signature_params(Model.from_preset, name="EncoderParams", skip=("name", "device"))

#: The encoder soma owns. Its class is resolved on first use (the store imports torch).
COMPOSITE = "composite"


def _composite_store_class() -> type:
    from soma.extract.dense.composite import CompositeDenseFeatureStore

    return CompositeDenseFeatureStore


class EncoderRegistry(Registry):
    """soma's ``encoders`` domain: slide2vec's encoders plus soma's ``composite``.

    Encoders belong to slide2vec, so a new one is registered there (an encoder
    provider), never here. Every slide2vec entry's params are ``Model.from_preset``'s
    keywords (:data:`EncoderParams`) and its metadata is slide2vec's, ``level``
    included. ``composite`` concatenates several of them (:class:`CompositeParams`).
    """

    def __init__(self) -> None:
        super().__init__("encoders")

    def register(self, name: str, cls: Any, **kwargs: Any) -> None:
        raise TypeError(
            f"encoder '{name}' cannot be registered with soma: encoders are slide2vec's; "
            "register it with slide2vec as an encoder provider"
        )

    def get(self, name: str) -> type:
        if name == COMPOSITE:
            return _composite_store_class()
        return _slide2vec_encoders.require(name)

    def __contains__(self, name: str) -> bool:
        return name == COMPOSITE or name in _slide2vec_encoders

    def list(self) -> list[str]:
        return [*_slide2vec_encoders.names(), COMPOSITE]

    def info(self, name: str) -> dict[str, Any]:
        if name == COMPOSITE:
            return self._describe(name, _composite_store_class(), CompositeParams, {"level": "tile"})
        metadata = _slide2vec_encoders.info(name)
        return self._describe(name, _slide2vec_encoders.require(name), EncoderParams, metadata)

    def canonical_params(self, name: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The base canonical params, with each composite member's own params canonical too."""
        canonical = super().canonical_params(name, params)
        if name == COMPOSITE:
            for member in canonical["members"]:
                member["params"] = self.canonical_params(member["name"], member["params"])
        return canonical

    def _lookup(self, name: str) -> tuple[Any, _Entry]:
        params = CompositeParams if name == COMPOSITE else EncoderParams
        return self.get(name), _Entry(cls=None, params=params, infer=None, metadata={})


#: The ``encoders`` domain of :func:`soma.components`.
encoder_registry = EncoderRegistry()


def resolve_aggregator(
    encoder_name: str,
    tile_recipe: Component,
) -> Component | None:
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
    "COMPOSITE",
    "CompositeMember",
    "CompositeParams",
    "EncoderParams",
    "EncoderRegistry",
    "encoder_registry",
    "resolve_aggregator",
    "resolve_encoder_precision",
    "resolve_preprocessing_config",
]
