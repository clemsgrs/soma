"""The ``encoders`` registry and the helpers that resolve what an encoder decides."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from slide2vec import Model
from slide2vec.encoders.registry import encoder_registry as _slide2vec_encoders

from soma.models.encoders.composite import CompositeMember, CompositeParams
from soma.models.encoders.validation import resolve_encoder_precision, resolve_preprocessing_config
from soma.registry import Component, Registry, _Entry, signature_params

@dataclass(frozen=True)
class EncoderLeaf(Component):
    """The frozen encoder: a slide2vec encoder, or ``composite`` for several concatenated.

    ``params`` are slide2vec's ``Model.from_preset`` keywords (``output_variant``,
    ``allow_non_recommended_settings``), or the composite's members
    (:class:`soma.models.encoders.CompositeParams`); the ``encoders`` registry validates them.
    """

    @property
    def output_variant(self) -> str | None:
        return self.params.get("output_variant")

    @property
    def allow_non_recommended_settings(self) -> bool:
        return bool(self.params.get("allow_non_recommended_settings", False))


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
            f"encoder '{name}' cannot be registered with soma: encoders are slide2vec's. "
            "Decorate the class with slide2vec.register_encoder in your script, or ship it "
            "as a slide2vec plugin package for the CLI and multi-GPU runs"
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
        return self.get(name), _Entry(cls=None, params=params, infer=None, fit=None, metadata={})


#: The ``encoders`` domain of :func:`soma.components`.
encoder_registry = EncoderRegistry()


__all__ = [
    "COMPOSITE",
    "CompositeMember",
    "CompositeParams",
    "EncoderParams",
    "EncoderRegistry",
    "encoder_registry",
    "resolve_encoder_precision",
    "resolve_preprocessing_config",
]
