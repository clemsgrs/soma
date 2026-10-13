"""``soma.components``: the one discovery function over every registry (ADR 0018).

Every swappable domain is a :class:`~soma.registry.Registry`. ``components()`` names the
domains; ``components(kind)`` lists a domain's entries with their params and defaults.
A dotted-path class is never registered, so it is never listed.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from typing import Any

from soma.registry import Registry

__all__ = ["components", "register", "registry"]

#: Each domain's registry, as ``(module that registers its entries, attribute)``.
_DOMAINS: dict[str, tuple[str, str]] = {
    "aggregators": ("soma.models.aggregators", "aggregator_registry"),
    "augmentations": ("soma.learn", "augment_registry"),
    "benchmarks": ("soma.benchmark", "benchmark_registry"),
    "callbacks": ("soma.learn", "callback_registry"),
    "decoders": ("soma.models.decoders", "decoder_registry"),
    "encoders": ("soma.models.encoders", "encoder_registry"),
    "heads": ("soma.models.heads", "task_registry"),
    "learners": ("soma.learn", "learner_registry"),
    "optimizers": ("soma.learn", "optimizer_registry"),
    "presets": ("soma.presets", "preset_registry"),
    "samplers": ("soma.learn", "sampler_registry"),
    "schedulers": ("soma.learn", "scheduler_registry"),
}


def registry(kind: str) -> Registry:
    """The registry of one component kind."""
    if kind not in _DOMAINS:
        raise ValueError(f"unknown component kind {kind!r}; the kinds are: {', '.join(sorted(_DOMAINS))}")
    module, attribute = _DOMAINS[kind]
    return getattr(importlib.import_module(module), attribute)


def components(kind: str | None = None) -> list[str] | list[dict[str, Any]]:
    """The component kinds, or one kind's entries.

    Without ``kind``: the domain names. With ``kind``: one dict per registered entry,
    sorted by name, holding its ``name``, its dotted ``class``, its ``params`` (each field
    mapped to its default; ``None`` for an entry registered without params) and its
    registration metadata.
    """
    if kind is None:
        return sorted(_DOMAINS)
    domain = registry(kind)
    return [domain.info(name) for name in sorted(domain.list())]


def register(
    kind: str,
    name: str,
    cls: Any,
    *,
    params: type | None = None,
    infer: Callable[[Any, Mapping[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Register ``cls`` under ``name`` in the ``kind`` registry, with its params dataclass
    and an optional ``infer(cohort, params) -> dict`` hook (see :meth:`Registry.register`)."""
    registry(kind).register(name, cls, params=params, infer=infer)
