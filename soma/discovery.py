"""``soma.components``: the one discovery function over every registry (ADR 0018).

Every swappable domain is a :class:`~soma.registry.Registry`, which announces its kind
when it is built. ``components()`` names the kinds whose module is imported (the core's
with ``import soma``, ``benchmarks`` once :mod:`soma.benchmark` is imported, ADR 0022);
``components(kind)`` lists a kind's entries with their params and defaults. A dotted-path
class is never registered, so it is never listed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from soma.registry import Registry, benchmark_import_hint, kinds

__all__ = ["components", "register", "registry"]


def registry(kind: str) -> Registry:
    """The registry of one component kind."""
    registries = kinds()
    if kind not in registries:
        raise ValueError(
            f"unknown component kind {kind!r}; the kinds are: {', '.join(sorted(registries))}."
            f"{benchmark_import_hint()}"
        )
    return registries[kind]


def components(kind: str | None = None) -> list[str] | list[dict[str, Any]]:
    """The component kinds, or one kind's entries.

    Without ``kind``: the names of the kinds whose registry is built. With ``kind``: one
    dict per registered entry, sorted by name (the presets keep the order they are
    registered in), holding its ``name``, its dotted ``class``, its ``params`` (each field
    mapped to its default; ``None`` for an entry registered without params) and its
    registration metadata.
    """
    if kind is None:
        return sorted(kinds())
    domain = registry(kind)
    names = domain.list()
    return [domain.info(name) for name in (sorted(names) if domain.listed_by_name else names)]


def register(
    kind: str,
    name: str,
    cls: Any,
    *,
    params: type | None = None,
    infer: Callable[[Any, Mapping[str, Any]], dict[str, Any]] | None = None,
    fit: Callable[[Sequence[Any], Mapping[str, Any]], dict[str, Any]] | None = None,
) -> None:
    """Register ``cls`` under ``name`` in the ``kind`` registry, with its params dataclass,
    an optional ``infer(cohort, params) -> dict`` hook and an optional
    ``fit(records, params) -> dict`` hook (see :meth:`Registry.register`)."""
    registry(kind).register(name, cls, params=params, infer=infer, fit=fit)
