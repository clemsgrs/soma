"""Generic component registry for soma: one per swappable component kind (ADR 0018)."""

from __future__ import annotations

import copy
import dataclasses
import functools
import importlib
import inspect
import sys
import warnings
import weakref
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# The params helpers a component author declares and checks its params with (ADR 0018):
# public here, for components outside core such as the benchmarks'.
from soma._internal.params import (  # noqa: F401
    NoParams,
    check_choice,
    check_non_negative,
    check_positive,
    check_probability,
)

#: The package that registers soma's benchmark components when it is imported. Core
#: never imports it (ADR 0022); it only names it in the hint of a failed lookup.
BENCHMARK_PACKAGE = "soma.benchmark"

#: The registry of each component kind, by kind: a :class:`Registry` announces itself
#: here when it is built, so :func:`soma.components` lists exactly the kinds whose module
#: is imported. Held weakly, so a throwaway registry leaves the listing with its last
#: reference.
_KINDS: weakref.WeakValueDictionary[str, Registry] = weakref.WeakValueDictionary()


def kinds() -> dict[str, Registry]:
    """Each component kind whose registry is built (its module imported), by kind."""
    return dict(_KINDS)


def benchmark_import_hint() -> str:
    """How to reach a benchmark's components while :data:`BENCHMARK_PACKAGE` is not imported."""
    if BENCHMARK_PACKAGE in sys.modules:
        return ""
    return (
        f" The benchmarks and their components register when {BENCHMARK_PACKAGE} is imported: "
        f"`import {BENCHMARK_PACKAGE}` first (the soma CLI does), or name a component by its dotted path."
    )


@dataclass(frozen=True)
class Component:
    """A ``{name, params}`` leaf: a registry key or a dotted import path, and its params."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a component leaf needs a non-empty name, got {self.name!r}")
        params = {} if self.params is None else self.params
        if not isinstance(params, Mapping):
            raise TypeError(f"the params of '{self.name}' must be a mapping, got {type(params).__name__}")
        object.__setattr__(self, "params", dict(params))

    @classmethod
    def of(cls, value: Any) -> Any:
        """``value`` as a leaf of this class: a leaf, a name or a ``{name, params}`` mapping."""
        if value is None or type(value) is cls:
            return value
        if isinstance(value, Component):
            return cls(value.name, value.params)
        if isinstance(value, str):
            return cls(value)
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - {"name", "params"})
            if unknown or "name" not in value:
                raise ValueError(f"a component leaf is a name or a {{name, params}} mapping, got {dict(value)!r}")
            return cls(value["name"], value.get("params") or {})
        raise TypeError(f"a component leaf is a name or a {{name, params}} mapping, got {value!r}")


class Registry:
    """Name → component, with the dataclass of the component's params.

    Each entry declares its params at registration: a dataclass whose fields mirror the
    constructor keywords a user sets, with their defaults, validated in its
    ``__post_init__`` (:meth:`validate`). A head may also declare
    ``infer(cohort, params) -> dict`` for the fields the cohort decides (:meth:`infer`),
    and ``fit(records, params) -> dict`` for the fields one fold's training records
    decide (:meth:`fit`).
    A dotted import path that names no registered class resolves too, unvalidated and
    unlisted, with a one-line warning (:meth:`resolve`). :func:`soma.components` lists
    the entries of every registry.

    A registry announces its ``domain`` (its component kind) when it is built: one live
    registry per kind. Entries register when the module defining them is imported, so a
    benchmark's private components are known once :mod:`soma.benchmark` is imported
    (ADR 0022); a dotted path imports its module itself.
    """

    def __init__(self, domain: str, *, listed_by_name: bool = True) -> None:
        if _KINDS.get(domain) is not None:
            raise ValueError(f"the {domain!r} kind already has a registry")
        self._domain = domain
        #: Whether :func:`soma.components` lists the entries by name, or else in the order
        #: they were registered.
        self.listed_by_name = listed_by_name
        self._entries: dict[str, _Entry] = {}
        self._warned: set[str] = set()
        _KINDS[domain] = self

    def register(
        self,
        name: str,
        cls: Any,
        *,
        params: type | None = None,
        infer: Callable[[Any, Mapping[str, Any]], dict[str, Any]] | None = None,
        fit: Callable[[Sequence[Any], Mapping[str, Any]], dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register ``cls`` under ``name`` with the dataclass of its params (ADR 0018)."""
        if name in self._entries:
            msg = f"'{name}' is already registered in the {self._domain} registry"
            raise ValueError(msg)
        if params is not None and not (isinstance(params, type) and dataclasses.is_dataclass(params)):
            raise TypeError(f"params of {self._domain} '{name}' must be a dataclass, got {params!r}")
        self._entries[name] = _Entry(cls=cls, params=params, infer=infer, fit=fit, metadata=metadata or {})

    def validate(self, name: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Check ``params`` against the entry's params dataclass and return them.

        Building the dataclass is the validation: an unknown field raises, and so does
        whatever its ``__post_init__`` refuses.
        """
        given = dict(params or {})
        _, entry = self._lookup(name)
        if entry is None or entry.params is None:
            return given
        fields = sorted(field.name for field in dataclasses.fields(entry.params))
        unknown = sorted(set(given) - set(fields))
        if unknown:
            raise ValueError(
                f"{self._domain} '{name}' has no param(s) {', '.join(map(repr, unknown))}; "
                f"its params are: {', '.join(fields) or '(none)'}"
            )
        entry.params(**given)
        return given

    def canonical_params(self, name: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """``params`` as the entry's params dataclass holds them: every field, defaults filled in.

        Spellings of the same params (a default left out or stated, a nested dataclass or
        its mapping) give one dict, so it is what an identity hashes. An entry without a
        params dataclass gives ``params`` as they are.
        """
        given = dict(params or {})
        _, entry = self._lookup(name)
        if entry is None or entry.params is None:
            return given
        return dataclasses.asdict(entry.params(**given))

    def infer(self, name: str, cohort: Any, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The fields the entry's ``infer`` hook decides from ``cohort`` (``{}`` without one).

        ``params`` are the explicitly set values; they win over the inferred ones when the
        caller merges the two.
        """
        _, entry = self._lookup(name)
        if entry is None or entry.infer is None:
            return {}
        return dict(entry.infer(cohort, dict(params or {})))

    def fit(self, name: str, records: Sequence[Any], params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The fields the entry's ``fit`` hook fits on one fold's training ``records``
        (``{}`` without one).

        ``infer`` runs once on the whole cohort; ``fit`` runs once per fold on the samples
        that fold trains on, so what it returns may differ between folds and is merged into
        that fold's params (the survival bin edges). A hook that only checks the fold
        returns ``{}``. ``params`` are the resolved params.
        """
        _, entry = self._lookup(name)
        if entry is None or entry.fit is None:
            return {}
        return dict(entry.fit(list(records), dict(params or {})))

    def get(self, name: str) -> Any:
        """Retrieve a registered class by name."""
        if name not in self._entries:
            available = ", ".join(sorted(self._entries)) or "(none)"
            msg = f"'{name}' not found in {self._domain} registry. Available: {available}.{benchmark_import_hint()}"
            raise KeyError(msg)
        return self._entries[name].cls

    def resolve(self, name: str) -> Any:
        """Resolve a ``{name, params}`` leaf's ``name``: a registry key or a dotted import path.

        A name containing a dot that is not registered is imported (``"pkg.module.Class"``),
        so users reference their own classes from a config without registering them. Such
        a class is never listed and its params pass through unvalidated, which a one-line
        warning says once per name. A dotted path naming a registered class is that entry.
        """
        return self._lookup(name)[0]

    def key_of(self, name: str) -> str:
        """The registry key ``name`` stands for.

        A dotted path naming a registered class gives that entry's key. Any other name
        (a key, an unregistered dotted path, an unknown name) comes back unchanged.
        """
        try:
            _, entry = self._lookup(name)
        except KeyError:
            return name
        if entry is None:
            return name
        return next((key for key, candidate in self._entries.items() if candidate is entry), name)

    def _lookup(self, name: str) -> tuple[Any, _Entry | None]:
        """``(object, entry)`` for a key or a dotted path; ``entry`` is ``None`` when unregistered."""
        if name in self:
            entry = self._entries[name]
            return entry.cls, entry
        if "." not in name:
            return self.get(name), None  # raises the not-found error
        module_name, _, attr = name.rpartition(".")
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise KeyError(
                f"'{name}' is neither a {self._domain} registry key nor an importable "
                f"dotted path ({error})"
            ) from error
        try:
            obj = getattr(module, attr)
        except AttributeError as error:
            raise KeyError(f"module '{module_name}' has no attribute '{attr}'") from error
        for entry in self._entries.values():
            if entry.cls is obj:
                return obj, entry
        if name not in self._warned:
            self._warned.add(name)
            warnings.warn(
                f"{self._domain} '{name}' is not registered: its params pass through "
                "unvalidated (register it to validate them and list it).",
                UserWarning,
                stacklevel=3,
            )
        return obj, None

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def list(self) -> list[str]:
        """Every registered name."""
        return list(self._entries.keys())

    def info(self, name: str) -> dict[str, Any]:
        """``name``, its dotted ``class``, its ``params`` defaults and its metadata.

        ``params`` maps each field of the params dataclass to its default (``None`` for a
        field without one); it is ``None`` for an entry registered without params.
        """
        if name not in self:
            msg = f"'{name}' not found in {self._domain} registry"
            raise KeyError(msg)
        entry = self._entries[name]
        return self._describe(name, entry.cls, entry.params, entry.metadata)

    @staticmethod
    def _describe(name: str, obj: Any, params: type | None, metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **metadata,
            "name": name,
            "class": _dotted_path(obj),
            "params": None if params is None else param_defaults(params),
        }


def signature_params(target: Callable[..., Any], *, name: str, skip: Collection[str] = ()) -> type:
    """A params dataclass mirroring ``target``'s keyword parameters.

    For callables soma registers but does not own (torch optimizers and schedulers, a
    benchmark's ``build_config``): deriving the fields keeps them in step with the
    upstream signature. ``skip`` names the parameters the caller supplies (the
    optimizer's ``params``, the scheduler's ``optimizer``); a parameter without a
    default defaults to ``None``, meaning the caller or the user must set it.
    """
    fields = []
    for parameter in inspect.signature(target).parameters.values():
        if parameter.name in skip or parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        default = None if parameter.default is parameter.empty else parameter.default
        if type(default).__hash__ is None:  # a mutable default needs a factory
            spec = dataclasses.field(default_factory=functools.partial(copy.deepcopy, default))
        else:
            spec = dataclasses.field(default=default)
        fields.append((parameter.name, Any, spec))
    return dataclasses.make_dataclass(name, fields)


def param_defaults(params: type) -> dict[str, Any]:
    """Each field of a params dataclass mapped to its default (``None`` when it has none)."""
    defaults: dict[str, Any] = {}
    for field in dataclasses.fields(params):
        if field.default is not dataclasses.MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not dataclasses.MISSING:
            defaults[field.name] = field.default_factory()
        else:
            defaults[field.name] = None
    return defaults


def _dotted_path(obj: Any) -> str:
    """The import path of a class or function, or of an instance's class."""
    target = obj if hasattr(obj, "__qualname__") else type(obj)
    return f"{target.__module__}.{target.__qualname__}"


class _Entry:
    __slots__ = ("cls", "params", "infer", "fit", "metadata")

    def __init__(
        self,
        cls: Any,
        params: type | None,
        infer: Callable[[Any, Mapping[str, Any]], dict[str, Any]] | None,
        fit: Callable[[Sequence[Any], Mapping[str, Any]], dict[str, Any]] | None,
        metadata: dict[str, Any],
    ) -> None:
        self.cls = cls
        self.params = params
        self.infer = infer
        self.fit = fit
        self.metadata = metadata
