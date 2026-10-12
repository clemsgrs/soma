"""The ``composite`` encoder: several frozen encoders whose dense grids are concatenated.

``composite`` is the one encoder soma owns. Each member is extracted on its own, into
its own dense cache, exactly as that encoder is extracted in a single-encoder run, and
:class:`~soma.dense.composite.CompositeDenseFeatureStore` concatenates the grids at load
time (ADR 0011). A member names a slide2vec encoder and may override the run's slide2vec
``DenseOptions`` and ``ExecutionOptions`` fields for itself (a native sliding window, a
feature kind, a precision); everything else it shares with the run.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from slide2vec import DenseOptions, ExecutionOptions

__all__ = ["CompositeMember", "CompositeParams", "member_norm_for"]

#: The run's read geometry, shared by every member: one spacing and one supervision size.
_SHARED_DENSE_FIELDS = frozenset({"spacing_um", "target_size", "tolerance", "backend"})
_MEMBER_DENSE_FIELDS = frozenset(f.name for f in dataclasses.fields(DenseOptions)) - _SHARED_DENSE_FIELDS
_MEMBER_EXECUTION_FIELDS = frozenset({"precision", "batch_size"})
_MEMBER_NORMS = (None, "none", "l2", "layernorm")
_CONCAT_RESOLUTIONS = ("grid", "target")


@dataclass(frozen=True)
class CompositeMember:
    """One member of a composite: a slide2vec encoder plus its own overrides.

    ``params`` are the member encoder's own params (``output_variant``,
    ``allow_non_recommended_settings``). ``dense`` overrides fields of the run's slide2vec
    ``DenseOptions`` for this member, all but the shared read geometry (``spacing_um``,
    ``target_size``, ``tolerance``, ``backend``). ``execution`` overrides ``precision`` or
    ``batch_size`` of the run's ``ExecutionOptions``. ``member_norm`` is the per-member
    normalization applied at load time before the concat (ADR 0011); ``None`` resolves to
    ``none`` for a ``cls_attention`` member and ``l2`` otherwise.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    dense: Mapping[str, Any] = field(default_factory=dict)
    execution: Mapping[str, Any] = field(default_factory=dict)
    member_norm: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"a composite member needs an encoder name, got {self.name!r}")
        for section, allowed in (("dense", _MEMBER_DENSE_FIELDS), ("execution", _MEMBER_EXECUTION_FIELDS)):
            unknown = sorted(set(getattr(self, section)) - allowed)
            if unknown:
                raise ValueError(
                    f"composite member '{self.name}' cannot set {section} field(s) {unknown}; "
                    f"a member may set: {sorted(allowed)}"
                )
        if self.member_norm not in _MEMBER_NORMS:
            raise ValueError(
                f"composite member '{self.name}' member_norm must be 'none', 'l2', 'layernorm' "
                f"or None, got {self.member_norm!r}"
            )
        from soma.encoders import encoder_registry

        if self.name == "composite" or self.name not in encoder_registry:
            raise ValueError(
                f"Unknown encoder name '{self.name}' in a composite's members; a member is a "
                "slide2vec encoder."
            )
        encoder_registry.validate(self.name, self.params)

    def dense_options(self, run: DenseOptions) -> DenseOptions:
        """The run's ``DenseOptions`` with this member's overrides."""
        overrides = dict(self.dense)
        if "attention_blocks" in overrides:
            overrides["attention_blocks"] = tuple(int(b) for b in overrides["attention_blocks"])
        return dataclasses.replace(run, **overrides)

    def execution_options(self, run: ExecutionOptions) -> ExecutionOptions:
        """The run's ``ExecutionOptions`` with this member's overrides."""
        return dataclasses.replace(run, **dict(self.execution)) if self.execution else run


def member_norm_for(member: CompositeMember, dense: DenseOptions) -> str:
    """``member.member_norm``, or the default for the member's feature kind."""
    if member.member_norm is not None:
        return member.member_norm
    return "none" if dense.feature_kind == "cls_attention" else "l2"


@dataclass(frozen=True)
class CompositeParams:
    """The params of the ``composite`` encoder (ADR 0019).

    ``members`` lists the member encoders (mappings or :class:`CompositeMember`).
    ``concat_resolution`` is ``grid`` (resample each member's token grid to a common grid,
    for decoders) or ``target`` (upsample to the supervision size); ``concat_grid_size`` is
    the common grid in ``grid`` mode, the largest member grid when ``None``.
    """

    members: Sequence[Any] = ()
    concat_resolution: str = "grid"
    concat_grid_size: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("the composite encoder needs at least one member")
        members = tuple(
            member if isinstance(member, CompositeMember) else CompositeMember(**dict(member))
            for member in self.members
        )
        object.__setattr__(self, "members", members)
        if self.concat_resolution not in _CONCAT_RESOLUTIONS:
            raise ValueError(
                f"composite concat_resolution must be one of {list(_CONCAT_RESOLUTIONS)}, "
                f"got {self.concat_resolution!r}"
            )
        if self.concat_grid_size is not None:
            height, width = (int(v) for v in self.concat_grid_size)
            if height <= 0 or width <= 0:
                raise ValueError(f"composite concat_grid_size must be positive, got {self.concat_grid_size}")
            object.__setattr__(self, "concat_grid_size", (height, width))

    @property
    def label(self) -> str:
        """The members' names joined with ``+``: the composite's identity as one name."""
        return "+".join(member.name for member in self.members)
