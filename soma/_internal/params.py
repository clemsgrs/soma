"""Field checks the params dataclasses of soma's components share (ADR 0018).

A component declares its params at registration as a dataclass whose fields mirror its
user-set constructor keywords; the dataclass's ``__post_init__`` is where those values
are validated, so a config is checked before anything is built.

A field the run decides is declared with :func:`run_decided`: a config spelling the run
turns into constructor arguments (a head's class scheme, HIPT's region grid), or a
keyword left ``None`` for the run to derive (a decoder's upsampling depth, an
augmentation's fill value). A plan records it like any other field.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import Field, dataclass, field
from numbers import Integral, Real
from typing import Any

__all__ = [
    "NoParams",
    "check_choice",
    "check_non_negative",
    "check_positive",
    "check_probability",
    "is_run_decided",
    "run_decided",
]

_RUN_DECIDED = "run_decided"


def run_decided(default: Any = None) -> Any:
    """A params field the run turns into constructor arguments or derives when ``None``."""
    return field(default=default, metadata={_RUN_DECIDED: True})


def is_run_decided(params_field: Field) -> bool:
    """Whether a params dataclass field was declared with :func:`run_decided`."""
    return bool(params_field.metadata.get(_RUN_DECIDED))


@dataclass
class NoParams:
    """The params of a component with no user-set constructor keywords."""


def check_probability(name: str, value: Any, *, below_one: bool = False) -> None:
    """``value`` is a real number in ``[0, 1]`` (``[0, 1)`` when ``below_one``)."""
    in_range = (
        not isinstance(value, bool)
        and isinstance(value, Real)
        and 0.0 <= value
        and (value < 1.0 if below_one else value <= 1.0)
    )
    if not in_range:
        interval = "[0, 1)" if below_one else "[0, 1]"
        raise ValueError(f"{name} must be a probability in {interval}, got {value!r}")


def check_positive(name: str, value: Any, *, integer: bool = False, optional: bool = False) -> None:
    """``value`` is a number above zero (an integer when ``integer``; ``None`` when ``optional``)."""
    if value is None and optional:
        return
    kind = Integral if integer else Real
    if isinstance(value, bool) or not isinstance(value, kind) or value <= 0:
        noun = "a positive integer" if integer else "a positive number"
        raise ValueError(f"{name} must be {noun}, got {value!r}")


def check_non_negative(name: str, value: Any, *, integer: bool = False) -> None:
    """``value`` is a number at or above zero (an integer when ``integer``)."""
    kind = Integral if integer else Real
    if isinstance(value, bool) or not isinstance(value, kind) or value < 0:
        noun = "a non-negative integer" if integer else "a non-negative number"
        raise ValueError(f"{name} must be {noun}, got {value!r}")


def check_choice(name: str, value: Any, choices: Collection[Any]) -> None:
    """``value`` is one of ``choices``."""
    if value not in choices:
        raise ValueError(f"{name} must be one of {sorted(map(str, choices))}, got {value!r}")
