"""Sections a config owner edits by assigning fields, until a plan seals them."""

from __future__ import annotations

import dataclasses
from typing import Any

__all__ = ["SEALED", "Editable"]

#: The instance attribute a sealed section carries.
SEALED = "_sealed"


class Editable:
    """A section its owner changes by assigning fields, until a plan seals it."""

    def __setattr__(self, name: str, value: Any) -> None:
        if self.__dict__.get(SEALED):
            raise dataclasses.FrozenInstanceError(
                f"cannot assign to field {name!r}: a plan is frozen (edit the config and resolve it again)"
            )
        object.__setattr__(self, name, value)
