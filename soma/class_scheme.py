"""The class scheme of a dense task: which annotated raw values form which class.

``task.params.classes`` maps each class name to the raw annotated value(s) that form it —
several values merge into one class — and the class index is the declaration order. A
second list names the raw values to exclude: ``ignore`` for segmentation (mask pixels left
out of loss and metrics), ``drop`` for detection (points removed from the supervised set).
No name is reserved, and a raw value belongs to one class or to the excluded list, never
two.

The scheme is a training-layer setting: it never enters a tiling or feature cache key, so
regrouping classes reuses cached features.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["assign_raw_values", "resolve_classes"]


def _raw_values(owner: str, values: Any, max_value: int | None) -> list[int]:
    """Normalize one ``classes`` entry / the excluded list to validated raw integers."""
    if not isinstance(values, (list, tuple)):
        values = [values]
    allowed = "non-negative integers" if max_value is None else f"integers in [0, {max_value}]"
    for value in values:
        is_integer = isinstance(value, (int, np.integer)) and not isinstance(value, bool)
        if not is_integer or value < 0 or (max_value is not None and value > max_value):
            raise ValueError(f"'{owner}' raw values must be {allowed}, got {value!r}.")
    return [int(value) for value in values]


def assign_raw_values(
    classes: Mapping[str, Any],
    *,
    excluded: Sequence[int] | int,
    excluded_name: str,
    max_value: int | None = None,
) -> tuple[dict[int, int], list[int]]:
    """Return ``({raw value: class index}, [excluded raw values])`` for a class scheme.

    ``excluded_name`` is the config key the excluded list came from (``ignore`` / ``drop``),
    used in error messages. ``max_value`` caps the raw values (255 for single-byte masks).
    """
    if not classes:
        raise ValueError("classes must name at least one class.")
    class_of: dict[int, int] = {}
    owners: dict[int, str] = {}

    def claim(owner: str, values: Any) -> list[int]:
        claimed = _raw_values(owner, values, max_value)
        for value in claimed:
            if value in owners:
                raise ValueError(
                    f"'{owners[value]}' and '{owner}' both list raw value {value}; a raw "
                    f"value belongs to exactly one class or to {excluded_name}."
                )
            owners[value] = owner
        return claimed

    for class_index, (name, values) in enumerate(classes.items()):
        if isinstance(values, (list, tuple)) and not values:
            raise ValueError(f"class '{name}' lists no raw values.")
        for value in claim(str(name), values):
            class_of[value] = class_index
    return class_of, claim(excluded_name, excluded)


def resolve_classes(
    task_params: Mapping[str, Any],
    *,
    excluded_key: str,
    subject: str,
    max_value: int | None = None,
) -> tuple[int, tuple[str, ...], dict[int, int] | None, tuple[int, ...]]:
    """Resolve ``(num_classes, class names, {raw value: class index}, excluded values)``.

    ``num_classes`` is derived from ``task.params.classes``; giving both is an error unless
    they agree. Without ``classes`` the annotations must already hold contiguous class
    indices: ``num_classes`` is required, the mapping is ``None`` and nothing is excluded.
    ``subject`` is the ``dataset_type`` named in error messages.
    """
    classes = task_params.get("classes")
    excluded = task_params.get(excluded_key)
    num_classes = task_params.get("num_classes")
    if classes is None:
        if excluded is not None:
            raise ValueError(f"task.params.{excluded_key} needs task.params.classes.")
        if num_classes is None:
            raise ValueError(
                f"dataset_type='{subject}' requires task.params.classes (or "
                "task.params.num_classes when the annotations already hold class indices)."
            )
        num_classes = int(num_classes)
        return num_classes, tuple(f"class_{index}" for index in range(num_classes)), None, ()

    if not isinstance(classes, Mapping):
        raise ValueError("task.params.classes must map class name → raw annotated value(s).")
    class_of, excluded_values = assign_raw_values(
        classes,
        excluded=() if excluded is None else excluded,
        excluded_name=excluded_key,
        max_value=max_value,
    )
    if num_classes is not None and int(num_classes) != len(classes):
        raise ValueError(
            f"task.params.num_classes={num_classes} disagrees with the {len(classes)} "
            "classes in task.params.classes; drop num_classes (it is derived)."
        )
    return len(classes), tuple(str(name) for name in classes), class_of, tuple(excluded_values)
