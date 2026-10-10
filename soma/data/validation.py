"""Shared manifest/cohort validation: safe ids, uniqueness, spacing, split names.

Every reader in :mod:`soma.data` validates only its own columns, but the rules they
share live here once (design §4.3).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Hashable, Iterator

import numpy as np
import pandas as pd

__all__ = [
    "SPLIT_TRAIN",
    "SPLIT_TUNE",
    "SPLIT_TEST_PREFIX",
    "is_filename_safe_id",
    "ensure_filename_safe_id",
    "is_valid_split_name",
    "iter_rows",
    "optional_text",
    "optional_path",
    "parse_spacing_at_level_0",
    "require_columns",
    "validate_sample_ids",
    "validate_patient_ids",
    "validate_spacing_declaration_columns",
]

SPLIT_TRAIN = "train"
SPLIT_TUNE = "tune"
SPLIT_TEST_PREFIX = "test"


def is_filename_safe_id(value: object) -> bool:
    """True if ``value`` is safe to use as a bare cache filename.

    ``sample_id`` and ``patient_id`` are written directly as ``<id>.pt`` (and sidecars)
    across every cache kind, so an id containing a path separator, ``..``, or an absolute
    path would write *outside* the intended directory (path traversal).
    """
    text = str(value)
    if not text or text in {".", ".."} or os.path.isabs(text):
        return False
    if "/" in text or "\\" in text or os.sep in text:
        return False
    if os.altsep is not None and os.altsep in text:
        return False
    return Path(text).name == text


def ensure_filename_safe_id(value: object, *, field: str = "sample_id") -> str:
    """Return ``str(value)`` if it is a safe cache filename, else raise ValueError."""
    if not is_filename_safe_id(value):
        raise ValueError(
            f"Unsafe {field} {value!r}: it is used as a cache filename, so it must be a "
            "bare name with no path separators, '..', or absolute path."
        )
    return str(value)


def is_valid_split_name(name: object) -> bool:
    """A split name is ``train``, ``tune``, or anything starting with ``test``."""
    return isinstance(name, str) and (
        name in (SPLIT_TRAIN, SPLIT_TUNE) or name.startswith(SPLIT_TEST_PREFIX)
    )


def iter_rows(df: pd.DataFrame) -> Iterator[tuple[Hashable, pd.Series]]:
    """``df.iterrows()`` with every value boxed to its column's Python type.

    A plain ``iterrows`` builds one Series per row in the common dtype of the frame, so a
    table with any float column reports integer ids and labels as floats (``1`` becomes
    ``"1.0"`` once stringified). Boxing to object first keeps ints as ints.
    """
    return df.astype(object).iterrows()


def require_columns(df: pd.DataFrame, columns: set[str] | frozenset[str], *, what: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(
            f"{what} is missing required column(s) {missing}. Available: {list(df.columns)}"
        )


def validate_sample_ids(df: pd.DataFrame, *, what: str = "manifest") -> None:
    """``sample_id`` present, non-blank, unique and filename-safe."""
    require_columns(df, {"sample_id"}, what=what)
    ids = df["sample_id"]
    if ids.isna().any():
        raise ValueError(f"{what} has blank sample_id value(s) at row(s) {ids[ids.isna()].index.tolist()}")
    if ids.duplicated().any():
        dupes = sorted({str(value) for value in ids[ids.duplicated()]})
        raise ValueError(f"Duplicate sample_id values in {what}: {dupes}")
    unsafe = sorted({str(value) for value in ids if not is_filename_safe_id(value)})
    if unsafe:
        raise ValueError(
            "Unsafe sample_id value(s) (used as cache filenames; no path separators, "
            f"'..', or absolute paths allowed): {unsafe}"
        )


def validate_patient_ids(df: pd.DataFrame) -> None:
    """``patient_id`` values, when the column exists, are filename-safe."""
    if "patient_id" not in df.columns:
        return
    unsafe = sorted(
        {str(value) for value in df["patient_id"].dropna() if not is_filename_safe_id(value)}
    )
    if unsafe:
        raise ValueError(
            "Unsafe patient_id value(s) (used as cache filenames; no path separators, "
            f"'..', or absolute paths allowed): {unsafe}"
        )


def optional_text(row: pd.Series, column: str) -> str | None:
    value = row.get(column)
    return str(value) if column in row.index and pd.notna(value) else None


def optional_path(row: pd.Series, column: str) -> Path | None:
    text = optional_text(row, column)
    return Path(text) if text is not None else None


def parse_spacing_at_level_0(value: object) -> float | None:
    """Parse a level-0 spacing declaration: positive finite float, or ``None`` when blank."""
    if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"invalid spacing_at_level_0 {value!r}")
    try:
        spacing = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid spacing_at_level_0 {value!r}") from None
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"invalid spacing_at_level_0 {value!r}")
    return spacing


def validate_spacing_declaration_columns(df: pd.DataFrame) -> None:
    """Validate the sole optional source-spacing declaration column."""
    if "level0_spacing" in df.columns:
        raise ValueError(
            "Manifest column 'level0_spacing' is retired; use 'spacing_at_level_0' instead."
        )
    if "spacing_at_level_0" not in df.columns:
        return
    invalid: list[str] = []
    for index, row in iter_rows(df):
        try:
            parse_spacing_at_level_0(row["spacing_at_level_0"])
        except ValueError:
            sample = row.get("sample_id", index)
            invalid.append(f"{sample}={row['spacing_at_level_0']!r}")
    if invalid:
        raise ValueError(
            "spacing_at_level_0 must be a positive, finite number or blank; "
            f"invalid sample(s): {invalid}."
        )
