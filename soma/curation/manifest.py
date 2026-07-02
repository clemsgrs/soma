"""Neutral home for the unified Manifest schema, its writer, and the Curator type.

Every soma curator — regardless of dataset or task — emits the **same** Manifest: a
``dataset.csv`` + ``splits.csv`` pair plus a ``summary.json``. This module owns that
contract so no single curator does (ADR 0004): curators are dataset-specific adapters
that are *never* interchangeable, so there is **no base class**. What they share is their
*output* (the :class:`CuratedManifest`) and their *machinery* (:func:`write_manifest`) —
shared data + functions, not a shared inheritance tree. The structural :class:`Curator`
Protocol names that shared shape without ever being subclassed.

Unified schema:

* ``dataset.csv`` — ``sample_id``, ``image_path``, then **exactly one** supervision
  column selected by ``dataset_type`` (:data:`SUPERVISION_COLUMN`): ``label`` for
  classification, ``mask_path`` for segmentation, ``points_path`` for detection. Optional
  recognized columns ``patient_id`` and ``level0_spacing`` follow; any further columns are
  preserved verbatim as per-sample metadata.
* ``splits.csv`` — ``sample_id``, ``split``, ``fold`` (single-fold curators emit
  ``fold=0`` for every row).
* ``summary.json`` — a free-form, curator-authored summary; always written.

Re-curating the same raw data yields **byte-identical** files: rows are written in the
order the curator supplies them, columns follow a fixed canonical order, and the summary
is serialized with sorted keys.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import pandas as pd

# dataset_type -> the single supervision column its Manifest carries. The three task
# families are mutually exclusive: classification is a scalar ``label``, segmentation a
# per-pixel ``mask_path`` raster, detection a per-sample ``points_path`` file.
SUPERVISION_COLUMN: dict[str, str] = {
    "tile": "label",
    "slide": "label",
    "patient": "label",
    "segmentation": "mask_path",
    "detection": "points_path",
}

# Every supervision column, used to enforce "exactly one" per Manifest.
_ALL_SUPERVISION_COLUMNS = frozenset(SUPERVISION_COLUMN.values())

# Fixed leading columns and the recognized optional columns, in canonical order.
_LEADING_COLUMNS = ("sample_id", "image_path")
_OPTIONAL_COLUMNS = ("patient_id", "level0_spacing")
_SPLITS_COLUMNS = ("sample_id", "split", "fold")


@dataclass(frozen=True)
class CuratedManifest:
    """Paths to a generated Soma Manifest (``dataset.csv`` + ``splits.csv`` + summary)."""

    dataset_csv: Path
    splits_csv: Path
    summary_json: Path | None = None


@runtime_checkable
class Curator(Protocol):
    """Structural type for a curator: a deterministic ``raw data -> CuratedManifest`` fn.

    There is **no base class** (ADR 0004). Curators are dataset-specific adapters with
    heterogeneous signatures (EVA takes a dataset ``name``, OCELOT a ``render_spacing_um``,
    …); the only thing they structurally share is being callables that return a
    :class:`CuratedManifest`. This Protocol documents that shared shape and lets a test
    assert conformance via ``isinstance`` without ever coupling curators by inheritance.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> CuratedManifest: ...


def supervision_column_for(dataset_type: str) -> str:
    """Return the supervision column a ``dataset_type`` Manifest must carry (fail-fast)."""
    try:
        return SUPERVISION_COLUMN[dataset_type]
    except KeyError:
        raise ValueError(
            f"Unknown dataset_type {dataset_type!r} for a Manifest; expected one of "
            f"{sorted(SUPERVISION_COLUMN)}."
        ) from None


def _ordered_dataset_columns(columns: Sequence[str], supervision: str) -> list[str]:
    leading = [*_LEADING_COLUMNS, supervision]
    optional = [c for c in _OPTIONAL_COLUMNS if c in columns]
    seen = set(leading) | set(optional)
    extras = [c for c in columns if c not in seen]
    return leading + optional + extras


def write_manifest(
    output_dir: str | Path,
    *,
    dataset_type: str,
    dataset_rows: Sequence[Mapping[str, Any]],
    split_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> CuratedManifest:
    """Write the unified Manifest (``dataset.csv`` + ``splits.csv`` + ``summary.json``).

    This is the single writer shared by every curator; it replaces the per-curator
    hand-rolled CSV writers so the emitted schema is identical everywhere.

    Args:
        output_dir: Directory to write the three files into (created if absent).
        dataset_type: Selects the required supervision column via
            :data:`SUPERVISION_COLUMN`.
        dataset_rows: One mapping per sample. Each must contain ``sample_id``,
            ``image_path`` and the supervision column for ``dataset_type``; may also
            carry ``patient_id`` / ``level0_spacing`` and arbitrary metadata columns. Rows
            are written in the given order; columns are reordered to the canonical layout.
        split_rows: One mapping per (sample, fold) with ``sample_id`` + ``split`` and an
            optional ``fold`` (defaults to ``0`` when omitted, per the single-fold rule).
        summary: A JSON-serializable summary, written verbatim (with sorted keys).

    Returns:
        A :class:`CuratedManifest` pointing at the three written files.
    """
    if not dataset_rows:
        raise ValueError("write_manifest requires at least one dataset row.")

    supervision = supervision_column_for(dataset_type)
    forbidden = _ALL_SUPERVISION_COLUMNS - {supervision}

    dataset_df = pd.DataFrame(list(dataset_rows))
    required = {*_LEADING_COLUMNS, supervision}
    missing = sorted(required - set(dataset_df.columns))
    if missing:
        raise ValueError(
            f"dataset rows for dataset_type={dataset_type!r} are missing required "
            f"column(s) {missing}; got columns {list(dataset_df.columns)}."
        )
    present_forbidden = sorted(forbidden & set(dataset_df.columns))
    if present_forbidden:
        raise ValueError(
            f"dataset rows for dataset_type={dataset_type!r} must carry exactly one "
            f"supervision column ({supervision!r}), but also contain {present_forbidden}."
        )
    dataset_df = dataset_df[_ordered_dataset_columns(list(dataset_df.columns), supervision)]

    split_df = pd.DataFrame(list(split_rows))
    if split_df.empty:
        raise ValueError("write_manifest requires at least one split row.")
    if "fold" not in split_df.columns:
        split_df = split_df.copy()
        split_df["fold"] = 0
    missing_split = sorted({"sample_id", "split"} - set(split_df.columns))
    if missing_split:
        raise ValueError(
            f"split rows are missing required column(s) {missing_split}; "
            f"got columns {list(split_df.columns)}."
        )
    extra_split = [c for c in split_df.columns if c not in _SPLITS_COLUMNS]
    split_df = split_df[[*_SPLITS_COLUMNS, *extra_split]]

    unknown = set(split_df["sample_id"].astype(str)) - set(dataset_df["sample_id"].astype(str))
    if unknown:
        raise ValueError(
            f"split rows reference sample_id(s) absent from dataset.csv: {sorted(unknown)}."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = output_dir / "dataset.csv"
    splits_csv = output_dir / "splits.csv"
    summary_json = output_dir / "summary.json"

    dataset_df.to_csv(dataset_csv, index=False)
    split_df.to_csv(splits_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return CuratedManifest(
        dataset_csv=dataset_csv, splits_csv=splits_csv, summary_json=summary_json
    )
