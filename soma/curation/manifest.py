"""Neutral home for the unified Manifest schema, its writer, and the Curator type.

Every soma curator — regardless of dataset or task — emits the **same** Manifest: a
``dataset.csv`` + ``splits.csv`` pair plus a ``summary.json``. This module owns that
contract so no single curator does (ADR 0004): curators are dataset-specific adapters
that are *never* interchangeable, so there is **no base class**. What they share is their
*output* (the :class:`CuratedManifest`) and their *machinery* (:func:`write_manifest`) —
shared data + functions, not a shared inheritance tree. The structural :class:`Curator`
Protocol names that shared shape without ever being subclassed.

Unified schema:

* ``dataset.csv`` — ``sample_id``, ``image_path``, an optional ``mask_path`` (precomputed
  tissue mask, valid for every dataset_type), then **exactly one** supervision column
  selected by ``dataset_type`` (:data:`SUPERVISION_COLUMN`): ``label`` for
  classification, ``label_mask_path`` for segmentation, ``points_path`` for detection,
  ``target_index`` for spatial_expression. Optional recognized columns ``patient_id`` and
  ``spacing_at_level_0`` follow; any further columns are preserved verbatim as per-sample
  metadata.
* ``splits.csv`` — ``sample_id``, ``split``, ``fold`` (single-fold curators emit
  ``fold=0`` for every row).
* ``summary.json`` — a free-form, curator-authored summary; always written.
* ``targets.npy`` + ``genes.json`` — spatial_expression **only**: the multi-target
  regression sidecars. ``target_index`` is an integer row-key into ``targets.npy`` (shape
  ``[n_rows, n_genes]``); ``genes.json`` is the ordered gene list. The loaded sample
  record carries its resolved vector (see :class:`soma.dataset.SpatialExpressionManifest`).

Re-curating the same raw data yields **byte-identical** files: rows are written in the
order the curator supplies them, columns follow a fixed canonical order, the summary is
serialized with sorted keys, and the sidecars use uncompressed ``.npy`` / plain ``.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd

from soma.dataset import (
    GENES_FILENAME,
    TARGET_MATRIX_FILENAME,
    validate_spacing_declaration_columns,
)

# dataset_type -> the single supervision column its Manifest carries. The task families
# are mutually exclusive: classification is a scalar ``label``, segmentation a per-pixel
# ``label_mask_path`` raster, detection a per-sample ``points_path`` file, and
# spatial_expression a ``target_index`` row-key into a sidecar target matrix (a vector).
# ``mask_path`` is NOT supervision: it is the optional precomputed tissue mask.
SUPERVISION_COLUMN: dict[str, str] = {
    "tile": "label",
    "slide": "label",
    "patient": "label",
    "segmentation": "label_mask_path",
    "detection": "points_path",
    "spatial_expression": "target_index",
}

# Every supervision column, used to enforce "exactly one" per Manifest.
_ALL_SUPERVISION_COLUMNS = frozenset(SUPERVISION_COLUMN.values())

# Fixed leading columns and the recognized optional columns, in canonical order.
# ``mask_path`` (tissue mask) sits right after ``image_path``, before the supervision column.
_LEADING_COLUMNS = ("sample_id", "image_path")
_OPTIONAL_LEADING_COLUMNS = ("mask_path",)
_OPTIONAL_COLUMNS = ("patient_id", "spacing_at_level_0")
_SPLITS_COLUMNS = ("sample_id", "split", "fold")


@dataclass(frozen=True)
class CuratedManifest:
    """Paths to a generated Soma Manifest (``dataset.csv`` + ``splits.csv`` + summary).

    ``target_matrix_path`` / ``genes_path`` are set only for a ``spatial_expression``
    Manifest, pointing at the two sidecars written beside ``dataset.csv``; they are
    ``None`` for every other dataset_type.
    """

    dataset_csv: Path
    splits_csv: Path
    summary_json: Path | None = None
    target_matrix_path: Path | None = None
    genes_path: Path | None = None


@runtime_checkable
class Curator(Protocol):
    """Structural type for a curator: a deterministic ``raw data -> CuratedManifest`` fn.

    There is **no base class** (ADR 0004). Curators are dataset-specific adapters with
    heterogeneous signatures (EVA takes a dataset ``name``, MONKEY a source-spacing
    declaration, …); the only thing they structurally share is being callables that return a
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
    leading = [*_LEADING_COLUMNS, *(c for c in _OPTIONAL_LEADING_COLUMNS if c in columns), supervision]
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
    target_matrix: "np.ndarray | Sequence[Sequence[float]] | None" = None,
    genes: Sequence[str] | None = None,
) -> CuratedManifest:
    """Write the unified Manifest (``dataset.csv`` + ``splits.csv`` + ``summary.json``).

    This is the single writer shared by every curator; it replaces the per-curator
    hand-rolled CSV writers so the emitted schema is identical everywhere.

    Args:
        output_dir: Directory to write the files into (created if absent).
        dataset_type: Selects the required supervision column via
            :data:`SUPERVISION_COLUMN`.
        dataset_rows: One mapping per sample. Each must contain ``sample_id``,
            ``image_path`` and the supervision column for ``dataset_type``; may also
            carry ``mask_path`` (precomputed tissue mask), ``patient_id`` /
            ``spacing_at_level_0`` and arbitrary metadata columns. Rows
            are written in the given order; columns are reordered to the canonical layout.
        split_rows: One mapping per (sample, fold) with ``sample_id`` + ``split`` and an
            optional ``fold`` (defaults to ``0`` when omitted, per the single-fold rule).
        summary: A JSON-serializable summary, written verbatim (with sorted keys).
        target_matrix: Required for ``dataset_type='spatial_expression'`` (forbidden
            otherwise): a ``[n_rows, n_genes]`` array written verbatim to the
            ``targets.npy`` sidecar. Row ``i`` is the ``target_index==i`` sample's vector.
        genes: Required for ``dataset_type='spatial_expression'`` (forbidden otherwise):
            the ordered gene list written to the ``genes.json`` sidecar; its length must
            equal the target matrix width.

    Returns:
        A :class:`CuratedManifest` pointing at the written files (with the sidecar paths
        populated for a ``spatial_expression`` Manifest).
    """
    if not dataset_rows:
        raise ValueError("write_manifest requires at least one dataset row.")

    supervision = supervision_column_for(dataset_type)
    forbidden = _ALL_SUPERVISION_COLUMNS - {supervision}

    is_spatial = dataset_type == "spatial_expression"
    if not is_spatial and (target_matrix is not None or genes is not None):
        raise ValueError(
            "target_matrix / genes sidecars are only valid for "
            f"dataset_type='spatial_expression', not {dataset_type!r}."
        )

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
    validate_spacing_declaration_columns(dataset_df)
    duplicated = dataset_df.loc[
        dataset_df["sample_id"].astype(str).duplicated(), "sample_id"
    ].astype(str)
    if not duplicated.empty:
        dupes = sorted(set(duplicated))
        raise ValueError(
            f"dataset rows contain {len(dupes)} duplicated sample_id value(s): "
            f"{dupes[:20]}{' ...' if len(dupes) > 20 else ''}. Each sample_id names one "
            "cache payload, so a duplicate would silently overwrite another sample."
        )

    matrix: np.ndarray | None = None
    if is_spatial:
        matrix = _validate_spatial_expression(dataset_df, target_matrix, genes)

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

    target_matrix_path: Path | None = None
    genes_path: Path | None = None
    if is_spatial:
        assert matrix is not None and genes is not None  # guaranteed by _validate_...
        target_matrix_path = output_dir / TARGET_MATRIX_FILENAME
        genes_path = output_dir / GENES_FILENAME
        # Uncompressed .npy + plain .json keep re-writes byte-identical (ADR 0004).
        with open(target_matrix_path, "wb") as fh:
            np.save(fh, matrix, allow_pickle=False)
        genes_path.write_text(json.dumps(list(genes), indent=2) + "\n")

    return CuratedManifest(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        summary_json=summary_json,
        target_matrix_path=target_matrix_path,
        genes_path=genes_path,
    )


def _validate_spatial_expression(
    dataset_df: pd.DataFrame,
    target_matrix: "np.ndarray | Sequence[Sequence[float]] | None",
    genes: Sequence[str] | None,
) -> np.ndarray:
    """Validate the spatial_expression sidecars and return the canonical target matrix.

    Fail-fast on: a missing sidecar input, a non-2D matrix, a gene/column-count mismatch,
    or a ``target_index`` value out of range. Returns the C-contiguous ``np.ndarray`` to
    write to ``targets.npy`` (a fixed memory order keeps re-writes byte-identical).
    """
    if target_matrix is None:
        raise ValueError(
            "dataset_type='spatial_expression' requires a target_matrix "
            "([n_rows, n_genes]) written to the targets.npy sidecar."
        )
    if genes is None:
        raise ValueError(
            "dataset_type='spatial_expression' requires an ordered genes list "
            "written to the genes.json sidecar."
        )
    matrix = np.ascontiguousarray(target_matrix)
    if matrix.ndim != 2:
        raise ValueError(
            f"target_matrix must be 2D [n_rows, n_genes]; got shape {matrix.shape}."
        )
    if matrix.shape[1] != len(genes):
        raise ValueError(
            f"gene list has {len(genes)} genes but the target_matrix has "
            f"{matrix.shape[1]} columns; they must match."
        )
    n_rows = matrix.shape[0]
    try:
        indices = [int(v) for v in dataset_df["target_index"]]
    except (TypeError, ValueError):
        raise ValueError(
            "target_index must be integer row-keys into the target matrix."
        ) from None
    oob = sorted({v for v in indices if v < 0 or v >= n_rows})
    if oob:
        raise ValueError(
            f"target_index value(s) {oob} out of range for target matrix with "
            f"{n_rows} rows."
        )
    return matrix
