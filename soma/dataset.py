"""Dataset and Splits — load dataset.csv and user-provided splits.csv."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Sidecar artifacts written beside a ``spatial_expression`` ``dataset.csv`` (the shared
# writer emits them, this loader reads them back): a ``[n_rows, n_genes]`` target matrix
# and the ordered gene list. Uncompressed ``.npy`` / plain ``.json`` keep re-writes
# byte-identical (ADR 0004); compressed formats are not bit-deterministic.
TARGET_MATRIX_FILENAME = "targets.npy"
GENES_FILENAME = "genes.json"


REQUIRED_DATASET_COLUMNS = {"sample_id", "image_path", "label"}
# ``mask_path`` is the optional precomputed *tissue* mask (a tile-sampling aid, valid for
# every dataset_type); ``label_mask_path`` is segmentation's per-pixel supervision raster.
# ``points_path`` (detection's per-sample point file) and ``spacing_at_level_0`` (the
# optional caller declaration for the source image's level-0 spacing) are recognized
# typed columns too. Other detection columns — ``source_wsi`` / ``tile_x``
# / ``tile_y`` (tile origin, retained for deferred WSI stitching) — carry no typed
# ``SampleRecord`` field and surface via ``metadata`` rather than being dropped.
KNOWN_DATASET_COLUMNS = REQUIRED_DATASET_COLUMNS | {
    "mask_path",
    "label_mask_path",
    "patient_id",
    "group_id",
    "points_path",
    # spatial_expression: integer row-key into the sidecar target matrix; resolved to a
    # vector on SampleRecord.target, so it is a typed column (not free metadata).
    "target_index",
    # Slide-manifest segmentation ROI origin (level-0 px); typed onto SampleRecord.region.
    "region_x",
    "region_y",
    # The parent slide an ROI was sampled from; typed onto SampleRecord.slide_id.
    "slide_id",
    "spacing_at_level_0",
}
REQUIRED_SPLITS_COLUMNS = {"sample_id", "split"}


def is_filename_safe_id(value: object) -> bool:
    """True if ``value`` is safe to use as a bare cache filename.

    ``sample_id`` and ``patient_id`` are written directly as ``<id>.pt`` (and
    sidecars) across every cache kind, so an id containing a path separator,
    ``..``, or an absolute path would write *outside* the intended directory
    (path traversal). Safe ids are non-empty bare names with no separators.
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


def _parse_spacing_at_level_0(value: object) -> float | None:
    if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError
    try:
        spacing = float(value)
    except (TypeError, ValueError):
        raise ValueError from None
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError
    return spacing


def validate_spacing_declaration_columns(df: pd.DataFrame) -> None:
    """Validate the sole optional source-spacing declaration in a Manifest table."""
    if "level0_spacing" in df.columns:
        raise ValueError(
            "Manifest column 'level0_spacing' is retired; use 'spacing_at_level_0' instead."
        )
    if "spacing_at_level_0" not in df.columns:
        return

    invalid: list[str] = []
    for index, row in df.iterrows():
        try:
            _parse_spacing_at_level_0(row["spacing_at_level_0"])
        except ValueError:
            sample = row.get("sample_id", index)
            invalid.append(f"{sample}={row['spacing_at_level_0']!r}")
    if invalid:
        raise ValueError(
            "spacing_at_level_0 must be a positive, finite number or blank; "
            f"invalid sample(s): {invalid}."
        )


def _spacing_at_level_0(row: pd.Series) -> float | None:
    return _parse_spacing_at_level_0(row.get("spacing_at_level_0"))


def _optional_text_column(row: pd.Series, column: str) -> str | None:
    value = row.get(column)
    return str(value) if column in row.index and pd.notna(value) else None


def _optional_path_column(row: pd.Series, column: str) -> Path | None:
    text = _optional_text_column(row, column)
    return Path(text) if text is not None else None


def _is_valid_split_name(name: str) -> bool:
    """A split name is valid if it is 'train', 'tune', or starts with 'test'."""
    return name in ("train", "tune") or name.startswith("test")


@dataclass(frozen=True)
class SampleRecord:
    """A single sample (one slide) with its path and label."""

    sample_id: str
    image_path: Path
    label: str | int | None  # None for segmentation/detection (supervision is dense)
    # Optional precomputed tissue mask (tile-sampling aid, never trained on). Any task.
    mask_path: Path | None = None
    # Segmentation: the per-pixel supervision raster. Not a tissue mask.
    label_mask_path: Path | None = None
    points_path: Path | None = None  # detection: per-sample point annotations
    patient_id: str | None = None
    # Literal non-independence group from the manifest. Optional for every task;
    # representation evaluation validates it only for the selected cohort.
    group_id: str | None = None
    # Optional caller declaration for ``image_path``'s physical level-0 pixel size.
    # Extraction resolves and persists the authoritative source spacing separately.
    spacing_at_level_0: float | None = None
    # Slide-manifest segmentation: an ROI's (x, y) top-left in level-0 pixel space.
    # image_path/label_mask_path then point at the parent *slide* (+ annotation slide), and
    # the run's spacing/tile size complete the region read. None for pre-cropped tiles.
    region: tuple[int, int] | None = None
    # Slide-manifest segmentation: the parent slide this ROI was sampled from. Recorded
    # explicitly because it is part of the ROI's on-disk address — slide2vec namespaces
    # dense grids as ``<slide_id>/<x>_<y>.pt`` — and reconstructing it by splitting the
    # ROI's own ``<slide>__x<X>_y<Y>`` id apart would make a naming convention
    # load-bearing (ADR 0007). None for every non-ROI row.
    slide_id: str | None = None
    # spatial_expression: the resolved multi-target regression vector for this spot,
    # attached from the sidecar target matrix (row ``target_index``). None for every
    # other dataset_type. Excluded from equality/hash — it is derived supervision data,
    # not sample identity, and array-valued fields have no scalar truth value.
    target: "np.ndarray | None" = field(default=None, compare=False)
    metadata: dict[str, Any] = field(default_factory=dict)


class Dataset:
    """Loads a dataset from a CSV with columns: sample_id, image_path, label.

    Optional columns: mask_path (precomputed tissue mask). Extra columns become metadata.
    """

    def __init__(self, dataset_csv: str | Path) -> None:
        self._path = Path(dataset_csv)
        df = pd.read_csv(self._path)
        self._validate_columns(df)
        self._samples = self._build_samples(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        validate_spacing_declaration_columns(df)
        if "tissue_mask_path" in df.columns:
            raise ValueError(
                "Use 'mask_path' (the tissue mask column) instead of 'tissue_mask_path'."
            )
        for col in REQUIRED_DATASET_COLUMNS:
            if col not in df.columns:
                msg = f"Required column '{col}' not found. Available: {list(df.columns)}"
                raise ValueError(msg)
        if df["sample_id"].duplicated().any():
            dupes = df["sample_id"][df["sample_id"].duplicated()].tolist()
            msg = f"Duplicate sample_id values: {dupes}"
            raise ValueError(msg)
        # sample_id / patient_id become cache filenames (<id>.pt); reject ids that
        # would escape the cache dir via path separators or '..' (path traversal).
        unsafe_ids = sorted({str(s) for s in df["sample_id"] if not is_filename_safe_id(s)})
        if unsafe_ids:
            raise ValueError(
                "Unsafe sample_id value(s) (used as cache filenames; no path "
                f"separators, '..', or absolute paths allowed): {unsafe_ids}"
            )
        if "patient_id" in df.columns:
            unsafe_patients = sorted(
                {str(p) for p in df["patient_id"].dropna() if not is_filename_safe_id(p)}
            )
            if unsafe_patients:
                raise ValueError(
                    "Unsafe patient_id value(s) (used as cache filenames; no path "
                    f"separators, '..', or absolute paths allowed): {unsafe_patients}"
                )

    def _build_samples(self, df: pd.DataFrame) -> dict[str, SampleRecord]:
        meta_columns = [c for c in df.columns if c not in KNOWN_DATASET_COLUMNS]
        samples: dict[str, SampleRecord] = {}
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            patient_id = _optional_text_column(row, "patient_id")
            group_id = _optional_text_column(row, "group_id")
            metadata = {c: row[c] for c in meta_columns}
            samples[sid] = SampleRecord(
                sample_id=sid,
                image_path=Path(str(row["image_path"])),
                label=row["label"],
                mask_path=_optional_path_column(row, "mask_path"),
                patient_id=patient_id,
                group_id=group_id,
                spacing_at_level_0=_spacing_at_level_0(row),
                metadata=metadata,
            )
        return samples

    @property
    def samples(self) -> dict[str, SampleRecord]:
        return self._samples

    @property
    def sample_ids(self) -> list[str]:
        return list(self._samples.keys())

    @property
    def has_patient_ids(self) -> bool:
        """True if any sample record has a patient_id."""
        return any(r.patient_id is not None for r in self._samples.values())

    @property
    def patient_groups(self) -> dict[str, list["SampleRecord"]]:
        """Group sample records by patient_id.

        Raises ValueError if any sample record is missing a patient_id.
        """
        groups: dict[str, list[SampleRecord]] = {}
        for record in self._samples.values():
            if record.patient_id is None:
                raise ValueError(
                    f"Sample '{record.sample_id}' is missing a patient_id. "
                    "All rows must have a patient_id for patient-level pipelines."
                )
            groups.setdefault(record.patient_id, []).append(record)
        return groups

    @property
    def patient_record_map(self) -> dict[str, "SampleRecord"]:
        """Map patient_id to a representative SampleRecord for that patient.

        Validates that all slides for a patient share the same label, then
        returns the first record per patient. The representative record carries
        the patient's label and metadata, so a task head's ``extract_targets``
        works identically for slide- and patient-level pipelines.

        Raises ValueError if patient_ids are missing or labels are inconsistent.
        """
        record_map: dict[str, SampleRecord] = {}
        for patient_id, records in self.patient_groups.items():
            labels = {r.label for r in records}
            if len(labels) > 1:
                raise ValueError(
                    f"Patient '{patient_id}' has inconsistent labels across slides: "
                    f"{sorted(str(l) for l in labels)}. "
                    "All slides for a patient must share the same label."
                )
            record_map[patient_id] = records[0]
        return record_map


class TileDataset(Dataset):
    """Scalar-supervised pre-cropped images with Given encoder geometry.

    The distinct type is intentionally behavior-free: it prevents persistent extraction
    from guessing whether a generic ``Dataset`` row names a whole slide or a tile by
    inspecting paths, pixels, or CSV columns.
    """


REQUIRED_SEGMENTATION_COLUMNS = {"sample_id", "image_path", "label_mask_path"}


class SegmentationManifest:
    """Loads a segmentation dataset CSV: sample_id, image_path, label_mask_path (required).

    Unlike :class:`Dataset`, ``label`` is NOT required — the supervision signal is
    the per-pixel ``label_mask_path`` raster, not a scalar label (a separate loader
    rather than relaxing :class:`Dataset`, whose ``label`` requirement guards every
    other task). ``label_mask_path`` must be present and non-null for every row.
    ``mask_path`` keeps its usual meaning — an optional precomputed *tissue* mask —
    so a segmentation row can carry both. ``label`` and ``patient_id`` are optional;
    extra columns become metadata. Exposes the same ``samples``/``sample_ids`` surface
    as :class:`Dataset`, so :class:`Splits` works against it unchanged.
    """

    def __init__(self, dataset_csv: str | Path) -> None:
        self._path = Path(dataset_csv)
        df = pd.read_csv(self._path)
        self._validate_columns(df)
        self._samples = self._build_samples(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        validate_spacing_declaration_columns(df)
        if "tissue_mask_path" in df.columns:
            raise ValueError(
                "Use 'mask_path' (the tissue mask column) instead of 'tissue_mask_path'."
            )
        if "label_mask_path" not in df.columns and "mask_path" in df.columns:
            # Pre-rename segmentation manifests carried the supervision raster as
            # ``mask_path``. Never reinterpret it silently as a tissue mask: say what
            # happened. Manifests are reproducible from their curator, so regenerate.
            raise ValueError(
                "Segmentation manifest has 'mask_path' but no 'label_mask_path': this looks "
                "like a pre-rename segmentation manifest (soma < 1.11 used 'mask_path' for "
                "the supervision raster; it now means an optional tissue mask). Regenerate "
                "it with its curator, or rename the column to 'label_mask_path'."
            )
        for col in REQUIRED_SEGMENTATION_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found. Available: {list(df.columns)}"
                )
        if df["sample_id"].duplicated().any():
            dupes = df["sample_id"][df["sample_id"].duplicated()].tolist()
            raise ValueError(f"Duplicate sample_id values: {dupes}")
        if df["label_mask_path"].isna().any():
            missing = df.loc[df["label_mask_path"].isna(), "sample_id"].tolist()
            raise ValueError(
                f"label_mask_path is required for every segmentation sample; missing for: {missing}"
            )
        # sample_id / patient_id become cache filenames; reject path-traversal ids.
        unsafe_ids = sorted({str(s) for s in df["sample_id"] if not is_filename_safe_id(s)})
        if unsafe_ids:
            raise ValueError(
                "Unsafe sample_id value(s) (used as cache filenames; no path "
                f"separators, '..', or absolute paths allowed): {unsafe_ids}"
            )
        if "patient_id" in df.columns:
            unsafe_patients = sorted(
                {str(p) for p in df["patient_id"].dropna() if not is_filename_safe_id(p)}
            )
            if unsafe_patients:
                raise ValueError(
                    "Unsafe patient_id value(s) (used as cache filenames; no path "
                    f"separators, '..', or absolute paths allowed): {unsafe_patients}"
                )

    def _build_samples(self, df: pd.DataFrame) -> dict[str, SampleRecord]:
        meta_columns = [c for c in df.columns if c not in KNOWN_DATASET_COLUMNS]
        samples: dict[str, SampleRecord] = {}
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            label = row["label"] if "label" in row.index and pd.notna(row.get("label")) else None
            patient_id = _optional_text_column(row, "patient_id")
            group_id = _optional_text_column(row, "group_id")
            metadata = {c: row[c] for c in meta_columns}
            region = None
            if "region_x" in row.index and pd.notna(row.get("region_x")):
                region = (int(row["region_x"]), int(row["region_y"]))
            slide_id = (
                str(row["slide_id"])
                if "slide_id" in row.index and pd.notna(row.get("slide_id"))
                else None
            )
            samples[sid] = SampleRecord(
                sample_id=sid,
                image_path=Path(str(row["image_path"])),
                label=label,  # optional for segmentation; supervision is label_mask_path
                mask_path=_optional_path_column(row, "mask_path"),
                label_mask_path=Path(str(row["label_mask_path"])),
                patient_id=patient_id,
                group_id=group_id,
                spacing_at_level_0=_spacing_at_level_0(row),
                region=region,
                slide_id=slide_id,
                metadata=metadata,
            )
        return samples

    @property
    def samples(self) -> dict[str, SampleRecord]:
        return self._samples

    @property
    def sample_ids(self) -> list[str]:
        return list(self._samples.keys())

    @property
    def has_patient_ids(self) -> bool:
        return any(r.patient_id is not None for r in self._samples.values())


REQUIRED_DETECTION_COLUMNS = {"sample_id", "image_path", "points_path"}


class DetectionManifest:
    """Loads a detection dataset CSV: sample_id, image_path, points_path (required).

    The detection counterpart of :class:`SegmentationManifest` (design §3): the
    supervision is a per-sample point file (``points_path``, level-0 ``x,y,class``),
    not a scalar ``label`` (optional) or a mask. Optional columns include ``mask_path``
    (precomputed tissue mask), ``spacing_at_level_0`` (the source image's optional µm/px
    declaration), ``source_wsi`` /
    ``tile_x`` / ``tile_y`` (retained now for deferred WSI stitching), ``label``,
    ``patient_id``. ``points_path`` must be present and non-null for every row. Exposes
    the same ``samples`` / ``sample_ids`` surface as :class:`Dataset`, so
    :class:`Splits` works against it unchanged.
    """

    def __init__(self, dataset_csv: str | Path) -> None:
        self._path = Path(dataset_csv)
        df = pd.read_csv(self._path)
        self._validate_columns(df)
        self._samples = self._build_samples(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        validate_spacing_declaration_columns(df)
        for col in REQUIRED_DETECTION_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found. Available: {list(df.columns)}"
                )
        if df["sample_id"].duplicated().any():
            dupes = df["sample_id"][df["sample_id"].duplicated()].tolist()
            raise ValueError(f"Duplicate sample_id values: {dupes}")
        if df["points_path"].isna().any():
            missing = df.loc[df["points_path"].isna(), "sample_id"].tolist()
            raise ValueError(
                f"points_path is required for every detection sample; missing for: {missing}"
            )
        unsafe_ids = sorted({str(s) for s in df["sample_id"] if not is_filename_safe_id(s)})
        if unsafe_ids:
            raise ValueError(
                "Unsafe sample_id value(s) (used as cache filenames; no path "
                f"separators, '..', or absolute paths allowed): {unsafe_ids}"
            )
        if "patient_id" in df.columns:
            unsafe_patients = sorted(
                {str(p) for p in df["patient_id"].dropna() if not is_filename_safe_id(p)}
            )
            if unsafe_patients:
                raise ValueError(
                    "Unsafe patient_id value(s) (used as cache filenames; no path "
                    f"separators, '..', or absolute paths allowed): {unsafe_patients}"
                )

    def _build_samples(self, df: pd.DataFrame) -> dict[str, SampleRecord]:
        # source_wsi / tile_x / tile_y are not typed yet and remain metadata for deferred
        # WSI stitching. Source spacing is the canonical typed declaration instead.
        meta_columns = [c for c in df.columns if c not in KNOWN_DATASET_COLUMNS]
        samples: dict[str, SampleRecord] = {}
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            label = row["label"] if "label" in row.index and pd.notna(row.get("label")) else None
            patient_id = _optional_text_column(row, "patient_id")
            group_id = _optional_text_column(row, "group_id")
            metadata = {c: row[c] for c in meta_columns if pd.notna(row.get(c))}
            samples[sid] = SampleRecord(
                sample_id=sid,
                image_path=Path(str(row["image_path"])),
                label=label,  # optional for detection; supervision is the points
                mask_path=_optional_path_column(row, "mask_path"),
                points_path=Path(str(row["points_path"])),
                patient_id=patient_id,
                group_id=group_id,
                spacing_at_level_0=_spacing_at_level_0(row),
                metadata=metadata,
            )
        return samples

    @property
    def samples(self) -> dict[str, SampleRecord]:
        return self._samples

    @property
    def sample_ids(self) -> list[str]:
        return list(self._samples.keys())

    @property
    def has_patient_ids(self) -> bool:
        return any(r.patient_id is not None for r in self._samples.values())


REQUIRED_SPATIAL_EXPRESSION_COLUMNS = {"sample_id", "image_path", "target_index"}


class SpatialExpressionManifest:
    """Loads a spatial_expression dataset CSV: sample_id, image_path, target_index.

    The multi-target-regression counterpart of :class:`SegmentationManifest` /
    :class:`DetectionManifest` (HEST-benchmark design §4): supervision is a **vector**,
    not a scalar ``label``. Each row's ``target_index`` is an integer row-key into a
    sidecar target matrix written beside ``dataset.csv``:

    * ``targets.npy`` — shape ``[n_rows, n_genes]``; row ``i`` is the ``target_index==i``
      sample's expression vector.
    * ``genes.json`` — the ordered gene list (``len == n_genes``).

    Both sidecars are required and fail-fast validated at construction: missing files, a
    missing ``target_index`` column, out-of-range indices, or a gene/column-count mismatch
    all raise before any expensive run. Each loaded :class:`SampleRecord` carries its
    resolved vector on ``target``; the ordered gene list is exposed via :attr:`genes` and
    the whole matrix via :attr:`target_matrix`. Exposes the same
    ``samples`` / ``sample_ids`` / ``has_patient_ids`` surface as :class:`Dataset`, so
    :class:`Splits` works against it unchanged.
    """

    def __init__(self, dataset_csv: str | Path) -> None:
        self._path = Path(dataset_csv)
        df = pd.read_csv(self._path)
        self._validate_columns(df)
        self._genes, self._target_matrix = self._load_sidecars()
        self._validate_targets(df)
        self._samples = self._build_samples(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        validate_spacing_declaration_columns(df)
        for col in REQUIRED_SPATIAL_EXPRESSION_COLUMNS:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found. Available: {list(df.columns)}"
                )
        if df["sample_id"].duplicated().any():
            dupes = df["sample_id"][df["sample_id"].duplicated()].tolist()
            raise ValueError(f"Duplicate sample_id values: {dupes}")
        unsafe_ids = sorted({str(s) for s in df["sample_id"] if not is_filename_safe_id(s)})
        if unsafe_ids:
            raise ValueError(
                "Unsafe sample_id value(s) (used as cache filenames; no path "
                f"separators, '..', or absolute paths allowed): {unsafe_ids}"
            )
        if "patient_id" in df.columns:
            unsafe_patients = sorted(
                {str(p) for p in df["patient_id"].dropna() if not is_filename_safe_id(p)}
            )
            if unsafe_patients:
                raise ValueError(
                    "Unsafe patient_id value(s) (used as cache filenames; no path "
                    f"separators, '..', or absolute paths allowed): {unsafe_patients}"
                )

    def _load_sidecars(self) -> tuple[list[str], np.ndarray]:
        sidecar_dir = self._path.parent
        targets_path = sidecar_dir / TARGET_MATRIX_FILENAME
        genes_path = sidecar_dir / GENES_FILENAME
        if not targets_path.exists():
            raise ValueError(
                f"spatial_expression Manifest is missing its target-matrix sidecar "
                f"'{TARGET_MATRIX_FILENAME}' (expected beside dataset.csv at {targets_path})."
            )
        if not genes_path.exists():
            raise ValueError(
                f"spatial_expression Manifest is missing its gene-list sidecar "
                f"'{GENES_FILENAME}' (expected beside dataset.csv at {genes_path})."
            )
        matrix = np.load(targets_path)
        genes = json.loads(genes_path.read_text())
        if matrix.ndim != 2:
            raise ValueError(
                f"target matrix '{TARGET_MATRIX_FILENAME}' must be 2D [n_rows, n_genes]; "
                f"got shape {matrix.shape}."
            )
        if not isinstance(genes, list) or matrix.shape[1] != len(genes):
            raise ValueError(
                f"gene list '{GENES_FILENAME}' ({len(genes)} genes) does not match the "
                f"target matrix width ({matrix.shape[1]} columns)."
            )
        return genes, matrix

    def _validate_targets(self, df: pd.DataFrame) -> None:
        n_rows = self._target_matrix.shape[0]
        try:
            indices = [int(v) for v in df["target_index"]]
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

    def _build_samples(self, df: pd.DataFrame) -> dict[str, SampleRecord]:
        meta_columns = [c for c in df.columns if c not in KNOWN_DATASET_COLUMNS]
        samples: dict[str, SampleRecord] = {}
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            patient_id = _optional_text_column(row, "patient_id")
            group_id = _optional_text_column(row, "group_id")
            metadata = {c: row[c] for c in meta_columns}
            target_index = int(row["target_index"])
            samples[sid] = SampleRecord(
                sample_id=sid,
                image_path=Path(str(row["image_path"])),
                label=None,  # supervision is the vector target, not a scalar label
                mask_path=_optional_path_column(row, "mask_path"),
                patient_id=patient_id,
                group_id=group_id,
                spacing_at_level_0=_spacing_at_level_0(row),
                target=self._target_matrix[target_index],
                metadata=metadata,
            )
        return samples

    @property
    def samples(self) -> dict[str, SampleRecord]:
        return self._samples

    @property
    def sample_ids(self) -> list[str]:
        return list(self._samples.keys())

    @property
    def has_patient_ids(self) -> bool:
        return any(r.patient_id is not None for r in self._samples.values())

    @property
    def genes(self) -> list[str]:
        """Ordered gene list backing the target-vector columns."""
        return list(self._genes)

    @property
    def target_matrix(self) -> np.ndarray:
        """The full ``[n_rows, n_genes]`` sidecar target matrix."""
        return self._target_matrix


def load_manifest(
    dataset_csv: str | Path, dataset_type: str
) -> "Dataset | TileDataset | SegmentationManifest | DetectionManifest | SpatialExpressionManifest":
    """Load + validate a ``dataset.csv`` with the loader selected by ``dataset_type``.

    This is the single load-time validator keyed on ``dataset_type``: each loader
    fail-fast validates its required supervision column (``label`` for classification,
    ``label_mask_path`` for segmentation, ``points_path`` for detection, ``target_index`` for
    spatial_expression) at construction time, so a malformed Manifest is rejected with a
    clear message before any expensive run.
    """
    if dataset_type == "segmentation":
        return SegmentationManifest(dataset_csv)
    if dataset_type == "detection":
        return DetectionManifest(dataset_csv)
    if dataset_type == "spatial_expression":
        return SpatialExpressionManifest(dataset_csv)
    if dataset_type == "tile":
        return TileDataset(dataset_csv)
    return Dataset(dataset_csv)


@dataclass(frozen=True)
class FoldSplit:
    """Sample IDs for each split within one fold.

    ``tests`` maps each test split name (e.g. ``"test"``, ``"test_external"``)
    to the corresponding tuple of sample IDs.  Every fold must contain at
    least one test split (any name that starts with ``"test"``).

    ``test_from_tune`` is ``True`` when the fold had no user-provided test
    split and the ``"test"`` entry was synthesized from the tune split (see
    ``TrainingConfig.tune_is_test``). In that case the test entry mirrors
    ``tune`` exactly, so leakage checks skip it to avoid false positives.
    """

    train: tuple[str, ...]
    tune: tuple[str, ...]
    tests: dict[str, tuple[str, ...]]
    test_from_tune: bool = False

    @property
    def test_split_names(self) -> list[str]:
        """Sorted list of test split names for this fold."""
        return sorted(self.tests.keys())


class Splits:
    """Loads user-provided splits from a CSV with columns: sample_id, split[, fold].

    The ``fold`` column is optional. When absent, all rows are treated as a
    single split (no cross-validation). Valid split names are ``"train"``,
    ``"tune"``, or any name starting with ``"test"`` (e.g. ``"test"``,
    ``"test_external"``, ``"test_prospective"``). Validates that all sample_ids
    exist in the dataset, split names are valid, and no sample appears twice
    within the same fold.

    Each fold must contain at least one test split. When ``tune_is_test`` is
    set, a fold may instead provide only a tune split, in which case the tune
    split is reused for test reporting (a ``"test"`` entry is synthesized from
    it); the fold must still provide either a tune or a test split.
    """

    def __init__(
        self,
        splits_csv: str | Path,
        dataset: Dataset,
        *,
        tune_is_test: bool = False,
    ) -> None:
        self._path = Path(splits_csv)
        self._tune_is_test = tune_is_test
        df = pd.read_csv(self._path)
        self._validate_columns(df)
        if "fold" not in df.columns:
            df = df.copy()
            df["fold"] = 0
        self._validate_values(df, dataset)
        self._folds = self._build_folds(df)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        for col in REQUIRED_SPLITS_COLUMNS:
            if col not in df.columns:
                msg = f"Required column '{col}' not found. Available: {list(df.columns)}"
                raise ValueError(msg)

    def _validate_values(self, df: pd.DataFrame, dataset: Dataset) -> None:
        # Check sample_ids exist in dataset
        known_ids = set(dataset.sample_ids)
        unknown = set(df["sample_id"].astype(str)) - known_ids
        if unknown:
            msg = f"Unknown sample_id(s) in splits.csv: {sorted(unknown)}"
            raise ValueError(msg)

        # Check split names
        invalid = {
            name
            for name in df["split"]
            if not isinstance(name, str) or not _is_valid_split_name(name)
        }
        if invalid:
            msg = (
                f"Invalid split name(s): {sorted(str(name) for name in invalid)}. "
                "Must be 'train', 'tune', or start with 'test'."
            )
            raise ValueError(msg)

        # Check no duplicate sample within a fold
        for fold_idx, group in df.groupby("fold"):
            dupes = group["sample_id"][group["sample_id"].duplicated()]
            if not dupes.empty:
                msg = f"Duplicate sample_id(s) in fold {fold_idx}: {dupes.tolist()}"
                raise ValueError(msg)
            has_test_split = group["split"].map(lambda name: name.startswith("test")).any()
            has_tune_split = (group["split"] == "tune").any()
            if not has_test_split and not self._tune_is_test:
                msg = (
                    f"Fold {fold_idx} must contain at least one test split "
                    "(a split name starting with 'test'). Set "
                    "training.tune_is_test=True to reuse the tune split "
                    "for test reporting."
                )
                raise ValueError(msg)
            if self._tune_is_test and has_tune_split and has_test_split:
                msg = (
                    f"Fold {fold_idx} provides both a tune and a test split, "
                    "but tune_is_test=True ties them to a single held-out split. "
                    "Provide only one of them, or set tune_is_test=False."
                )
                raise ValueError(msg)
            if not has_test_split and self._tune_is_test and not has_tune_split:
                msg = (
                    f"Fold {fold_idx} has no test split and no tune split; "
                    "tune_is_test=True requires either a tune or a test "
                    "split to reuse for both roles."
                )
                raise ValueError(msg)

    def _build_folds(self, df: pd.DataFrame) -> list[FoldSplit]:
        folds = []
        for fold_idx, group in sorted(df.groupby("fold")):
            train_ids = tuple(str(s) for s in group.loc[group["split"] == "train", "sample_id"])
            tune_ids = tuple(str(s) for s in group.loc[group["split"] == "tune", "sample_id"])
            tests: dict[str, tuple[str, ...]] = {
                split_name: tuple(
                    str(s) for s in group.loc[group["split"] == split_name, "sample_id"]
                )
                for split_name in sorted(group["split"].unique())
                if split_name.startswith("test")
            }
            test_from_tune = False
            if not tests and self._tune_is_test:
                logger.warning(
                    "Fold %s has no test split; reusing the tune split for test "
                    "reporting because tune_is_test=True. Reported 'test' "
                    "metrics are measured on the tune samples.",
                    fold_idx,
                )
                tests = {"test": tune_ids}
                test_from_tune = True
            folds.append(
                FoldSplit(
                    train=train_ids,
                    tune=tune_ids,
                    tests=tests,
                    test_from_tune=test_from_tune,
                )
            )
        return folds

    @property
    def folds(self) -> list[FoldSplit]:
        return self._folds

    @property
    def num_folds(self) -> int:
        return len(self._folds)

    def project(self, dataset: "Dataset") -> "Splits":
        """Project assignments onto an effective dataset.

        Existing sample IDs retain their own assignment. A derived ROI that has no direct
        assignment inherits exactly one hop through its explicit ``slide_id``. No sample-id
        parsing or generic parent convention participates in the projection.
        """

        projected_folds: list[FoldSplit] = []
        for fold_index, fold in enumerate(self._folds):
            source_locations: dict[str, set[tuple[str, str | None]]] = {}

            def record_location(
                sample_ids: tuple[str, ...], kind: str, name: str | None = None
            ) -> None:
                for sample_id in sample_ids:
                    source_locations.setdefault(sample_id, set()).add((kind, name))

            record_location(fold.train, "train")
            record_location(fold.tune, "tune")
            for test_name, sample_ids in fold.tests.items():
                record_location(sample_ids, "test", test_name)

            selected: dict[str, set[tuple[str, str | None]]] = {}
            for sample_id, record in dataset.samples.items():
                direct = source_locations.get(sample_id, set())
                inherited = (
                    source_locations.get(str(record.slide_id), set())
                    if record.slide_id is not None
                    else set()
                )
                if direct and inherited and direct != inherited:
                    raise ValueError(
                        f"Conflicting split ancestry for sample '{sample_id}' in fold "
                        f"{fold_index}: direct={sorted(direct)}, "
                        f"slide_id '{record.slide_id}'={sorted(inherited)}."
                    )
                locations = direct or inherited
                if not locations and record.slide_id is not None:
                    raise ValueError(
                        f"Unresolved split ancestry for sample '{sample_id}' in fold "
                        f"{fold_index}: slide_id '{record.slide_id}' has no assignment."
                    )
                if locations:
                    selected[sample_id] = locations

            ordered_ids = list(dataset.sample_ids)
            projected_folds.append(
                FoldSplit(
                    train=tuple(
                        sample_id
                        for sample_id in ordered_ids
                        if ("train", None) in selected.get(sample_id, set())
                    ),
                    tune=tuple(
                        sample_id
                        for sample_id in ordered_ids
                        if ("tune", None) in selected.get(sample_id, set())
                    ),
                    tests={
                        test_name: tuple(
                            sample_id
                            for sample_id in ordered_ids
                            if ("test", test_name) in selected.get(sample_id, set())
                        )
                        for test_name in fold.tests
                    },
                    test_from_tune=fold.test_from_tune,
                )
            )

        projected = object.__new__(Splits)
        projected._path = self._path
        projected._tune_is_test = self._tune_is_test
        projected._folds = projected_folds
        return projected

    def validate_no_patient_leakage(self, dataset: "Dataset") -> None:
        """Validate that no patient appears in more than one split within any fold.

        Raises ValueError if a patient's slides are assigned to different splits
        in the same fold (e.g., some in train, some in test). This is required
        for patient-level pipelines to prevent data leakage.
        """
        sample_to_patient = {
            sid: record.patient_id
            for sid, record in dataset.samples.items()
            if record.patient_id is not None
        }
        if not sample_to_patient:
            raise ValueError(
                "Dataset has no patient_id column. "
                "Patient-level pipelines require a patient_id column in the dataset CSV."
            )
        for fold_idx, fold_split in enumerate(self._folds):
            patient_splits: dict[str, set[str]] = {}
            # Skip the synthesized test entry: it mirrors tune by construction,
            # so counting it would flag every tune patient as leaked.
            test_items = [] if fold_split.test_from_tune else list(fold_split.tests.items())
            all_splits = [
                ("train", fold_split.train),
                ("tune", fold_split.tune),
                *test_items,
            ]
            for split_name, sample_ids in all_splits:
                for sid in sample_ids:
                    pid = sample_to_patient.get(sid)
                    if pid is None:
                        continue
                    patient_splits.setdefault(pid, set()).add(split_name)
            leaked = {pid: splits for pid, splits in patient_splits.items() if len(splits) > 1}
            if leaked:
                details = "; ".join(
                    f"patient '{pid}' in {sorted(splits)}" for pid, splits in sorted(leaked.items())
                )
                raise ValueError(
                    f"Patient leakage detected in fold {fold_idx}: {details}. "
                    "All slides for a patient must be assigned to the same split."
                )
