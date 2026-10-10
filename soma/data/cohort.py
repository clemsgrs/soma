"""Cohort = records + folds, immutable and validated on construction (design §4.1)."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soma.data.records import SampleRecord, targets_equal
from soma.data.validation import (
    SPLIT_TEST_PREFIX,
    SPLIT_TRAIN,
    SPLIT_TUNE,
    is_filename_safe_id,
    is_valid_split_name,
    optional_text,
    require_columns,
    validate_patient_ids,
    validate_sample_ids,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Cohort",
    "FoldSplit",
    "EXPRESSION_MATRIX_FILENAME",
    "EXPRESSION_NAMES_FILENAME",
    "MANIFEST_COLUMNS",
    "SPLIT_COLUMNS",
]

# Sidecars written beside a dataset CSV by the HEST curator: a ``[n_rows, n_genes]``
# target matrix indexed by the ``target_index`` column, and the ordered gene list.
EXPRESSION_MATRIX_FILENAME = "targets.npy"
EXPRESSION_NAMES_FILENAME = "genes.json"
EXPRESSION_INDEX_COLUMN = "target_index"

SPLIT_COLUMNS = frozenset({"split", "fold"})
#: Columns owned by the image / annotation manifests (and the legacy ROI dataset writer).
#: They never reach ``SampleRecord.metadata``: a record carries no file paths.
MANIFEST_COLUMNS = frozenset(
    {
        "image_path",
        "mask_path",
        "label_mask_path",
        "points_path",
        "ignore_mask_path",
        "spacing_at_level_0",
        "coordinates_path",
        "label_mask_crop_path",
        "region_x",
        "region_y",
        "slide_id",
        EXPRESSION_INDEX_COLUMN,
    }
)
_IDENTITY_COLUMNS = frozenset({"sample_id", "patient_id"})


@dataclass(frozen=True)
class FoldSplit:
    """Sample ids of each split within one fold.

    ``tests`` maps each test split name (``"test"``, ``"test_external"`` ...) to its
    sample ids. ``test_from_tune`` marks a ``"test"`` entry synthesized from ``tune`` by
    :meth:`Cohort.with_test_from_tune`; leakage checks skip it because it mirrors tune.
    """

    train: tuple[str, ...]
    tune: tuple[str, ...]
    tests: dict[str, tuple[str, ...]]
    test_from_tune: bool = False

    @property
    def test_split_names(self) -> list[str]:
        return sorted(self.tests.keys())

    @property
    def all_sample_ids(self) -> tuple[str, ...]:
        return (*self.train, *self.tune, *(sid for ids in self.tests.values() for sid in ids))


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (np.ndarray, list, tuple)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _check_dtype(key: str, value: Any, dtype: type, sample_id: str) -> None:
    ok = True
    if dtype is float:
        ok = isinstance(value, Real) and not isinstance(value, bool)
    elif dtype is int:
        ok = (isinstance(value, Integral) and not isinstance(value, bool)) or (
            isinstance(value, float) and float(value).is_integer()
        )
    elif dtype is str:
        ok = isinstance(value, str)
    elif dtype is np.ndarray:
        ok = isinstance(value, (np.ndarray, list, tuple))
    if not ok:
        raise ValueError(
            f"Target {key!r} of sample {sample_id!r} must be {dtype.__name__}, got {value!r}."
        )


def _resolve_targets(
    targets: Sequence[str] | Mapping[str, str] | None,
) -> dict[str, str]:
    if targets is None:
        return {}
    if isinstance(targets, Mapping):
        return {str(key): str(column) for key, column in targets.items()}
    return {str(key): str(key) for key in targets}


def _load_expression_sidecar(sidecar_dir: Path) -> tuple[np.ndarray, list[str]]:
    matrix_path = sidecar_dir / EXPRESSION_MATRIX_FILENAME
    names_path = sidecar_dir / EXPRESSION_NAMES_FILENAME
    if not matrix_path.is_file() or not names_path.is_file():
        raise ValueError(
            f"Target 'expression' needs the sidecars {EXPRESSION_MATRIX_FILENAME} and "
            f"{EXPRESSION_NAMES_FILENAME} beside the dataset CSV in {sidecar_dir}."
        )
    matrix = np.load(matrix_path)
    names = json.loads(names_path.read_text())
    if matrix.ndim != 2:
        raise ValueError(f"{EXPRESSION_MATRIX_FILENAME} must be 2-D, got shape {matrix.shape}.")
    if not isinstance(names, list) or len(names) != matrix.shape[1]:
        raise ValueError(
            f"{EXPRESSION_NAMES_FILENAME} ({len(names)} names) does not match the target "
            f"matrix width ({matrix.shape[1]})."
        )
    return matrix, [str(name) for name in names]


def _expression_column(frame: pd.DataFrame, sidecar_dir: Path) -> tuple[pd.Series, list[str]]:
    require_columns(frame, {EXPRESSION_INDEX_COLUMN}, what="expression target")
    matrix, names = _load_expression_sidecar(sidecar_dir)
    try:
        indices = [int(value) for value in frame[EXPRESSION_INDEX_COLUMN]]
    except (TypeError, ValueError):
        raise ValueError(f"{EXPRESSION_INDEX_COLUMN} must be integer row-keys.") from None
    out_of_range = sorted({i for i in indices if i < 0 or i >= matrix.shape[0]})
    if out_of_range:
        raise ValueError(
            f"{EXPRESSION_INDEX_COLUMN} value(s) {out_of_range} out of range for a target "
            f"matrix with {matrix.shape[0]} rows."
        )
    return pd.Series([matrix[i] for i in indices], index=frame.index, dtype=object), names


def _split_frame(records_df: pd.DataFrame, splits_df: pd.DataFrame | None) -> pd.DataFrame:
    if splits_df is None:
        if "split" not in records_df.columns:
            raise ValueError(
                "No splits given: pass a splits table or add a 'split' column (and an "
                "optional 'fold' column) to the records table."
            )
        columns = ["sample_id", "split"] + (["fold"] if "fold" in records_df.columns else [])
        splits_df = records_df[columns]
    require_columns(splits_df, {"sample_id", "split"}, what="splits table")
    splits_df = splits_df.copy()
    if "fold" not in splits_df.columns:
        splits_df["fold"] = 0
    blank = splits_df.loc[splits_df["fold"].isna(), "sample_id"].astype(str).tolist()
    if blank:
        raise ValueError(
            f"Blank 'fold' value for {len(blank)} row(s): {blank[:20]}"
            f"{' ...' if len(blank) > 20 else ''}. Every row must name its fold (or drop "
            "the column for a single-fold split table)."
        )
    invalid = sorted({str(name) for name in splits_df["split"] if not is_valid_split_name(name)})
    if invalid:
        raise ValueError(
            f"Invalid split name(s): {invalid}. Must be 'train', 'tune', or start with 'test'."
        )
    return splits_df


def _folds_from_frame(splits_df: pd.DataFrame) -> list[FoldSplit]:
    folds: list[FoldSplit] = []
    for fold_index, group in sorted(splits_df.groupby("fold"), key=lambda item: item[0]):
        dupes = group["sample_id"][group["sample_id"].duplicated()]
        if not dupes.empty:
            raise ValueError(
                f"Duplicate sample_id(s) in fold {fold_index}: "
                f"{sorted({str(v) for v in dupes})}"
            )
        ids = group["sample_id"].astype(str)
        splits = group["split"].astype(str)
        folds.append(
            FoldSplit(
                train=tuple(ids[splits == SPLIT_TRAIN]),
                tune=tuple(ids[splits == SPLIT_TUNE]),
                tests={
                    name: tuple(ids[splits == name])
                    for name in sorted(splits.unique())
                    if name.startswith(SPLIT_TEST_PREFIX)
                },
            )
        )
    return folds


class Cohort:
    """Immutable records plus folds, validated once at construction.

    ``unit`` is the sampling unit leakage checks protect: ``"patient_id"`` (default
    whenever records carry patient ids) or ``"sample_id"``. Construction rejects unsafe
    or duplicate ids, unknown split ids, a unit crossing splits within a fold, and a
    fold without a test split; it warns when a split lacks a class of a ``label`` target.
    """

    def __init__(
        self,
        records: Sequence[SampleRecord],
        folds: Sequence[FoldSplit],
        *,
        unit: str | None = None,
        dtypes: Mapping[str, type] | None = None,
        allow_missing_test: bool = False,
        target_names: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._records: tuple[SampleRecord, ...] = tuple(records)
        self._by_id: dict[str, SampleRecord] = {}
        for record in self._records:
            if not is_filename_safe_id(record.sample_id):
                raise ValueError(f"Unsafe sample_id {record.sample_id!r}.")
            if record.sample_id in self._by_id:
                raise ValueError(f"Duplicate sample_id {record.sample_id!r}.")
            if record.patient_id is not None and not is_filename_safe_id(record.patient_id):
                raise ValueError(f"Unsafe patient_id {record.patient_id!r}.")
            self._by_id[record.sample_id] = record
        self._folds: tuple[FoldSplit, ...] = tuple(folds)
        self._unit = self._resolve_unit(unit)
        self._target_names = {
            str(key): [str(name) for name in names] for key, names in (target_names or {}).items()
        }
        self._validate_targets(dtypes or {})
        self._validate_folds(allow_missing_test=allow_missing_test)
        self._validate_no_unit_leakage()
        self._warn_on_missing_class_coverage()

    # --- construction --------------------------------------------------------------- #

    @classmethod
    def from_frames(
        cls,
        records_df: pd.DataFrame,
        splits_df: pd.DataFrame | None = None,
        *,
        targets: Sequence[str] | Mapping[str, str] | None = None,
        dtypes: Mapping[str, type] | None = None,
        unit: str | None = None,
        allow_missing_test: bool = False,
        sidecar_dir: str | Path | None = None,
    ) -> "Cohort":
        """Build a cohort from a records table and a splits table.

        ``targets`` maps target keys to columns (``{"label": "diagnosis"}``) or lists
        keys whose columns share their names. The ``expression`` key may instead be
        served from the ``target_index`` column and the ``targets.npy`` / ``genes.json``
        sidecars in ``sidecar_dir``. ``splits_df`` may be omitted when ``records_df``
        carries ``split`` (and optionally ``fold``) columns.
        """
        validate_sample_ids(records_df, what="records table")
        validate_patient_ids(records_df)
        target_columns = _resolve_targets(targets)
        frame = records_df.copy()
        target_names: dict[str, list[str]] = {}
        for key, column in target_columns.items():
            if column in frame.columns:
                continue
            if key == "expression" and EXPRESSION_INDEX_COLUMN in frame.columns:
                if sidecar_dir is None:
                    raise ValueError(
                        "Target 'expression' from a target_index column needs sidecar_dir "
                        "(the directory holding targets.npy and genes.json)."
                    )
                frame[column], target_names[key] = _expression_column(frame, Path(sidecar_dir))
                continue
            raise ValueError(
                f"Target {key!r} needs column {column!r}, which the records table lacks. "
                f"Available: {list(records_df.columns)}"
            )
        splits = _split_frame(records_df, splits_df)
        known = set(frame["sample_id"].astype(str))
        unknown = sorted(set(splits["sample_id"].astype(str)) - known)
        if unknown:
            raise ValueError(f"Unknown sample_id(s) in splits: {unknown}")

        excluded = _IDENTITY_COLUMNS | SPLIT_COLUMNS | MANIFEST_COLUMNS | set(target_columns.values())
        metadata_columns = [column for column in frame.columns if column not in excluded]
        records: list[SampleRecord] = []
        for _, row in frame.iterrows():
            sample_id = str(row["sample_id"])
            record_targets: dict[str, Any] = {}
            for key, column in target_columns.items():
                value = row[column]
                if _is_blank(value):
                    raise ValueError(
                        f"Target {key!r} (column {column!r}) is blank for sample {sample_id!r}."
                    )
                record_targets[key] = value.item() if isinstance(value, np.generic) else value
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    targets=record_targets,
                    patient_id=optional_text(row, "patient_id"),
                    metadata={column: row[column] for column in metadata_columns},
                )
            )
        return cls(
            records,
            _folds_from_frame(splits),
            unit=unit,
            dtypes=dtypes,
            allow_missing_test=allow_missing_test,
            target_names=target_names,
        )

    @classmethod
    def from_csv(
        cls,
        records_csv: str | Path,
        splits_csv: str | Path | None = None,
        *,
        targets: Sequence[str] | Mapping[str, str] | None = None,
        dtypes: Mapping[str, type] | None = None,
        unit: str | None = None,
        allow_missing_test: bool = False,
    ) -> "Cohort":
        """:meth:`from_frames` over CSV files; expression sidecars sit beside ``records_csv``."""
        records_csv = Path(records_csv)
        return cls.from_frames(
            pd.read_csv(records_csv),
            None if splits_csv is None else pd.read_csv(splits_csv),
            targets=targets,
            dtypes=dtypes,
            unit=unit,
            allow_missing_test=allow_missing_test,
            sidecar_dir=records_csv.parent,
        )

    # --- access --------------------------------------------------------------------- #

    @property
    def records(self) -> tuple[SampleRecord, ...]:
        return self._records

    @property
    def sample_ids(self) -> list[str]:
        return [record.sample_id for record in self._records]

    def record(self, sample_id: str) -> SampleRecord:
        try:
            return self._by_id[sample_id]
        except KeyError:
            raise KeyError(f"Unknown sample_id {sample_id!r}.") from None

    def __contains__(self, sample_id: object) -> bool:
        return sample_id in self._by_id

    def __len__(self) -> int:
        return len(self._records)

    @property
    def folds(self) -> tuple[FoldSplit, ...]:
        return self._folds

    @property
    def num_folds(self) -> int:
        return len(self._folds)

    @property
    def unit(self) -> str:
        return self._unit

    @property
    def target_keys(self) -> tuple[str, ...]:
        if not self._records:
            return ()
        return tuple(self._records[0].targets)

    @property
    def target_names(self) -> dict[str, list[str]]:
        """Per-key component names for vector targets (the gene list of ``expression``)."""
        return {key: list(names) for key, names in self._target_names.items()}

    @property
    def has_patient_ids(self) -> bool:
        return any(record.patient_id is not None for record in self._records)

    # --- transforms ----------------------------------------------------------------- #

    def with_test_from_tune(self) -> "Cohort":
        """A cohort whose folds report on their tune split (``tune_is_test``)."""
        folds: list[FoldSplit] = []
        for index, fold in enumerate(self._folds):
            if fold.tests:
                raise ValueError(
                    f"Fold {index} provides both a tune and a test split; with_test_from_tune "
                    "reuses tune for test reporting, so drop one of them."
                )
            if not fold.tune:
                raise ValueError(f"Fold {index} has no tune split to reuse as test.")
            logger.warning(
                "Fold %d has no test split; reporting 'test' metrics on the tune samples.", index
            )
            folds.append(replace(fold, tests={"test": fold.tune}, test_from_tune=True))
        return Cohort(self._records, folds, unit=self._unit, target_names=self._target_names)

    def collapse(self, unit: str) -> "Cohort":
        """One record per ``unit`` member group, carrying the group's shared targets.

        Members of a group must agree on every target; the first member's metadata is
        kept. Folds are rewritten onto unit ids in order of first appearance.
        """
        if unit == "sample_id":
            return self
        if unit != "patient_id":
            raise ValueError(f"Unknown sampling unit {unit!r}; expected 'patient_id' or 'sample_id'.")
        groups: dict[str, list[SampleRecord]] = {}
        for record in self._records:
            if record.patient_id is None:
                raise ValueError(
                    f"Sample {record.sample_id!r} has no patient_id; every record needs one "
                    "to collapse the cohort to patients."
                )
            groups.setdefault(record.patient_id, []).append(record)
        records: list[SampleRecord] = []
        for patient_id, members in groups.items():
            head = members[0]
            for member in members[1:]:
                if not targets_equal(head.targets, member.targets):
                    raise ValueError(
                        f"Patient {patient_id!r} has inconsistent targets across its samples "
                        f"({head.sample_id!r}: {head.targets} vs {member.sample_id!r}: "
                        f"{member.targets}); members of a unit must agree."
                    )
            records.append(
                SampleRecord(
                    sample_id=patient_id,
                    targets=dict(head.targets),
                    patient_id=patient_id,
                    metadata=dict(head.metadata),
                )
            )
        unit_of = {record.sample_id: record.patient_id for record in self._records}

        def units(ids: tuple[str, ...]) -> tuple[str, ...]:
            return tuple(dict.fromkeys(unit_of[sid] for sid in ids))

        folds = [
            FoldSplit(
                train=units(fold.train),
                tune=units(fold.tune),
                tests={name: units(ids) for name, ids in fold.tests.items()},
                test_from_tune=fold.test_from_tune,
            )
            for fold in self._folds
        ]
        return Cohort(
            records,
            folds,
            unit="sample_id",
            allow_missing_test=True,
            target_names=self._target_names,
        )

    # --- validation ----------------------------------------------------------------- #

    def _resolve_unit(self, unit: str | None) -> str:
        if unit is None:
            return "patient_id" if self.has_patient_ids else "sample_id"
        if unit not in ("patient_id", "sample_id"):
            raise ValueError(f"Unknown sampling unit {unit!r}; expected 'patient_id' or 'sample_id'.")
        if unit == "patient_id":
            missing = [r.sample_id for r in self._records if r.patient_id is None]
            if missing:
                raise ValueError(
                    f"unit='patient_id' but {len(missing)} record(s) have no patient_id: "
                    f"{missing[:20]}{' ...' if len(missing) > 20 else ''}."
                )
        return unit

    def _validate_targets(self, dtypes: Mapping[str, type]) -> None:
        keys = self.target_keys
        for record in self._records:
            if tuple(record.targets) != keys:
                raise ValueError(
                    f"Sample {record.sample_id!r} declares targets {tuple(record.targets)}, "
                    f"but the cohort's targets are {keys}."
                )
            for key, value in record.targets.items():
                if _is_blank(value):
                    raise ValueError(f"Target {key!r} is blank for sample {record.sample_id!r}.")
                if key in dtypes:
                    _check_dtype(key, value, dtypes[key], record.sample_id)

    def _validate_folds(self, *, allow_missing_test: bool) -> None:
        for index, fold in enumerate(self._folds):
            seen: set[str] = set()
            tests = {} if fold.test_from_tune else fold.tests
            for sample_id in (*fold.train, *fold.tune, *(sid for ids in tests.values() for sid in ids)):
                if sample_id not in self._by_id:
                    raise ValueError(f"Unknown sample_id {sample_id!r} in fold {index}.")
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample_id {sample_id!r} in fold {index}.")
                seen.add(sample_id)
            if not fold.tests and not allow_missing_test:
                raise ValueError(
                    f"Fold {index} must contain at least one test split (a split name "
                    "starting with 'test'), or be derived with with_test_from_tune()."
                )

    def _validate_no_unit_leakage(self) -> None:
        if self._unit != "patient_id":
            return
        unit_of = {record.sample_id: record.patient_id for record in self._records}
        for index, fold in enumerate(self._folds):
            tests = {} if fold.test_from_tune else fold.tests
            membership: dict[str, set[str]] = {}
            for split_name, ids in (("train", fold.train), ("tune", fold.tune), *tests.items()):
                for sample_id in ids:
                    membership.setdefault(unit_of[sample_id], set()).add(split_name)
            leaked = {unit: splits for unit, splits in membership.items() if len(splits) > 1}
            if leaked:
                details = "; ".join(
                    f"patient {unit!r} in {sorted(splits)}" for unit, splits in sorted(leaked.items())
                )
                raise ValueError(
                    f"Leakage in fold {index}: {details}. All samples of a patient must share "
                    "one split."
                )

    def _warn_on_missing_class_coverage(self) -> None:
        if "label" not in self.target_keys:
            return
        labels = {record.sample_id: record.targets["label"] for record in self._records}
        if any(isinstance(value, (np.ndarray, list, tuple)) for value in labels.values()):
            return
        classes = {str(value) for value in labels.values()}
        if len(classes) < 2:
            return
        for index, fold in enumerate(self._folds):
            for split_name, ids in (("train", fold.train), ("tune", fold.tune), *fold.tests.items()):
                if not ids:
                    continue
                present = {str(labels[sid]) for sid in ids}
                missing = sorted(classes - present)
                if missing:
                    logger.warning(
                        "Fold %d split '%s' has no samples for class(es) %s (present: %s). "
                        "Threshold-free metrics such as AUROC are undefined on a "
                        "single-class split.",
                        index,
                        split_name,
                        missing,
                        sorted(present),
                    )
