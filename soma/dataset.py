"""Dataset and Splits — load dataset.csv and user-provided splits.csv."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_DATASET_COLUMNS = {"sample_id", "image_path", "label"}
KNOWN_DATASET_COLUMNS = REQUIRED_DATASET_COLUMNS | {"mask_path", "patient_id"}
REQUIRED_SPLITS_COLUMNS = {"fold", "sample_id", "split"}
VALID_SPLIT_NAMES = {"train", "tune", "test"}


@dataclass(frozen=True)
class SampleRecord:
    """A single sample (one slide) with its path and label."""

    sample_id: str
    image_path: Path
    label: str | int
    mask_path: Path | None = None
    patient_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Dataset:
    """Loads a dataset from a CSV with columns: sample_id, image_path, label.

    Optional columns: mask_path. Extra columns become metadata.
    """

    def __init__(self, dataset_csv: str | Path) -> None:
        self._path = Path(dataset_csv)
        df = pd.read_csv(self._path)
        self._validate_columns(df)
        self._samples = self._build_samples(df)
        self._label_map = self._build_label_map()

    def _validate_columns(self, df: pd.DataFrame) -> None:
        if "tissue_mask_path" in df.columns:
            raise ValueError("Use 'mask_path' instead of 'tissue_mask_path'.")
        for col in REQUIRED_DATASET_COLUMNS:
            if col not in df.columns:
                msg = f"Required column '{col}' not found. Available: {list(df.columns)}"
                raise ValueError(msg)
        if df["sample_id"].duplicated().any():
            dupes = df["sample_id"][df["sample_id"].duplicated()].tolist()
            msg = f"Duplicate sample_id values: {dupes}"
            raise ValueError(msg)

    def _build_samples(self, df: pd.DataFrame) -> dict[str, SampleRecord]:
        meta_columns = [c for c in df.columns if c not in KNOWN_DATASET_COLUMNS]
        samples: dict[str, SampleRecord] = {}
        for _, row in df.iterrows():
            sid = str(row["sample_id"])
            mask_path = (
                Path(str(row["mask_path"]))
                if "mask_path" in row.index and pd.notna(row.get("mask_path"))
                else None
            )
            patient_id = (
                str(row["patient_id"])
                if "patient_id" in row.index and pd.notna(row.get("patient_id"))
                else None
            )
            metadata = {c: row[c] for c in meta_columns}
            samples[sid] = SampleRecord(
                sample_id=sid,
                image_path=Path(str(row["image_path"])),
                label=row["label"],
                mask_path=mask_path,
                patient_id=patient_id,
                metadata=metadata,
            )
        return samples

    def _build_label_map(self) -> dict[str | int, int]:
        unique_labels = sorted(set(r.label for r in self._samples.values()))
        return {label: idx for idx, label in enumerate(unique_labels)}

    @property
    def samples(self) -> dict[str, SampleRecord]:
        return self._samples

    @property
    def sample_ids(self) -> list[str]:
        return list(self._samples.keys())

    @property
    def label_map(self) -> dict[str | int, int]:
        """Sorted unique labels → integer encoding."""
        return self._label_map

    @property
    def num_classes(self) -> int:
        return len(self._label_map)

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
    def patient_label_map(self) -> dict[str, str | int]:
        """Map patient_id to label. Validates all slides per patient share the same label.

        Raises ValueError if patient_ids are missing or labels are inconsistent.
        """
        patient_label: dict[str, str | int] = {}
        for patient_id, records in self.patient_groups.items():
            labels = {r.label for r in records}
            if len(labels) > 1:
                raise ValueError(
                    f"Patient '{patient_id}' has inconsistent labels across slides: {sorted(str(l) for l in labels)}. "
                    "All slides for a patient must share the same label."
                )
            patient_label[patient_id] = records[0].label
        return patient_label


@dataclass(frozen=True)
class FoldSplit:
    """Sample IDs for each split within one fold."""

    train: tuple[str, ...]
    tune: tuple[str, ...]
    test: tuple[str, ...]


class Splits:
    """Loads user-provided splits from a CSV with columns: fold, sample_id, split.

    Validates that all sample_ids exist in the dataset, split names are valid
    (train/tune/test), and no sample appears twice within the same fold.
    """

    def __init__(self, splits_csv: str | Path, dataset: Dataset) -> None:
        self._path = Path(splits_csv)
        df = pd.read_csv(self._path)
        self._validate_columns(df)
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
        invalid = set(df["split"]) - VALID_SPLIT_NAMES
        if invalid:
            msg = f"Invalid split name(s): {sorted(invalid)}. Must be one of {sorted(VALID_SPLIT_NAMES)}"
            raise ValueError(msg)

        # Check no duplicate sample within a fold
        for fold_idx, group in df.groupby("fold"):
            dupes = group["sample_id"][group["sample_id"].duplicated()]
            if not dupes.empty:
                msg = f"Duplicate sample_id(s) in fold {fold_idx}: {dupes.tolist()}"
                raise ValueError(msg)

    def _build_folds(self, df: pd.DataFrame) -> list[FoldSplit]:
        folds = []
        for _, group in sorted(df.groupby("fold")):
            train_ids = tuple(
                str(s) for s in group.loc[group["split"] == "train", "sample_id"]
            )
            tune_ids = tuple(
                str(s) for s in group.loc[group["split"] == "tune", "sample_id"]
            )
            test_ids = tuple(
                str(s) for s in group.loc[group["split"] == "test", "sample_id"]
            )
            folds.append(FoldSplit(train=train_ids, tune=tune_ids, test=test_ids))
        return folds

    @property
    def folds(self) -> list[FoldSplit]:
        return self._folds

    @property
    def num_folds(self) -> int:
        return len(self._folds)

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
            for split_name, sample_ids in (
                ("train", fold_split.train),
                ("tune", fold_split.tune),
                ("test", fold_split.test),
            ):
                for sid in sample_ids:
                    pid = sample_to_patient.get(sid)
                    if pid is None:
                        continue
                    patient_splits.setdefault(pid, set()).add(split_name)
            leaked = {
                pid: splits
                for pid, splits in patient_splits.items()
                if len(splits) > 1
            }
            if leaked:
                details = "; ".join(
                    f"patient '{pid}' in {sorted(splits)}"
                    for pid, splits in sorted(leaked.items())
                )
                raise ValueError(
                    f"Patient leakage detected in fold {fold_idx}: {details}. "
                    "All slides for a patient must be assigned to the same split."
                )
