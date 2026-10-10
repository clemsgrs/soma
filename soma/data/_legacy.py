"""Legacy bridge: fat records for ``pipeline.py`` and the extraction stack.

**Delete in slice 7 (#581).** Until ``pipeline.py`` is rewritten over the verbs and
``soma.extract`` reads :class:`~soma.data.ImageManifest` directly, the pipeline, the
cache layer, the extractors and the curators still consume one record per sample that
carries file paths next to targets. This module rebuilds that record
(:class:`LegacyRecord`) from the slice-1 contracts — :class:`~soma.data.Cohort`,
:class:`~soma.data.ImageManifest`, :class:`~soma.data.AnnotationManifest` — so no public
type carries a path. Nothing under ``soma/tasks`` or the set-shape training datasets
may import this module.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from soma.data.cohort import (
    EXPRESSION_INDEX_COLUMN,
    MANIFEST_COLUMNS,
    SPLIT_COLUMNS,
    Cohort,
    FoldSplit,
    _expression_column,
)
from soma.data.manifests import AnnotationManifest, ImageManifest
from soma.data.records import SampleRecord, targets_equal
from soma.data.validation import optional_text, validate_patient_ids, validate_sample_ids

__all__ = [
    "legacy_folds_from_csv",
    "bind_detection_targets",
    "bind_segmentation_targets",
    "LegacyFolds",
    "LegacyRecord",
    "LegacySamples",
    "legacy_samples_from_csv",
    "project_folds",
    "records_for_pipeline",
]

_ANNOTATION_COLUMNS = ("label_mask_path", "points_path", "ignore_mask_path")


@dataclass(frozen=True, eq=False)
class LegacyRecord(SampleRecord):
    """A :class:`SampleRecord` plus the file paths ``pipeline.py`` still reads.

    ``image_path`` / ``mask_path`` / ``spacing_at_level_0`` / ``coordinates_path`` come
    from the image manifest; ``label_mask_path`` / ``points_path`` / ``ignore_mask_path``
    from the annotation manifest. ``region`` / ``label_mask_crop_path`` / ``slide_id``
    describe an annotation-sampled ROI (its origin in the parent slide's level-0 frame,
    its stored mask crop, its parent); ``None`` for every non-ROI row.
    """

    image_path: Path | None = None
    mask_path: Path | None = None
    label_mask_path: Path | None = None
    points_path: Path | None = None
    ignore_mask_path: Path | None = None
    spacing_at_level_0: float | None = None
    coordinates_path: Path | None = None
    region: tuple[int, int] | None = None
    label_mask_crop_path: Path | None = None
    slide_id: str | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LegacyRecord):
            return NotImplemented
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if f.name == "targets":
                if not targets_equal(mine, theirs):
                    return False
            elif mine != theirs:
                return False
        return True

    def __hash__(self) -> int:
        return hash((self.sample_id, self.patient_id, self.slide_id, self.region))


class LegacySamples:
    """The ``samples`` / ``sample_ids`` surface the extraction stack still expects."""

    def __init__(
        self,
        records: Iterable[LegacyRecord],
        *,
        target_names: Mapping[str, Sequence[str]] | None = None,
        path: str | Path | None = None,
    ) -> None:
        # The CSV these records came from, for run provenance (None when built in memory).
        self.path = None if path is None else Path(path)
        self._samples: dict[str, LegacyRecord] = {}
        for record in records:
            if record.sample_id in self._samples:
                raise ValueError(f"Duplicate sample_id {record.sample_id!r}.")
            self._samples[record.sample_id] = record
        self._target_names = {k: list(v) for k, v in (target_names or {}).items()}

    @property
    def samples(self) -> dict[str, LegacyRecord]:
        return self._samples

    @property
    def records(self) -> list[LegacyRecord]:
        return list(self._samples.values())

    @property
    def sample_ids(self) -> list[str]:
        return list(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def has_patient_ids(self) -> bool:
        return any(r.patient_id is not None for r in self._samples.values())

    @property
    def supplies_coordinates(self) -> bool:
        return any(r.coordinates_path is not None for r in self._samples.values())

    def with_coordinates(self, coordinates_paths: Mapping[str, Path]) -> "LegacySamples":
        clone = copy.copy(self)
        clone._samples = {
            sid: replace(record, coordinates_path=Path(coordinates_paths[sid]))
            for sid, record in self._samples.items()
        }
        return clone

    @property
    def patient_groups(self) -> dict[str, list[LegacyRecord]]:
        groups: dict[str, list[LegacyRecord]] = {}
        for record in self._samples.values():
            if record.patient_id is None:
                raise ValueError(
                    f"Sample '{record.sample_id}' is missing a patient_id. "
                    "All rows must have a patient_id for patient-level pipelines."
                )
            groups.setdefault(record.patient_id, []).append(record)
        return groups

    @property
    def patient_record_map(self) -> dict[str, LegacyRecord]:
        """One representative record per patient; members must share their targets."""
        record_map: dict[str, LegacyRecord] = {}
        for patient_id, records in self.patient_groups.items():
            for other in records[1:]:
                if not targets_equal(records[0].targets, other.targets):
                    raise ValueError(
                        f"Patient '{patient_id}' has inconsistent targets across slides: "
                        f"{records[0].targets} vs {other.targets}. "
                        "All slides for a patient must share the same targets."
                    )
            record_map[patient_id] = records[0]
        return record_map

    @property
    def target_names(self) -> dict[str, list[str]]:
        return {k: list(v) for k, v in self._target_names.items()}

    @property
    def genes(self) -> list[str]:
        return list(self._target_names.get("expression", []))

    def subset(self, sample_ids: Iterable[str]) -> "LegacySamples":
        wanted = set(sample_ids)
        return LegacySamples(
            (r for sid, r in self._samples.items() if sid in wanted),
            target_names=self._target_names,
            path=self.path,
        )


def _legacy_record(
    record: SampleRecord,
    images: ImageManifest | None,
    annotations: AnnotationManifest | None,
) -> LegacyRecord:
    image = images[record.sample_id] if images is not None and record.sample_id in images else None
    annotation = (
        annotations[record.sample_id]
        if annotations is not None and record.sample_id in annotations
        else None
    )
    return LegacyRecord(
        sample_id=record.sample_id,
        targets=dict(record.targets),
        patient_id=record.patient_id
        if record.patient_id is not None
        else (image.patient_id if image is not None else None),
        metadata=dict(record.metadata),
        image_path=None if image is None else image.image_path,
        mask_path=None if image is None else image.mask_path,
        spacing_at_level_0=None if image is None else image.spacing_at_level_0,
        coordinates_path=None if image is None else image.coordinates_path,
        label_mask_path=None if annotation is None else annotation.label_mask_path,
        points_path=None if annotation is None else annotation.points_path,
        ignore_mask_path=None if annotation is None else annotation.ignore_mask_path,
    )


def records_for_pipeline(
    cohort: Cohort,
    image_manifest: ImageManifest,
    annotation_manifest: AnnotationManifest | None = None,
    *,
    path: str | Path | None = None,
) -> LegacySamples:
    """Join a cohort with its manifests into the fat records ``pipeline.py`` consumes."""
    missing = [sid for sid in cohort.sample_ids if sid not in image_manifest]
    if missing:
        raise ValueError(
            f"{len(missing)} cohort sample(s) have no image manifest row: "
            f"{missing[:20]}{' ...' if len(missing) > 20 else ''}."
        )
    return LegacySamples(
        (_legacy_record(record, image_manifest, annotation_manifest) for record in cohort.records),
        target_names=cohort.target_names,
        path=path,
    )


def legacy_samples_from_csv(
    dataset_csv: str | Path,
    *,
    pixel_mapping: Mapping[str, int] | None = None,
) -> LegacySamples:
    """Fat records straight from a dataset CSV, for extraction without a task.

    Reads the image manifest columns, the annotation columns when present, a ``label``
    column into ``targets["label"]`` when present, an ``expression`` sidecar when a
    ``target_index`` column is present, and the ROI columns a legacy effective dataset
    carries (``region_x`` / ``region_y`` / ``slide_id`` / ``label_mask_crop_path``).
    """
    dataset_csv = Path(dataset_csv)
    frame = pd.read_csv(dataset_csv)
    validate_sample_ids(frame, what="dataset CSV")
    validate_patient_ids(frame)
    images = ImageManifest.from_frame(frame)
    annotations = (
        AnnotationManifest.from_frame(frame, pixel_mapping=pixel_mapping)
        if any(column in frame.columns for column in _ANNOTATION_COLUMNS)
        else None
    )
    target_names: dict[str, list[str]] = {}
    # Without a head to declare target keys, the bridge reads every built-in head's
    # same-named target column that is present.
    target_columns: dict[str, str] = {
        key: key for key in ("label", "value", "time", "event", "bin") if key in frame.columns
    }
    if EXPRESSION_INDEX_COLUMN in frame.columns:
        frame["expression"], target_names["expression"] = _expression_column(
            frame, dataset_csv.parent
        )
        target_columns["expression"] = "expression"
    excluded = {"sample_id", "patient_id", "expression", *target_columns} | SPLIT_COLUMNS | MANIFEST_COLUMNS
    metadata_columns = [c for c in frame.columns if c not in excluded]
    records: list[LegacyRecord] = []
    for _, row in frame.iterrows():
        targets: dict[str, Any] = {}
        for key, column in target_columns.items():
            value = row[column]
            if isinstance(value, np.generic):
                value = value.item()
            if not isinstance(value, (np.ndarray, list)) and pd.isna(value):
                continue
            targets[key] = value
        slim = SampleRecord(
            sample_id=str(row["sample_id"]),
            targets=targets,
            patient_id=optional_text(row, "patient_id"),
            metadata={c: row[c] for c in metadata_columns},
        )
        record = _legacy_record(slim, images, annotations)
        region = None
        if "region_x" in row.index and pd.notna(row.get("region_x")):
            region = (int(row["region_x"]), int(row["region_y"]))
        crop = optional_text(row, "label_mask_crop_path")
        records.append(
            replace(
                record,
                region=region,
                label_mask_crop_path=None if crop is None else Path(crop),
                slide_id=optional_text(row, "slide_id"),
            )
        )
    return LegacySamples(records, target_names=target_names, path=dataset_csv)


def project_folds(folds: Sequence[FoldSplit], samples: LegacySamples) -> list[FoldSplit]:
    """Project fold assignments onto an effective (ROI) sample set.

    Existing sample ids keep their own assignment. A derived ROI with no direct
    assignment inherits exactly one hop through its explicit ``slide_id``.
    """
    projected: list[FoldSplit] = []
    for fold_index, fold in enumerate(folds):
        locations: dict[str, set[tuple[str, str | None]]] = {}
        for sid in fold.train:
            locations.setdefault(sid, set()).add(("train", None))
        for sid in fold.tune:
            locations.setdefault(sid, set()).add(("tune", None))
        for name, ids in fold.tests.items():
            for sid in ids:
                locations.setdefault(sid, set()).add(("test", name))
        selected: dict[str, set[tuple[str, str | None]]] = {}
        for sample_id, record in samples.samples.items():
            direct = locations.get(sample_id, set())
            inherited = (
                locations.get(str(record.slide_id), set()) if record.slide_id is not None else set()
            )
            if direct and inherited and direct != inherited:
                raise ValueError(
                    f"Conflicting split ancestry for sample '{sample_id}' in fold {fold_index}: "
                    f"direct={sorted(direct)}, slide_id '{record.slide_id}'={sorted(inherited)}."
                )
            chosen = direct or inherited
            if not chosen:
                detail = (
                    f"slide_id '{record.slide_id}' has no assignment"
                    if record.slide_id is not None
                    else "the sample has no direct assignment or slide_id"
                )
                raise ValueError(
                    f"Unresolved split ancestry for sample '{sample_id}' in fold {fold_index}: {detail}."
                )
            selected[sample_id] = chosen
        ordered = list(samples.sample_ids)
        projected.append(
            FoldSplit(
                train=tuple(s for s in ordered if ("train", None) in selected.get(s, set())),
                tune=tuple(s for s in ordered if ("tune", None) in selected.get(s, set())),
                tests={
                    name: tuple(s for s in ordered if ("test", name) in selected.get(s, set()))
                    for name in fold.tests
                },
                test_from_tune=fold.test_from_tune,
            )
        )
    return projected


def legacy_folds_from_csv(
    splits_csv: str | Path,
    samples: LegacySamples,
    *,
    tune_is_test: bool = False,
    unit: str | None = None,
) -> "LegacyFolds":
    """Folds for ``samples`` from a splits CSV, validated through :class:`Cohort`."""
    cohort = Cohort(
        [
            SampleRecord(r.sample_id, dict(r.targets), r.patient_id, dict(r.metadata))
            for r in samples.records
        ],
        Cohort.from_frames(
            pd.DataFrame(
                {
                    "sample_id": samples.sample_ids,
                    "patient_id": [r.patient_id for r in samples.records],
                }
            ),
            pd.read_csv(splits_csv),
            unit="sample_id",
            allow_missing_test=True,
        ).folds,
        unit=unit,
        allow_missing_test=tune_is_test,
    )
    if tune_is_test:
        cohort = cohort.with_test_from_tune()
    return LegacyFolds.from_cohort(cohort)


class LegacyFolds:
    """The ``folds`` / ``num_folds`` / ``project`` surface ``pipeline.py`` still expects."""

    def __init__(self, folds: Sequence[FoldSplit]) -> None:
        self._folds = list(folds)

    @classmethod
    def from_cohort(cls, cohort: Cohort) -> "LegacyFolds":
        return cls(cohort.folds)

    @property
    def folds(self) -> list[FoldSplit]:
        return list(self._folds)

    @property
    def num_folds(self) -> int:
        return len(self._folds)

    def project(self, samples: LegacySamples) -> "LegacyFolds":
        return LegacyFolds(project_folds(self._folds, samples))


def bind_segmentation_targets(head, records: Iterable[LegacyRecord]):
    """Bind a label-map source built from legacy records to a ``SegmentationHead``.

    Annotation-sampled ROIs (``region`` set) read their stored mask crops through a
    :class:`~soma.data.CachedLabelMapSource`; pre-cropped tiles read their rasters
    through a :class:`~soma.data.LabelMapSource`.
    """
    from soma.data.targets import CachedLabelMapSource, LabelMapEntry, LabelMapSource

    records = list(records)
    params = head.label_map_read_params
    if records and any(record.region is not None for record in records):
        crops: dict[str, Path] = {}
        for record in records:
            if record.region is None or record.label_mask_crop_path is None:
                raise ValueError(
                    f"segmentation sample '{record.sample_id}': a slide-manifest ROI needs its "
                    "stored mask crop (label_mask_crop_path), which ROI sampling writes."
                )
            crops[record.sample_id] = record.label_mask_crop_path
        source = CachedLabelMapSource(
            crops,
            size=params["size"],
            label_remap=params["label_remap"],
            ignore_index=head.ignore_index,
        )
    else:
        entries = []
        for record in records:
            if record.label_mask_path is None:
                raise ValueError(
                    f"segmentation sample '{record.sample_id}' has no label_mask_path"
                )
            entries.append(
                LabelMapEntry(
                    sample_id=record.sample_id,
                    label_mask_path=record.label_mask_path,
                    reference_path=record.image_path,
                    spacing_at_level_0=record.spacing_at_level_0,
                )
            )
        source = LabelMapSource(entries, **params)
    head.bind_targets(source)
    return source


def bind_detection_targets(head, records: Iterable[LegacyRecord]):
    """Bind a :class:`~soma.data.PointSource` built from legacy records to a ``DetectionHead``."""
    from soma.data.targets import PointEntry, PointSource

    entries = []
    for record in records:
        if record.points_path is None:
            raise ValueError(f"detection sample '{record.sample_id}' has no points_path")
        entries.append(
            PointEntry(
                sample_id=record.sample_id,
                points_path=record.points_path,
                ignore_mask_path=record.ignore_mask_path,
            )
        )
    source = PointSource(entries)
    head.bind_targets(source)
    return source
