"""The pipeline's fat-record bridge: everything that still reads ``LegacyRecord`` fields.

The builders consume the legacy records ``records_for_pipeline`` derives from a
:class:`~soma.data.Cohort` and its manifests (paths, ROI regions, spacing) through the
helpers here, so the dependency on the bridge sits in one module. Slice 7 deletes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from soma.class_scheme import resolve_classes
from soma.config import TaskConfig
from soma.data import (
    AnnotationManifest,
    CachedLabelMapSource,
    Cohort,
    ImageManifest,
    LabelMapEntry,
    LabelMapSource,
    PointEntry,
    PointSource,
    TileImageSource,
)
from soma.data._legacy import LegacyFolds, LegacyRecord, LegacySamples, records_for_pipeline
from soma.data.manifests import ImageEntry

__all__ = [
    "LegacyFolds",
    "LegacyRecord",
    "LegacySamples",
    "detection_points",
    "flat_tile_images",
    "legacy_inputs",
    "overlay_images",
    "patient_ids_for_records",
    "records_for_sample_ids",
    "segmentation_label_maps",
    "segmentation_run_records",
]


def legacy_inputs(
    cohort: Cohort,
    image_manifest: ImageManifest,
    annotation_manifest: AnnotationManifest | None,
    *,
    path: str | Path,
) -> tuple[LegacySamples, LegacyFolds]:
    """The fat records and folds the builders still consume."""
    dataset = records_for_pipeline(cohort, image_manifest, annotation_manifest, path=path)
    return dataset, LegacyFolds.from_cohort(cohort)


def records_for_sample_ids(
    dataset: LegacySamples, sample_ids: Sequence[str], allowed_sample_ids: set[str]
) -> list[LegacyRecord]:
    return [dataset.samples[sample_id] for sample_id in sample_ids if sample_id in allowed_sample_ids]


def patient_ids_for_records(records: list[LegacyRecord]) -> list[str]:
    seen: set[str] = set()
    patient_ids: list[str] = []
    for record in records:
        if record.patient_id is None:
            raise ValueError(
                f"Sample '{record.sample_id}' has no patient_id. "
                "All samples must have a patient_id for dataset_type='patient'."
            )
        if record.patient_id not in seen:
            seen.add(record.patient_id)
            patient_ids.append(record.patient_id)
    return patient_ids


def segmentation_run_records(dataset: LegacySamples, splits: LegacyFolds) -> list[LegacyRecord]:
    """Every ROI named by the run's folds, once, in manifest order."""
    sample_ids = {
        sample_id
        for fold_split in splits.folds
        for sample_id in (
            *fold_split.train,
            *fold_split.tune,
            *(sample_id for ids in fold_split.tests.values() for sample_id in ids),
        )
    }
    return [record for sample_id, record in dataset.samples.items() if sample_id in sample_ids]


def segmentation_label_maps(head, records: Sequence[LegacyRecord]):
    """The label-map target source of a segmentation run's records.

    Annotation-sampled ROIs (``region`` set) read the mask crops ROI sampling stored
    through a :class:`~soma.data.CachedLabelMapSource`; pre-cropped tiles read their
    rasters through a :class:`~soma.data.LabelMapSource`. Both fold the class scheme in.
    """
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
        return CachedLabelMapSource(
            crops, size=params["size"], label_remap=params["label_remap"], ignore_index=head.ignore_index
        )
    entries = []
    for record in records:
        if record.label_mask_path is None:
            raise ValueError(f"segmentation sample '{record.sample_id}' has no label_mask_path")
        entries.append(
            LabelMapEntry(
                sample_id=record.sample_id,
                label_mask_path=record.label_mask_path,
                reference_path=record.image_path,
                spacing_at_level_0=record.spacing_at_level_0,
            )
        )
    return LabelMapSource(entries, **params)


def detection_points(task: TaskConfig, records: Sequence[LegacyRecord]):
    """``(num_classes, class names, PointSource)`` of a detection run's records."""
    num_classes, class_names, class_remap, dropped = resolve_classes(
        task.params, excluded_key="drop", subject="detection"
    )
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
    return num_classes, class_names, PointSource(entries, class_remap=class_remap, drop=dropped)


def _image_entries(records: Sequence[LegacyRecord]) -> list[ImageEntry]:
    return [
        ImageEntry(
            sample_id=record.sample_id,
            image_path=record.image_path,
            spacing_at_level_0=record.spacing_at_level_0,
        )
        for record in records
    ]


def flat_tile_images(
    records: Sequence[LegacyRecord],
    *,
    size,
    spacing_um: float | None = None,
    spacing_policy: str = "strict",
    backend: str = "auto",
    tolerance: float = 0.05,
):
    """A :class:`~soma.data.TileImageSource` over pre-cropped tiles, or ``None`` for
    slide-manifest ROIs (their pixels are windows of a whole slide)."""
    if any(record.region is not None for record in records):
        return None
    return TileImageSource(
        _image_entries(records),
        size=size,
        spacing_um=spacing_um,
        spacing_policy=spacing_policy,
        backend=backend,
        tolerance=tolerance,
    )


def overlay_images(records: Sequence[LegacyRecord], *, size, preprocessing):
    """The pixels :class:`~soma.evaluation.SaveOverlays` blends over: the flat tiles, or
    each slide-manifest ROI's window of its whole slide at the run spacing."""
    spacing_um = preprocessing.requested_spacing_um if preprocessing is not None else None
    spacing_policy = preprocessing.spacing_policy if preprocessing is not None else "strict"
    backend = preprocessing.backend if preprocessing is not None else "auto"
    tolerance = float(preprocessing.tolerance) if preprocessing is not None else 0.05
    if spacing_um is None or not any(record.region is not None for record in records):
        return flat_tile_images(
            records,
            size=size,
            spacing_um=spacing_um,
            spacing_policy=spacing_policy,
            backend=backend,
            tolerance=tolerance,
        )
    from soma.data.images import RegionImageSource

    return RegionImageSource(
        _image_entries(records),
        regions={record.sample_id: record.region for record in records},
        size=size,
        spacing_um=spacing_um,
        spacing_policy=spacing_policy,
        backend=backend,
        tolerance=tolerance,
    )
