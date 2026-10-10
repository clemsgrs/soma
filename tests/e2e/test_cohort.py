"""One cohort scenario per shape: a dataset CSV enters soma only through the data contracts.

The set scenario feeds a slide cohort through ``Cohort`` + ``ImageManifest``, derives
patient bags with ``group_by`` / ``collapse`` and checks every source against the
conformance suite. The grid scenario feeds a segmentation and a detection table through
``Cohort`` + ``AnnotationManifest`` and reads the dense supervision back through the
target sources. No record ever carries a file path; nothing here is task-typed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from soma.data import (
    AnnotationManifest,
    Cohort,
    ImageManifest,
    LabelMapSource,
    PointSource,
    Points,
    SampleRecord,
    from_arrays,
    from_directory,
    group_by,
    require_coverage,
)
from soma.testing import check_set_source, check_target_source
from tests.e2e.synthetic import make_dense_dataset, make_slide_cohort


def test_set_shape_cohort_from_one_csv_to_patient_bags(tmp_path: Path) -> None:
    slides = make_slide_cohort(tmp_path / "cohort", n=16)
    dataset_csv = slides.manifest("binary_classification")
    frame = pd.read_csv(dataset_csv)
    frame["patient_id"] = [f"p{i // 2}" for i in range(len(frame))]
    # Pairs share a patient, so each pair must share a split: re-deal by patient.
    splits = pd.read_csv(slides.splits_csv).set_index("sample_id")["split"]
    frame["split"] = [splits[frame.loc[i - i % 2, "sample_id"]] for i in range(len(frame))]
    frame["label"] = [int(frame.loc[i - i % 2, "label"]) for i in range(len(frame))]
    frame.to_csv(dataset_csv, index=False)

    cohort = Cohort.from_csv(dataset_csv, targets=["label"])  # split column, no splits.csv
    images = ImageManifest.from_csv(dataset_csv)

    assert cohort.unit == "patient_id"
    assert cohort.sample_ids == images.sample_ids
    record = cohort.record(cohort.sample_ids[0])
    assert isinstance(record, SampleRecord) and set(record.targets) == {"label"}
    assert not hasattr(record, "image_path")
    assert images[record.sample_id].image_path.suffix == ".tif"

    # Features enter through a source, never through the record.
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for index, sample_id in enumerate(cohort.sample_ids):
        torch.save(torch.full((3 + index % 2, 8), float(index)), feature_dir / f"{sample_id}.pt")
    tiles = from_directory(feature_dir)
    check_set_source(tiles)
    require_coverage(tiles, cohort.sample_ids)

    patients = cohort.collapse("patient_id")
    bags = group_by(tiles, cohort, unit="patient_id", how="concat")
    check_set_source(bags)
    assert bags.sample_ids == patients.sample_ids == [f"p{i}" for i in range(8)]
    assert bags.load("p0").shape == (7, 8)
    assert patients.record("p0").targets == cohort.record(cohort.sample_ids[0]).targets
    for fold in patients.folds:
        assert fold.tests
        members = set(fold.train) | set(fold.tune) | {s for ids in fold.tests.values() for s in ids}
        assert members == set(patients.sample_ids)

    vectors = from_arrays({sid: tiles.load(sid).mean(0) for sid in cohort.sample_ids})
    stacked = group_by(vectors, cohort, unit="patient_id", how="stack")
    check_set_source(stacked)
    assert stacked.rank == 2 and stacked.load("p3").shape == (2, 8)


def test_grid_shape_cohort_serves_label_maps_and_points(tmp_path: Path) -> None:
    seg_csv, seg_splits = make_dense_dataset(tmp_path / "seg", kind="segmentation", n=8, size=64)
    det_csv, det_splits = make_dense_dataset(tmp_path / "det", kind="detection", n=8, size=64)

    seg_cohort = Cohort.from_csv(seg_csv, seg_splits)  # dense supervision: no record targets
    assert seg_cohort.target_keys == ()
    assert seg_cohort.folds[0].tests == {"test": tuple(seg_cohort.sample_ids[-2:])}
    annotations = AnnotationManifest.from_csv(seg_csv)
    images = ImageManifest.from_csv(seg_csv)
    assert annotations.has_label_masks and not annotations.has_points
    label_maps = LabelMapSource.from_manifests(
        annotations,
        images,
        size=(64, 64),
        mask_vocabulary={"background": 0, "disc_a": 1, "disc_b": 2},
    )
    check_target_source(label_maps)
    mask = label_maps.load(seg_cohort.sample_ids[0])
    assert mask.shape == (64, 64) and mask.dtype == torch.long
    assert set(torch.unique(mask).tolist()) <= {0, 1, 2}

    det_cohort = Cohort.from_csv(det_csv, det_splits)
    points = PointSource.from_manifest(AnnotationManifest.from_csv(det_csv))
    check_target_source(points)
    for sample_id in det_cohort.folds[0].train:
        loaded = points.load(sample_id)
        assert isinstance(loaded, Points)
        assert loaded.xy.shape == (len(loaded), 2)
        assert set(loaded.classes.tolist()) <= {0, 1}
