"""ImageManifest / AnnotationManifest: task-neutral file tables (design §4.3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from soma.data import AnnotationManifest, Cohort, ImageManifest


def _single_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "patient_id": ["p", "q"],
            "image_path": ["/img/a.tif", "/img/b.tif"],
            "mask_path": ["/tissue/a.png", None],
            "label_mask_path": ["/ann/a.png", "/ann/b.png"],
            "spacing_at_level_0": [0.25, None],
            "label": [0, 1],
            "split": ["train", "test"],
            "site": ["x", "y"],
        }
    )
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)
    return path


def test_one_csv_feeds_image_manifest_annotation_manifest_and_cohort(tmp_path: Path) -> None:
    csv = _single_csv(tmp_path)

    images = ImageManifest.from_csv(csv)
    annotations = AnnotationManifest.from_csv(csv)
    cohort = Cohort.from_csv(csv, targets=["label"])

    assert images.sample_ids == ["a", "b"] == annotations.sample_ids == cohort.sample_ids
    entry = images["a"]
    assert entry.image_path == Path("/img/a.tif")
    assert entry.mask_path == Path("/tissue/a.png")
    assert entry.spacing_at_level_0 == 0.25
    assert entry.patient_id == "p"
    assert images["b"].mask_path is None and images["b"].spacing_at_level_0 is None
    assert annotations["b"].label_mask_path == Path("/ann/b.png")
    assert annotations["b"].points_path is None
    assert cohort.record("a").metadata == {"site": "x"}


def test_image_manifest_validates_only_its_own_columns(tmp_path: Path) -> None:
    pd.DataFrame({"sample_id": ["a"], "image_path": ["/img/a.tif"]}).to_csv(
        tmp_path / "images.csv", index=False
    )
    manifest = ImageManifest.from_csv(tmp_path / "images.csv")
    assert manifest["a"].patient_id is None
    assert not manifest.supplies_coordinates

    with pytest.raises(ValueError, match="image_path"):
        ImageManifest.from_frame(pd.DataFrame({"sample_id": ["a"], "label": [1]}))


def test_image_manifest_rejects_retired_and_unsafe_inputs() -> None:
    with pytest.raises(ValueError, match="mask_path"):
        ImageManifest.from_frame(
            pd.DataFrame({"sample_id": ["a"], "image_path": ["x"], "tissue_mask_path": ["m"]})
        )
    with pytest.raises(ValueError, match="spacing_at_level_0"):
        ImageManifest.from_frame(
            pd.DataFrame({"sample_id": ["a"], "image_path": ["x"], "spacing_at_level_0": [-1.0]})
        )
    with pytest.raises(ValueError, match="Unsafe"):
        ImageManifest.from_frame(pd.DataFrame({"sample_id": ["../a"], "image_path": ["x"]}))
    with pytest.raises(ValueError, match="image_path"):
        ImageManifest.from_frame(pd.DataFrame({"sample_id": ["a"], "image_path": [None]}))


def test_coordinates_column_is_all_or_nothing() -> None:
    with pytest.raises(ValueError, match="coordinates_path"):
        ImageManifest.from_frame(
            pd.DataFrame(
                {
                    "sample_id": ["a", "b"],
                    "image_path": ["x", "y"],
                    "coordinates_path": ["a.npz", None],
                }
            )
        )
    manifest = ImageManifest.from_frame(
        pd.DataFrame({"sample_id": ["a"], "image_path": ["x"], "coordinates_path": ["a.npz"]})
    )
    assert manifest.supplies_coordinates
    assert manifest["a"].coordinates_path == Path("a.npz")
    rebound = manifest.with_coordinates({"a": Path("/run/a.npz")})
    assert rebound["a"].coordinates_path == Path("/run/a.npz")
    assert manifest["a"].coordinates_path == Path("a.npz")


def test_annotation_manifest_requires_an_annotation_column() -> None:
    with pytest.raises(ValueError, match="label_mask_path.*points_path"):
        AnnotationManifest.from_frame(pd.DataFrame({"sample_id": ["a"], "image_path": ["x"]}))


def test_annotation_manifest_reads_points_and_ignore_masks_with_pixel_mapping() -> None:
    manifest = AnnotationManifest.from_frame(
        pd.DataFrame(
            {
                "sample_id": ["a", "b"],
                "points_path": ["a.csv", "b.csv"],
                "ignore_mask_path": ["a_ign.png", None],
            }
        ),
        pixel_mapping={"background": 0, "tumor": 1},
    )
    assert manifest["a"].points_path == Path("a.csv")
    assert manifest["a"].ignore_mask_path == Path("a_ign.png")
    assert manifest["b"].ignore_mask_path is None
    assert manifest.pixel_mapping == {"background": 0, "tumor": 1}
    assert manifest.has_points and not manifest.has_label_masks


def test_annotation_manifest_refuses_a_pre_rename_segmentation_table() -> None:
    with pytest.raises(ValueError, match="pre-rename"):
        AnnotationManifest.from_frame(
            pd.DataFrame({"sample_id": ["a"], "image_path": ["x"], "mask_path": ["m.png"]})
        )


def test_blank_annotation_path_is_rejected() -> None:
    with pytest.raises(ValueError, match="label_mask_path.*b"):
        AnnotationManifest.from_frame(
            pd.DataFrame({"sample_id": ["a", "b"], "label_mask_path": ["a.png", None]})
        )
