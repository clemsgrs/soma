"""Tests for EVA patch-level dataset curation."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soma.curation.eva import (
    _gleason_arvaniti_core_archives,
    _gleason_arvaniti_raw_present,
    curate_eva_patch_dataset,
)
from soma.dataset import Dataset, Splits


CRC_CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
BREAKHIS_CLASS_NAMES = ("TA", "MC", "F", "DC")
GLEASON_ARVANITI_CLASS_NAMES = ("benign", "gleason_3", "gleason_4", "gleason_5")

# The raw Harvard Dataverse download (DOI 10.7910/DVN/OCYCMP), if present on this host.
GLEASON_REAL_ROOT = Path(
    "/data/pathology/projects/vlfm/data/vision/tile_level_datasets/datasets/GleasonArvaniti"
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _make_gleason_core(directory: Path, name: str, value: int = 100) -> None:
    """An 800x800 RGB TMA core (mean well under the white cutoff so its one patch is kept)."""
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((800, 800, 3), value, np.uint8)).save(directory / f"{name}.jpg")


def _make_gleason_mask(directory: Path, name: str, class_id: int) -> None:
    """A single-grade mask; real masks are palette PNGs, but asarray yields the same ids."""
    directory.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((800, 800), class_id, np.uint8), mode="L").save(
        directory / f"mask_{name}.png"
    )


def _build_gleason_extracted_raw(raw_root: Path) -> None:
    """Raw layout as left by unpacking the tarballs: per-section core dirs + a masks dir."""
    _make_gleason_core(raw_root / "ZT76_39_A", "ZT76_39_A_1_1")  # -> val cohort
    _make_gleason_core(raw_root / "ZT111_4_A", "ZT111_4_A_1_1")  # -> train cohort
    masks = raw_root / "Gleason_masks_train"
    _make_gleason_mask(masks, "ZT76_39_A_1_1", 2)  # gleason_4
    _make_gleason_mask(masks, "ZT111_4_A_1_1", 0)  # benign


def test_curate_mhist_preserves_eva_binary_targets_and_reserves_test(tmp_path: Path):
    raw_root = tmp_path / "mhist_raw"
    (raw_root / "images").mkdir(parents=True)
    annotations = pd.DataFrame(
        {
            "Image Name": [
                "ssa_train_0.png",
                "ssa_train_1.png",
                "hp_train_0.png",
                "hp_train_1.png",
                "ssa_test.png",
                "hp_test.png",
            ],
            "Majority Vote Label": ["SSA", "SSA", "HP", "HP", "SSA", "HP"],
            "Partition": ["train", "train", "train", "train", "test", "test"],
        }
    )
    annotations.to_csv(raw_root / "annotations.csv", index=False)
    for image_name in annotations["Image Name"]:
        _touch(raw_root / "images" / image_name)

    manifest = curate_eva_patch_dataset(
        "mhist",
        raw_root,
        tmp_path / "curated",
        tune_fraction=0.5,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv).sort_values("sample_id")
    splits_df = pd.read_csv(manifest.splits_csv).sort_values("sample_id")

    assert dataset_df[["sample_id", "label", "class_name", "eva_split"]].to_dict("records") == [
        {
            "sample_id": "mhist_images_hp_test",
            "label": 1,
            "class_name": "HP",
            "eva_split": "test",
        },
        {
            "sample_id": "mhist_images_hp_train_0",
            "label": 1,
            "class_name": "HP",
            "eva_split": "train",
        },
        {
            "sample_id": "mhist_images_hp_train_1",
            "label": 1,
            "class_name": "HP",
            "eva_split": "train",
        },
        {
            "sample_id": "mhist_images_ssa_test",
            "label": 0,
            "class_name": "SSA",
            "eva_split": "test",
        },
        {
            "sample_id": "mhist_images_ssa_train_0",
            "label": 0,
            "class_name": "SSA",
            "eva_split": "train",
        },
        {
            "sample_id": "mhist_images_ssa_train_1",
            "label": 0,
            "class_name": "SSA",
            "eva_split": "train",
        },
    ]
    assert splits_df.to_dict("records") == [
        {"sample_id": "mhist_images_hp_test", "split": "test", "fold": 0},
        {"sample_id": "mhist_images_hp_train_0", "split": "train", "fold": 0},
        {"sample_id": "mhist_images_hp_train_1", "split": "tune", "fold": 0},
        {"sample_id": "mhist_images_ssa_test", "split": "test", "fold": 0},
        {"sample_id": "mhist_images_ssa_train_0", "split": "train", "fold": 0},
        {"sample_id": "mhist_images_ssa_train_1", "split": "tune", "fold": 0},
    ]

    dataset = Dataset(manifest.dataset_csv)
    splits = Splits(manifest.splits_csv, dataset)
    assert splits.num_folds == 1


def test_curate_mhist_allows_full_train_for_eva_reproduction(tmp_path: Path):
    raw_root = tmp_path / "mhist_raw"
    (raw_root / "images").mkdir(parents=True)
    annotations = pd.DataFrame(
        {
            "Image Name": [
                "ssa_train_0.png",
                "ssa_train_1.png",
                "hp_train_0.png",
                "hp_train_1.png",
                "ssa_test.png",
                "hp_test.png",
            ],
            "Majority Vote Label": ["SSA", "SSA", "HP", "HP", "SSA", "HP"],
            "Partition": ["train", "train", "train", "train", "test", "test"],
        }
    )
    annotations.to_csv(raw_root / "annotations.csv", index=False)
    for image_name in annotations["Image Name"]:
        _touch(raw_root / "images" / image_name)

    manifest = curate_eva_patch_dataset(
        "mhist",
        raw_root,
        tmp_path / "curated",
        tune_fraction=0.0,
    )

    splits_df = pd.read_csv(manifest.splits_csv)

    assert len(splits_df[splits_df["split"] == "train"]) == 4
    assert len(splits_df[splits_df["split"] == "tune"]) == 0
    assert len(splits_df[splits_df["split"] == "test"]) == 2

    dataset = Dataset(manifest.dataset_csv)
    splits = Splits(manifest.splits_csv, dataset)
    assert splits.folds[0].tune == ()


def test_curate_breakhis_uses_eva_40x_patient_split(tmp_path: Path):
    raw_root = tmp_path / "breakhis_raw"
    train_patient = "00001"
    val_patient = "18842D"
    for class_name in BREAKHIS_CLASS_NAMES:
        for patient_id in (train_patient, val_patient):
            _touch(
                raw_root
                / "BreaKHis_v1"
                / "histology_slides"
                / "breast"
                / "benign"
                / "SOB"
                / class_name
                / patient_id
                / "40X"
                / f"SOB_{class_name}-14-{patient_id}-001.png"
            )
        _touch(
            raw_root
            / "BreaKHis_v1"
            / "histology_slides"
            / "breast"
            / "benign"
            / "SOB"
            / class_name
            / train_patient
            / "100X"
            / f"SOB_{class_name}-14-{train_patient}-100.png"
        )
    _touch(
        raw_root
        / "BreaKHis_v1"
        / "histology_slides"
        / "breast"
        / "benign"
        / "SOB"
        / "PT"
        / train_patient
        / "40X"
        / f"SOB_PT-14-{train_patient}-001.png"
    )

    manifest = curate_eva_patch_dataset(
        "breakhis",
        raw_root,
        tmp_path / "curated",
        tune_fraction=0.0,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv)
    splits_df = pd.read_csv(manifest.splits_csv)

    assert len(dataset_df) == 8
    assert set(dataset_df["class_name"]) == set(BREAKHIS_CLASS_NAMES)
    assert set(dataset_df.loc[dataset_df["class_name"] == "TA", "label"]) == {0}
    assert set(dataset_df.loc[dataset_df["class_name"] == "MC", "label"]) == {1}
    assert set(dataset_df.loc[dataset_df["class_name"] == "F", "label"]) == {2}
    assert set(dataset_df.loc[dataset_df["class_name"] == "DC", "label"]) == {3}
    assert len(splits_df[splits_df["split"] == "train"]) == 4
    assert len(splits_df[splits_df["split"] == "test"]) == 4
    assert len(splits_df[splits_df["split"] == "tune"]) == 0
    assert dataset_df["image_path"].str.contains("/40X/").all()

    dataset = Dataset(manifest.dataset_csv)
    splits = Splits(manifest.splits_csv, dataset)
    assert splits.folds[0].tune == ()


def test_curate_crc_splits_eva_train_and_reserves_eva_val_as_test(tmp_path: Path):
    raw_root = tmp_path / "crc_raw"
    for class_name in CRC_CLASS_NAMES:
        for i in range(3):
            _touch(raw_root / "NCT-CRC-HE-100K" / class_name / f"{class_name}_{i}.tif")
        _touch(raw_root / "CRC-VAL-HE-7K" / class_name / f"{class_name}_val.tif")

    manifest = curate_eva_patch_dataset(
        "crc",
        raw_root,
        tmp_path / "curated",
        tune_fraction=1 / 3,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv)
    splits_df = pd.read_csv(manifest.splits_csv)

    assert set(dataset_df["class_name"]) == set(CRC_CLASS_NAMES)
    assert set(dataset_df.loc[dataset_df["class_name"] == "ADI", "label"]) == {0}
    assert set(dataset_df.loc[dataset_df["class_name"] == "BACK", "label"]) == {1}
    assert len(splits_df[splits_df["split"] == "train"]) == 18
    assert len(splits_df[splits_df["split"] == "tune"]) == 9
    assert len(splits_df[splits_df["split"] == "test"]) == 9

    dataset = Dataset(manifest.dataset_csv)
    Splits(manifest.splits_csv, dataset)


def test_curate_crc_accepts_original_subdirectories(tmp_path: Path):
    raw_root = tmp_path / "crc_raw"
    for class_name in CRC_CLASS_NAMES:
        for i in range(3):
            _touch(
                raw_root
                / "NCT-CRC-HE-100K"
                / "original"
                / class_name
                / f"{class_name}_{i}.tif"
            )
        _touch(
            raw_root
            / "CRC-VAL-HE-7K"
            / "original"
            / class_name
            / f"{class_name}_val.tif"
        )

    manifest = curate_eva_patch_dataset(
        "crc",
        raw_root,
        tmp_path / "curated",
        tune_fraction=1 / 3,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv)
    splits_df = pd.read_csv(manifest.splits_csv)

    assert len(dataset_df) == 36
    assert len(splits_df[splits_df["split"] == "test"]) == 9
    assert dataset_df["image_path"].str.contains("/original/").all()


def test_curate_gleason_arvaniti_reports_on_validation_and_ignores_test_patches(
    tmp_path: Path,
):
    raw_root = tmp_path / "gleason_raw"
    _touch(
        raw_root
        / "train_validation_patches_750"
        / "ZT111_01_A_1_1"
        / "ZT111_01_A_1_1_patch_1_class_0.jpg"
    )
    _touch(
        raw_root
        / "train_validation_patches_750"
        / "ZT199_02_A_1_1"
        / "ZT199_02_A_1_1_patch_1_class_1.jpg"
    )
    _touch(
        raw_root
        / "train_validation_patches_750"
        / "ZT76_03_A_1_1"
        / "ZT76_03_A_1_1_patch_1_class_2.jpg"
    )
    # test_patches_750 exists on disk, but EVA does not use it (its test split "leads to
    # unstable evaluation results"); the curator must ignore it entirely.
    _touch(
        raw_root
        / "test_patches_750"
        / "patho_1"
        / "ZT80_04_A_1_1"
        / "ZT80_04_A_1_1_patch_1_class_3.jpg"
    )

    manifest = curate_eva_patch_dataset(
        "gleason_arvaniti",
        raw_root,
        tmp_path / "curated",
        tune_fraction=0.0,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv).sort_values("sample_id")
    splits_df = pd.read_csv(manifest.splits_csv)

    # Only ZT111/ZT199 (train) and ZT76 (val) are curated — no test_patches rows.
    assert dataset_df[["label", "class_name", "eva_split"]].to_dict("records") == [
        {"label": 0, "class_name": "benign", "eva_split": "train"},
        {"label": 1, "class_name": "gleason_3", "eva_split": "train"},
        {"label": 2, "class_name": "gleason_4", "eva_split": "val"},
    ]
    assert not dataset_df["image_path"].str.contains("test_patches_750").any()

    # tune_fraction=0.0 → EVA val (ZT76) becomes the soma test split with no tune split,
    # so the benchmark's tune_is_test=True has a single held-out split (no collision).
    assert splits_df["split"].value_counts().to_dict() == {"train": 2, "test": 1}
    test_ids = splits_df.loc[splits_df["split"] == "test", "sample_id"].tolist()
    assert len(test_ids) == 1 and "ZT76" in test_ids[0]

    dataset = Dataset(manifest.dataset_csv)
    # Previously raised "provides both a tune and a test split"; must now construct cleanly.
    splits = Splits(manifest.splits_csv, dataset, tune_is_test=True)
    assert len(splits.folds[0].train) == 2


def test_gleason_materializes_from_extracted_section_dirs(tmp_path: Path):
    raw_root = tmp_path / "gleason_raw"
    _build_gleason_extracted_raw(raw_root)
    assert not (raw_root / "train_validation_patches_750").exists()

    manifest = curate_eva_patch_dataset(
        "gleason_arvaniti", raw_root, tmp_path / "curated", tune_fraction=0.0
    )

    # Patches were materialized in place, one per synthetic core, with the mask's grade.
    patches = sorted((raw_root / "train_validation_patches_750").glob("**/*.jpg"))
    assert [p.name for p in patches] == [
        "ZT111_4_A_1_1_patch_0_class_0.jpg",
        "ZT76_39_A_1_1_patch_0_class_2.jpg",
    ]
    # No transient extraction staging is left behind.
    assert not (raw_root / ".gleason_arvaniti_staging").exists()

    dataset_df = pd.read_csv(manifest.dataset_csv).sort_values("sample_id")
    assert dataset_df[["label", "class_name", "eva_split"]].to_dict("records") == [
        {"label": 0, "class_name": "benign", "eva_split": "train"},
        {"label": 2, "class_name": "gleason_4", "eva_split": "val"},
    ]
    dataset = Dataset(manifest.dataset_csv)
    Splits(manifest.splits_csv, dataset, tune_is_test=True)


def test_gleason_materializes_from_tarball_archives(tmp_path: Path):
    src = tmp_path / "src"
    _build_gleason_extracted_raw(src)
    raw_root = tmp_path / "gleason_raw"
    raw_root.mkdir()
    # Repack the way Harvard Dataverse ships the download: per-section + mask tarballs only.
    for section in ("ZT76_39_A", "ZT111_4_A"):
        with tarfile.open(raw_root / f"{section}.tar.gz", "w:gz") as tar:
            tar.add(src / section, arcname=section)
    with tarfile.open(raw_root / "Gleason_masks_train.tar.gz", "w:gz") as tar:
        tar.add(src / "Gleason_masks_train", arcname="Gleason_masks_train")

    manifest = curate_eva_patch_dataset(
        "gleason_arvaniti", raw_root, tmp_path / "curated", tune_fraction=0.0
    )

    patches = sorted((raw_root / "train_validation_patches_750").glob("**/*.jpg"))
    assert [p.name for p in patches] == [
        "ZT111_4_A_1_1_patch_0_class_0.jpg",
        "ZT76_39_A_1_1_patch_0_class_2.jpg",
    ]
    assert not (raw_root / ".gleason_arvaniti_staging").exists()
    assert len(pd.read_csv(manifest.dataset_csv)) == 2


def test_gleason_prefers_existing_patches_over_raw_materialization(tmp_path: Path):
    raw_root = tmp_path / "gleason_raw"
    _build_gleason_extracted_raw(raw_root)  # raw ingredients are present...
    # ...but a pre-made patch dir already exists and must be used verbatim (cores untouched).
    _touch(
        raw_root
        / "train_validation_patches_750"
        / "ZT76_09_A_1_1"
        / "ZT76_09_A_1_1_patch_1_class_3.jpg"
    )

    manifest = curate_eva_patch_dataset(
        "gleason_arvaniti", raw_root, tmp_path / "curated", tune_fraction=0.0
    )

    dataset_df = pd.read_csv(manifest.dataset_csv)
    # Exactly the one pre-existing patch — no cores were re-materialized over it.
    assert len(dataset_df) == 1
    assert dataset_df.iloc[0]["label"] == 3
    assert dataset_df.iloc[0]["eva_split"] == "val"


def test_gleason_materialization_clears_stale_partial(tmp_path: Path):
    raw_root = tmp_path / "gleason_raw"
    _build_gleason_extracted_raw(raw_root)
    stale = raw_root / "train_validation_patches_750.partial"
    (stale / "leftover").mkdir(parents=True)
    (stale / "leftover" / "junk_patch_0_class_9.jpg").write_bytes(b"")

    curate_eva_patch_dataset(
        "gleason_arvaniti", raw_root, tmp_path / "curated", tune_fraction=0.0
    )

    # A crashed run's staging is discarded; only freshly materialized patches remain.
    assert not stale.exists()
    names = sorted(
        p.name for p in (raw_root / "train_validation_patches_750").glob("**/*.jpg")
    )
    assert names == [
        "ZT111_4_A_1_1_patch_0_class_0.jpg",
        "ZT76_39_A_1_1_patch_0_class_2.jpg",
    ]


def test_gleason_raises_without_patches_or_raw(tmp_path: Path):
    raw_root = tmp_path / "empty"
    raw_root.mkdir()
    with pytest.raises(FileNotFoundError, match="Harvard Dataverse"):
        curate_eva_patch_dataset(
            "gleason_arvaniti", raw_root, tmp_path / "curated", tune_fraction=0.0
        )


def test_gleason_raises_when_masks_missing(tmp_path: Path):
    raw_root = tmp_path / "gleason_raw"
    _make_gleason_core(raw_root / "ZT76_39_A", "ZT76_39_A_1_1")  # cores present, masks absent
    with pytest.raises(FileNotFoundError, match="Harvard Dataverse"):
        curate_eva_patch_dataset(
            "gleason_arvaniti", raw_root, tmp_path / "curated", tune_fraction=0.0
        )


@pytest.mark.skipif(
    not _gleason_arvaniti_core_archives(GLEASON_REAL_ROOT),
    reason="raw GleasonArvaniti Dataverse download not present on this host",
)
def test_gleason_real_download_is_detected():
    assert _gleason_arvaniti_raw_present(GLEASON_REAL_ROOT)
    # The four train/validation TMAs ship as 9 core archives: ZT76 x2, ZT111 x3, ZT199 x2, ZT204 x2.
    assert len(_gleason_arvaniti_core_archives(GLEASON_REAL_ROOT)) == 9
    assert (GLEASON_REAL_ROOT / "Gleason_masks_train.tar.gz").is_file()


@pytest.mark.skipif(
    not os.environ.get("SOMA_RUN_SLOW_DATA_TESTS")
    or not (GLEASON_REAL_ROOT / "ZT76_39_A.tar.gz").is_file(),
    reason="slow real-data check; set SOMA_RUN_SLOW_DATA_TESTS=1 with the Dataverse download present",
)
def test_gleason_materializes_zt76_to_eva_validation_count(tmp_path: Path):
    # Materialize only the ZT76 sections against the real tarballs and confirm the patch
    # count matches EVA's validation cohort (2482) — the byte-for-byte reproduction check.
    raw_root = tmp_path / "gleason_raw"
    raw_root.mkdir()
    for section in ("ZT76_39_A", "ZT76_39_B"):
        (raw_root / f"{section}.tar.gz").symlink_to(GLEASON_REAL_ROOT / f"{section}.tar.gz")
    (raw_root / "Gleason_masks_train.tar.gz").symlink_to(
        GLEASON_REAL_ROOT / "Gleason_masks_train.tar.gz"
    )

    manifest = curate_eva_patch_dataset(
        "gleason_arvaniti", raw_root, tmp_path / "curated", tune_fraction=0.0
    )

    dataset_df = pd.read_csv(manifest.dataset_csv)
    assert set(dataset_df["eva_split"]) == {"val"}
    assert len(dataset_df) == 2482


def test_curate_patch_camelyon_accepts_split_class_folders(tmp_path: Path):
    raw_root = tmp_path / "patch_camelyon_raw"
    for split in ("train", "val", "test"):
        _touch(raw_root / split / "normal" / f"{split}_normal.png")
        _touch(raw_root / split / "tumor" / f"{split}_tumor.png")

    manifest = curate_eva_patch_dataset(
        "patch_camelyon",
        raw_root,
        tmp_path / "curated",
        tune_fraction=0.0,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv)
    splits_df = pd.read_csv(manifest.splits_csv)

    assert len(dataset_df) == 6
    assert set(dataset_df["class_name"]) == {"no_tumor", "tumor"}
    assert set(dataset_df.loc[dataset_df["class_name"] == "no_tumor", "label"]) == {0}
    assert set(dataset_df.loc[dataset_df["class_name"] == "tumor", "label"]) == {1}
    assert splits_df["split"].value_counts().to_dict() == {
        "train": 2,
        "tune": 2,
        "test": 2,
    }

    dataset = Dataset(manifest.dataset_csv)
    splits = Splits(manifest.splits_csv, dataset)
    assert len(splits.folds[0].tune) == 2


def test_curate_bach_uses_eva_index_ranges_with_train_split_recut(tmp_path: Path):
    raw_root = tmp_path / "bach_raw"
    photos = raw_root / "ICIAR2018_BACH_Challenge" / "Photos"
    for class_name in ("Benign", "InSitu", "Invasive", "Normal"):
        for i in range(100):
            _touch(photos / class_name / f"{class_name}_{i:03d}.tif")

    manifest = curate_eva_patch_dataset(
        "bach",
        raw_root,
        tmp_path / "curated",
        tune_fraction=0.25,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv)
    splits_df = pd.read_csv(manifest.splits_csv)

    assert len(dataset_df) == 400
    assert len(splits_df[splits_df["split"] == "test"]) == 132
    assert len(splits_df[splits_df["split"] == "tune"]) > 0
    assert len(splits_df[splits_df["split"] == "train"]) + len(splits_df[splits_df["split"] == "tune"]) == 268
    assert set(dataset_df["label"]) == {0, 1, 2, 3}

    dataset = Dataset(manifest.dataset_csv)
    Splits(manifest.splits_csv, dataset)


def test_curate_bach_accepts_pre_split_train_test_layout(tmp_path: Path):
    raw_root = tmp_path / "bach_raw"
    challenge = raw_root / "ICIAR2018_BACH_Challenge"
    class_counts = {
        "train": {"Benign": 2, "InSitu": 2, "Invasive": 2, "Normal": 2},
        "test": {"Benign": 1, "InSitu": 1, "Invasive": 1, "Normal": 1},
    }
    for split, counts in class_counts.items():
        for class_name, count in counts.items():
            for i in range(count):
                _touch(challenge / split / class_name / f"{class_name}_{split}_{i}.tif")

    manifest = curate_eva_patch_dataset(
        "bach",
        raw_root,
        tmp_path / "curated",
        tune_fraction=0.5,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv)
    splits_df = pd.read_csv(manifest.splits_csv)

    assert len(dataset_df) == 12
    assert len(splits_df[splits_df["split"] == "train"]) == 4
    assert len(splits_df[splits_df["split"] == "tune"]) == 4
    assert len(splits_df[splits_df["split"] == "test"]) == 4
    assert dataset_df["image_path"].str.contains("/train/|/test/").all()
