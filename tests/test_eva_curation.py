"""Tests for EVA patch-level dataset curation."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soma.curation.eva import curate_eva_patch_dataset
from soma.dataset import Dataset, Splits


CRC_CLASS_NAMES = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")
BREAKHIS_CLASS_NAMES = ("TA", "MC", "F", "DC")
GLEASON_ARVANITI_CLASS_NAMES = ("benign", "gleason_3", "gleason_4", "gleason_5")

# The full PatchCamelyon HDF5 download, if it happens to be available locally, lets us
# assert the materializer's reader assumptions against the real files (cheaply, without
# decoding all 327k tiles).
PCAM_REAL_ROOT = Path(
    "/data/pathology/projects/vlfm/data/vision/tile_level_datasets/datasets/PatchCam"
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _pcam_image(base: int) -> np.ndarray:
    """An 8×8 RGB tile with per-channel-distinct values (catches channel-order bugs)."""
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    for channel in range(3):
        image[..., channel] = (base + channel) % 256
    return image


def _write_pcam_hdf5(
    raw_root: Path, splits: dict[str, list[tuple[np.ndarray, int]]]
) -> None:
    """Write EVA's six official ``camelyonpatch_level_2_split_<split>_<x|y>.h5`` files."""
    raw_root.mkdir(parents=True, exist_ok=True)
    for eva_split, items in splits.items():
        x = np.stack([image for image, _ in items]).astype(np.uint8)
        y = np.asarray([label for _, label in items], dtype=np.uint8).reshape(-1, 1, 1, 1)
        with h5py.File(raw_root / f"camelyonpatch_level_2_split_{eva_split}_x.h5", "w") as f:
            f.create_dataset("x", data=x)
        with h5py.File(raw_root / f"camelyonpatch_level_2_split_{eva_split}_y.h5", "w") as f:
            f.create_dataset("y", data=y)


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


def test_curate_gleason_arvaniti_uses_eva_microarray_splits(tmp_path: Path):
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
    _touch(
        raw_root
        / "test_patches_750"
        / "patho_1"
        / "ZT80_04_A_1_1"
        / "ZT80_04_A_1_1_patch_1_class_3.jpg"
    )
    _touch(
        raw_root
        / "test_patches_750"
        / "patho_2"
        / "ZT80_04_A_1_1"
        / "ZT80_04_A_1_1_patch_1_class_0.jpg"
    )

    manifest = curate_eva_patch_dataset(
        "gleason_arvaniti",
        raw_root,
        tmp_path / "curated",
        tune_fraction=0.0,
    )

    dataset_df = pd.read_csv(manifest.dataset_csv).sort_values("sample_id")
    splits_df = pd.read_csv(manifest.splits_csv).sort_values("sample_id")

    assert dataset_df[["label", "class_name", "eva_split"]].to_dict("records") == [
        {"label": 3, "class_name": "gleason_5", "eva_split": "test"},
        {"label": 0, "class_name": "benign", "eva_split": "train"},
        {"label": 1, "class_name": "gleason_3", "eva_split": "train"},
        {"label": 2, "class_name": "gleason_4", "eva_split": "val"},
    ]
    assert splits_df["split"].value_counts().to_dict() == {
        "train": 2,
        "tune": 1,
        "test": 1,
    }
    assert not dataset_df["image_path"].str.contains("/patho_2/").any()

    dataset = Dataset(manifest.dataset_csv)
    splits = Splits(manifest.splits_csv, dataset)
    assert len(splits.folds[0].tune) == 1


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


def _tiny_pcam_splits() -> dict[str, list[tuple[np.ndarray, int]]]:
    return {
        # labels 0,1,0,1 (bases 10..13); label 0 -> normal, 1 -> tumor.
        "train": [(_pcam_image(10 + i), i % 2) for i in range(4)],
        "valid": [(_pcam_image(20), 1), (_pcam_image(21), 0)],
        "test": [(_pcam_image(30), 0), (_pcam_image(31), 1)],
    }


def test_curate_patch_camelyon_materializes_hdf5_when_folders_absent(tmp_path: Path):
    raw_root = tmp_path / "patch_camelyon_raw"
    _write_pcam_hdf5(raw_root, _tiny_pcam_splits())

    manifest = curate_eva_patch_dataset(
        "patch_camelyon", raw_root, tmp_path / "curated", tune_fraction=0.0
    )

    # HDF5 was materialized into the class-folder layout under the (writable) raw root,
    # with no staging directory left behind.
    for split in ("train", "val", "test"):
        assert (raw_root / split / "normal").is_dir()
        assert (raw_root / split / "tumor").is_dir()
        assert not (raw_root / f"{split}.partial").exists()

    dataset_df = pd.read_csv(manifest.dataset_csv)
    splits_df = pd.read_csv(manifest.splits_csv)

    assert len(dataset_df) == 8
    assert set(dataset_df["class_name"]) == {"no_tumor", "tumor"}
    # 4 normal (train 2 + val 1 + test 1) and 4 tumor.
    assert dataset_df["label"].value_counts().to_dict() == {0: 4, 1: 4}
    # tune_is_test protocol for pcam: eva val -> soma tune, eva test -> soma test.
    assert splits_df["split"].value_counts().to_dict() == {"train": 4, "tune": 2, "test": 2}

    # Materialized PNGs round-trip pixel-exact (lossless) and keep RGB channel order.
    assert np.array_equal(
        np.array(Image.open(raw_root / "train" / "normal" / "image_0.png")),
        _pcam_image(10),
    )
    assert np.array_equal(
        np.array(Image.open(raw_root / "train" / "tumor" / "image_1.png")),
        _pcam_image(11),
    )


def test_curate_patch_camelyon_hdf5_materialization_is_idempotent(tmp_path: Path):
    raw_root = tmp_path / "patch_camelyon_raw"
    _write_pcam_hdf5(raw_root, _tiny_pcam_splits())

    first = curate_eva_patch_dataset(
        "patch_camelyon", raw_root, tmp_path / "curated_a", tune_fraction=0.0
    )
    materialized = raw_root / "train" / "normal" / "image_0.png"
    mtime_after_first = materialized.stat().st_mtime_ns

    # Second curation finds the folders present and must not re-decode the HDF5.
    second = curate_eva_patch_dataset(
        "patch_camelyon", raw_root, tmp_path / "curated_b", tune_fraction=0.0
    )

    assert materialized.stat().st_mtime_ns == mtime_after_first
    pd.testing.assert_frame_equal(
        pd.read_csv(first.dataset_csv), pd.read_csv(second.dataset_csv)
    )


def test_curate_patch_camelyon_resumes_after_interrupted_final_split(tmp_path: Path):
    raw_root = tmp_path / "patch_camelyon_raw"
    _write_pcam_hdf5(raw_root, _tiny_pcam_splits())

    curate_eva_patch_dataset(
        "patch_camelyon", raw_root, tmp_path / "curated_a", tune_fraction=0.0
    )

    # Simulate a crash while the *last* split was being written: its final directory is
    # gone and a stale ``.partial`` staging dir remains. train/val stay complete.
    import shutil

    shutil.rmtree(raw_root / "test")
    stale = raw_root / "test.partial" / "normal"
    stale.mkdir(parents=True)
    (stale / "image_0.png").write_bytes(b"corrupt")

    manifest = curate_eva_patch_dataset(
        "patch_camelyon", raw_root, tmp_path / "curated_b", tune_fraction=0.0
    )

    assert not (raw_root / "test.partial").exists()
    splits_df = pd.read_csv(manifest.splits_csv)
    assert splits_df["split"].value_counts().to_dict() == {"train": 4, "tune": 2, "test": 2}
    # The re-decoded test split is pixel-exact, not the corrupt stale staging content.
    assert np.array_equal(
        np.array(Image.open(raw_root / "test" / "normal" / "image_0.png")),
        _pcam_image(30),
    )


@pytest.mark.skipif(
    not (PCAM_REAL_ROOT / "camelyonpatch_level_2_split_test_x.h5").is_file(),
    reason="real PatchCamelyon HDF5 download not available",
)
def test_patch_camelyon_real_hdf5_matches_materializer_reader(tmp_path: Path):
    x_path = PCAM_REAL_ROOT / "camelyonpatch_level_2_split_test_x.h5"
    y_path = PCAM_REAL_ROOT / "camelyonpatch_level_2_split_test_y.h5"

    with h5py.File(x_path, "r") as x_file, h5py.File(y_path, "r") as y_file:
        # The materializer reads dataset key "x"/"y", (N,96,96,3) uint8 tiles, 0/1 labels.
        assert "x" in x_file and "y" in y_file
        images = x_file["x"]
        assert images.shape[1:] == (96, 96, 3)
        assert images.dtype == np.uint8
        image0 = np.asarray(images[0])
        label0 = int(np.asarray(y_file["y"][0]).reshape(-1)[0])

    assert label0 in (0, 1)
    out = tmp_path / "image0.png"
    Image.fromarray(image0).save(out)
    assert np.array_equal(np.array(Image.open(out)), image0)  # PNG is lossless


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
