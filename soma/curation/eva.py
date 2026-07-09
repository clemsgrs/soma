"""Curators for EVA patch-level pathology classification datasets.

The curators emit Soma's standard ``dataset.csv`` and ``splits.csv`` manifests
from locally prepared raw EVA datasets. They intentionally do not download data;
license and access requirements differ across the EVA benchmark.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
import tarfile
from typing import Callable, Iterable

import pandas as pd

from soma.curation.manifest import CuratedManifest, write_manifest

logger = logging.getLogger(__name__)

# EVA patch datasets are tile-level classification; their supervision column is ``label``.
_EVA_DATASET_TYPE = "tile"


BACH_CLASSES = {"Benign": 0, "InSitu": 1, "Invasive": 2, "Normal": 3}
BACH_TRAIN_RANGES = [
    (0, 41),
    (59, 60),
    (90, 139),
    (169, 240),
    (258, 260),
    (273, 345),
    (368, 400),
]
BACH_VAL_RANGES = [
    (41, 59),
    (60, 90),
    (139, 169),
    (240, 258),
    (260, 273),
    (345, 368),
]

CRC_CLASSES = {
    "ADI": 0,
    "BACK": 1,
    "DEB": 2,
    "LYM": 3,
    "MUC": 4,
    "MUS": 5,
    "NORM": 6,
    "STR": 7,
    "TUM": 8,
}

GLEASON_ARVANITI_CLASSES = {
    "benign": 0,
    "gleason_3": 1,
    "gleason_4": 2,
    "gleason_5": 3,
}
GLEASON_ARVANITI_TRAIN_ARRAY_IDS = {"ZT111", "ZT199", "ZT204"}
GLEASON_ARVANITI_VAL_ARRAY_ID = "ZT76"
# Filename prefixes of the four train/validation TMAs on Harvard Dataverse (DOI
# 10.7910/DVN/OCYCMP). The test cohort (ZT80) is intentionally excluded — EVA/Soma
# report on the ZT76 validation cohort and ignore the unstable test split.
GLEASON_ARVANITI_TMA_PREFIXES = ("ZT76_39", "ZT111_4", "ZT199_1", "ZT204_6")
GLEASON_ARVANITI_PATCHES_DIRNAME = "train_validation_patches_750"
GLEASON_ARVANITI_MASK_DIRNAME = "Gleason_masks_train"
GLEASON_ARVANITI_PATCH_SIZE = 750
# gleason_CNN drops patches whose mean pixel value exceeds this (mostly-background tissue).
GLEASON_ARVANITI_WHITE_LIMIT = 180

PATCH_CAMELYON_CLASSES = {"no_tumor": 0, "tumor": 1}
PATCH_CAMELYON_FOLDER_CLASS_ALIASES = {"normal": "no_tumor", "no_tumor": "no_tumor", "tumor": "tumor"}

BREAKHIS_CLASSES = {"TA": 0, "MC": 1, "F": 2, "DC": 3}
BREAKHIS_VAL_PATIENT_IDS = {
    "18842D",
    "19979",
    "15275",
    "15792",
    "16875",
    "3909",
    "5287",
    "16716",
    "2773",
    "5695",
    "16184CD",
    "23060CD",
    "21998CD",
    "21998EF",
}

MHIST_CLASSES = {"SSA": 0, "HP": 1}

EVA_PATCH_CLASSIFICATION_DATASETS = (
    "bach",
    "mhist",
    "crc",
    "breakhis",
    "gleason_arvaniti",
    "patch_camelyon",
)


@dataclass(frozen=True)
class _Sample:
    sample_id: str
    image_path: Path
    label: int
    eva_split: str
    class_name: str
    source_dataset: str


def curate_eva_patch_dataset(
    name: str,
    raw_root: str | Path,
    output_dir: str | Path,
    *,
    tune_fraction: float = 0.2,
) -> CuratedManifest:
    """Curate one EVA patch-level classification dataset.

    Args:
        name: Dataset name. Supported names include ``"bach"``, ``"mhist"``,
            ``"crc"``, ``"breakhis"``, ``"gleason_arvaniti"``, and
            ``"patch_camelyon"``.
        raw_root: Local raw dataset root in the layout expected by EVA.
        output_dir: Directory where ``dataset.csv`` and ``splits.csv`` will be
            written.
        tune_fraction: Fraction of EVA train samples reserved as Soma ``tune``.
            Set to ``0.0`` to keep all EVA train samples as Soma ``train`` for
            tune-is-test benchmark reproduction. EVA validation/test samples
            are reserved as Soma ``test``.
    """

    normalized_name = _normalize_dataset_name(name)
    builders: dict[str, Callable[[Path], list[_Sample]]] = {
        "bach": _bach_samples,
        "mhist": _mhist_samples,
        "crc": _crc_samples,
        "breakhis": _breakhis_samples,
        "gleason_arvaniti": _gleason_arvaniti_samples,
        "patch_camelyon": _patch_camelyon_samples,
    }
    try:
        samples = builders[normalized_name](Path(raw_root))
    except KeyError as exc:
        supported = ", ".join(EVA_PATCH_CLASSIFICATION_DATASETS)
        raise ValueError(
            f"Unsupported EVA patch dataset '{name}'. Supported: {supported}"
        ) from exc
    return _write_manifests(
        dataset_name=normalized_name,
        samples=samples,
        output_dir=Path(output_dir),
        tune_fraction=tune_fraction,
    )


def curate_eva_patch_datasets(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    dataset_names: Iterable[str] = EVA_PATCH_CLASSIFICATION_DATASETS,
    tune_fraction: float = 0.2,
) -> dict[str, CuratedManifest]:
    """Curate multiple EVA patch-level datasets from sibling raw roots.

    ``raw_root`` is expected to contain one child directory per dataset using
    the normalized names accepted by :func:`curate_eva_patch_dataset`.
    """

    raw_root = Path(raw_root)
    output_root = Path(output_root)
    manifests: dict[str, CuratedManifest] = {}
    for name in dataset_names:
        normalized_name = _normalize_dataset_name(name)
        manifests[normalized_name] = curate_eva_patch_dataset(
            normalized_name,
            raw_root / normalized_name,
            output_root / normalized_name,
            tune_fraction=tune_fraction,
        )
    return manifests


def _write_manifests(
    *,
    dataset_name: str,
    samples: list[_Sample],
    output_dir: Path,
    tune_fraction: float,
) -> CuratedManifest:
    if not 0.0 <= tune_fraction < 1.0:
        raise ValueError("tune_fraction must be in [0, 1)")
    if not samples:
        raise ValueError(f"No samples found for EVA dataset '{dataset_name}'")

    dataset_rows = [
        {
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "label": sample.label,
            "source_dataset": sample.source_dataset,
            "eva_split": sample.eva_split,
            "class_name": sample.class_name,
        }
        for sample in sorted(samples, key=lambda s: s.sample_id)
    ]

    train_ids = [sample.sample_id for sample in samples if sample.eva_split == "train"]
    has_eva_test = any(sample.eva_split == "test" for sample in samples)
    labels_by_id = {sample.sample_id: sample.label for sample in samples}
    if has_eva_test:
        tune_ids = sorted(sample.sample_id for sample in samples if sample.eva_split == "val")
        tune_ids.extend(_stratified_tune_ids(train_ids, labels_by_id, tune_fraction))
        test_ids = [sample.sample_id for sample in samples if sample.eva_split == "test"]
    else:
        tune_ids = _stratified_tune_ids(train_ids, labels_by_id, tune_fraction)
        test_ids = [sample.sample_id for sample in samples if sample.eva_split != "train"]
    tune_set = set(tune_ids)
    # Single fold ⇒ fold=0 for every row (write_manifest fills the fold column).
    split_rows = [
        {"sample_id": sample_id, "split": "tune" if sample_id in tune_set else "train", "fold": 0}
        for sample_id in sorted(train_ids)
    ]
    split_rows.extend(
        {"sample_id": sample_id, "split": "tune", "fold": 0}
        for sample_id in sorted(tune_set - set(train_ids))
    )
    split_rows.extend(
        {"sample_id": sample_id, "split": "test", "fold": 0} for sample_id in sorted(test_ids)
    )

    class_names_by_label = {sample.label: sample.class_name for sample in samples}
    summary = {
        "dataset": dataset_name,
        "dataset_type": _EVA_DATASET_TYPE,
        "num_classes": len(class_names_by_label),
        "class_names": [class_names_by_label[label] for label in sorted(class_names_by_label)],
        "total_samples": len(samples),
        "tune_fraction": tune_fraction,
        "splits": {
            "train": sum(1 for r in split_rows if r["split"] == "train"),
            "tune": sum(1 for r in split_rows if r["split"] == "tune"),
            "test": sum(1 for r in split_rows if r["split"] == "test"),
        },
    }

    return write_manifest(
        output_dir,
        dataset_type=_EVA_DATASET_TYPE,
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary=summary,
    )


def _stratified_tune_ids(
    train_ids: list[str],
    labels_by_id: dict[str, int],
    tune_fraction: float,
) -> list[str]:
    """Select a deterministic stratified tune subset from EVA train samples."""
    if tune_fraction == 0.0:
        return []

    by_label: dict[int, list[str]] = {}
    for sample_id in sorted(train_ids):
        by_label.setdefault(labels_by_id[sample_id], []).append(sample_id)

    tune_ids: list[str] = []
    for ids in by_label.values():
        n_tune = max(1, round(len(ids) * tune_fraction)) if len(ids) > 1 else 0
        tune_ids.extend(ids[-n_tune:] if n_tune else [])
    if not tune_ids and train_ids:
        tune_ids.append(sorted(train_ids)[-1])
    return sorted(tune_ids)


def _bach_samples(root: Path) -> list[_Sample]:
    dataset_path = root / "ICIAR2018_BACH_Challenge"
    photos_path = dataset_path / "Photos"
    if not photos_path.is_dir() and (root / "Photos").is_dir():
        photos_path = root / "Photos"
    if photos_path.is_dir():
        return _bach_photos_samples(photos_path)

    split_root = dataset_path if dataset_path.is_dir() else root
    train_path = split_root / "train"
    test_path = split_root / "test"
    if train_path.is_dir() and test_path.is_dir():
        return _class_folder_samples(
            dataset_name="bach",
            base_dir=split_root,
            split_dirs={"train": train_path, "val": test_path},
            class_to_label=BACH_CLASSES,
            extension=".tif",
        )

    raise FileNotFoundError(
        "BACH raw data must contain either "
        "ICIAR2018_BACH_Challenge/Photos/<class>/*.tif or "
        "ICIAR2018_BACH_Challenge/{train,test}/<class>/*.tif"
    )


def _bach_photos_samples(photos_path: Path) -> list[_Sample]:
    all_samples: list[tuple[Path, str]] = []
    for class_name in BACH_CLASSES:
        all_samples.extend(
            (path, class_name)
            for path in sorted((photos_path / class_name).glob("*.tif"))
        )
    all_samples = sorted(all_samples, key=lambda item: str(item[0]))
    train_indices = set(_ranges_to_indices(BACH_TRAIN_RANGES))
    val_indices = set(_ranges_to_indices(BACH_VAL_RANGES))
    samples: list[_Sample] = []
    for idx, (image_path, class_name) in enumerate(all_samples):
        if idx in train_indices:
            eva_split = "train"
        elif idx in val_indices:
            eva_split = "val"
        else:
            continue
        samples.append(
            _sample(
                "bach",
                photos_path,
                image_path,
                class_name,
                BACH_CLASSES[class_name],
                eva_split,
            )
        )
    return samples


def _crc_samples(root: Path) -> list[_Sample]:
    split_dirs = {
        "train": _first_existing_dir(
            root / "NCT-CRC-HE-100K" / "original",
            root / "NCT-CRC-HE-100K",
        ),
        "val": _first_existing_dir(
            root / "CRC-VAL-HE-7K" / "original",
            root / "CRC-VAL-HE-7K",
        ),
    }
    return _class_folder_samples(
        dataset_name="crc",
        base_dir=root,
        split_dirs=split_dirs,
        class_to_label=CRC_CLASSES,
        extension=".tif",
    )


def _mhist_samples(root: Path) -> list[_Sample]:
    annotations_path = root / "annotations.csv"
    images_dir = root / "images"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"MHIST annotations not found: {annotations_path}")

    annotations = pd.read_csv(annotations_path)
    required = {"Image Name", "Majority Vote Label", "Partition"}
    missing = sorted(required - set(annotations.columns))
    if missing:
        raise ValueError(f"MHIST annotations missing required column(s): {missing}")

    samples: list[_Sample] = []
    for _, row in annotations.sort_values("Image Name").iterrows():
        class_name = str(row["Majority Vote Label"])
        if class_name not in MHIST_CLASSES:
            raise ValueError(f"Unsupported MHIST class '{class_name}'")
        source_split = str(row["Partition"])
        if source_split not in {"train", "test"}:
            raise ValueError(f"Unsupported MHIST partition '{source_split}'")
        image_path = images_dir / str(row["Image Name"])
        samples.append(
            _sample(
                "mhist",
                root,
                image_path,
                class_name,
                MHIST_CLASSES[class_name],
                source_split,
            )
        )
    return samples


def _gleason_arvaniti_samples(root: Path) -> list[_Sample]:
    # EVA reports GleasonArvaniti on the validation cohort (TMA ``ZT76``) and trains on
    # ``ZT111``/``ZT199``/``ZT204``. It deliberately does not use ``test_patches_750``:
    # EVA's dataset card documents that its test split "leads to unstable evaluation
    # results" and recommends the validation split, and the packaged reference band is the
    # val number. So the reproduction reads only ``train_validation_patches_750``. Emitting
    # ``test_patches_750`` as a soma ``test`` split would also collide with the benchmark's
    # ``tune_is_test=True`` (it forbids a fold carrying both a tune and a test split).
    patches_dir = root / GLEASON_ARVANITI_PATCHES_DIRNAME
    if not patches_dir.is_dir():
        if _gleason_arvaniti_raw_present(root):
            _materialize_gleason_arvaniti_patches(root, patches_dir)
        else:
            raise FileNotFoundError(
                "GleasonArvaniti raw data must contain either pre-made patches at "
                f"{patches_dir} (produced by gleason_CNN's create_patches.py), or the "
                "official Harvard Dataverse download so Soma can materialize them: the "
                "train/validation TMA archives ZT{76,111,199,204}_*.tar.gz (or their "
                "extracted folders, or a TMA_images/ dir) plus Gleason_masks_train.tar.gz "
                "(or an extracted Gleason_masks_train/ dir). Source: "
                "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OCYCMP"
            )

    image_paths = sorted(patches_dir.glob("**/*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"GleasonArvaniti images not found under {patches_dir}")

    samples: list[_Sample] = []
    for image_path in image_paths:
        class_label = _gleason_arvaniti_class_label(image_path)
        class_name = _class_name_from_label(GLEASON_ARVANITI_CLASSES, class_label)
        array_id = image_path.stem.split("_")[0]
        if array_id == GLEASON_ARVANITI_VAL_ARRAY_ID:
            eva_split = "val"
        elif array_id in GLEASON_ARVANITI_TRAIN_ARRAY_IDS:
            eva_split = "train"
        else:
            raise ValueError(f"Invalid GleasonArvaniti microarray ID for file: {image_path}")
        samples.append(
            _sample(
                "gleason_arvaniti",
                root,
                image_path,
                class_name,
                class_label,
                eva_split,
            )
        )
    return samples


def _gleason_arvaniti_class_label(image_path: Path) -> int:
    label = int(image_path.stem.split("_")[-1])
    if label not in GLEASON_ARVANITI_CLASSES.values():
        raise ValueError(f"Unsupported GleasonArvaniti class label '{label}' in {image_path}")
    return label


def _gleason_arvaniti_core_archives(root: Path) -> list[Path]:
    """Train/validation TMA core archives (``ZT{76,111,199,204}_*.tar.gz``) under ``root``."""
    archives: list[Path] = []
    for prefix in GLEASON_ARVANITI_TMA_PREFIXES:
        archives.extend(sorted(root.glob(f"{prefix}*.tar.gz")))
    return archives


def _gleason_arvaniti_has_extracted_cores(root: Path) -> bool:
    tma_images = root / "TMA_images"
    if tma_images.is_dir() and next(tma_images.glob("**/*.jpg"), None) is not None:
        return True
    for prefix in GLEASON_ARVANITI_TMA_PREFIXES:
        for section in root.glob(f"{prefix}*"):
            if section.is_dir() and next(section.glob("*.jpg"), None) is not None:
                return True
    return False


def _gleason_arvaniti_raw_present(root: Path) -> bool:
    """True when ``root`` holds the raw ingredients to materialize the patch dataset."""
    has_images = _gleason_arvaniti_has_extracted_cores(root) or bool(
        _gleason_arvaniti_core_archives(root)
    )
    has_masks = (root / GLEASON_ARVANITI_MASK_DIRNAME).is_dir() or (
        root / f"{GLEASON_ARVANITI_MASK_DIRNAME}.tar.gz"
    ).is_file()
    return has_images and has_masks


def _extract_tar(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(dest, filter="data")  # path-traversal guard (Python >= 3.12)
        except TypeError:  # pragma: no cover - older Pythons lack the ``filter`` kwarg
            tar.extractall(dest)


def _gleason_arvaniti_core_paths(root: Path, staging_root: Path) -> list[Path]:
    """Resolve the TMA core jpgs, extracting the archives into ``staging_root`` if needed."""
    tma_images = root / "TMA_images"
    if tma_images.is_dir():
        paths = sorted(tma_images.glob("**/*.jpg"))
        if paths:
            return paths
    section_paths: list[Path] = []
    for prefix in GLEASON_ARVANITI_TMA_PREFIXES:
        for section in sorted(root.glob(f"{prefix}*")):
            if section.is_dir():
                section_paths.extend(sorted(section.glob("**/*.jpg")))
    if section_paths:
        return section_paths
    archives = _gleason_arvaniti_core_archives(root)
    if not archives:
        raise FileNotFoundError(f"No GleasonArvaniti TMA cores or archives found under {root}")
    dest = staging_root / "tma_images"
    dest.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        logger.info("gleason_arvaniti: extracting %s", archive.name)
        _extract_tar(archive, dest)
    return sorted(dest.glob("**/*.jpg"))


def _gleason_arvaniti_masks_dir(root: Path, staging_root: Path) -> Path:
    """Resolve the train-mask dir, extracting ``Gleason_masks_train.tar.gz`` if needed."""
    extracted = root / GLEASON_ARVANITI_MASK_DIRNAME
    if extracted.is_dir():
        return extracted
    archive = root / f"{GLEASON_ARVANITI_MASK_DIRNAME}.tar.gz"
    if not archive.is_file():
        raise FileNotFoundError(
            f"GleasonArvaniti train masks not found: expected {extracted} or {archive}"
        )
    dest = staging_root / "masks"
    dest.mkdir(parents=True, exist_ok=True)
    logger.info("gleason_arvaniti: extracting %s", archive.name)
    _extract_tar(archive, dest)
    inner = dest / GLEASON_ARVANITI_MASK_DIRNAME
    return inner if inner.is_dir() else dest


def _gleason_arvaniti_patch_coords(size_x: int, size_y: int, patch_size: int) -> list[tuple[int, int]]:
    """Overlapping grid of top-left patch coords (verbatim from gleason_CNN)."""
    step = patch_size // 2
    coords: list[tuple[int, int]] = []
    for y in range(0, size_y - patch_size, step):
        for x in range(0, size_x - patch_size, step):
            coords.append((x, y))
    return coords


def _gleason_arvaniti_patch_grade(i0, j0, patch_size, mask, n_class, np) -> int:
    """Grade of a patch = the single Gleason class covering its central third, else ignore."""
    window = patch_size // 3
    central = mask[(i0 + window):(i0 + 2 * window), (j0 + window):(j0 + 2 * window)]
    grades = np.unique(central)
    grades = grades[grades < n_class]
    return int(grades[0]) if len(grades) == 1 else n_class


def _write_gleason_arvaniti_patches(core_paths, masks_dir, out_dir, np, Image) -> int:
    patch_size = GLEASON_ARVANITI_PATCH_SIZE
    n_class = len(GLEASON_ARVANITI_CLASSES)
    written = 0
    for core_path in core_paths:
        name = core_path.stem
        if not any(name.startswith(prefix) for prefix in GLEASON_ARVANITI_TMA_PREFIXES):
            continue
        mask_path = masks_dir / f"mask_{name}.png"
        if not mask_path.is_file():
            continue
        image = np.asarray(Image.open(core_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path))  # palette indices == class ids (0..4)
        size_y, size_x = image.shape[0], image.shape[1]
        subdir = out_dir / name
        for index, (i0, j0) in enumerate(
            _gleason_arvaniti_patch_coords(size_x, size_y, patch_size)
        ):
            patch = image[i0:i0 + patch_size, j0:j0 + patch_size]
            grade = _gleason_arvaniti_patch_grade(i0, j0, patch_size, mask, n_class, np)
            if grade < n_class and float(np.mean(patch)) <= GLEASON_ARVANITI_WHITE_LIMIT:
                subdir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(patch).save(subdir / f"{name}_patch_{index}_class_{grade}.jpg")
                written += 1
    return written


def _materialize_gleason_arvaniti_patches(root: Path, patches_dir: Path) -> None:
    """Build ``train_validation_patches_750`` from the raw Harvard Dataverse download.

    Faithful port of eiriniar/gleason_CNN ``utils/create_patches.py``
    (``create_annotated_patches``) for the four train/validation TMAs. The upstream script
    imports the long-removed ``scipy.misc.imread``; Soma vendors the ~40 lines of patch
    geometry + labelling so the benchmark stays reproducible without that dependency. Only
    ``train_validation_patches_750`` is produced — the ZT80 test cohort (joint
    two-pathologist ``test_patches_750``) is intentionally skipped, matching what the
    curator reads. Materialization is atomic: patches land in a ``.partial`` staging dir
    that is renamed into place only once complete, so an interrupted run is retriable.
    """
    try:
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - NumPy/Pillow are core deps
        raise ImportError(
            "Materializing GleasonArvaniti patches requires NumPy and Pillow."
        ) from exc

    staging_root = root / ".gleason_arvaniti_staging"
    partial_dir = patches_dir.with_name(f"{patches_dir.name}.partial")
    if staging_root.exists():
        shutil.rmtree(staging_root)
    if partial_dir.exists():
        shutil.rmtree(partial_dir)
    try:
        core_paths = _gleason_arvaniti_core_paths(root, staging_root)
        masks_dir = _gleason_arvaniti_masks_dir(root, staging_root)
        partial_dir.mkdir(parents=True)
        written = _write_gleason_arvaniti_patches(core_paths, masks_dir, partial_dir, np, Image)
        if written == 0:
            raise RuntimeError(
                "GleasonArvaniti materialization produced no patches; check that the TMA "
                f"cores and masks under {root} pair up (mask_<core>.png beside <core>.jpg)."
            )
        partial_dir.rename(patches_dir)
        logger.info("gleason_arvaniti: materialized %d patches under %s", written, patches_dir)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        if partial_dir.exists():
            shutil.rmtree(partial_dir)


def _patch_camelyon_samples(root: Path) -> list[_Sample]:
    split_dirs = {split: root / split for split in ("train", "val", "test")}
    if all(path.is_dir() for path in split_dirs.values()):
        return _patch_camelyon_folder_samples(root, split_dirs)
    raise FileNotFoundError(
        "PatchCamelyon raw data must contain class-folder splits at "
        "<root>/{train,val,test}/{normal|no_tumor,tumor}. "
        "EVA's official HDF5 files are not directly usable as Soma image_path "
        "manifests without first materializing images."
    )


def _patch_camelyon_folder_samples(root: Path, split_dirs: dict[str, Path]) -> list[_Sample]:
    samples: list[_Sample] = []
    for eva_split, split_dir in split_dirs.items():
        for folder_name, class_name in PATCH_CAMELYON_FOLDER_CLASS_ALIASES.items():
            class_dir = split_dir / folder_name
            if not class_dir.is_dir():
                continue
            label = PATCH_CAMELYON_CLASSES[class_name]
            for image_path in _iter_images(class_dir, (".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                samples.append(
                    _sample("patch_camelyon", root, image_path, class_name, label, eva_split)
                )
    if not samples:
        raise ValueError(f"No PatchCamelyon class-folder images found under: {root}")
    return samples


def _breakhis_samples(root: Path) -> list[_Sample]:
    image_paths = sorted(root.glob("**/40X/*.png"))
    if not image_paths:
        raise FileNotFoundError(f"BreaKHis 40X images not found under: {root}")

    samples: list[_Sample] = []
    for image_path in image_paths:
        class_name = _breakhis_class_name(image_path)
        if class_name not in BREAKHIS_CLASSES:
            continue
        patient_id = _breakhis_patient_id(image_path)
        eva_split = "val" if patient_id in BREAKHIS_VAL_PATIENT_IDS else "train"
        samples.append(
            _sample(
                "breakhis",
                root,
                image_path,
                class_name,
                BREAKHIS_CLASSES[class_name],
                eva_split,
            )
        )
    if not samples:
        raise ValueError(f"No supported BreaKHis class images found under: {root}")
    return samples


def _breakhis_class_name(image_path: Path) -> str:
    return image_path.name.split("-")[0].split("_")[-1]


def _breakhis_patient_id(image_path: Path) -> str:
    return image_path.name.split("-")[2]


def _sample(
    dataset_name: str,
    base_dir: Path,
    image_path: Path,
    class_name: str,
    label: int,
    eva_split: str,
) -> _Sample:
    relative_path = image_path.relative_to(base_dir)
    sample_id = _make_sample_id(dataset_name, relative_path)
    return _Sample(
        sample_id=sample_id,
        image_path=base_dir.absolute() / relative_path,
        label=label,
        eva_split=eva_split,
        class_name=class_name,
        source_dataset=dataset_name,
    )


def _class_folder_samples(
    *,
    dataset_name: str,
    base_dir: Path,
    split_dirs: dict[str, Path],
    class_to_label: dict[str, int],
    extension: str,
) -> list[_Sample]:
    samples: list[_Sample] = []
    for eva_split, split_dir in split_dirs.items():
        if split_dir is None or not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory for {dataset_name}: {split_dir}")
        for class_name, label in class_to_label.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing class directory for {dataset_name}: {class_dir}"
                )
            for image_path in sorted(class_dir.glob(f"*{extension}")):
                samples.append(
                    _sample(dataset_name, base_dir, image_path, class_name, label, eva_split)
                )
    return samples


def _iter_images(directory: Path, extensions: tuple[str, ...]) -> Iterable[Path]:
    lower_extensions = tuple(extension.lower() for extension in extensions)
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in lower_extensions:
            yield path


def _class_name_from_label(class_to_label: dict[str, int], label: int) -> str:
    for class_name, class_label in class_to_label.items():
        if class_label == label:
            return class_name
    raise ValueError(f"Unknown class label: {label}")


def _first_existing_dir(*paths: Path) -> Path:
    for path in paths:
        if path.is_dir():
            return path
    return paths[0]


def _ranges_to_indices(ranges: Iterable[tuple[int, int]]) -> list[int]:
    indices: list[int] = []
    for start, stop in ranges:
        indices.extend(range(start, stop))
    return indices


def _make_sample_id(dataset_name: str, relative_path: Path) -> str:
    stem = str(relative_path.with_suffix(""))
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_")
    return f"{dataset_name}_{safe_stem}"


def _normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace("-", "_")
    aliases = {
        "bach": "bach",
        "mhist": "mhist",
        "crc": "crc",
        "nct_crc": "crc",
        "breakhis": "breakhis",
        "break_his": "breakhis",
        "gleason": "gleason_arvaniti",
        "gleason_arvaniti": "gleason_arvaniti",
        "arvaniti": "gleason_arvaniti",
        "patchcam": "patch_camelyon",
        "patch_cam": "patch_camelyon",
        "patch_camelyon": "patch_camelyon",
        "pcam": "patch_camelyon",
    }
    return aliases.get(normalized, normalized)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m soma.curation.eva",
        description="Curate an EVA patch-level classification dataset into a Soma Manifest.",
    )
    parser.add_argument(
        "--name",
        required=True,
        help=f"dataset name, one of: {', '.join(EVA_PATCH_CLASSIFICATION_DATASETS)}",
    )
    parser.add_argument("--raw-root", type=Path, required=True, help="local raw dataset root")
    parser.add_argument("--output-dir", type=Path, required=True, help="curated output dir")
    parser.add_argument(
        "--tune-fraction",
        type=float,
        default=0.2,
        help="fraction of EVA train reserved as Soma tune (0.0 keeps all train for tune-is-test)",
    )
    args = parser.parse_args(argv)

    manifest = curate_eva_patch_dataset(
        args.name, args.raw_root, args.output_dir, tune_fraction=args.tune_fraction
    )
    print(f"curated: {manifest.dataset_csv}")
    print(f"         {manifest.splits_csv}")
    if manifest.summary_json is not None:
        print(f"         {manifest.summary_json}")


if __name__ == "__main__":
    main()
