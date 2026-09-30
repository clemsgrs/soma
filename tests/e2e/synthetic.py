"""Synthetic cohorts whose labels are a learnable function of tissue colour.

Every scenario runs the real preprocessing and extraction stack over images written
to disk, with a weight-free encoder that returns the mean RGB of what it reads (see
``tests/dense_literal_encoder.py``). The generators therefore put the supervision
signal in the colour: a model that learns nothing scores at chance, a pipeline that
feeds the wrong pixels, labels or features to the model cannot score well.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

SPACING_UM = 0.5
_RESOLUTION = (1e4 / SPACING_UM, 1e4 / SPACING_UM)  # pixels per cm
_BACKGROUND = np.array([240, 238, 240])
_LIGHT = np.array([215, 150, 205])
_DARK = np.array([95, 40, 125])


def write_tiff(path: Path, image: np.ndarray) -> None:
    import tifffile

    tile = min(256, image.shape[0])
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        tile=(tile, tile),
        resolution=_RESOLUTION,
        resolutionunit="CENTIMETER",
    )


def _tissue_colour(severity: float) -> np.ndarray:
    return _LIGHT + severity * (_DARK - _LIGHT)


def _noisy(image: np.ndarray, rng: np.random.Generator, amplitude: int) -> np.ndarray:
    noise = rng.integers(-amplitude, amplitude, image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@dataclass(frozen=True)
class SlideCohort:
    """One set of whole-slide images; each task reads its own manifest over them."""

    root: Path
    sample_ids: list[str]
    severity: np.ndarray
    splits_csv: Path

    def manifest(self, task: str) -> Path:
        """Write ``dataset_<task>.csv`` whose labels derive from ``severity``."""
        z = self.severity
        frame = pd.DataFrame(
            {
                "sample_id": self.sample_ids,
                "image_path": [str(self.root / "slides" / f"{s}.tif") for s in self.sample_ids],
            }
        )
        if task == "binary_classification":
            frame["label"] = (z > 0.5).astype(int)
        elif task in {"multiclass_classification", "ordinal_classification"}:
            frame["label"] = np.digitize(z, [1 / 3, 2 / 3])
        elif task == "regression":
            frame["label"] = np.round(10.0 * z, 4)
        elif task == "survival":
            # Higher severity -> earlier event; every fourth sample is censored early.
            time = np.round(1.0 + 20.0 * (1.0 - z), 4)
            event = np.ones(len(z), dtype=int)
            event[::4] = 0
            time[::4] = np.round(time[::4] * 0.8, 4)
            frame["label"] = time
            frame["event"] = event
            frame["bin"] = pd.qcut(time, 4, labels=False)
        else:
            raise ValueError(task)
        path = self.root / f"dataset_{task}.csv"
        frame.to_csv(path, index=False)
        return path

    def cv_splits(self, n_folds: int) -> Path:
        """``splits_cv.csv``: each fold deals the severity ranking with rotated roles."""
        ranked = np.argsort(self.severity)
        rows = []
        for fold in range(n_folds):
            for rank, index in enumerate(ranked):
                role = ("train", "train", "tune", "test")[(rank + 2 * fold) % 4]
                rows.append((fold, self.sample_ids[index], role))
        path = self.root / "splits_cv.csv"
        pd.DataFrame(rows, columns=["fold", "sample_id", "split"]).to_csv(path, index=False)
        return path


def make_slide_cohort(root: Path, n: int = 32, size: int = 384, seed: int = 0) -> SlideCohort:
    """``n`` tiled TIFF slides: a tissue disc whose colour darkens with severity.

    Severities are evenly spread and interleaved across splits, so each split spans
    the whole range and every derived label is present in every split.
    """
    rng = np.random.default_rng(seed)
    slides = root / "slides"
    slides.mkdir(parents=True)
    severity = np.linspace(0.02, 0.98, n)
    order = rng.permutation(n)
    severity = severity[order]
    sample_ids = [f"slide{i:02d}" for i in range(n)]

    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < (size * 0.4) ** 2
    for sample_id, z in zip(sample_ids, severity):
        image = np.empty((size, size, 3), dtype=np.int16)
        image[:] = _BACKGROUND
        image[disc] = _tissue_colour(z)
        write_tiff(slides / f"{sample_id}.tif", _noisy(image, rng, 15))

    # Deal the severity-ranked slides round-robin: 1/2 train, 1/4 tune, 1/4 test.
    ranked = np.argsort(severity)
    split = np.empty(n, dtype=object)
    for rank, index in enumerate(ranked):
        split[index] = ("train", "train", "tune", "test")[rank % 4]
    splits_csv = root / "splits.csv"
    pd.DataFrame({"sample_id": sample_ids, "split": split}).to_csv(splits_csv, index=False)
    return SlideCohort(root=root, sample_ids=sample_ids, severity=severity, splits_csv=splits_csv)


def make_tile_dataset(root: Path, n: int = 32, size: int = 64, seed: int = 0) -> tuple[Path, Path]:
    """Patch images for tile-level binary classification (light vs dark tissue)."""
    rng = np.random.default_rng(seed)
    tiles = root / "tiles"
    tiles.mkdir(parents=True)
    sample_ids = [f"tile{i:02d}" for i in range(n)]
    labels = [i % 2 for i in range(n)]
    for sample_id, label in zip(sample_ids, labels):
        severity = rng.uniform(0.6, 1.0) if label else rng.uniform(0.0, 0.4)
        image = np.empty((size, size, 3), dtype=np.int16)
        image[:] = _tissue_colour(severity)
        Image.fromarray(_noisy(image, rng, 30)).save(tiles / f"{sample_id}.png")
    dataset_csv = root / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": [str(tiles / f"{s}.png") for s in sample_ids],
            "label": labels,
        }
    ).to_csv(dataset_csv, index=False)
    split = [("train", "train", "tune", "test")[(i // 2) % 4] for i in range(n)]
    splits_csv = root / "splits.csv"
    pd.DataFrame({"sample_id": sample_ids, "split": split}).to_csv(splits_csv, index=False)
    return dataset_csv, splits_csv


_CLASS_COLOURS = {1: _LIGHT, 2: _DARK}


def make_dense_dataset(
    root: Path, kind: str, n: int = 12, size: int = 96, seed: int = 0
) -> tuple[Path, Path]:
    """ROIs with coloured structures on background, labelled as masks or points.

    ``kind="segmentation"`` draws discs of two classes and writes a label mask;
    ``kind="detection"`` draws small cells of two classes and writes their centres.
    """
    if kind not in {"segmentation", "detection"}:
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    rois = root / "rois"
    labels = root / "labels"
    rois.mkdir(parents=True)
    labels.mkdir()
    sample_ids = [f"roi{i:02d}" for i in range(n)]
    yy, xx = np.ogrid[:size, :size]
    for sample_id in sample_ids:
        image = np.empty((size, size, 3), dtype=np.int16)
        image[:] = _BACKGROUND
        mask = np.zeros((size, size), dtype=np.uint8)
        points = []
        for cls in (1, 2):
            for _ in range(3):
                x, y = (int(v) for v in rng.integers(12, size - 12, 2))
                if kind == "segmentation":
                    radius = int(rng.integers(8, 16))
                    disc = (xx - x) ** 2 + (yy - y) ** 2 < radius**2
                    mask[disc] = cls
                    image[disc] = _CLASS_COLOURS[cls]
                else:
                    image[y - 5 : y + 6, x - 5 : x + 6] = _CLASS_COLOURS[cls]
                    points.append((x, y, cls - 1))
        write_tiff(rois / f"{sample_id}.tif", _noisy(image, rng, 10))
        if kind == "segmentation":
            Image.fromarray(mask).save(labels / f"{sample_id}.png")
        else:
            pd.DataFrame(points, columns=["x", "y", "class"]).to_csv(
                labels / f"{sample_id}.csv", index=False
            )

    column, suffix = ("label_mask_path", "png") if kind == "segmentation" else ("points_path", "csv")
    dataset_csv = root / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": [str(rois / f"{s}.tif") for s in sample_ids],
            column: [str(labels / f"{s}.{suffix}") for s in sample_ids],
        }
    ).to_csv(dataset_csv, index=False)
    n_train = n - 4
    split = ["train"] * n_train + ["tune"] * 2 + ["test"] * 2
    splits_csv = root / "splits.csv"
    pd.DataFrame({"sample_id": sample_ids, "split": split}).to_csv(splits_csv, index=False)
    return dataset_csv, splits_csv
