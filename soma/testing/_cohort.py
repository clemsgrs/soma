"""``synthetic_cohort``: whole-slide images whose labels are a learnable function of colour."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["synthetic_cohort"]

#: Level-0 spacing of every synthetic slide, in µm/px (written in the TIFF resolution tags).
SPACING_UM = 0.5
_SIZE_PX = 384
_BACKGROUND = np.array([240, 238, 240])
_LIGHT = np.array([215, 150, 205])
_DARK = np.array([95, 40, 125])
#: The split of the k-th slide by severity rank: half train, a quarter tune, a quarter test.
_DEAL = ("train", "train", "tune", "test")


def _write_tiff(path: Path, image: np.ndarray) -> None:
    import tifffile

    tile = min(256, image.shape[0])
    resolution = 1e4 / SPACING_UM  # pixels per cm
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        tile=(tile, tile),
        resolution=(resolution, resolution),
        resolutionunit="CENTIMETER",
    )


def synthetic_cohort(root: str | Path, n: int = 32, seed: int = 0) -> tuple[Path, Path]:
    """Write ``n`` synthetic slides and the two CSVs that describe them; return their paths.

    Each slide is a tiled TIFF (``root/slides/slideNN.tif``, 384 px at 0.5 µm/px) holding a
    tissue disc on a light background. The disc darkens with the slide's ``severity``
    (spread evenly over ``(0, 1)``, in an order drawn from ``seed``), and ``label`` is
    ``severity > 0.5``: a model scores above chance only if the right pixels, features and
    labels reach it.

    ``root/cohort.csv`` holds ``sample_id``, ``image_path``, ``patient_id``, ``label`` and
    ``severity``; ``root/splits.csv`` holds ``sample_id`` and ``split``. The slides are
    dealt by severity rank, half to ``train`` and a quarter each to ``tune`` and ``test``,
    so every split spans the whole range; two slides of neighbouring severity in one split
    share a patient. The same ``n`` and ``seed`` write the same files.
    """
    root = Path(root).resolve()
    rng = np.random.default_rng(seed)
    slides = root / "slides"
    slides.mkdir(parents=True)
    severity = np.linspace(0.02, 0.98, n)[rng.permutation(n)]
    sample_ids = [f"slide{i:02d}" for i in range(n)]

    yy, xx = np.mgrid[0:_SIZE_PX, 0:_SIZE_PX]
    disc = ((xx - _SIZE_PX / 2) ** 2 + (yy - _SIZE_PX / 2) ** 2) < (_SIZE_PX * 0.4) ** 2
    for sample_id, z in zip(sample_ids, severity):
        image = np.empty((_SIZE_PX, _SIZE_PX, 3), dtype=np.int16)
        image[:] = _BACKGROUND
        image[disc] = _LIGHT + z * (_DARK - _LIGHT)
        noise = rng.integers(-15, 15, image.shape)
        _write_tiff(slides / f"{sample_id}.tif", np.clip(image + noise, 0, 255).astype(np.uint8))

    ranked = np.argsort(severity)
    split = np.empty(n, dtype=object)
    for rank, index in enumerate(ranked):
        split[index] = _DEAL[rank % len(_DEAL)]
    patient = np.empty(n, dtype=object)
    count = 0
    for name in dict.fromkeys(_DEAL):
        members = [index for index in ranked if split[index] == name]
        for k in range(0, len(members), 2):
            for index in members[k : k + 2]:
                patient[index] = f"patient{count:02d}"
            count += 1

    records = root / "cohort.csv"
    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": [str(slides / f"{s}.tif") for s in sample_ids],
            "patient_id": patient,
            "label": (severity > 0.5).astype(int),
            "severity": severity,
        }
    ).to_csv(records, index=False)
    splits = root / "splits.csv"
    pd.DataFrame({"sample_id": sample_ids, "split": split}).to_csv(splits, index=False)
    return records, splits
