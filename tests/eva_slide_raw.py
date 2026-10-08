"""Synthetic raw roots in the layouts the EVA Cam16Small / PANDASmall curators read.

Each slide is a small tiled pyramidal TIFF (OpenSlide reads it, and its resolution tags
give the ``openslide.mpp-*`` properties EVA's spacing rule uses) holding saturated
tissue ellipses on a near-white background. The label is a function of the tissue
colour, so a weight-free mean-RGB encoder can learn it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

_BACKGROUND = (242, 242, 242)
#: Tissue colours from light to dark pink, all well above EVA's saturation threshold.
_LIGHT = np.array([215, 150, 205])
_DARK = np.array([95, 40, 125])


def tissue_colour(severity: float) -> tuple[int, int, int]:
    return tuple(int(v) for v in _LIGHT + severity * (_DARK - _LIGHT))


def write_pyramid(
    path: Path,
    *,
    size: tuple[int, int],
    mpp: float,
    levels: int = 4,
    seed: int = 0,
    blobs: int = 6,
    colour: tuple[int, int, int] = (205, 120, 190),
) -> Path:
    """A tiled pyramidal TIFF (level ``k`` is ``2**k`` smaller) with tissue ellipses.

    The background's HSV saturation is below EVA's threshold of 20; each ellipse is
    ``colour`` with noise, so tiles at the ellipse edges fall on both sides of EVA's 0.35
    foreground ratio.
    """
    width, height = size
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = _BACKGROUND
    yy, xx = np.mgrid[0:height, 0:width]
    for _ in range(blobs):
        cx, cy = rng.integers(0, width), rng.integers(0, height)
        rx, ry = rng.integers(width // 12, width // 4), rng.integers(height // 12, height // 4)
        inside = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1.0
        image[inside] = colour
    noise = rng.integers(-6, 7, image.shape)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    resolution = 1e4 / mpp  # pixels per centimetre
    path.parent.mkdir(parents=True, exist_ok=True)
    with tifffile.TiffWriter(path) as writer:
        for level in range(levels):
            factor = 2**level
            writer.write(
                np.ascontiguousarray(image[::factor, ::factor]),
                photometric="rgb",
                tile=(128, 128),
                resolution=(resolution / factor, resolution / factor),
                resolutionunit="CENTIMETER",
                subfiletype=0 if level == 0 else 1,
            )
    return path


#: A Camelyon16 subset touching every split: train (normal_001-004, tumor_002-004),
#: EVA's validation slides normal_010, normal_013, tumor_001, tumor_005, and test_001-004.
CAMELYON16_SUBSET: dict[str, str] = {
    **{f"normal_{i:03d}": "normal" for i in (1, 2, 3, 4, 10, 13)},
    **{f"tumor_{i:03d}": "tumor" for i in (1, 2, 3, 4, 5)},
    **{f"test_{i:03d}": t for i, t in ((1, "tumor"), (2, "normal"), (3, "normal"), (4, "tumor"))},
}


def write_fake_camelyon16(root: Path, *, mpp: float = 0.2431, size=(1400, 1200)) -> Path:
    """``images/<slide>.tif`` + ``evaluation/reference.csv`` over :data:`CAMELYON16_SUBSET`.

    Tumor slides carry dark tissue, normal slides light tissue.
    """
    rows = []
    for index, (slide_id, slide_type) in enumerate(sorted(CAMELYON16_SUBSET.items())):
        severity = 0.9 if slide_type == "tumor" else 0.1
        write_pyramid(
            root / "images" / f"{slide_id}.tif",
            size=size,
            mpp=mpp,
            seed=index,
            blobs=4,
            colour=tissue_colour(severity),
        )
        reference_class = "macro" if slide_type == "tumor" else "negative"
        rows.append((f"{slide_id}.tif", slide_type, reference_class, index % 2))
    (root / "evaluation").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["image", "type", "class", "center"]).to_csv(
        root / "evaluation" / "reference.csv", index=False
    )
    return root


def write_fake_panda(
    root: Path, *, per_grade: int = 4, mpp: float = 0.4862, size=(1200, 1000)
) -> Path:
    """``train_images/<id>.tiff`` + ``train_with_noisy_labels.csv``, ``per_grade`` per ISUP grade.

    PANDASmall's floor-or-1 split puts one slide of each grade in each of train / tune /
    test; one extra slide per grade is marked noisy (``noise_ratio_10 == 0``) and is
    filtered out. Tissue darkens with the grade.
    """
    rng = np.random.default_rng(0)
    rows = []
    for grade in range(6):
        for k in range(per_grade + 1):
            image_id = rng.bytes(16).hex()
            noisy = k == per_grade
            write_pyramid(
                root / "train_images" / f"{image_id}.tiff",
                size=size,
                mpp=mpp,
                seed=grade * 100 + k,
                blobs=4,
                colour=tissue_colour(grade / 5),
            )
            rows.append(
                {
                    "image_id": image_id,
                    "data_provider": "radboud" if k % 2 else "karolinska",
                    "isup_grade": grade,
                    "gleason_score": "0+0",
                    "noise_ratio_10": 0 if noisy else 1,
                }
            )
    pd.DataFrame(rows).to_csv(root / "train_with_noisy_labels.csv")
    return root
