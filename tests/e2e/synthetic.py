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

    def flat_manifest(self, task: str) -> Path:
        """``manifest(task)`` over PNG copies of the slides.

        A PNG carries no spacing, so the manifest declares ``spacing_at_level_0``.
        """
        import tifffile

        flat = self.root / "slides_png"
        flat.mkdir(exist_ok=True)
        frame = pd.read_csv(self.manifest(task))
        for sample_id in self.sample_ids:
            png = flat / f"{sample_id}.png"
            if not png.is_file():
                Image.fromarray(tifffile.imread(self.root / "slides" / f"{sample_id}.tif")).save(png)
        frame["image_path"] = [str(flat / f"{s}.png") for s in frame["sample_id"]]
        frame["spacing_at_level_0"] = SPACING_UM
        path = self.root / f"dataset_{task}_png.csv"
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


_MARKER_TILE_PX = 32
#: Brighter than tissue in green and darker in red, so a marker moves one feature up
#: (visible to max pooling) and one down. It has the tissue's HSV saturation (0.30):
#: tissue detection thresholds saturation, and a marker that stood apart from tissue
#: there would be split off from it.
_MARKER = np.array([150, 215, 205])


@dataclass(frozen=True)
class MarkerCohort:
    """Slides whose label is "contains marker tiles", with rigged, noise-free features.

    Every tile is one flat colour, so the literal encoder returns exactly that colour as
    the tile's feature: ordinary tissue tiles are jittered shades of ``_LIGHT``, marker
    tiles jittered shades of ``_MARKER``. Slides come in pairs of equal bag size, one with
    markers and one without, so bag size, sample order and split carry no label signal.
    """

    root: Path
    sample_ids: list[str]
    has_marker: np.ndarray
    #: Pair index of each slide; the two slides of a pair share a bag size and a split.
    pair: np.ndarray
    bag_size: np.ndarray
    splits_csv: Path

    def manifest(self, *, control: bool = False) -> Path:
        """``label`` = has_marker; with ``control``, labels independent of the markers.

        The control flips the labels of every other round of the train/train/tune/test
        deal, so within every split half the positives carry markers and half do not: a
        model that reads the tiles scores at chance, and only a leak (labels reaching the
        model some other way) scores higher.
        """
        label = self.has_marker.astype(int)
        if control:
            label = np.where((self.pair // 4) % 2 == 1, 1 - label, label)
        path = self.root / ("dataset_control.csv" if control else "dataset.csv")
        pd.DataFrame(
            {
                "sample_id": self.sample_ids,
                "image_path": [str(self.root / "slides" / f"{s}.tif") for s in self.sample_ids],
                "label": label,
            }
        ).to_csv(path, index=False)
        return path


def make_marker_cohort(root: Path, n_pairs: int = 24, seed: int = 0) -> MarkerCohort:
    """``2 * n_pairs`` slides with a tissue block of 4 to 64 flat-colour tiles.

    Blocks are aligned to a 64 px grid so that both 32 px tiles and HIPT's 2x2-tile
    regions cover whole tiles. Each tile's colour is jittered by at most 10 per channel:
    enough to make every tile distinct, too little to drop its saturation below the
    tissue threshold (a desaturated tile is cut out of the tissue mask as a hole). A slide with markers has 25-50% marker tiles at random
    positions. Pairs are dealt train/train/tune/test round-robin.
    """
    rng = np.random.default_rng(seed)
    slides = root / "slides"
    slides.mkdir(parents=True)
    tile = _MARKER_TILE_PX
    sides = np.array([2, 4, 6, 8])
    size = 2 * tile + sides.max() * tile + 2 * tile

    names = rng.permutation(2 * n_pairs)
    sample_ids, has_marker, pair, bag_size = [], [], [], []
    for index in range(n_pairs):
        rows, cols = (int(v) for v in rng.choice(sides, 2))
        if index < 4:
            # The first deal round gives every split a 4-tile bag, so a batch there
            # mixes a bag smaller than typical pseudo-bag / grid sizes with larger ones.
            rows, cols = 2, 2
        for marked in (index % 2 == 0, index % 2 == 1):
            sample_id = f"slide{names[len(sample_ids)]:02d}"
            n_tiles = rows * cols
            colours = _LIGHT + rng.integers(-10, 11, (n_tiles, 3))
            if marked:
                n_markers = max(1, int(round(rng.uniform(0.25, 0.5) * n_tiles)))
                where = rng.choice(n_tiles, n_markers, replace=False)
                colours[where] = _MARKER + rng.integers(-10, 11, (n_markers, 3))
            image = np.empty((size, size, 3), dtype=np.int16)
            image[:] = _BACKGROUND
            for k, colour in enumerate(colours):
                y = 2 * tile + (k // cols) * tile
                x = 2 * tile + (k % cols) * tile
                image[y : y + tile, x : x + tile] = colour
            write_tiff(slides / f"{sample_id}.tif", image.astype(np.uint8))
            sample_ids.append(sample_id)
            has_marker.append(marked)
            pair.append(index)
            bag_size.append(n_tiles)

    pair = np.array(pair)
    split = np.array([("train", "train", "tune", "test")[p % 4] for p in pair])
    splits_csv = root / "splits.csv"
    pd.DataFrame({"sample_id": sample_ids, "split": split}).to_csv(splits_csv, index=False)
    return MarkerCohort(
        root=root,
        sample_ids=sample_ids,
        has_marker=np.array(has_marker),
        pair=pair,
        bag_size=np.array(bag_size),
        splits_csv=splits_csv,
    )


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


def write_coordinates_artifact(
    output_dir: Path,
    *,
    sample_id: str,
    image_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    tile_size_px: int,
    spacing_um: float = SPACING_UM,
    slide_size: int,
    spacing_at_level_0: float | None = None,
) -> Path:
    """Hand-write one hs2p tiling artifact (level-0 tile origins) and return its ``.npz``.

    It is what a user who brings their own tile set writes: hs2p's own artifact format,
    built without running hs2p tiling. Every tile is read at level 0, at the size asked for.
    """
    from hs2p import TileGeometry, TilingResult
    from hs2p.artifacts import save_tiling_result

    n = len(x)
    tiles = TileGeometry(
        x=np.asarray(x, dtype=np.int64),
        y=np.asarray(y, dtype=np.int64),
        tissue_fractions=np.ones(n, dtype=np.float32),
        tile_index=np.arange(n, dtype=np.int32),
        requested_tile_size_px=tile_size_px,
        requested_spacing_um=spacing_um,
        read_level=0,
        read_tile_size_px=tile_size_px,
        read_spacing_um=spacing_um,
        tile_size_lv0=tile_size_px,
        is_within_tolerance=True,
        base_spacing_um=spacing_um,
        slide_dimensions=[slide_size, slide_size],
        level_downsamples=[1.0],
        overlap=0.0,
        min_tissue_fraction=0.0,
    )
    result = TilingResult(
        tiles=tiles,
        sample_id=sample_id,
        image_path=image_path,
        backend="openslide",
        requested_backend="openslide",
        spacing_at_level_0=spacing_at_level_0,
        tolerance=0.05,
        step_px_lv0=tile_size_px,
        tissue_method="user",
        requested_seg_downsample=1,
        seg_downsample=1,
        seg_level=0,
        seg_spacing_um=spacing_um,
        seg_sthresh=0,
        seg_sthresh_up=255,
        seg_mthresh=0,
        seg_close=0,
        ref_tile_size_px=tile_size_px,
        a_t=0,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=255,
        black_threshold=0,
        fraction_threshold=0.0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = save_tiling_result(result, output_dir, tiles_dir=output_dir)
    assert artifacts.coordinates_npz_path is not None
    return Path(artifacts.coordinates_npz_path)
