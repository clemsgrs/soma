"""Curators for the kaiko-ai/eva slide-level datasets (Camelyon16Small, PANDASmall).

EVA's offline slide-level configs embed a fixed set of tiles per slide and train an
attention-MIL head on the bags. These curators choose the same tiles EVA chooses and
write them as hs2p tiling artifacts, one per slide, named by the manifest's
``coordinates_path`` column. soma then skips its own tiling and slide2vec embeds exactly
those tiles (see "Bring your own coordinates" in the preprocessing docs).

Tile selection ports EVA's ``PatchCoordinates.from_file`` with a
``ForegroundGridSampler`` (kaiko-ai/eva @ f5d80152):

* **Spacing.** EVA's openslide rule: the mean of ``openslide.mpp-x`` and
  ``openslide.mpp-y``, else the TIFF resolution tags (:func:`eva_slide_mpp`). It is
  written as the manifest's ``spacing_at_level_0`` and into each artifact, so soma and
  slide2vec use EVA's value, not hs2p's own spacing resolution (to the last few units
  in the last place: see :func:`manifest_spacing`).
* **Grid.** Level 0, no overlap, cell side ``int(target_mpp / mpp * 224)``. Cells are
  listed x-outer / y-inner and their indices shuffled with ``np.random.default_rng(42)``.
* **Foreground.** HSV saturation above 20 (no blur, no hole filling) at the coarsest
  pyramid level where a cell still covers at least 9 mask pixels. A cell is kept when
  at least 35 % of its mask pixels are foreground. The first ``max_samples`` kept cells,
  in shuffled order, are the slide's tiles.
* **Read plan.** Tiles are read at ``get_closest_level(target_mpp)`` (the coarsest level
  whose spacing does not exceed the target, else the finest level) with side
  ``int(target_mpp / (mpp * downsample) * 224)``. slide2vec resizes each read to 224 px.
  For both datasets the read is at level 0, 222 to 247 px wide.

The artifact stores the selected tiles sorted by position (hs2p's canonical order); the
attention-MIL head does not depend on tile order.

Splits and labels:

* **camelyon16_small** (raw root: ``images/<slide>.tif`` + ``evaluation/reference.csv``):
  the 399 slides listed in ``reference.csv``, labelled by its ``type`` column
  (``normal`` = 0, ``tumor`` = 1, EVA's ``class_to_idx``; the ``class`` column grades the
  metastasis as ``negative`` / ``micro`` / ``macro`` and is kept as metadata). ``test_*``
  slides are soma ``test``; the 54
  training slides EVA holds out for validation (the PatchCamelyon validation slides) are
  soma ``tune``; the rest are ``train`` (216 / 54 / 129).
* **panda_small** (raw root: ``train_images/<image_id>.tiff`` +
  ``train_with_noisy_labels.csv``): the image ids in the label CSV, sorted, minus the
  ones EVA filters as noisy (it keeps ``noise_ratio_10 != 0``), labelled by
  ``isup_grade``, then EVA's ``stratified_split(train=0.1, val=0.05, test=0.05,
  seed=42)``: 952 / 475 / 475 slides out of 9555. EVA's ``val`` is soma ``tune``.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from soma.curation.manifest import CuratedManifest, write_manifest

logger = logging.getLogger(__name__)

_DATASET_TYPE = "slide"
#: EVA's ``width`` / ``height``: the tile side at the target spacing, and the encoder input.
TILE_PX = 224
#: ``ForegroundGridSampler`` defaults.
MIN_FOREGROUND_RATIO = 0.35
SAMPLER_SEED = 42
#: ``get_mask`` / ``get_mask_level`` defaults.
SATURATION_THRESHOLD = 20
MIN_MASK_PATCH_PIXELS = 3 * 3
#: hs2p's default spacing tolerance, used only for the artifact's ``is_within_tolerance``.
_TOLERANCE = 0.05

#: EVA ``Camelyon16._val_slides`` (the PatchCamelyon validation slides).
CAMELYON16_VAL_SLIDES: tuple[str, ...] = (
    "normal_010", "normal_013", "normal_016", "normal_017", "normal_019", "normal_020",
    "normal_025", "normal_030", "normal_031", "normal_032", "normal_052", "normal_056",
    "normal_057", "normal_067", "normal_076", "normal_079", "normal_085", "normal_095",
    "normal_098", "normal_099", "normal_101", "normal_102", "normal_105", "normal_106",
    "normal_109", "normal_129", "normal_132", "normal_137", "normal_142", "normal_143",
    "normal_148", "normal_152", "tumor_001", "tumor_005", "tumor_011", "tumor_012",
    "tumor_013", "tumor_019", "tumor_031", "tumor_037", "tumor_043", "tumor_046",
    "tumor_057", "tumor_065", "tumor_069", "tumor_071", "tumor_073", "tumor_079",
    "tumor_080", "tumor_081", "tumor_082", "tumor_085", "tumor_097", "tumor_109",
)  # fmt: skip
CAMELYON16_CLASSES: dict[str, int] = {"normal": 0, "tumor": 1}
#: EVA's expected split sizes (``Camelyon16.validate``), recorded in the summary.
CAMELYON16_SPLIT_SIZES: dict[str, int] = {"train": 216, "tune": 54, "test": 129}

#: The noisy-label CSV EVA downloads for PANDA (``PANDA._resources``).
PANDA_LABELS_CSV = "train_with_noisy_labels.csv"
PANDA_LABELS_MD5 = "5e4bfc78bda9603d2e2faf3ed4b21dfa"
PANDA_NUM_CLASSES = 6
#: ``PANDASmall`` split ratios and ``PANDA``'s split seed.
PANDA_SMALL_RATIOS: tuple[float, float, float] = (0.1, 0.05, 0.05)
PANDA_SPLIT_SEED = 42


@dataclass(frozen=True)
class SlideDatasetSpec:
    """The EVA sampling parameters of one slide-level dataset."""

    target_mpp: float
    max_samples: int  # EVA's ``N_PATCHES``: the sampler's cap and the head's pad size


SLIDE_DATASETS: dict[str, SlideDatasetSpec] = {
    "camelyon16_small": SlideDatasetSpec(target_mpp=0.25, max_samples=1000),
    "panda_small": SlideDatasetSpec(target_mpp=0.5, max_samples=200),
}


# --- EVA spacing, mask and level rules ------------------------------------------------

#: ``eva.vision.data.wsi.backends.openslide._conversion_factor_to_micrometer``.
_MICROMETERS_PER_UNIT: dict[str, float] = {
    "meter": 10**6,
    "decimeter": 10**5,
    "centimeter": 10**4,
    "millimeter": 10**3,
    "micrometer": 1,
    "nanometer": 10**-3,
    "picometer": 10**-6,
    "femtometer": 10**-9,
}


def eva_slide_mpp(properties: Mapping[str, str]) -> float:
    """Level-0 µm/px by EVA's openslide rule (``WsiOpenslide.mpp``).

    The mean of ``openslide.mpp-x`` / ``openslide.mpp-y`` when both are set, else the
    TIFF ``XResolution`` / ``YResolution`` tags converted by their unit.
    """
    if properties.get("openslide.mpp-x") and properties.get("openslide.mpp-y"):
        x_mpp = float(properties["openslide.mpp-x"])
        y_mpp = float(properties["openslide.mpp-y"])
    elif (
        properties.get("tiff.XResolution")
        and properties.get("tiff.YResolution")
        and properties.get("tiff.ResolutionUnit")
    ):
        unit = properties["tiff.ResolutionUnit"]
        if unit not in _MICROMETERS_PER_UNIT:
            raise ValueError(f"Unit {unit} not supported.")
        factor = float(_MICROMETERS_PER_UNIT[unit])
        x_mpp = factor / float(properties["tiff.XResolution"])
        y_mpp = factor / float(properties["tiff.YResolution"])
    else:
        raise ValueError("`mpp` cannot be obtained for this slide.")
    return (x_mpp + y_mpp) / 2.0


def manifest_spacing(mpp: float) -> float:
    """The value of ``mpp`` that the slide manifest declares, unchanged by its CSV round trip.

    ``write_manifest`` writes floats exactly, but ``Dataset`` reads the manifest with
    pandas' default float parser, which is not exact: ``0.24309399999999998`` reads back
    as ``0.2430939999999999``. soma's staging compares the manifest's
    ``spacing_at_level_0`` with the artifact's exactly, so both declare the value the
    parser returns (at most a few units in the last place from EVA's). Tile selection
    still uses EVA's exact spacing.
    """
    import io

    value = float(mpp)
    for _ in range(8):
        text = pd.DataFrame({"spacing_at_level_0": [value]}).to_csv(index=False)
        parsed = float(pd.read_csv(io.StringIO(text))["spacing_at_level_0"].iloc[0])
        if parsed == value:
            return value
        value = parsed
    raise ValueError(f"No CSV-stable spacing found near {mpp!r}.")


def eva_closest_level(mpp: float, level_downsamples: Sequence[float], target_mpp: float) -> int:
    """EVA ``Wsi.get_closest_level``: the coarsest level not coarser than the target.

    When every level is coarser than ``target_mpp``, the finest level.
    """
    level_mpps = mpp * np.array(level_downsamples)
    filtered = level_mpps.copy()
    filtered[filtered > target_mpp] = 0
    if filtered.max() == 0:
        return int(np.argmin(level_mpps))
    return int(np.argmax(filtered))


def eva_mask_level(
    mpp: float,
    level_downsamples: Sequence[float],
    target_mpp: float,
    *,
    tile_px: int = TILE_PX,
) -> int:
    """EVA ``get_mask_level``: the coarsest level where a tile covers >= 9 mask pixels."""
    level_mpps = mpp * np.array(level_downsamples)
    for level, level_mpp in reversed(list(enumerate(level_mpps))):
        ratio = target_mpp / level_mpp
        if int(ratio * tile_px) * int(ratio * tile_px) >= MIN_MASK_PATCH_PIXELS:
            return level
    raise ValueError("No level with the specified minimum number of patch pixels available.")


def eva_foreground_mask(rgba: np.ndarray) -> np.ndarray:
    """EVA ``get_mask`` on one level read: ``1`` where HSV saturation exceeds 20.

    ``rgba`` is an openslide ``read_region`` result. Fully transparent pixels are white,
    as in EVA's ``Wsi._read_postprocess``.
    """
    import cv2

    data = np.array(rgba, copy=True)
    if data.shape[2] == 4:
        data[data[:, :, 3] == 0] = 255
    hsv = cv2.cvtColor(np.ascontiguousarray(data[:, :, :3]), cv2.COLOR_RGB2HSV)
    _, mask = cv2.threshold(hsv[:, :, 1], SATURATION_THRESHOLD, 1, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


@dataclass(frozen=True)
class EvaSlideCoordinates:
    """The tiles EVA samples from one slide, in EVA's (shuffled) order."""

    x: np.ndarray  # level-0 tile origins
    y: np.ndarray
    foreground_fractions: np.ndarray
    mpp: float  # EVA's level-0 spacing
    tile_size_lv0: int  # grid cell side at level 0
    read_level: int
    read_tile_size_px: int
    mask_level: int
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]

    @property
    def read_spacing_um(self) -> float:
        return float(self.mpp * self.level_downsamples[self.read_level])


def _open_slide(path: Path):
    try:
        import openslide
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The EVA slide-level curators read slides with OpenSlide, as EVA does. "
            "Install it with `pip install openslide-python openslide-bin`."
        ) from exc
    return openslide.OpenSlide(str(path))


def sample_eva_coordinates(
    image_path: str | Path,
    *,
    target_mpp: float,
    max_samples: int,
    tile_px: int = TILE_PX,
) -> EvaSlideCoordinates:
    """EVA's ``PatchCoordinates.from_file`` with a ``ForegroundGridSampler``."""
    slide = _open_slide(Path(image_path))
    try:
        mpp = eva_slide_mpp(slide.properties)
        dimensions = tuple(tuple(int(v) for v in d) for d in slide.level_dimensions)
        downsamples = tuple(float(d) for d in slide.level_downsamples)
        mask_level = eva_mask_level(mpp, downsamples, target_mpp, tile_px=tile_px)
        rgba = np.array(slide.read_region((0, 0), mask_level, dimensions[mask_level]))
    finally:
        slide.close()
    mask = eva_foreground_mask(rgba)

    size_lv0 = int(target_mpp / mpp * tile_px)
    width_lv0, height_lv0 = dimensions[0]
    if size_lv0 > width_lv0 or size_lv0 > height_lv0:
        raise ValueError("The width / height cannot be bigger than the layer shape.")
    x_range = range(0, width_lv0 - size_lv0 + 1, size_lv0)
    y_range = range(0, height_lv0 - size_lv0 + 1, size_lv0)
    indices = list(range(len(x_range) * len(y_range)))
    np.random.default_rng(SAMPLER_SEED).shuffle(indices)

    scale_x = dimensions[0][0] / dimensions[mask_level][0]
    scale_y = dimensions[0][1] / dimensions[mask_level][1]
    mask_w, mask_h = int(size_lv0 / scale_x), int(size_lv0 / scale_y)
    xs: list[int] = []
    ys: list[int] = []
    fractions: list[float] = []
    for index in indices:
        if len(xs) >= max_samples:
            break
        # x-outer / y-inner listing: cell ``index`` is (x_range[i // ny], y_range[i % ny]).
        x = x_range[index // len(y_range)]
        y = y_range[index % len(y_range)]
        mx, my = int(x / scale_x), int(y / scale_y)
        patch = mask[my : my + mask_h, mx : mx + mask_w]
        if patch.size == 0:
            continue
        fraction = patch.sum() / patch.size
        if fraction >= MIN_FOREGROUND_RATIO:
            xs.append(x)
            ys.append(y)
            fractions.append(float(fraction))

    read_level = eva_closest_level(mpp, downsamples, target_mpp)
    read_ratio = target_mpp / (mpp * downsamples[read_level])
    return EvaSlideCoordinates(
        x=np.asarray(xs, dtype=np.int64),
        y=np.asarray(ys, dtype=np.int64),
        foreground_fractions=np.asarray(fractions, dtype=np.float32),
        mpp=mpp,
        tile_size_lv0=size_lv0,
        read_level=read_level,
        read_tile_size_px=int(read_ratio * tile_px),
        mask_level=mask_level,
        level_dimensions=dimensions,
        level_downsamples=downsamples,
    )


# --- hs2p tiling artifact ---------------------------------------------------------------


def write_eva_tiling_artifact(
    coordinates: EvaSlideCoordinates,
    *,
    sample_id: str,
    image_path: str | Path,
    target_mpp: float,
    output_dir: Path,
    spacing_at_level_0: float | None = None,
    tile_px: int = TILE_PX,
) -> Path:
    """Write ``coordinates`` as an hs2p tiling artifact and return its ``.coordinates.npz``.

    The artifact declares EVA's read plan, so slide2vec reads each tile at ``read_level``
    with side ``read_tile_size_px`` and resizes it to ``tile_px``. Its level-0 spacing is
    ``spacing_at_level_0`` (default: EVA's ``coordinates.mpp``), which must equal the
    manifest's declaration.
    """
    declared = coordinates.mpp if spacing_at_level_0 is None else float(spacing_at_level_0)
    from hs2p import TileGeometry, TilingResult
    from hs2p.artifacts import save_tiling_result

    n = len(coordinates.x)
    read_spacing = coordinates.read_spacing_um
    tiles = TileGeometry(
        x=coordinates.x,
        y=coordinates.y,
        tissue_fractions=coordinates.foreground_fractions,
        tile_index=np.arange(n, dtype=np.int32),
        requested_tile_size_px=tile_px,
        requested_spacing_um=float(target_mpp),
        read_level=coordinates.read_level,
        read_tile_size_px=coordinates.read_tile_size_px,
        read_spacing_um=read_spacing,
        tile_size_lv0=coordinates.tile_size_lv0,
        is_within_tolerance=abs(read_spacing - target_mpp) / target_mpp <= _TOLERANCE,
        base_spacing_um=declared,
        slide_dimensions=list(coordinates.level_dimensions[0]),
        level_downsamples=list(coordinates.level_downsamples),
        overlap=0.0,
        min_tissue_fraction=MIN_FOREGROUND_RATIO,
    )
    mask_downsample = coordinates.level_downsamples[coordinates.mask_level]
    result = TilingResult(
        tiles=tiles,
        sample_id=sample_id,
        image_path=Path(image_path),
        backend="openslide",
        requested_backend="openslide",
        spacing_at_level_0=declared,
        tolerance=_TOLERANCE,
        step_px_lv0=coordinates.tile_size_lv0,
        tissue_method="eva_saturation",
        requested_seg_downsample=int(round(mask_downsample)),
        seg_downsample=int(round(mask_downsample)),
        seg_level=coordinates.mask_level,
        seg_spacing_um=coordinates.mpp * mask_downsample,
        seg_sthresh=SATURATION_THRESHOLD,
        seg_sthresh_up=255,
        seg_mthresh=0,
        seg_close=0,
        ref_tile_size_px=tile_px,
        a_t=0,
        a_h=0,
        filter_white=False,
        filter_black=False,
        white_threshold=255,
        black_threshold=0,
        fraction_threshold=MIN_FOREGROUND_RATIO,
    )
    artifacts = save_tiling_result(result, output_dir, tiles_dir=output_dir)
    assert artifacts.coordinates_npz_path is not None
    return Path(artifacts.coordinates_npz_path)


# --- splits -----------------------------------------------------------------------------


def eva_stratified_split(
    samples: Sequence[Any],
    targets: Sequence[Any],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int] | None]:
    """EVA ``eva.core.data.splitting.stratified_split`` (ungrouped), line for line.

    Per class (in sorted class order), shuffle the class's indices with one shared
    ``np.random.default_rng(seed)`` and take ``floor(ratio * n) or 1`` of them for each
    split in turn.
    """
    samples = list(samples)
    targets = list(targets)
    if train_ratio + val_ratio + test_ratio > 1.0:
        raise ValueError("The sum of the ratios must be lower or equal to 1")
    if len(samples) != len(targets):
        raise ValueError("The number of samples and targets must be equal.")

    use_all_samples = train_ratio + val_ratio + test_ratio == 1
    random_generator = np.random.default_rng(seed)
    unique_classes, y_indices = np.unique(targets, return_inverse=True)
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    for c in range(unique_classes.shape[0]):
        class_indices = np.where(y_indices == c)[0]
        random_generator.shuffle(class_indices)
        n_train = int(np.floor(train_ratio * len(class_indices))) or 1
        n_val = (
            len(class_indices) - n_train
            if test_ratio == 0.0 and use_all_samples
            else int(np.floor(val_ratio * len(class_indices))) or 1
        )
        train_indices.extend(int(i) for i in class_indices[:n_train])
        val_indices.extend(int(i) for i in class_indices[n_train : n_train + n_val])
        if test_ratio > 0.0:
            n_test = (
                len(class_indices) - n_train - n_val
                if use_all_samples
                else int(np.floor(test_ratio * len(class_indices))) or 1
            )
            test_indices.extend(
                int(i) for i in class_indices[n_train + n_val : n_train + n_val + n_test]
            )
    return train_indices, val_indices, test_indices or None


def panda_small_split(labels: pd.DataFrame) -> dict[str, list[str]]:
    """EVA ``PANDASmall``'s slides per split, as ``{"train"|"tune"|"test": image_ids}``.

    ``labels`` is the noisy-label CSV indexed by ``image_id``. EVA lists the images
    sorted, keeps the ones whose ``noise_ratio_10`` is not 0 (``_filter_noisy_labels``
    names the kept set "non noisy", but this is the filter it applies) and splits them
    stratified by ``isup_grade``. EVA's ``val`` is soma's ``tune``.
    """
    image_ids = sorted(str(i) for i in labels.index)
    kept = set(labels.index[labels["noise_ratio_10"] != 0].astype(str))
    image_ids = [image_id for image_id in image_ids if image_id in kept]
    targets = [int(labels.loc[image_id, "isup_grade"]) for image_id in image_ids]
    train, val, test = eva_stratified_split(
        image_ids, targets, *PANDA_SMALL_RATIOS, seed=PANDA_SPLIT_SEED
    )
    return {
        "train": [image_ids[i] for i in train],
        "tune": [image_ids[i] for i in val],
        "test": [image_ids[i] for i in test or []],
    }


def camelyon16_small_split(slide_ids: Sequence[str]) -> dict[str, str]:
    """EVA ``Camelyon16``'s split of each slide id (EVA's ``val`` is soma's ``tune``)."""
    val = set(CAMELYON16_VAL_SLIDES)
    return {
        slide_id: "test" if slide_id.startswith("test") else "tune" if slide_id in val else "train"
        for slide_id in slide_ids
    }


# --- curators ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SlideRow:
    """One slide to curate: its manifest identity, label and split."""

    sample_id: str
    image_path: Path
    label: int
    split: str
    metadata: dict[str, Any]


def _default_workers() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def _curate_slides(
    rows: Sequence[SlideRow],
    *,
    spec: SlideDatasetSpec,
    output_dir: Path,
    summary: dict[str, Any],
    workers: int | None,
) -> CuratedManifest:
    """Sample every slide, write its artifact, and write the slide manifest."""
    coordinates_dir = output_dir / "coordinates"
    coordinates_dir.mkdir(parents=True, exist_ok=True)

    def _sample(row: SlideRow) -> tuple[EvaSlideCoordinates, Path | None]:
        coordinates = sample_eva_coordinates(
            row.image_path, target_mpp=spec.target_mpp, max_samples=spec.max_samples
        )
        if len(coordinates.x) == 0:
            return coordinates, None
        path = write_eva_tiling_artifact(
            coordinates,
            sample_id=row.sample_id,
            image_path=row.image_path,
            target_mpp=spec.target_mpp,
            output_dir=coordinates_dir,
            spacing_at_level_0=manifest_spacing(coordinates.mpp),
        )
        return coordinates, path

    worker_count = workers if workers is not None else _default_workers()
    if worker_count > 1:
        # OpenSlide and OpenCV release the GIL while they read and convert pixels.
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            sampled = list(pool.map(_sample, rows))
    else:
        sampled = [_sample(row) for row in rows]

    dataset_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    empty: list[str] = []
    read_sizes: dict[str, int] = {}
    for row, (coordinates, path) in zip(rows, sampled):
        if path is None:
            empty.append(row.sample_id)
            continue
        dataset_rows.append(
            {
                "sample_id": row.sample_id,
                "image_path": str(row.image_path),
                "label": row.label,
                "spacing_at_level_0": manifest_spacing(coordinates.mpp),
                "coordinates_path": str(path),
                "num_tiles": len(coordinates.x),
                "read_level": coordinates.read_level,
                "read_tile_size_px": coordinates.read_tile_size_px,
                **row.metadata,
            }
        )
        split_rows.append({"sample_id": row.sample_id, "split": row.split, "fold": 0})
        key = str(coordinates.read_tile_size_px)
        read_sizes[key] = read_sizes.get(key, 0) + 1
    if empty:
        logger.warning(
            "%d slide(s) have no EVA foreground tile and are left out: %s",
            len(empty),
            empty[:20],
        )
    split_counts: dict[str, int] = {}
    for split_row in split_rows:
        split_counts[split_row["split"]] = split_counts.get(split_row["split"], 0) + 1
    summary = {
        **summary,
        "target_mpp": spec.target_mpp,
        "max_tiles_per_slide": spec.max_samples,
        "tile_px": TILE_PX,
        "num_slides": len(dataset_rows),
        "split_counts": split_counts,
        "read_tile_size_px_counts": read_sizes,
        "slides_without_foreground": empty,
    }
    return write_manifest(
        output_dir,
        dataset_type=_DATASET_TYPE,
        dataset_rows=dataset_rows,
        split_rows=split_rows,
        summary=summary,
    )


def camelyon16_small_rows(raw_root: str | Path) -> list[SlideRow]:
    """EVA Camelyon16Small's slides, labels and splits, sorted by slide id."""
    raw_root = Path(raw_root)
    reference = pd.read_csv(raw_root / "evaluation" / "reference.csv")
    missing_columns = {"image", "type"} - set(reference.columns)
    if missing_columns:
        raise ValueError(
            f"{raw_root / 'evaluation' / 'reference.csv'} lacks column(s) "
            f"{sorted(missing_columns)}; expected the official Camelyon16 reference table."
        )
    types = reference["type"].astype(str).str.lower()
    unknown = sorted(set(types) - set(CAMELYON16_CLASSES))
    if unknown:
        raise ValueError(f"Unknown Camelyon16 type value(s) {unknown}; expected normal/tumor.")
    slide_ids = [str(name).removesuffix(".tif") for name in reference["image"]]
    images = {slide_id: raw_root / "images" / f"{slide_id}.tif" for slide_id in slide_ids}
    missing = sorted(slide_id for slide_id, path in images.items() if not path.is_file())
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} Camelyon16 slide(s) listed in reference.csv are missing from "
            f"{raw_root / 'images'}: {missing[:20]}"
        )
    splits = camelyon16_small_split(slide_ids)
    rows = []
    for slide_id, slide_type, (_, record) in sorted(
        zip(slide_ids, types, reference.iterrows()), key=lambda item: item[0]
    ):
        if not slide_id.startswith("test") and slide_id.split("_")[0] != slide_type:
            # EVA labels training slides by their file name; the table must agree.
            raise ValueError(
                f"reference.csv labels training slide {slide_id!r} as {slide_type!r}, "
                "which contradicts its file name."
            )
        metadata = {"reference_class": str(record["class"])} if "class" in record else {}
        rows.append(
            SlideRow(
                sample_id=slide_id,
                image_path=images[slide_id],
                label=CAMELYON16_CLASSES[slide_type],
                split=splits[slide_id],
                metadata=metadata,
            )
        )
    counts = {split: sum(r.split == split for r in rows) for split in CAMELYON16_SPLIT_SIZES}
    if counts != CAMELYON16_SPLIT_SIZES:
        logger.warning(
            "Camelyon16Small split sizes %s differ from EVA's %s: is the raw root complete?",
            counts,
            CAMELYON16_SPLIT_SIZES,
        )
    return rows


def curate_camelyon16_small(
    raw_root: str | Path, output_dir: str | Path, *, workers: int | None = None
) -> CuratedManifest:
    """Curate EVA's Camelyon16Small: 399 slides, at most 1000 tiles each at 0.25 µm/px."""
    return _curate_slides(
        camelyon16_small_rows(raw_root),
        spec=SLIDE_DATASETS["camelyon16_small"],
        output_dir=Path(output_dir),
        summary={
            "dataset": "camelyon16_small",
            "source": "kaiko-ai/eva Camelyon16 (configs/vision/pathology/offline/classification/camelyon16_small.yaml)",
            "classes": list(CAMELYON16_CLASSES),
            "expected_split_counts": CAMELYON16_SPLIT_SIZES,
        },
        workers=workers,
    )


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panda_small_rows(raw_root: str | Path) -> tuple[list[SlideRow], str]:
    """EVA PANDASmall's slides, labels and splits (sorted by id), and the label CSV md5."""
    raw_root = Path(raw_root)
    labels_csv = raw_root / PANDA_LABELS_CSV
    labels = pd.read_csv(labels_csv, index_col="image_id")
    labels_md5 = _md5(labels_csv)
    if labels_md5 != PANDA_LABELS_MD5:
        logger.warning(
            "%s has md5 %s, not EVA's %s: the split will differ from EVA's.",
            labels_csv,
            labels_md5,
            PANDA_LABELS_MD5,
        )
    image_dir = raw_root / "train_images"
    missing = sorted(
        str(image_id)
        for image_id in labels.index
        if not (image_dir / f"{image_id}.tiff").is_file()
    )
    if missing:
        # EVA lists the images on disk and requires one per label row; a missing image
        # would silently shift the stratified split.
        raise FileNotFoundError(
            f"{len(missing)} PANDA image(s) listed in {PANDA_LABELS_CSV} are missing from "
            f"{image_dir}: {missing[:20]}"
        )
    split_ids = panda_small_split(labels)
    rows = [
        SlideRow(
            sample_id=image_id,
            image_path=image_dir / f"{image_id}.tiff",
            label=int(labels.loc[image_id, "isup_grade"]),
            split=split,
            metadata={"data_provider": str(labels.loc[image_id, "data_provider"])}
            if "data_provider" in labels.columns
            else {},
        )
        for split in ("train", "tune", "test")
        for image_id in split_ids[split]
    ]
    rows.sort(key=lambda row: row.sample_id)
    return rows, labels_md5


def curate_panda_small(
    raw_root: str | Path, output_dir: str | Path, *, workers: int | None = None
) -> CuratedManifest:
    """Curate EVA's PANDASmall: 1902 slides, at most 200 tiles each at 0.5 µm/px."""
    rows, labels_md5 = panda_small_rows(raw_root)
    return _curate_slides(
        rows,
        spec=SLIDE_DATASETS["panda_small"],
        output_dir=Path(output_dir),
        summary={
            "dataset": "panda_small",
            "source": "kaiko-ai/eva PANDASmall (configs/vision/pathology/offline/classification/panda_small.yaml)",
            "labels_csv_md5": labels_md5,
            "labels_csv_matches_eva": labels_md5 == PANDA_LABELS_MD5,
            "num_classes": PANDA_NUM_CLASSES,
        },
        workers=workers,
    )


EVA_SLIDE_DATASETS: tuple[str, ...] = tuple(SLIDE_DATASETS)


def curate_eva_slide_dataset(
    name: str, raw_root: str | Path, output_dir: str | Path, *, workers: int | None = None
) -> CuratedManifest:
    """Curate one EVA slide-level dataset (``"camelyon16_small"`` or ``"panda_small"``)."""
    builders: dict[str, Callable[..., CuratedManifest]] = {
        "camelyon16_small": curate_camelyon16_small,
        "panda_small": curate_panda_small,
    }
    try:
        builder = builders[name.strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unsupported EVA slide dataset '{name}'. Supported: {', '.join(EVA_SLIDE_DATASETS)}"
        ) from None
    return builder(raw_root, output_dir, workers=workers)
