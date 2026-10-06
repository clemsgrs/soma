"""Curators for the kaiko-ai/eva patch-segmentation datasets (CoNSeP, MoNuSAC).

Both curators materialise EVA's exact sample geometry as flat 224 px RGB PNG tiles with
8-bit class-index PNG masks, and write the unified Manifest (``dataset.csv`` with
``label_mask_path``, ``splits.csv``, ``summary.json``). soma's dense path reads flat PNGs
spacing-less, so what the decoder sees is what EVA's decoder saw.

* **CoNSeP** (HoVer-Net layout ``Train|Test/Images/*.png`` + ``Labels/*.mat``): EVA's
  ``GridSampler`` over each 1000x1000 image with 250x250 tiles at the native 0.25 µm/px
  (16 tiles per image, no overlap, full tiles only), each tile ``Resize(224)`` +
  ``CenterCrop(224)``; the ``type_map`` classes are merged 7 -> 5 as HoVer-Net does
  (3 & 4 -> epithelial, 5-7 -> spindle-shaped). ``Train`` is soma ``train``; ``Test`` is
  EVA's validation split, which it reports on, so it is soma ``test``.
* **MoNuSAC** (challenge layout: one ``.tif`` + ``.xml`` per image under
  ``MoNuSAC_images_and_annotations`` / ``MoNuSAC Testing Data and Annotations``): the XML
  polygons are rasterised to a semantic mask (annotation name -> class, anything else ->
  ``Ambiguous`` = 5), then the whole image is ``Resize(224)`` on its short side +
  ``CenterCrop(224)``. Train is soma ``train``, test is soma ``test``.

Images are resized bilinearly with antialiasing (PIL), masks with nearest-neighbour, the
same interpolation torchvision v2 applies to an ``Image`` and a ``Mask``.

CoNSeP's official download link is dead; the archive is mirrored from the HoVer-Net
repository's issue #267. MoNuSAC is two Google Drive archives distributed under
CC BY-NC-SA 4.0 (https://monusac-2020.grand-challenge.org/Data/).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from soma.curation.manifest import CuratedManifest, write_manifest

_DATASET_TYPE = "segmentation"
#: Side of every curated sample (EVA ``RESIZE_DIM``).
OUTPUT_PX = 224
#: Spacing declared on every curated PNG. The tiles are EVA's resized samples and carry no
#: physical scale of their own; the benchmark requests this same value, so the dense path
#: reads them pixel for pixel without resampling.
NOMINAL_SPACING_UM = 0.25

CONSEP_CLASSES: tuple[str, ...] = (
    "background",
    "other",
    "inflammatory",
    "epithelial",
    "spindle-shaped",
)
#: EVA ``CoNSeP(width=250, height=250, target_mpp=0.25)`` with ``overwrite_mpp=0.25``.
CONSEP_TILE_PX = 250
CONSEP_SPLIT_DIRS: dict[str, str] = {"Train": "train", "Test": "test"}

MONUSAC_CLASSES: tuple[str, ...] = (
    "Background",
    "Epithelial",
    "Lymphocyte",
    "Neutrophil",
    "Macrophage",
    "Ambiguous",
)
#: ``Ambiguous`` is test-only and excluded from the loss and the metrics.
MONUSAC_IGNORE_INDEX = 5
MONUSAC_SPLIT_DIRS: dict[str, str] = {
    "MoNuSAC_images_and_annotations": "train",
    "MoNuSAC Testing Data and Annotations": "test",
}

EVA_SEGMENTATION_DATASETS: tuple[str, ...] = ("consep", "monusac")


def curate_eva_segmentation_dataset(
    name: str, raw_root: str | Path, output_dir: str | Path
) -> CuratedManifest:
    """Curate one EVA segmentation dataset (``"consep"`` or ``"monusac"``)."""
    builders: dict[str, Callable[[Path, Path], CuratedManifest]] = {
        "consep": curate_consep,
        "monusac": curate_monusac,
    }
    normalized = name.strip().lower()
    try:
        builder = builders[normalized]
    except KeyError:
        supported = ", ".join(EVA_SEGMENTATION_DATASETS)
        raise ValueError(
            f"Unsupported EVA segmentation dataset '{name}'. Supported: {supported}"
        ) from None
    return builder(Path(raw_root), Path(output_dir))


# --- CoNSeP ---------------------------------------------------------------------------


def consep_merge_classes(type_map: np.ndarray) -> np.ndarray:
    """HoVer-Net's 7 -> 5 class merge: ``4 -> 3`` (epithelial), ``5, 6, 7 -> 4`` (spindle)."""
    array = np.asarray(type_map).astype(np.int64)
    array = np.where(array == 4, 3, array)
    array = np.where(array > 4, 4, array)
    return array


def grid_origins(width: int, height: int, tile: int) -> list[tuple[int, int]]:
    """EVA ``GridSampler`` origins: full tiles only, no overlap, column-major from top-left."""
    xs = range(0, width - tile + 1, tile)
    ys = range(0, height - tile + 1, tile)
    return [(x, y) for x in xs for y in ys]


def curate_consep(raw_root: str | Path, output_dir: str | Path) -> CuratedManifest:
    """Curate CoNSeP into 16 resized 224 px tiles per image under the EVA protocol."""
    from scipy.io import loadmat

    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    samples: list[dict[str, Any]] = []
    for split_dir, split in CONSEP_SPLIT_DIRS.items():
        images_dir = raw_root / split_dir / "Images"
        labels_dir = raw_root / split_dir / "Labels"
        image_paths = sorted(images_dir.glob("*.png"))
        if not image_paths:
            raise FileNotFoundError(
                f"CoNSeP raw root {raw_root} has no '{split_dir}/Images/*.png'; expected "
                "the HoVer-Net layout Train|Test/Images/*.png + Labels/*.mat."
            )
        for image_path in image_paths:
            mask_path = labels_dir / f"{image_path.stem}.mat"
            if not mask_path.is_file():
                raise FileNotFoundError(f"CoNSeP label file missing for {image_path}: {mask_path}")
            image = np.asarray(Image.open(image_path).convert("RGB"))
            type_map = consep_merge_classes(loadmat(mask_path)["type_map"])
            if type_map.shape != image.shape[:2]:
                raise ValueError(
                    f"CoNSeP mask {mask_path} is {type_map.shape}, image is {image.shape[:2]}."
                )
            height, width = type_map.shape
            for x, y in grid_origins(width, height, CONSEP_TILE_PX):
                tile = image[y : y + CONSEP_TILE_PX, x : x + CONSEP_TILE_PX]
                mask_tile = type_map[y : y + CONSEP_TILE_PX, x : x + CONSEP_TILE_PX]
                sample_id = f"consep_{image_path.stem}_x{x}_y{y}"
                tile_png, mask_png = _write_sample(
                    output_dir, split, sample_id, tile, mask_tile
                )
                samples.append(
                    {
                        "sample_id": sample_id,
                        "image_path": str(tile_png),
                        "label_mask_path": str(mask_png),
                        "spacing_at_level_0": NOMINAL_SPACING_UM,
                        "eva_split": split,
                        "source_image": image_path.stem,
                        "tile_x": x,
                        "tile_y": y,
                    }
                )
    summary = {
        "dataset": "consep",
        "class_names": list(CONSEP_CLASSES),
        "num_classes": len(CONSEP_CLASSES),
        "ignore_index": None,
        "tile_px": CONSEP_TILE_PX,
        "output_px": OUTPUT_PX,
        "protocol": "eva-grid-250px-resize-224",
    }
    return _write_segmentation_manifest(samples, output_dir, summary)


# --- MoNuSAC --------------------------------------------------------------------------


def polygon_pixels(
    xs: np.ndarray, ys: np.ndarray, *, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Integer pixel coordinates ``(rows, cols)`` inside a polygon, like ``skimage.draw.polygon``.

    Tests every lattice point of the polygon's bounding box (clipped to the image) with a
    point-in-polygon rule, so a vertex list given in image ``(x, y)`` coordinates fills the
    pixels whose centres fall inside it.
    """
    from matplotlib.path import Path as MplPath

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    x0 = max(int(np.floor(xs.min())), 0)
    x1 = min(int(np.ceil(xs.max())), width - 1)
    y0 = max(int(np.floor(ys.min())), 0)
    y1 = min(int(np.ceil(ys.max())), height - 1)
    if x1 < x0 or y1 < y0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    grid_y, grid_x = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
    points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    inside = MplPath(np.column_stack([xs, ys])).contains_points(points)
    return grid_y.ravel()[inside].astype(np.int64), grid_x.ravel()[inside].astype(np.int64)


def rasterize_monusac_annotations(
    xml_path: str | Path, *, width: int, height: int
) -> np.ndarray:
    """Semantic mask from a MoNuSAC XML: annotation ``Name`` -> class, unknown -> Ambiguous.

    Follows EVA's reader: each ``Annotation`` takes the first ``Attribute`` name as its
    class, every ``Region`` polygon is filled with that class, later annotations overwrite
    earlier ones.
    """
    class_to_idx = {name: index for index, name in enumerate(MONUSAC_CLASSES)}
    root = ElementTree.parse(xml_path).getroot()
    labels = np.zeros((height, width), dtype=np.uint8)
    for annotation in root:
        names = [item.attrib["Name"] for item in annotation[0]]
        if not names:
            raise ValueError(f"MoNuSAC annotation without a class name in {xml_path}")
        class_id = class_to_idx.get(names[0], MONUSAC_IGNORE_INDEX)
        regions = [item for child in annotation for item in child if item.tag == "Region"]
        for region in regions:
            vertices = np.array(
                [(float(v.attrib["X"]), float(v.attrib["Y"])) for v in region[1]],
                dtype=float,
            )
            rows, cols = polygon_pixels(vertices[:, 0], vertices[:, 1], width=width, height=height)
            labels[rows, cols] = class_id
    return labels


def curate_monusac(raw_root: str | Path, output_dir: str | Path) -> CuratedManifest:
    """Curate MoNuSAC into one resized + centre-cropped 224 px sample per image."""
    raw_root = Path(raw_root)
    output_dir = Path(output_dir)
    samples: list[dict[str, Any]] = []
    for split_dir, split in MONUSAC_SPLIT_DIRS.items():
        split_root = raw_root / split_dir
        image_paths = sorted(split_root.rglob("*.tif"))
        if not image_paths:
            raise FileNotFoundError(
                f"MoNuSAC raw root {raw_root} has no '{split_dir}/**/*.tif'; expected the "
                "two extracted challenge archives."
            )
        for image_path in image_paths:
            xml_path = image_path.with_suffix(".xml")
            if not xml_path.is_file():
                raise FileNotFoundError(f"MoNuSAC annotation missing for {image_path}: {xml_path}")
            image = np.asarray(Image.open(image_path).convert("RGB"))
            height, width = image.shape[:2]
            mask = rasterize_monusac_annotations(xml_path, width=width, height=height)
            sample_id = "monusac_" + _relative_stem(image_path, split_root)
            tile_png, mask_png = _write_sample(output_dir, split, sample_id, image, mask)
            samples.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(tile_png),
                    "label_mask_path": str(mask_png),
                    "spacing_at_level_0": NOMINAL_SPACING_UM,
                    "eva_split": split,
                    "source_image": image_path.stem,
                }
            )
    summary = {
        "dataset": "monusac",
        "class_names": list(MONUSAC_CLASSES),
        "num_classes": MONUSAC_IGNORE_INDEX,
        "ignore_index": MONUSAC_IGNORE_INDEX,
        "output_px": OUTPUT_PX,
        "protocol": "eva-whole-image-resize-224-center-crop",
    }
    return _write_segmentation_manifest(samples, output_dir, summary)


# --- Shared machinery -----------------------------------------------------------------


def resize_and_center_crop(array: np.ndarray, *, size: int, nearest: bool) -> np.ndarray:
    """torchvision ``Resize(size)`` (short side, aspect kept) + ``CenterCrop(size)``.

    ``nearest`` selects the mask interpolation; images are resized bilinearly with PIL's
    antialiasing. The output is always ``(size, size[, 3])``.
    """
    height, width = array.shape[:2]
    short, long = (width, height) if width <= height else (height, width)
    if short != size:
        new_short = size
        new_long = int(size * long / short)
        new_w, new_h = (new_short, new_long) if width <= height else (new_long, new_short)
        resample = Image.NEAREST if nearest else Image.BILINEAR
        array = np.asarray(Image.fromarray(array).resize((new_w, new_h), resample=resample))
        height, width = array.shape[:2]
    top = int(round((height - size) / 2.0))
    left = int(round((width - size) / 2.0))
    return np.ascontiguousarray(array[top : top + size, left : left + size])


def _write_sample(
    output_dir: Path, split: str, sample_id: str, image: np.ndarray, mask: np.ndarray
) -> tuple[Path, Path]:
    image_out = resize_and_center_crop(image.astype(np.uint8), size=OUTPUT_PX, nearest=False)
    mask_out = resize_and_center_crop(mask.astype(np.uint8), size=OUTPUT_PX, nearest=True)
    images_dir = output_dir / "images" / split
    masks_dir = output_dir / "masks" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    tile_png = images_dir / f"{sample_id}.png"
    mask_png = masks_dir / f"{sample_id}.png"
    Image.fromarray(image_out, mode="RGB").save(tile_png)
    Image.fromarray(mask_out, mode="L").save(mask_png)
    return tile_png, mask_png


def _relative_stem(path: Path, root: Path) -> str:
    relative = path.relative_to(root).with_suffix("")
    return "_".join(part.replace(" ", "-") for part in relative.parts)


def _write_segmentation_manifest(
    samples: list[dict[str, Any]], output_dir: Path, summary: dict[str, Any]
) -> CuratedManifest:
    if not samples:
        raise ValueError(f"No samples curated for EVA dataset '{summary['dataset']}'")
    ordered = sorted(samples, key=lambda s: s["sample_id"])
    split_rows = [
        {"sample_id": s["sample_id"], "split": s["eva_split"], "fold": 0} for s in ordered
    ]
    counts = {split: sum(1 for r in split_rows if r["split"] == split) for split in ("train", "test")}
    full_summary = {
        **summary,
        "dataset_type": _DATASET_TYPE,
        "nominal_spacing_um": NOMINAL_SPACING_UM,
        "total_samples": len(ordered),
        # EVA reports on its validation split; the run selects on it with tune_is_test.
        "splits": {"train": counts["train"], "tune": 0, "test": counts["test"]},
        "tune_is_test": True,
    }
    return write_manifest(
        output_dir,
        dataset_type=_DATASET_TYPE,
        dataset_rows=ordered,
        split_rows=split_rows,
        summary=full_summary,
    )


__all__ = [
    "CONSEP_CLASSES",
    "CONSEP_TILE_PX",
    "EVA_SEGMENTATION_DATASETS",
    "MONUSAC_CLASSES",
    "MONUSAC_IGNORE_INDEX",
    "NOMINAL_SPACING_UM",
    "OUTPUT_PX",
    "consep_merge_classes",
    "curate_consep",
    "curate_eva_segmentation_dataset",
    "curate_monusac",
    "grid_origins",
    "polygon_pixels",
    "rasterize_monusac_annotations",
    "resize_and_center_crop",
]
