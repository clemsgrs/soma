"""Curators for the kaiko-ai/eva patch-segmentation datasets (CoNSeP, MoNuSAC).

Both curators materialise EVA's online-protocol sample geometry as flat RGB PNGs with
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
  ``Ambiguous`` = 5). EVA's online config trains on a ``RandomResizedCrop(224)`` of the
  whole image drawn every step, so train images are written whole at their native size
  (the benchmark crops them on soma's live path); test images are ``Resize(224)`` on
  their short side + ``CenterCrop(224)`` (EVA's ``ResizeAndCrop``). Train is soma
  ``train``, test is soma ``test``.

Resizing and cropping go through torchvision v2 on a ``tv_tensors.Image`` and a
``tv_tensors.Mask`` exactly as EVA's ``ResizeAndCrop`` does (antialiased bilinear for the
image, nearest for the mask), and MoNuSAC polygons are filled with ``skimage.draw.polygon``
as EVA's reader does, so the curated pixels match EVA's byte for byte.

CoNSeP's official Warwick download requires a login; the Kaggle mirror
``karthikperupogu/consep`` carries the same HoVer-Net layout. MoNuSAC is two Google Drive archives distributed under
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
    """Integer pixel coordinates ``(rows, cols)`` filled by ``skimage.draw.polygon``.

    EVA's MoNuSAC reader calls ``draw.polygon(X, Y, (width, height))`` and indexes the
    mask as ``[Y, X]``; this is the same fill expressed row-major, so boundary pixels and
    vertex winding are handled exactly as upstream (clipped to the image).
    """
    from skimage import draw

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size < 3:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rows, cols = draw.polygon(ys, xs, (height, width))
    return rows.astype(np.int64), cols.astype(np.int64)


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
    """Curate MoNuSAC: whole train images, resized + centre-cropped 224 px test images."""
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
            tile_png, mask_png = _write_sample(
                output_dir, split, sample_id, image, mask, resize=split == "test"
            )
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
        "protocol": "eva-online-train-whole-image-test-resize-224-center-crop",
    }
    return _write_segmentation_manifest(samples, output_dir, summary)


# --- Shared machinery -----------------------------------------------------------------


def resize_and_center_crop(array: np.ndarray, *, size: int, nearest: bool) -> np.ndarray:
    """torchvision v2 ``Resize(size)`` (short side, aspect kept) + ``CenterCrop(size)``.

    ``nearest=True`` treats ``array`` as a ``tv_tensors.Mask`` (nearest-neighbour, like
    EVA's mask target); otherwise it is a ``tv_tensors.Image`` (antialiased bilinear on
    the uint8 tensor, like EVA's image). The output is always ``(size, size[, 3])``.
    """
    import torch
    from torchvision import tv_tensors
    from torchvision.transforms import v2

    transform = v2.Compose([v2.Resize(size), v2.CenterCrop(size)])
    if nearest:
        tensor = tv_tensors.Mask(torch.from_numpy(np.ascontiguousarray(array)).to(torch.int64))
        return transform(tensor).numpy().astype(array.dtype)
    chw = np.ascontiguousarray(np.moveaxis(array, -1, 0))
    tensor = tv_tensors.Image(torch.from_numpy(chw))
    return np.ascontiguousarray(np.moveaxis(transform(tensor).numpy(), 0, -1))


def _write_sample(
    output_dir: Path,
    split: str,
    sample_id: str,
    image: np.ndarray,
    mask: np.ndarray,
    *,
    resize: bool = True,
) -> tuple[Path, Path]:
    image_out = image.astype(np.uint8)
    mask_out = mask.astype(np.uint8)
    if resize:
        image_out = resize_and_center_crop(image_out, size=OUTPUT_PX, nearest=False)
        mask_out = resize_and_center_crop(mask_out, size=OUTPUT_PX, nearest=True)
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
