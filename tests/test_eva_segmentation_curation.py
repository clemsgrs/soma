"""Curators for the EVA segmentation datasets reproduce EVA's sample geometry (issue #522)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soma.curation.eva_segmentation import (
    CONSEP_CLASSES,
    MONUSAC_CLASSES,
    MONUSAC_IGNORE_INDEX,
    OUTPUT_PX,
    consep_merge_classes,
    curate_consep,
    curate_eva_segmentation_dataset,
    curate_monusac,
    grid_origins,
    polygon_pixels,
    rasterize_monusac_annotations,
    resize_and_center_crop,
)


# --- fixtures ---------------------------------------------------------------------------


def write_consep_raw(root: Path, *, train: int = 2, test: int = 1, size: int = 1000) -> None:
    """HoVer-Net layout with a ``type_map`` that uses every raw class 0..7."""
    from scipy.io import savemat

    rng = np.random.default_rng(0)
    for split_dir, count, prefix in (("Train", train, "train"), ("Test", test, "test")):
        images = root / split_dir / "Images"
        labels = root / split_dir / "Labels"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(1, count + 1):
            image = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
            Image.fromarray(image).save(images / f"{prefix}_{i}.png")
            type_map = np.zeros((size, size), dtype=np.uint8)
            # One vertical band per raw class 1..7 so every tile row sees all of them.
            band = size // 8
            for raw in range(1, 8):
                type_map[:, raw * band : (raw + 1) * band] = raw
            savemat(labels / f"{prefix}_{i}.mat", {"type_map": type_map, "inst_map": type_map})


def write_monusac_raw(root: Path) -> dict[str, tuple[int, int]]:
    """Challenge layout: one ``.tif`` + ``.xml`` per image under a patient folder."""
    sizes = {}
    layout = {
        "MoNuSAC_images_and_annotations": [("TCGA-A", "img_a", (300, 200)), ("TCGA-B", "img_b", (224, 224))],
        "MoNuSAC Testing Data and Annotations": [("TCGA-C", "img_c", (200, 320))],
    }
    for split_dir, images in layout.items():
        for patient, stem, (width, height) in images:
            folder = root / split_dir / patient
            folder.mkdir(parents=True)
            Image.fromarray(np.full((height, width, 3), 200, dtype=np.uint8)).save(folder / f"{stem}.tif")
            (folder / f"{stem}.xml").write_text(_monusac_xml(width, height))
            sizes[stem] = (width, height)
    return sizes


def _monusac_xml(width: int, height: int) -> str:
    """Two annotations: an Epithelial rectangle and an unknown name (-> Ambiguous)."""
    def polygon(points: list[tuple[float, float]]) -> str:
        vertices = "".join(f'<Vertex X="{x}" Y="{y}" />' for x, y in points)
        return f"<Region><Attributes/><Vertices>{vertices}</Vertices></Region>"

    w, h = width, height
    epithelial = polygon([(10, 10), (w / 2, 10), (w / 2, h / 2), (10, h / 2)])
    # Both regions sit inside the central square so the centre crop keeps them for any aspect.
    unknown = polygon([(0.55 * w, 0.55 * h), (0.75 * w, 0.55 * h), (0.75 * w, 0.75 * h), (0.55 * w, 0.75 * h)])
    return (
        '<?xml version="1.0"?><Annotations>'
        '<Annotation><Attributes><Attribute Name="Epithelial" /></Attributes>'
        f"<Regions>{epithelial}</Regions></Annotation>"
        '<Annotation><Attributes><Attribute Name="Mystery" /></Attributes>'
        f"<Regions>{unknown}</Regions></Annotation>"
        "</Annotations>"
    )


# --- geometry helpers -------------------------------------------------------------------


def test_consep_class_merge_follows_hovernet():
    raw = np.arange(8).reshape(2, 4)
    merged = consep_merge_classes(raw)
    assert merged.tolist() == [[0, 1, 2, 3], [3, 4, 4, 4]]


def test_grid_origins_are_full_tiles_without_overlap():
    assert len(grid_origins(1000, 1000, 250)) == 16
    assert grid_origins(600, 500, 250) == [(0, 0), (0, 250), (250, 0), (250, 250)]  # column-major
    assert grid_origins(200, 200, 250) == []


def test_resize_and_center_crop_matches_torchvision_short_side_rule():
    image = np.zeros((300, 200, 3), dtype=np.uint8)  # portrait: short side = width
    out = resize_and_center_crop(image, size=224, nearest=False)
    assert out.shape == (224, 224, 3)
    mask = np.zeros((200, 320), dtype=np.uint8)
    mask[:, 160:] = 1  # right half; after resize the crop is centred, so both halves survive
    out = resize_and_center_crop(mask, size=224, nearest=True)
    assert out.shape == (224, 224)
    assert set(np.unique(out).tolist()) == {0, 1}
    assert out[:, :100].max() == 0 and out[:, 124:].min() == 1
    square = np.ones((224, 224), dtype=np.uint8)
    assert resize_and_center_crop(square, size=224, nearest=True).shape == (224, 224)


def test_polygon_pixels_fill_interior_lattice_points():
    rows, cols = polygon_pixels(np.array([2, 6, 6, 2]), np.array([1, 1, 4, 4]), width=10, height=10)
    filled = set(zip(rows.tolist(), cols.tolist()))
    assert (2, 3) in filled and (3, 5) in filled
    assert (0, 0) not in filled and (5, 7) not in filled
    assert cols.max() <= 6 and rows.max() <= 4


def test_rasterize_monusac_assigns_classes_and_ambiguous(tmp_path: Path):
    xml = tmp_path / "a.xml"
    xml.write_text(_monusac_xml(100, 80))
    mask = rasterize_monusac_annotations(xml, width=100, height=80)
    assert mask.shape == (80, 100)
    assert mask[20, 20] == MONUSAC_CLASSES.index("Epithelial")
    assert mask[52, 65] == MONUSAC_IGNORE_INDEX  # unknown annotation name
    assert mask[0, 0] == 0


# --- curators ---------------------------------------------------------------------------


def test_curate_consep_writes_sixteen_resized_tiles_per_image(tmp_path: Path):
    raw = tmp_path / "raw"
    write_consep_raw(raw, train=2, test=1)
    manifest = curate_consep(raw, tmp_path / "out")

    dataset = pd.read_csv(manifest.dataset_csv)
    splits = pd.read_csv(manifest.splits_csv)
    assert len(dataset) == 3 * 16
    assert list(dataset.columns[:4]) == ["sample_id", "image_path", "label_mask_path", "spacing_at_level_0"]
    assert set(dataset.spacing_at_level_0) == {0.25}
    assert splits["split"].value_counts().to_dict() == {"train": 32, "test": 16}
    assert set(splits["fold"]) == {0}
    # EVA Train -> soma train, EVA Test (its validation split) -> soma test.
    merged = dataset.merge(splits, on="sample_id")
    assert set(merged.loc[merged.source_image.str.startswith("train"), "split"]) == {"train"}
    assert set(merged.loc[merged.source_image.str.startswith("test"), "split"]) == {"test"}

    row = dataset.iloc[0]
    tile = np.asarray(Image.open(row.image_path))
    mask = np.asarray(Image.open(row.label_mask_path))
    assert tile.shape == (OUTPUT_PX, OUTPUT_PX, 3)
    assert mask.shape == (OUTPUT_PX, OUTPUT_PX) and mask.dtype == np.uint8
    # The raw classes 0..7 collapse onto 0..4 (two bands of 125 px per raw class, so every
    # 250 px tile sees two raw classes and the merge is visible per tile).
    values = set()
    for path in dataset.label_mask_path:
        values |= set(np.unique(np.asarray(Image.open(path))).tolist())
    assert values == set(range(len(CONSEP_CLASSES)))

    summary = json.loads(manifest.summary_json.read_text())
    assert summary["dataset_type"] == "segmentation"
    assert summary["num_classes"] == 5 and summary["ignore_index"] is None
    assert summary["splits"] == {"train": 32, "tune": 0, "test": 16}
    assert summary["tune_is_test"] is True


def test_curate_consep_is_byte_stable(tmp_path: Path):
    raw = tmp_path / "raw"
    write_consep_raw(raw, train=1, test=1)
    first = curate_consep(raw, tmp_path / "a")
    second = curate_consep(raw, tmp_path / "b")
    a = first.dataset_csv.read_text().replace(str(tmp_path / "a"), "X")
    b = second.dataset_csv.read_text().replace(str(tmp_path / "b"), "X")
    assert a == b


def test_curate_consep_requires_hovernet_layout(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Train/Images"):
        curate_consep(tmp_path / "empty", tmp_path / "out")


def test_curate_monusac_resizes_whole_images_and_keeps_ambiguous(tmp_path: Path):
    raw = tmp_path / "raw"
    write_monusac_raw(raw)
    manifest = curate_monusac(raw, tmp_path / "out")

    dataset = pd.read_csv(manifest.dataset_csv)
    splits = pd.read_csv(manifest.splits_csv)
    assert len(dataset) == 3
    assert splits["split"].value_counts().to_dict() == {"train": 2, "test": 1}
    for path in dataset.image_path:
        assert np.asarray(Image.open(path)).shape == (OUTPUT_PX, OUTPUT_PX, 3)
    masks = {Path(p).stem: np.asarray(Image.open(p)) for p in dataset.label_mask_path}
    assert all(m.shape == (OUTPUT_PX, OUTPUT_PX) for m in masks.values())
    # Every image carries the Epithelial rectangle and the unknown-name region (Ambiguous=5).
    for mask in masks.values():
        values = set(np.unique(mask).tolist())
        assert MONUSAC_CLASSES.index("Epithelial") in values
        assert MONUSAC_IGNORE_INDEX in values
        assert values <= set(range(len(MONUSAC_CLASSES)))

    summary = json.loads(manifest.summary_json.read_text())
    assert summary["num_classes"] == 5 and summary["ignore_index"] == 5
    assert summary["class_names"] == list(MONUSAC_CLASSES)


def test_curate_monusac_requires_both_archives(tmp_path: Path):
    raw = tmp_path / "raw"
    (raw / "MoNuSAC_images_and_annotations" / "p").mkdir(parents=True)
    Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(
        raw / "MoNuSAC_images_and_annotations" / "p" / "a.tif"
    )
    (raw / "MoNuSAC_images_and_annotations" / "p" / "a.xml").write_text(_monusac_xml(50, 50))
    with pytest.raises(FileNotFoundError, match="Testing Data"):
        curate_monusac(raw, tmp_path / "out")


def test_curate_eva_segmentation_dataset_dispatches_by_name(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported EVA segmentation dataset 'bach'"):
        curate_eva_segmentation_dataset("bach", tmp_path, tmp_path / "out")
    raw = tmp_path / "raw"
    write_consep_raw(raw, train=1, test=1, size=500)
    manifest = curate_eva_segmentation_dataset("CoNSeP", raw, tmp_path / "out")
    assert len(pd.read_csv(manifest.dataset_csv)) == 2 * 4
