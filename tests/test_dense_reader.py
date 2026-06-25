from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soma.dense.reader import (
    build_label_remap,
    read_image_at_spacing,
    read_mask_at_spacing,
)


def test_non_flat_reader_uses_hs2p_spacing_apis(tmp_path: Path, monkeypatch):
    """Non-flat inputs must route through hs2p's spacing-aware reader APIs.

    This catches CI environments with an older hs2p package: the monkeypatches use
    ``raising=True`` on the real API names, so the test fails if either API is absent.
    """
    import hs2p.wsi.masks as masks_mod
    import hs2p.wsi.wsi as wsi_mod

    calls: list[tuple] = []

    def fake_init(self, path, *, backend="auto"):
        calls.append(("init", Path(path).name, backend))

    def fake_read_full_at_spacing(self, spacing_um, *, tolerance, interpolation):
        calls.append(("image", spacing_um, tolerance, interpolation))
        return np.array([[[1, 2, 3, 255], [4, 5, 6, 255]]], dtype=np.uint8)

    def fake_read_label_at_spacing(wsi, spacing_um, *, tolerance):
        calls.append(("mask", spacing_um, tolerance, type(wsi).__name__))
        return np.array([[0, 1]], dtype=np.uint8)

    monkeypatch.setattr(wsi_mod.WSI, "__init__", fake_init)
    monkeypatch.setattr(wsi_mod.WSI, "read_full_at_spacing", fake_read_full_at_spacing)
    monkeypatch.setattr(masks_mod, "read_label_at_spacing", fake_read_label_at_spacing)

    tif_path = tmp_path / "roi.tif"
    image = read_image_at_spacing(
        tif_path,
        spacing_um=0.5,
        backend="openslide",
        tolerance=0.02,
        interpolation="area",
    )
    mask = read_mask_at_spacing(
        tif_path,
        spacing_um=0.5,
        backend="openslide",
        tolerance=0.02,
    )

    assert image.shape == (1, 2, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image, np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8))
    np.testing.assert_array_equal(mask, np.array([[0, 1]], dtype=np.uint8))
    assert calls == [
        ("init", "roi.tif", "openslide"),
        ("image", 0.5, 0.02, "area"),
        ("init", "roi.tif", "openslide"),
        ("mask", 0.5, 0.02, "WSI"),
    ]


# --------------------------------------------------------------------------- #
# Label remap: raw annotation-mask pixel values -> soma class indices.
#
# The slide-manifest segmentation path reads the *raw* annotation raster (whose
# pixel values are the dataset's own vocabulary, e.g. BEETLE's {0,1,2,3,4}), so
# soma must remap those onto contiguous class indices [0, num_classes) with the
# unannotated ``background`` value collapsed to ``ignore_index``. The remap is
# derived from ``masks.pixel_mapping`` (class name -> raw pixel value), with the
# non-background classes taking class index = their order in the mapping.
# --------------------------------------------------------------------------- #


def test_build_label_remap_beetle_contract():
    # BEETLE: 0 unannotated (background -> ignore), 1 other, 2 non-invasive,
    # 3 invasive, 4 necrosis -> classes 0..3, unannotated -> 255.
    pixel_mapping = {
        "background": 0,
        "other": 1,
        "non_invasive_epithelium": 2,
        "invasive_epithelium": 3,
        "necrosis": 4,
    }
    lut, num_classes = build_label_remap(pixel_mapping, num_classes=4, ignore_index=255)
    assert num_classes == 4
    raw = np.array([[0, 1, 2], [3, 4, 0]], dtype=np.int64)
    remapped = lut[raw]
    np.testing.assert_array_equal(
        remapped, np.array([[255, 0, 1], [2, 3, 255]], dtype=lut.dtype)
    )


def test_build_label_remap_unmapped_values_go_to_ignore():
    # A raw pixel value absent from pixel_mapping must collapse to ignore_index,
    # not silently alias an in-range class.
    lut, num_classes = build_label_remap(
        {"background": 0, "tumor": 1, "stroma": 2}, ignore_index=255
    )
    assert num_classes == 2
    raw = np.array([0, 1, 2, 7, 9], dtype=np.int64)
    np.testing.assert_array_equal(lut[raw], np.array([255, 0, 1, 255, 255]))


def test_build_label_remap_preserves_mapping_order():
    # Class index follows insertion order of the non-background classes, not the
    # raw pixel value's numeric order.
    lut, num_classes = build_label_remap(
        {"background": 0, "necrosis": 4, "tumor": 1}, ignore_index=255
    )
    assert num_classes == 2
    assert int(lut[4]) == 0  # necrosis listed first -> class 0
    assert int(lut[1]) == 1  # tumor listed second -> class 1
    assert int(lut[0]) == 255


def test_build_label_remap_background_as_real_class():
    # When num_classes == len(pixel_mapping), background is a real class (index 0),
    # not the ignore label — the existing slide-manifest contract (raw values already
    # equal class indices). The LUT is then the identity on the mapped values.
    lut, num_classes = build_label_remap(
        {"background": 0, "tumor": 1}, num_classes=2, ignore_index=255
    )
    assert num_classes == 2
    np.testing.assert_array_equal(lut[np.array([0, 1])], np.array([0, 1]))


def test_build_label_remap_rejects_class_count_mismatch():
    with pytest.raises(ValueError, match="num_classes"):
        build_label_remap({"background": 0, "tumor": 1, "stroma": 2}, num_classes=4)


def test_build_label_remap_background_absent_every_label_is_a_class():
    # No reserved 'background' name: with class-count == mapping size, every named
    # label is a real class (index = order) and any unlisted raw value -> ignore.
    lut, num_classes = build_label_remap(
        {"tumor": 1, "stroma": 2}, num_classes=2, ignore_index=255
    )
    assert num_classes == 2
    raw = np.array([0, 1, 2, 7], dtype=np.int64)
    # 0 is unlisted -> ignore; 1 -> tumor (class 0); 2 -> stroma (class 1); 7 -> ignore.
    np.testing.assert_array_equal(lut[raw], np.array([255, 0, 1, 255]))


def test_build_label_remap_background_absent_infers_num_classes():
    # num_classes omitted: with no 'background', every label is a class, so the
    # resolved class count is the mapping size.
    lut, num_classes = build_label_remap({"tumor": 2}, ignore_index=255)
    assert num_classes == 1
    assert int(lut[2]) == 0  # the single named label -> class 0
    assert int(lut[0]) == 255  # unlisted -> ignore


def test_build_label_remap_background_absent_rejects_class_count_mismatch():
    # Without a 'background' name there is no ignore-label mode: class-count must
    # equal the mapping size, so a smaller num_classes is a clear error.
    with pytest.raises(ValueError, match="num_classes"):
        build_label_remap({"tumor": 1, "stroma": 2}, num_classes=1)
