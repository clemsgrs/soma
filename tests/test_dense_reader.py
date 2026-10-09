from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from soma.dense.reader import (
    UNDECLARED_LABEL,
    build_label_remap,
    read_image_at_spacing,
    read_image_region_at_spacing,
    resolve_class_scheme,
)


def test_non_flat_image_reader_uses_hs2p_spacing_apis(tmp_path: Path, monkeypatch):
    """Non-flat images must route through hs2p's spacing-aware reader APIs, keyword-only.

    The fakes take keyword-only arguments like hs2p 5, so a positional call fails here as
    it would against the real reader; ``raising=True`` fails if an API is absent. The
    mask reads are exercised against real pyramidal TIFFs in
    ``test_dense_slide_sampling_integration.py``.
    """
    import hs2p.wsi.wsi as wsi_mod

    calls: list[tuple] = []

    def fake_init(self, *, path, backend="auto"):
        calls.append(("init", Path(path).name, backend))

    def fake_read_full_at_spacing(self, spacing_um, *, tolerance, interpolation):
        calls.append(("full", spacing_um, tolerance, interpolation))
        return np.array([[[1, 2, 3, 255], [4, 5, 6, 255]]], dtype=np.uint8)

    def fake_read_region_at_spacing(
        self, *, location, requested_spacing_um, size, tolerance, interpolation
    ):
        calls.append(("region", location, requested_spacing_um, size, tolerance, interpolation))
        return np.array([[[7, 8, 9, 255]]], dtype=np.uint8)

    monkeypatch.setattr(wsi_mod.WSI, "__init__", fake_init)
    monkeypatch.setattr(wsi_mod.WSI, "read_full_at_spacing", fake_read_full_at_spacing)
    monkeypatch.setattr(wsi_mod.WSI, "read_region_at_spacing", fake_read_region_at_spacing)

    tif_path = tmp_path / "roi.tif"
    image = read_image_at_spacing(
        tif_path,
        spacing_um=0.5,
        backend="openslide",
        tolerance=0.02,
        interpolation="area",
    )
    region = read_image_region_at_spacing(
        tif_path,
        location=(4, 2),
        size=(1, 1),
        spacing_um=0.5,
        backend="openslide",
        tolerance=0.02,
    )

    np.testing.assert_array_equal(image, np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8))
    np.testing.assert_array_equal(region, np.array([[[7, 8, 9]]], dtype=np.uint8))
    assert calls == [
        ("init", "roi.tif", "openslide"),
        ("full", 0.5, 0.02, "area"),
        ("init", "roi.tif", "openslide"),
        ("region", (4, 2), 0.5, (1, 1), 0.02, "area"),
    ]


# --------------------------------------------------------------------------- #
# Label remap: raw annotation-mask pixel values -> soma class indices.
#
# Annotation rasters carry the dataset's own pixel vocabulary (e.g. BEETLE's
# {0,1,2,3,4}); ``task.params.classes`` (class name -> raw value(s), class index =
# order) and ``task.params.ignore`` (raw values excluded from loss and metrics) say
# how to turn them into training targets. Several raw values may form one class.
# --------------------------------------------------------------------------- #


def test_build_label_remap_merges_values_into_one_class():
    lut = build_label_remap(
        {"tumor": [1, 2], "stroma": [3], "muscle": [4, 5, 6]}, ignore=[0], ignore_index=255
    )
    raw = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    np.testing.assert_array_equal(lut[raw], np.array([255, 0, 0, 1, 2, 2, 2]))


def test_build_label_remap_class_index_follows_declaration_order():
    # Not the raw value's numeric order.
    lut = build_label_remap({"necrosis": [4], "tumor": [1]}, ignore_index=255)
    np.testing.assert_array_equal(lut[np.array([4, 1])], np.array([0, 1]))


def test_build_label_remap_accepts_scalar_value():
    lut = build_label_remap({"tumor": 2, "stroma": [3]}, ignore_index=255)
    np.testing.assert_array_equal(lut[np.array([2, 3])], np.array([0, 1]))


def test_build_label_remap_background_is_an_ordinary_class_name():
    lut = build_label_remap({"background": [0], "tumor": [1]}, ignore_index=255)
    np.testing.assert_array_equal(lut[np.array([0, 1])], np.array([0, 1]))


def test_build_label_remap_marks_undeclared_values():
    # A raw value in neither ``classes`` nor ``ignore`` must not alias a class or be
    # silently ignored: it gets the sentinel the segmentation head fails loud on.
    lut = build_label_remap({"tumor": [1]}, ignore=[0], ignore_index=255)
    np.testing.assert_array_equal(
        lut[np.array([0, 1, 7])], np.array([255, 0, UNDECLARED_LABEL])
    )


def test_build_label_remap_rejects_value_in_two_classes():
    with pytest.raises(ValueError, match="'tumor' and 'stroma' both list raw value 2"):
        build_label_remap({"tumor": [1, 2], "stroma": [2]})


def test_build_label_remap_rejects_value_both_class_and_ignored():
    with pytest.raises(ValueError, match="'tumor' and 'ignore' both list raw value 0"):
        build_label_remap({"tumor": [0, 1]}, ignore=[0])


def test_build_label_remap_rejects_empty_class():
    with pytest.raises(ValueError, match="'tumor' lists no raw values"):
        build_label_remap({"tumor": []})


def test_build_label_remap_rejects_no_classes():
    with pytest.raises(ValueError, match="at least one class"):
        build_label_remap({})


@pytest.mark.parametrize("value", [-1, 256, 1.5, True, "1"])
def test_build_label_remap_rejects_value_outside_single_byte_integers(value):
    with pytest.raises(ValueError, match=r"integers in \[0, 255\]"):
        build_label_remap({"tumor": [value]})


# --------------------------------------------------------------------------- #
# task.params.classes / ignore: the training-side class scheme. Independent of the
# sampling-side ``preprocessing.masks.pixel_mapping`` (which only selects ROIs).
# --------------------------------------------------------------------------- #


def test_class_scheme_derives_class_count_names_and_remap_from_classes():
    num_classes, names, lut = resolve_class_scheme(
        {"classes": {"tumor": [1, 2], "stroma": [3]}, "ignore": [0], "ignore_index": 255},
        annotation_rasters=False,
    )
    assert num_classes == 2
    assert names == ("tumor", "stroma")
    np.testing.assert_array_equal(lut[np.array([0, 1, 2, 3])], np.array([255, 0, 0, 1]))


def test_class_scheme_without_classes_keeps_contiguous_masks_unremapped():
    assert resolve_class_scheme({"num_classes": 3}, annotation_rasters=False) == (
        3,
        ("class_0", "class_1", "class_2"),
        None,
    )


def test_class_scheme_rejects_num_classes_disagreeing_with_classes():
    with pytest.raises(ValueError, match="num_classes=3 disagrees with the 2 classes"):
        resolve_class_scheme(
            {"num_classes": 3, "classes": {"tumor": [1], "stroma": [2]}}, annotation_rasters=False
        )


def test_class_scheme_requires_classes_for_annotation_rasters():
    with pytest.raises(ValueError, match=r"requires task\.params\.classes"):
        resolve_class_scheme({"num_classes": 1}, annotation_rasters=True)


def test_class_scheme_rejects_ignore_without_classes():
    with pytest.raises(ValueError, match=r"task\.params\.ignore needs task\.params\.classes"):
        resolve_class_scheme({"num_classes": 2, "ignore": [0]}, annotation_rasters=False)


def test_resolve_class_scheme_accepts_a_scalar_zero_ignore():
    _, _, lut = resolve_class_scheme(
        {"classes": {"tumor": [1]}, "ignore": 0}, annotation_rasters=True
    )

    assert lut[0] == 255
    assert lut[1] == 0


def test_mask_crop_keeps_raw_values_above_255_in_a_16_bit_png(tmp_path: Path):
    from PIL import Image

    from soma.dense.reader import read_mask_crop, write_mask_crop

    raw = np.array([[0, 255], [256, 65535]], dtype=np.int64)
    path = tmp_path / "masks" / "slide" / "0_0.png"
    write_mask_crop(path, raw)

    with Image.open(path) as image:
        assert image.format == "PNG"
        assert np.asarray(image).max() == 65535
    labels, inside = read_mask_crop(path, size=(2, 2))
    assert inside is None
    assert labels.tolist() == [[0, 255], [256, 65535]]


def test_mask_crop_pads_an_in_slide_rectangle_back_to_its_window(tmp_path: Path):
    from soma.dense.reader import read_mask_crop, write_mask_crop

    path = tmp_path / "edge.png"
    write_mask_crop(path, np.array([[7, 7, 3]], dtype=np.uint8))

    labels, inside = read_mask_crop(path, size=(4, 2))
    assert labels.tolist() == [[7, 7, 3, 0], [0, 0, 0, 0]]
    assert inside.tolist() == [[True, True, True, False], [False, False, False, False]]
    with pytest.raises(ValueError, match="exceed"):
        read_mask_crop(path, size=(2, 2))


@pytest.mark.parametrize("value", [-1, 65536])
def test_mask_crop_rejects_raw_values_a_png_cannot_hold(tmp_path: Path, value):
    from soma.dense.reader import write_mask_crop

    path = tmp_path / "bad.png"
    with pytest.raises(ValueError, match=r"\[0, 65535\]"):
        write_mask_crop(path, np.array([[0, value]], dtype=np.int64))
    assert not path.exists()


@pytest.mark.parametrize("in_slide_shape", [(2, 0), (0, 3), (0, 0)])
def test_mask_crop_stores_a_roi_with_no_in_slide_pixel(tmp_path: Path, in_slide_shape):
    """An edge ROI whose overhang holds less than one target pixel on an axis has an
    empty in-slide rectangle: it is stored all the same, and reads back as all outside."""
    from soma.dense.reader import pad_in_slide_labels, read_mask_crop, write_mask_crop

    path = tmp_path / "empty.png"
    write_mask_crop(path, np.zeros(in_slide_shape, dtype=np.uint8))

    labels, inside = read_mask_crop(path, size=(3, 2))
    expected_labels, expected_inside = pad_in_slide_labels(
        np.zeros(in_slide_shape, dtype=np.uint8), size=(3, 2)
    )
    np.testing.assert_array_equal(labels, expected_labels)
    np.testing.assert_array_equal(inside, expected_inside)
    assert not inside.any()


def test_a_one_pixel_mask_crop_is_not_read_as_empty(tmp_path: Path):
    from soma.dense.reader import read_mask_crop, write_mask_crop

    path = tmp_path / "one.png"
    write_mask_crop(path, np.array([[0]], dtype=np.uint8))

    labels, inside = read_mask_crop(path, size=(2, 2))
    assert inside.tolist() == [[True, False], [False, False]]
