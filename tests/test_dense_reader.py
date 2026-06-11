from __future__ import annotations

from pathlib import Path

import numpy as np

from soma.dense.reader import read_image_at_spacing, read_mask_at_spacing


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
