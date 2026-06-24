"""Un-stubbed slide-manifest sampling seam: real hs2p 4.1.1 ``merged`` mode + mask read.

The companion to the (deliberately stubbed) ``test_pipeline_segmentation_slide_manifest``:
where that test monkeypatches ``sample_slide_rois`` / ``read_mask_region_at_spacing`` to stay
offline, this one drives the genuine soma↔hs2p contract end to end against a small,
soma-owned **synthetic pyramidal WSI + multiresolution label mask** fixture.

It proves what the stub hid: that ``sample_slide_rois`` resolves a sampling spec, runs hs2p
``tile_slide`` in ``CoordinateOutputMode.MERGED``, and consumes the ``{None: merged}``
per-slide collapse without an ``AttributeError`` — and that the same fixture's mask region
reads back a non-empty label window registered to a sampled ROI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from soma.config import MasksConfig, PreprocessingConfig, SamplingConfig

TARGET = 64
SPACING_UM = 0.5
PIXEL_MAPPING = {"background": 0, "tumor": 1}


def _write_pyramidal_tiff(path: Path, array: np.ndarray, *, photometric: str) -> None:
    """Write a small multiresolution (subifds) TIFF carrying a level-0 spacing tag.

    ``resolution`` is pixels-per-cm, so spacing(µm/px) = 1e4 / res — readers (cucim/openslide)
    recover ``SPACING_UM`` at level 0, which is all the slide-manifest sampling path needs.
    """
    res = 1e4 / SPACING_UM  # pixels per centimeter
    levels = [array]
    cur = array
    for _ in range(2):
        cur = cur[::2, ::2]
        levels.append(cur)
    opts = dict(photometric=photometric, tile=(32, 32), resolution=(res, res), resolutionunit="CENTIMETER")
    with tifffile.TiffWriter(path, bigtiff=False) as writer:
        writer.write(levels[0], subifds=len(levels) - 1, **opts)
        for level in levels[1:]:
            writer.write(level, subfiletype=1, **opts)


def _make_fixture(root: Path) -> tuple[Path, Path]:
    """A 256x256 synthetic slide + label mask, tumor in the top-left 128x128 quadrant."""
    size = 256
    image = np.full((size, size, 3), 220, np.uint8)
    image[0:128, 0:128] = (150, 60, 80)  # darker "tissue" so tumor tiles pass tissue checks
    mask = np.zeros((size, size), np.uint8)
    mask[0:128, 0:128] = 1  # tumor label
    slide_path = root / "slide.tif"
    mask_path = root / "mask.tif"
    _write_pyramidal_tiff(slide_path, image, photometric="rgb")
    _write_pyramidal_tiff(mask_path, mask, photometric="minisblack")
    return slide_path, mask_path


def _segmentation_manifest(root: Path, slide_path: Path, mask_path: Path):
    from soma.dataset import SegmentationManifest

    manifest = root / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,mask_path\n" f"s0,{slide_path},{mask_path}\n",
        encoding="utf-8",
    )
    return SegmentationManifest(manifest)


def test_sample_slide_rois_runs_real_hs2p_merged_mode(tmp_path: Path):
    """Real hs2p 4.1.1 merged-mode sampling: no AttributeError, non-empty ROI coords."""
    from soma.dense_slide_extraction import sample_slide_rois

    slide_path, mask_path = _make_fixture(tmp_path)
    dataset = _segmentation_manifest(tmp_path, slide_path, mask_path)
    masks = MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0})
    sampling = SamplingConfig(strategy="joint", output_mode="merged")
    preprocessing = PreprocessingConfig(
        backend="auto",
        requested_tile_size_px=TARGET,
        requested_spacing_um=SPACING_UM,
        tolerance=0.07,
        min_coverage={"tissue": 0.0},
        overlap=0.0,
    )

    coords_by_slide = sample_slide_rois(
        dataset, masks=masks, sampling=sampling, preprocessing=preprocessing
    )

    assert set(coords_by_slide) == {"s0"}
    coords = coords_by_slide["s0"]
    assert coords, "merged-mode sampling returned no ROI coordinates for an annotated slide"
    # Coordinates are level-0 origins on a TARGET grid within the 256px slide.
    assert all(isinstance(x, int) and isinstance(y, int) for x, y in coords)
    assert all(0 <= x < 256 and 0 <= y < 256 for x, y in coords)


def test_mask_region_read_back_from_fixture(tmp_path: Path):
    """The same fixture's label mask reads back a non-empty window at a sampled ROI origin."""
    from soma.dense.reader import read_mask_region_at_spacing
    from soma.dense_slide_extraction import sample_slide_rois

    slide_path, mask_path = _make_fixture(tmp_path)
    dataset = _segmentation_manifest(tmp_path, slide_path, mask_path)
    masks = MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0})
    sampling = SamplingConfig(strategy="joint", output_mode="merged")
    preprocessing = PreprocessingConfig(
        backend="auto",
        requested_tile_size_px=TARGET,
        requested_spacing_um=SPACING_UM,
        tolerance=0.07,
        min_coverage={"tissue": 0.0},
        overlap=0.0,
    )

    coords = sample_slide_rois(
        dataset, masks=masks, sampling=sampling, preprocessing=preprocessing
    )["s0"]
    # Read the label window registered to the first sampled ROI.
    x, y = coords[0]
    region = read_mask_region_at_spacing(
        mask_path,
        location=(x, y),
        size=(TARGET, TARGET),
        spacing_um=SPACING_UM,
        backend="auto",
        tolerance=0.07,
    )
    assert region.shape == (TARGET, TARGET)
    assert region.size > 0
    assert np.issubdtype(region.dtype, np.integer)
    # The tumor quadrant guarantees at least one tile whose window contains tumor labels.
    union_labels = set()
    for x, y in coords:
        win = read_mask_region_at_spacing(
            mask_path,
            location=(x, y),
            size=(TARGET, TARGET),
            spacing_um=SPACING_UM,
            backend="auto",
            tolerance=0.07,
        )
        union_labels.update(int(v) for v in np.unique(win))
    assert 1 in union_labels, "no sampled ROI window contained the tumor label"
