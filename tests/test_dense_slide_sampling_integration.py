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


def _write_pyramidal_tiff(
    path: Path, array: np.ndarray, *, photometric: str, spacing_um: float = SPACING_UM
) -> None:
    """Write a small multiresolution (subifds) TIFF carrying a level-0 spacing tag.

    ``resolution`` is pixels-per-cm, so spacing(µm/px) = 1e4 / res — readers (cucim/openslide)
    recover ``spacing_um`` at level 0, which is all the slide-manifest sampling path needs.
    """
    res = 1e4 / spacing_um  # pixels per centimeter
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
    label_mask_path = root / "mask.tif"
    _write_pyramidal_tiff(slide_path, image, photometric="rgb")
    _write_pyramidal_tiff(label_mask_path, mask, photometric="minisblack")
    return slide_path, label_mask_path


def _segmentation_manifest(root: Path, slide_path: Path, label_mask_path: Path):
    from soma.dataset import SegmentationManifest

    manifest = root / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path\n" f"s0,{slide_path},{label_mask_path}\n",
        encoding="utf-8",
    )
    return SegmentationManifest(manifest)


def test_sample_slide_rois_runs_real_hs2p_merged_mode(tmp_path: Path):
    """Real hs2p 4.1.1 merged-mode sampling: no AttributeError, non-empty ROI coords."""
    from soma.dense_slide_extraction import sample_slide_rois

    slide_path, label_mask_path = _make_fixture(tmp_path)
    dataset = _segmentation_manifest(tmp_path, slide_path, label_mask_path)
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

    slide_path, label_mask_path = _make_fixture(tmp_path)
    dataset = _segmentation_manifest(tmp_path, slide_path, label_mask_path)
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
        label_mask_path,
        location=(x, y),
        size=(TARGET, TARGET),
        spacing_um=SPACING_UM,
        reference_path=slide_path,
        pixel_mapping=PIXEL_MAPPING,
        backend="auto",
    )
    assert region.shape == (TARGET, TARGET)
    assert region.size > 0
    assert np.issubdtype(region.dtype, np.integer)
    # The tumor quadrant guarantees at least one tile whose window contains tumor labels.
    union_labels = set()
    for x, y in coords:
        win = read_mask_region_at_spacing(
            label_mask_path,
            location=(x, y),
            size=(TARGET, TARGET),
            spacing_um=SPACING_UM,
            reference_path=slide_path,
            pixel_mapping=PIXEL_MAPPING,
            backend="auto",
        )
        union_labels.update(int(v) for v in np.unique(win))
    assert 1 in union_labels, "no sampled ROI window contained the tumor label"


def _make_half_resolution_mask_fixture(root: Path) -> tuple[Path, Path]:
    """A 256x256 slide at 0.5 µm/px and its label mask at half that resolution.

    The 128x128 mask (1.0 µm/px) marks tumor in the slide's top-right quadrant
    (slide x in [128, 256), y in [0, 128)), i.e. mask x in [64, 128), y in [0, 64).
    """
    image = np.full((256, 256, 3), 220, np.uint8)
    mask = np.zeros((128, 128), np.uint8)
    mask[0:64, 64:128] = 1
    slide_path = root / "slide.tif"
    label_mask_path = root / "mask_half.tif"
    _write_pyramidal_tiff(slide_path, image, photometric="rgb")
    _write_pyramidal_tiff(
        label_mask_path, mask, photometric="minisblack", spacing_um=2 * SPACING_UM
    )
    return slide_path, label_mask_path


def test_mask_region_location_is_in_slide_pixels_for_a_coarser_mask(tmp_path: Path):
    """A region's location is in the *slide's* level-0 pixels, even when the mask's level
    0 is coarser. hs2p 4.x read it in mask-file pixels, so slide (128, 0) landed at slide
    (256, 0) — off the tumor quadrant."""
    from soma.dense.reader import read_mask_region_at_spacing

    slide_path, label_mask_path = _make_half_resolution_mask_fixture(tmp_path)

    def region(x: int, y: int) -> np.ndarray:
        return read_mask_region_at_spacing(
            label_mask_path,
            location=(x, y),
            size=(TARGET, TARGET),
            spacing_um=SPACING_UM,
            reference_path=slide_path,
            pixel_mapping=PIXEL_MAPPING,
            backend="auto",
        )

    np.testing.assert_array_equal(region(128, 0), np.ones((TARGET, TARGET), np.uint8))
    np.testing.assert_array_equal(region(64, 0), np.zeros((TARGET, TARGET), np.uint8))
    np.testing.assert_array_equal(region(128, 128), np.zeros((TARGET, TARGET), np.uint8))


def test_full_mask_read_registers_a_coarser_mask_to_the_image_grid(tmp_path: Path):
    """A pre-cropped pyramidal mask coarser than its image is read on the image's grid."""
    from soma.dense.reader import read_mask_at_spacing

    slide_path, label_mask_path = _make_half_resolution_mask_fixture(tmp_path)
    labels = read_mask_at_spacing(
        label_mask_path,
        spacing_um=SPACING_UM,
        size=(256, 256),
        reference_path=slide_path,
        reference_backend="openslide",
        pixel_mapping=PIXEL_MAPPING,
        backend="auto",
    )
    expected = np.zeros((256, 256), np.uint8)
    expected[0:128, 128:256] = 1
    np.testing.assert_array_equal(labels, expected)


def test_mask_read_rejects_a_value_outside_the_declared_vocabulary(tmp_path: Path):
    from soma.dense.reader import read_mask_region_at_spacing

    slide_path, label_mask_path = _make_half_resolution_mask_fixture(tmp_path)
    with pytest.raises(ValueError, match="undeclared label IDs"):
        read_mask_region_at_spacing(
            label_mask_path,
            location=(128, 0),
            size=(TARGET, TARGET),
            spacing_um=SPACING_UM,
            reference_path=slide_path,
            pixel_mapping={"background": 0},
            backend="auto",
        )


def test_mask_read_rejects_a_mask_that_does_not_cover_the_slide(tmp_path: Path):
    from soma.dense.reader import read_mask_at_spacing

    slide_path, _ = _make_half_resolution_mask_fixture(tmp_path)
    narrow = tmp_path / "narrow_mask.tif"
    _write_pyramidal_tiff(
        narrow, np.zeros((128, 96), np.uint8), photometric="minisblack", spacing_um=2 * SPACING_UM
    )
    with pytest.raises(ValueError, match="Mask alignment failed"):
        read_mask_at_spacing(
            narrow,
            spacing_um=SPACING_UM,
            size=(256, 256),
            reference_path=slide_path,
            pixel_mapping=PIXEL_MAPPING,
            backend="auto",
        )


def test_segmentation_coverage_summarizes_every_label_of_the_fixture(tmp_path: Path):
    """The coverage driver against real hs2p: every pixel_mapping label, background too."""
    import pandas as pd

    from soma.curation.segmentation_coverage import summarize_coverage

    slide_path, label_mask_path = _make_fixture(tmp_path)
    manifest = pd.DataFrame(
        {
            "sample_id": ["s0"],
            "image_path": [str(slide_path)],
            "label_mask_path": [str(label_mask_path)],
        }
    )

    row = summarize_coverage(
        manifest,
        pixel_mapping=PIXEL_MAPPING,
        min_coverage={"tumor": 0.5},
        tile_size_px=TARGET,
        spacing_um=SPACING_UM,
        seg_downsample=4,
    ).iloc[0]

    # 256 px at 0.5 µm/px is 0.128 mm a side; tumor fills the top-left quarter.
    assert row["area_mm2_tumor"] == pytest.approx(0.128**2 / 4)
    assert row["area_mm2_background"] == pytest.approx(0.128**2 * 3 / 4)
    assert row["est_tiles_tumor"] == 4


def _make_split_tumor_fixture(root: Path) -> tuple[Path, Path]:
    """A 256x256 slide whose top-left 64 px tile is 20 columns of value 1, 20 of value 2
    and 24 of background: 31% and 31% apart, 62.5% together. Nothing else is annotated."""
    image = np.full((256, 256, 3), 220, np.uint8)
    image[0:64, 0:64] = (150, 60, 80)
    mask = np.zeros((256, 256), np.uint8)
    mask[0:64, 0:20] = 1
    mask[0:64, 20:40] = 2
    slide_path = root / "slide.tif"
    label_mask_path = root / "mask.tif"
    _write_pyramidal_tiff(slide_path, image, photometric="rgb")
    _write_pyramidal_tiff(label_mask_path, mask, photometric="minisblack")
    return slide_path, label_mask_path


def _sample_split_tumor(tmp_path: Path, masks: MasksConfig) -> list[tuple[int, int]]:
    from soma.dense_slide_extraction import sample_slide_rois

    tmp_path.mkdir()
    slide_path, label_mask_path = _make_split_tumor_fixture(tmp_path)
    dataset = _segmentation_manifest(tmp_path, slide_path, label_mask_path)
    preprocessing = PreprocessingConfig(
        backend="auto",
        requested_tile_size_px=TARGET,
        requested_spacing_um=SPACING_UM,
        tolerance=0.07,
        min_coverage={"tissue": 0.0},
        overlap=0.0,
    )
    return sample_slide_rois(
        dataset,
        masks=masks,
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        preprocessing=preprocessing,
    )["s0"]


def test_merged_label_samples_a_tile_only_its_summed_coverage_qualifies(tmp_path: Path):
    """#484: a list-valued pixel_mapping entry is one sampling label whose coverage is the
    sum of its values, so a tile 31% value 1 + 31% value 2 passes min_coverage 0.5."""
    from soma.dense.reader import read_mask_region_at_spacing

    merged = MasksConfig(
        pixel_mapping={"background": 0, "tumor": [1, 2]}, min_coverage={"tumor": 0.5}
    )
    coords = _sample_split_tumor(tmp_path / "merged", merged)
    assert coords == [(0, 0)]

    # The sampled ROI's mask reads back against the same list-valued vocabulary.
    region = read_mask_region_at_spacing(
        tmp_path / "merged" / "mask.tif",
        location=coords[0],
        size=(TARGET, TARGET),
        spacing_um=SPACING_UM,
        reference_path=tmp_path / "merged" / "slide.tif",
        pixel_mapping=merged.pixel_mapping,
        backend="auto",
    )
    assert sorted(np.unique(region).tolist()) == [0, 1, 2]

    separate = MasksConfig(
        pixel_mapping={"background": 0, "tumor_a": 1, "tumor_b": 2},
        min_coverage={"tumor_a": 0.5, "tumor_b": 0.5},
    )
    assert _sample_split_tumor(tmp_path / "separate", separate) == []
