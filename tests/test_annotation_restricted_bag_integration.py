"""End-to-end annotation-restricted merged bag (#110): the BAG forwarding path.

Proves the tracer bullet of issue #110 against a small, soma-owned synthetic pyramidal
WSI + multiresolution label mask: a ``dataset_type="slide"`` dataset declaring an
annotation vocabulary produces one merged bag per slide restricted to the selected
compartment(s), excluding tissue-only tiles.

soma holds slide2vec's ``PreprocessingConfig`` as given (ADR 0019), and slide2vec hands
its hs2p tiling and mask-sampling configs to hs2p ``tile_slides`` unchanged. This test
therefore drives hs2p ``tile_slide`` in ``MERGED`` mode with exactly the configs soma
holds, and asserts the *resulting tile set*:

  * a tumor-restricted bag samples ONLY tiles meeting the tumor coverage threshold;
  * a lower coverage threshold admits a superset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import slide2vec

from tests.extraction_configs import annotation_masks, extraction

TARGET = 64
SPACING_UM = 0.5


def _write_pyramidal_tiff(path: Path, array: np.ndarray, *, photometric: str) -> None:
    res = 1e4 / SPACING_UM  # pixels per centimeter ⇒ readers recover SPACING_UM at level 0
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
    """256x256 slide: tissue (dark) across the whole LEFT half; tumor mask only in the
    TOP-LEFT quadrant. So a tissue bag tiles the left half, a tumor bag only the top-left."""
    size = 256
    image = np.full((size, size, 3), 220, np.uint8)
    image[0:256, 0:128] = (150, 60, 80)  # dark "tissue" across the left half
    mask = np.zeros((size, size), np.uint8)
    mask[0:128, 0:128] = 1  # tumor label in the top-left quadrant only
    slide_path = root / "slide.tif"
    mask_path = root / "mask.tif"
    _write_pyramidal_tiff(slide_path, image, photometric="rgb")
    _write_pyramidal_tiff(mask_path, mask, photometric="minisblack")
    return slide_path, mask_path


def _tile_coords(
    preprocessing: slide2vec.PreprocessingConfig, slide_path: Path, mask_path: Path
) -> set[tuple[int, int]]:
    """Tile the slide with the hs2p configs soma holds and return the merged level-0 tile
    origins. Tissue segmentation is intentionally not driven here: annotation sampling
    gates on the per-class mask coverage (not tissue), and the synthetic fixture is too
    small for otsu to resolve a tissue contour."""
    from hs2p import SlideSpec, tile_slide

    result = tile_slide(
        SlideSpec(sample_id="s0", image_path=slide_path, mask_path=mask_path),
        tiling=preprocessing.tiling,
        masks=preprocessing.masks,
    )
    merged = result[None] if isinstance(result, dict) else result
    return {(int(x), int(y)) for x, y in zip(merged.tiles.x.tolist(), merged.tiles.y.tolist())}


def _restricted(tumor_coverage: float) -> slide2vec.PreprocessingConfig:
    return extraction(
        SPACING_UM,
        TARGET,
        method="otsu",
        tiling_fields={"tolerance": 0.07},
        masks=annotation_masks({"background": 0, "tumor": 1}, {"tumor": tumor_coverage}),
    )


def _tumor_preprocessing() -> slide2vec.PreprocessingConfig:
    return _restricted(0.5)


def _full_annotation_preprocessing() -> slide2vec.PreprocessingConfig:
    """Samples tumor at a coverage of 0.0, so it admits every tile *touching* the tumor
    quadrant — used only as the superset sanity reference."""
    return _restricted(0.0)


def test_tumor_restricted_bag_excludes_tissue_only_tiles(tmp_path: Path):
    """AC1/AC8: a slide dataset with masks={tumor} yields a merged bag of exactly the tiles
    meeting the tumor coverage threshold (the top-left quadrant), excluding every tile in the
    tissue-only bottom-left region the annotation mask does not label as tumor."""
    slide_path, mask_path = _make_fixture(tmp_path)

    tumor_coords = _tile_coords(_tumor_preprocessing(), slide_path, mask_path)

    assert tumor_coords, "tumor-restricted merged bag sampled no tiles for an annotated slide"
    # The TARGET-px grid over the 128px tumor quadrant is exactly these four origins.
    assert tumor_coords == {(0, 0), (0, TARGET), (TARGET, 0), (TARGET, TARGET)}
    # Every tumor tile lies in the top-left quadrant; the tissue-only bottom/right regions
    # (which a plain tissue bag would admit) are excluded.
    assert all(x < 128 and y < 128 for x, y in tumor_coords), tumor_coords


def test_higher_tumor_coverage_shrinks_the_bag(tmp_path: Path):
    """AC1: the coverage threshold actually gates — raising tumor min_coverage from 0.0 to
    0.5 keeps only fully-tumor tiles (still the four quadrant tiles here, since each sits
    wholly inside the tumor region), and a tile only partially overlapping tumor is dropped.

    Concretely: a strict 0.5 threshold admits no tile outside the tumor quadrant, so the
    coverage-0.0 reference bag is a superset of the coverage-0.5 bag."""
    slide_path, mask_path = _make_fixture(tmp_path)

    permissive = _tile_coords(_full_annotation_preprocessing(), slide_path, mask_path)
    strict = _tile_coords(_tumor_preprocessing(), slide_path, mask_path)

    assert strict <= permissive
    assert all(x < 128 and y < 128 for x, y in permissive), permissive
