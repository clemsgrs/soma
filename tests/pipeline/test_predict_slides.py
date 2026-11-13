"""``soma.predict_slides``: one label TIFF per slide, streamed tile by tile.

The slides are synthetic TIFFs, the bundles carry the weight-free ``fake`` encoder's
recipe, and everything runs on the CPU. A map is read back through hs2p; its value 0 means
"not predicted" and class ``k`` is stored as ``k + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

import soma
from soma.data.images import RegionImageSource
from soma.data.manifests import ImageEntry
from soma.learn import DenseLearner, FitResult
from soma.learn.grid import GridBatch
from soma.models import DenseModel
from soma.models.decoders.base import Decoder
from soma.pipeline.build import build_grid_model, bundle_encoder_stage

SPACING_UM = 0.5
TILE = 32
#: The recipe of the encoder a bundle's grids came from: ``fake`` pools 8 px patches.
RECIPE = {"encoder": "fake", "target_size": TILE, "spacing_um": SPACING_UM, "precision": "fp32"}


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


def _px_per_cm(spacing_um: float) -> float:
    return 1e4 / spacing_um


def write_slide(path, image: np.ndarray, *, spacing_um: float = SPACING_UM):
    """A single-level tiled RGB TIFF at ``spacing_um``."""
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        tile=(16, 16),
        resolution=(_px_per_cm(spacing_um), _px_per_cm(spacing_um)),
        resolutionunit="CENTIMETER",
    )
    return path


def write_mask(path, mask: np.ndarray, *, spacing_um: float = SPACING_UM):
    """A single-level tiled tissue mask (1 = tissue) at ``spacing_um``."""
    tifffile.imwrite(
        path,
        mask.astype(np.uint8),
        tile=(16, 16),
        resolution=(_px_per_cm(spacing_um), _px_per_cm(spacing_um)),
        resolutionunit="CENTIMETER",
    )
    return path


def blocky(height: int, width: int, *, seed: int = 0, block: int = 8) -> np.ndarray:
    """An RGB image of random ``block`` px squares: every patch the encoder pools differs."""
    rng = np.random.default_rng(seed)
    cells = rng.integers(0, 256, size=(-(-height // block), -(-width // block), 3), dtype=np.uint8)
    return np.ascontiguousarray(cells.repeat(block, 0).repeat(block, 1)[:height, :width])


def write_records(path, rows: list[dict]) -> str:
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def save_bundle(path, *, seed: int = 0, num_classes: int = 3, decoder=None, **preprocessing):
    """A segmentation bundle over the ``fake`` encoder's grids, with random weights."""
    stage = soma.extract.live_encoder(**RECIPE)
    torch.manual_seed(seed)
    model = build_grid_model(
        feature_dim=stage.feature_dim,
        decoder=decoder or {"name": "linear", "params": {}},
        head={"name": "segmentation", "params": {"num_classes": num_classes, "geometry": stage.geometry}},
    )
    fit = FitResult(DenseLearner(model), pd.DataFrame(), 0, ("loss", 0.0), None, 0)
    recorded = {
        "shape": "grid",
        "task_family": "segmentation",
        "num_classes": num_classes,
        "class_names": [f"class_{k}" for k in range(num_classes)],
        "tile_size": [TILE, TILE],
        "spacing_um": SPACING_UM,
        "encoder": dict(RECIPE),
        **preprocessing,
    }
    return soma.save(fit, path, preprocessing=recorded)


def read_map(path) -> np.ndarray:
    """The base level of a written label TIFF, read through hs2p."""
    from hs2p.wsi.wsi import WSI

    wsi = WSI(path=path, backend="tifffile")
    width, height = wsi.level_dimensions[0]
    region = np.asarray(wsi.read_region((0, 0), 0, (width, height)))
    return region[..., 0] if region.ndim == 3 else region


def model_of(bundle_path) -> DenseModel:
    """The bundle's model behind its frozen encoder stage, as ``predict_slides`` runs it."""
    bundle = soma.load(bundle_path)
    stage = bundle_encoder_stage(bundle)
    model = bundle.model
    return DenseModel(model.decoder, model.head, encoder=stage, adaptor=model.adaptor).eval()


def tile_pixels(slide_path, regions: dict[str, tuple[int, int]]) -> RegionImageSource:
    entries = {tid: ImageEntry(sample_id=tid, image_path=slide_path) for tid in regions}
    return RegionImageSource(entries, regions=regions, size=TILE, spacing_um=SPACING_UM, backend="tifffile")


def test_overlap_zero_writes_what_soma_predict_returns_for_each_tile(tmp_path):
    """With no overlap, each kept tile's whole output is written: the map read back
    through hs2p, minus 1, is ``soma.predict`` over the same tiles, pixel for pixel, and
    every pixel outside the kept tiles is 0."""
    height, width = 120, 150
    slide = write_slide(tmp_path / "s1.tif", blocky(height, width))
    tissue = np.zeros((height, width), dtype=np.uint8)
    tissue[0:96, 32:128] = 1  # exactly the tiles x in {32, 64, 96}, y in {0, 32, 64}
    mask = write_mask(tmp_path / "s1_mask.tif", tissue)
    records = write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )
    bundle = save_bundle(tmp_path / "run" / "bundle")

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions",
        bundles=[bundle],
        records=records,
        extraction={"tiling": {"overlap": 0.0, "backend": "tifffile"}, "masks": {"min_coverage": {"tissue": 0.5}}},
        batch_size=4,
    )

    (row,) = index.itertuples()
    label_map = read_map(tmp_path / "predictions" / row.path)
    assert label_map.shape == (height, width)
    regions = {f"{x}_{y}": (x, y) for x in (32, 64, 96) for y in (0, 32, 64)}
    pixels = tile_pixels(slide, regions)
    ids = list(regions)
    batches = [
        GridBatch(
            features=None,
            pixels=torch.stack([pixels.load(tid) for tid in ids[start : start + 4]]),
            targets={},
            sample_ids=tuple(ids[start : start + 4]),
        )
        for start in range(0, len(ids), 4)
    ]
    expected = soma.predict(model_of(bundle), batches, device="cpu")
    predicted = np.zeros_like(label_map, dtype=bool)
    for tid, (x, y) in regions.items():
        np.testing.assert_array_equal(label_map[y : y + TILE, x : x + TILE].astype(int) - 1, expected[tid])
        predicted[y : y + TILE, x : x + TILE] = True
    assert not label_map[~predicted].any()
    assert row.num_tiles == 9
    assert row.not_predicted_fraction == pytest.approx(1 - 9 * TILE * TILE / (height * width))


# --- pick one: which tile each pixel comes from --------------------------------------- #

#: Classes of the provenance decoder: its value at a pixel names the tile and the patch.
PROVENANCE_CLASSES = 250


def provenance_class(origin_block_x, origin_block_y, patch_x, patch_y):
    """The class the provenance decoder gives patch ``(patch_x, patch_y)`` of the tile whose
    first patch is block ``(origin_block_x, origin_block_y)`` of the read level."""
    return (13 * origin_block_x + 29 * origin_block_y + 3 * patch_x + 5 * patch_y) % PROVENANCE_CLASSES


@dataclass
class ProvenanceParams:
    """The provenance decoder has no params."""


class ProvenanceDecoder(Decoder):
    """Reads where its tile is from the pixels and paints each 8 px patch with
    :func:`provenance_class`.

    The provenance slide is one colour per 8 px block of the read level, red the block's
    column and green its row, so the ``fake`` encoder's patch means are those colours
    exactly: patch ``(0, 0)`` names the tile's origin, and a patch's index its place in
    the tile.
    """

    def __init__(self, *, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self._num_classes = int(num_classes)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def forward(self, X):
        colours = torch.round((X * 0.25 + 0.5) * 255).long()  # the fake encoder's normalisation
        batch, _, rows, cols = X.shape
        patch_y, patch_x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
        classes = provenance_class(
            colours[:, 0, :1, :1], colours[:, 1, :1, :1], patch_x[None], patch_y[None]
        )
        logits = torch.nn.functional.one_hot(classes, self._num_classes).permute(0, 3, 1, 2).float()
        return logits.repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)


@pytest.fixture
def provenance_decoder(monkeypatch):
    """Register :class:`ProvenanceDecoder` for one test, then restore the registry."""
    from soma.models import decoder_registry

    monkeypatch.setattr(decoder_registry, "_entries", dict(decoder_registry._entries))
    soma.register("decoders", "provenance", ProvenanceDecoder, params=ProvenanceParams)
    return {"name": "provenance", "params": {}}


def write_pyramid(path, read_level: np.ndarray, *, downsample: int, read_spacing_um: float = SPACING_UM):
    """A two-level tiled RGB TIFF: ``read_level`` at ``read_spacing_um`` is level 1, and
    level 0 repeats each of its pixels ``downsample`` times on each axis."""
    level0 = read_level.repeat(downsample, 0).repeat(downsample, 1)
    base_spacing = read_spacing_um / downsample
    with tifffile.TiffWriter(path) as tiff:
        options = {"photometric": "rgb", "tile": (16, 16), "resolutionunit": "CENTIMETER"}
        resolution = _px_per_cm(base_spacing)
        tiff.write(level0, subifds=1, resolution=(resolution, resolution), **options)
        resolution = _px_per_cm(read_spacing_um)
        tiff.write(read_level, subfiletype=1, resolution=(resolution, resolution), **options)
    return path, level0.shape[:2], base_spacing


def block_coordinates(height: int, width: int) -> np.ndarray:
    """RGB: red the column and green the row of each 8 px block."""
    rows, cols = np.mgrid[0:height, 0:width]
    return np.stack([cols // 8, rows // 8, np.full_like(rows, 128)], axis=-1).astype(np.uint8)


def test_each_pixel_comes_from_the_tile_the_pick_one_rule_names(tmp_path, provenance_decoder):
    """Overlap 0.5 on 32 px tiles: a 16 px stride and an 8 px margin. The slide is read at
    a level of downsample 3, so the level-0 stride is 48 px and no tile origin divides
    into map pixels by a power of two. Every map pixel, including the first row and
    column and the right and bottom edges, holds the class of the tile the rule names:
    tile 0 for the first 24 px of an axis, then tile ``(p - 8) // 16``."""
    read_h, read_w = 104, 134  # 6.5 by 8.4 strides: the last tiles overhang the edges
    slide, (height, width), base_spacing = write_pyramid(tmp_path / "s1.tif", block_coordinates(read_h, read_w), downsample=3)
    mask = write_mask(tmp_path / "s1_mask.tif", np.ones((height, width)), spacing_um=base_spacing)
    records = write_records(
        tmp_path / "dataset.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )
    bundle = save_bundle(tmp_path / "run" / "bundle", num_classes=PROVENANCE_CLASSES, decoder=provenance_decoder)

    index = soma.predict_slides(
        output_dir=tmp_path / "predictions",
        bundles=[bundle],
        records=records,
        extraction={"tiling": {"overlap": 0.5, "backend": "tifffile", "mask_backend": "tifffile"}},
    )

    (row,) = index.itertuples()
    # The lattice: origins every 48 level-0 px (16 px at the read level, downsample 3),
    # so the map is the read level's pixel grid.
    assert (row.width, row.height) == (read_w, read_h)
    assert row.spacing_um == pytest.approx(SPACING_UM)
    assert row.num_tiles == len(range(0, width, 48)) * len(range(0, height, 48))
    label_map = read_map(tmp_path / "predictions" / row.path)
    stride, margin = 16, 8

    def tile_of(position: np.ndarray) -> np.ndarray:
        return np.where(position < stride + margin, 0, (position - margin) // stride)

    rows, cols = np.mgrid[0:read_h, 0:read_w]
    tile_x, tile_y = tile_of(cols), tile_of(rows)
    expected = provenance_class(
        tile_x * stride // 8, tile_y * stride // 8, (cols - tile_x * stride) // 8, (rows - tile_y * stride) // 8
    )
    np.testing.assert_array_equal(label_map.astype(int) - 1, expected)
