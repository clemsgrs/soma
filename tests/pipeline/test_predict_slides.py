"""``soma.predict_slides``: one label TIFF per slide, streamed tile by tile.

The slides are synthetic TIFFs, the bundles carry the weight-free ``fake`` encoder's
recipe, and everything runs on the CPU. A map is read back through hs2p; its value 0 means
"not predicted" and class ``k`` is stored as ``k + 1``.
"""

from __future__ import annotations

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
