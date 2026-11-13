"""Synthetic whole slides, masks and segmentation bundles for the whole-slide tests.

The slides are small tiled TIFFs read through hs2p's ``tifffile`` reader (never cuCIM,
which crashes on some synthetic TIFFs), and the bundles carry the weight-free ``fake``
encoder's recipe, so a whole-slide prediction runs on the CPU in seconds.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch

import soma
from soma.learn import DenseLearner, FitResult
from soma.pipeline.build import build_grid_model

SPACING_UM = 0.5
TILE = 32
#: hs2p's readers for the slides and the masks: never cuCIM, which crashes on some
#: synthetic TIFFs.
TIFFFILE = {"backend": "tifffile", "mask_backend": "tifffile"}
#: The recipe of the encoder a bundle's grids came from: ``fake`` pools 8 px patches.
RECIPE = {"encoder": "fake", "target_size": TILE, "spacing_um": SPACING_UM, "precision": "fp32"}


def px_per_cm(spacing_um: float) -> float:
    return 1e4 / spacing_um


def write_slide(path, image: np.ndarray, *, spacing_um: float = SPACING_UM):
    """A single-level tiled RGB TIFF at ``spacing_um``."""
    tifffile.imwrite(
        path,
        image,
        photometric="rgb",
        tile=(16, 16),
        resolution=(px_per_cm(spacing_um), px_per_cm(spacing_um)),
        resolutionunit="CENTIMETER",
    )
    return path


def write_mask(path, mask: np.ndarray, *, spacing_um: float = SPACING_UM):
    """A single-level tiled uint8 raster at ``spacing_um``: a tissue mask (1 = tissue) or
    an annotation raster."""
    tifffile.imwrite(
        path,
        mask.astype(np.uint8),
        tile=(16, 16),
        resolution=(px_per_cm(spacing_um), px_per_cm(spacing_um)),
        resolutionunit="CENTIMETER",
    )
    return path


def write_pyramid(path, read_level: np.ndarray, *, downsample: int, read_spacing_um: float = SPACING_UM):
    """A two-level tiled RGB TIFF: ``read_level`` at ``read_spacing_um`` is level 1, and
    level 0 repeats each of its pixels ``downsample`` times on each axis."""
    level0 = read_level.repeat(downsample, 0).repeat(downsample, 1)
    base_spacing = read_spacing_um / downsample
    with tifffile.TiffWriter(path) as tiff:
        options = {"photometric": "rgb", "tile": (16, 16), "resolutionunit": "CENTIMETER"}
        resolution = px_per_cm(base_spacing)
        tiff.write(level0, subifds=1, resolution=(resolution, resolution), **options)
        resolution = px_per_cm(read_spacing_um)
        tiff.write(read_level, subfiletype=1, resolution=(resolution, resolution), **options)
    return path, level0.shape[:2], base_spacing


def write_large_slide(path, side: int, *, tile: int = 512):
    """A ``side`` x ``side`` px uniform RGB pyramid at 0.5 um/px, written tile by tile,
    with levels down to below 512 px so hs2p reads the mask on a small grid."""
    downsamples = [4**k for k in range(8) if side // 4**k >= 512]
    block = np.full((tile, tile, 3), 200, dtype=np.uint8)
    with tifffile.TiffWriter(path, bigtiff=True) as tiff:
        for k, downsample in enumerate(downsamples):
            size = side // downsample
            resolution = px_per_cm(SPACING_UM * downsample)
            tiles = (block for _ in range((-(-size // tile)) ** 2))
            options = {"subifds": len(downsamples) - 1} if k == 0 else {"subfiletype": 1}
            tiff.write(
                tiles,
                shape=(size, size, 3),
                dtype=np.uint8,
                tile=(tile, tile),
                photometric="rgb",
                compression="zlib",
                resolution=(resolution, resolution),
                resolutionunit="CENTIMETER",
                **options,
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


def save_bundle(path, *, seed: int = 0, num_classes: int = 3, decoder=None, head=None, **preprocessing):
    """A segmentation bundle over the ``fake`` encoder's grids, with random weights.

    ``head`` adds segmentation head params, such as a class scheme (``label_remap``,
    ``pixel_mapping`` and ``class_names``); ``preprocessing`` overrides what the bundle
    records in ``preprocessing.json``.
    """
    stage = soma.extract.live_encoder(**RECIPE)
    torch.manual_seed(seed)
    head = dict(head or {})
    model = build_grid_model(
        feature_dim=stage.feature_dim,
        decoder=decoder or {"name": "linear", "params": {}},
        head={"name": "segmentation", "params": {"num_classes": num_classes, "geometry": stage.geometry, **head}},
    )
    fit = FitResult(DenseLearner(model), pd.DataFrame(), 0, ("loss", 0.0), None, 0)
    remap = model.head.label_remap
    recorded = {
        "shape": "grid",
        "task_family": "segmentation",
        "num_classes": num_classes,
        "class_names": list(model.head.class_names),
        "ignore_index": int(model.head.ignore_index),
        "label_remap": None if remap is None else remap.tolist(),
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


def peak_rss_of(child: str, *args: str, timeout: int = 300) -> dict:
    """Run ``child``, a Python script that prints a JSON line last, in a new process that
    imports this checkout's soma, and return that line parsed."""
    import json

    checkout = str(Path(soma.__file__).resolve().parents[1])
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [checkout, os.environ.get("PYTHONPATH")])),
        "CUDA_VISIBLE_DEVICES": "",
    }
    done = subprocess.run(
        [sys.executable, "-c", child, *args], capture_output=True, text=True, env=env, timeout=timeout
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


#: The start of a memory child script: ``status(key)`` reads ``/proc/self/status`` in
#: bytes, and ``reset_peak()`` restarts the peak RSS from the current one.
MEMORY_PRELUDE = r"""
import json, re, sys
from pathlib import Path

import soma

def status(key):
    text = Path("/proc/self/status").read_text()
    return int(re.search(rf"^{key}:\s+(\d+) kB", text, re.M).group(1)) * 1024

def reset_peak():
    try:
        Path("/proc/self/clear_refs").write_text("5")  # the peak restarts from here
    except OSError:
        pass  # the peak then includes what came before, which only makes a bound harder
"""
