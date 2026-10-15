"""Rung 0 of the detection decoder ladder: ``decoder: attention_map`` through ``soma run``.

The grids stand in for a ``cls_attention`` cache: four channels per token, of which only
channel 0 (the "cell head") is high where a cell lies; channels 1-3 are noise. The run
trains nothing (the decoder has no parameter), freezes its threshold on tune and is
scored on test. Head selection is the decoder's ``heads`` param: the cell head alone
finds the cells, the noise heads do not.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file

import soma
from soma.data.geometry import compute_dense_geometry
from soma.extract.dense import dense_grid_metadata, write_dense_grid
from tests.e2e.harness import CPU_LOADER, run_soma
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset

ROI_PX = 96
PATCH_PX = 8
HEADS = 4


def write_attention_grids(dataset_csv: Path, out_dir: Path, *, seed: int = 0) -> Path:
    """``(HEADS, 12, 12)`` grids: channel 0 is 1 on the token under each cell, the rest noise."""
    rng = np.random.default_rng(seed)
    frame = pd.read_csv(dataset_csv)
    grid_px = ROI_PX // PATCH_PX
    geometry = compute_dense_geometry(target_size=ROI_PX, patch_size=PATCH_PX)
    for sample_id, points_path in zip(frame["sample_id"], frame["points_path"]):
        grid = rng.uniform(0.0, 0.3, size=(HEADS, grid_px, grid_px)).astype(np.float32)
        for x, y in pd.read_csv(points_path)[["x", "y"]].itertuples(index=False):
            grid[0, int(y) // PATCH_PX, int(x) // PATCH_PX] = 1.0
        metadata = dense_grid_metadata(geometry, feature_dim=HEADS, pad_mode="reflect", spacing_um=SPACING_UM)
        metadata.update({"source_spacing_um": SPACING_UM, "effective_spacing_um": SPACING_UM})
        write_dense_grid(out_dir, str(sample_id), torch.from_numpy(grid), metadata)
    return out_dir


def write_cells(dataset_csv: Path, *, seed: int = 0) -> None:
    """One class (attention is class-blind), cells on a jittered 2 x 2 layout, so no two
    share a token or its neighbour; the grids, not the pixels, carry them."""
    rng = np.random.default_rng(seed)
    for points_path in pd.read_csv(dataset_csv)["points_path"]:
        points = [
            (cx + int(rng.integers(-4, 5)), cy + int(rng.integers(-4, 5)), 0)
            for cx in (24, 72)
            for cy in (24, 72)
            if rng.random() >= 0.2
        ]
        pd.DataFrame(points, columns=["x", "y", "class"]).to_csv(points_path, index=False)


def attention_map_config(tmp_path: Path, dataset_csv: Path, splits_csv: Path, heads: list[int]) -> dict:
    return {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "kind": "cached",
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": ROI_PX,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
            },
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": {
            "name": "dense",
            "params": {
                "decoder": {"name": "attention_map", "params": {"heads": heads}},
                "head": {"name": "detection", "params": {"num_classes": 1, "match_distance": 5.0, "sigma": 1.5}},
            },
        },
        "trainer": {"max_epochs": 5, "callbacks": [{"name": "checkpoint"}]},
        "evaluation": {"metrics": ["mean_f1"]},
    }


@pytest.mark.parametrize(("label", "heads"), [("cell_head", [0]), ("noise_heads", [1, 2, 3])])
def test_attention_map_scores_the_attention_without_training(label, heads, tmp_path, new_artifact):
    artifact = new_artifact(f"dense_attention_map_{label}")
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "detection", size=ROI_PX)
    write_cells(dataset_csv)
    grid_dir = write_attention_grids(dataset_csv, tmp_path / "grids")
    config = attention_map_config(tmp_path, dataset_csv, splits_csv, heads)
    run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(grid_dir))
    artifact.record_run("run", run)

    # Nothing is trained: no epoch ran and the bundle holds no weight.
    artifact.check_equal("history_epochs", run.histories()[""], [])
    weights = load_file(run.run_dir / "bundle" / "weights.safetensors")
    artifact.check_equal("bundle_weights", sorted(weights), [])
    bundle = soma.load(run.run_dir / "bundle")
    artifact.check_equal("bundle_decoder_heads", bundle.model.decoder.heads, heads)

    test_f1 = json.loads((run.run_dir / "metrics.json").read_text())["test"]["mean_f1"]
    if label == "cell_head":
        artifact.check_at_least("test_mean_f1", test_f1, 0.8)
    else:
        artifact.check_at_most("test_mean_f1", test_f1, 0.5)
    artifact.assert_passed()
