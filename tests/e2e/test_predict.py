"""Whole-slide prediction: a segmentation bundle trained on ROIs, then ``soma predict``.

The run trains on the dense segmentation ROIs (discs of two colours on background) over
the weight-free ``fake`` encoder's cached grids. ``soma predict`` then cuts larger slides
with the same discs into hs2p's slide grid, runs the bundle over every tile behind its
frozen encoder stage, and writes one label TIFF per slide.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tifffile
import yaml

from soma.cli import main as soma_main
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, learner_section, run_soma, sha256, trainer_section
from tests.e2e.synthetic import BACKGROUND, CLASS_COLOURS, SPACING_UM, make_dense_dataset, noisy, write_tiff

ROI_PX = 96


def segmentation_config(tmp_path, encoder_name):
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "segmentation", size=ROI_PX)
    return {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "kind": "cached",
            "encoder": {"name": encoder_name},
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": ROI_PX,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
            },
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 8},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": learner_section(
            "dense",
            {
                "decoder": {"name": "lightweight_conv"},
                "head": {"name": "segmentation", "params": {"num_classes": 3}},
            },
            learning_rate=3e-3,
            normalization={"method": "zscore"},
        ),
        "trainer": trainer_section(20),
        "evaluation": {"metrics": ["mean_dice"]},
    }


def write_disc_slide(root, sample_id, *, height, width, seed):
    """A slide of discs of both classes on background, a mask of tissue everywhere, and
    the slide's class of each pixel (0 background, 1 and 2 the discs)."""
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), dtype=np.int16)
    image[:] = BACKGROUND
    truth = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.ogrid[:height, :width]
    for cls in (1, 2):
        for _ in range(8):
            x, y = int(rng.integers(16, width - 16)), int(rng.integers(16, height - 16))
            disc = (xx - x) ** 2 + (yy - y) ** 2 < int(rng.integers(8, 16)) ** 2
            truth[disc] = cls
            image[disc] = CLASS_COLOURS[cls]
    write_tiff(root / f"{sample_id}.tif", noisy(image, rng, 10))
    resolution = (1e4 / SPACING_UM, 1e4 / SPACING_UM)
    tifffile.imwrite(
        root / f"{sample_id}_mask.tif",
        np.ones((height, width), dtype=np.uint8),
        tile=(16, 16),
        resolution=resolution,
        resolutionunit="CENTIMETER",
    )
    return {"sample_id": sample_id, "image_path": f"slides/{sample_id}.tif", "mask_path": f"slides/{sample_id}_mask.tif"}, truth


def mean_dice(predicted: np.ndarray, truth: np.ndarray) -> float:
    scores = []
    for cls in (0, 1, 2):
        both = np.logical_and(predicted == cls, truth == cls).sum()
        scores.append(2 * both / ((predicted == cls).sum() + (truth == cls).sum()))
    return float(np.mean(scores))


def test_predict_writes_a_label_tiff_per_slide_from_a_trained_bundle(
    tmp_path, monkeypatch, encoder_name, encoded_images, new_artifact
):
    artifact = new_artifact("predict_segmentation_slides")
    run = run_soma(segmentation_config(tmp_path, encoder_name), tmp_path / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)

    (tmp_path / "slides").mkdir()
    rows, truths = [], {}
    for k, (height, width) in enumerate(((288, 384), (272, 330))):  # the second is no multiple of the tile
        row, truths[f"slide{k}"] = write_disc_slide(tmp_path / "slides", f"slide{k}", height=height, width=width, seed=k)
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "slides.csv", index=False)
    config = {
        "predict": {"output_dir": "predictions", "bundles": str(run.run_dir.relative_to(tmp_path)), "batch_size": 8},
        "cohort": {"records": "slides.csv"},
        "extraction": {"tiling": {"overlap": 0.5, "backend": "openslide", "mask_backend": "tifffile"}},
    }
    (tmp_path / "predict.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.chdir(tmp_path)  # the config's relative paths resolve against the working directory
    encoded_images.clear()

    soma_main(["predict", "predict.yaml"])

    output_dir = tmp_path / "predictions"
    plan = yaml.safe_load((output_dir / "plan.yaml").read_text())
    # The run directory stands for its bundle, and every path is absolute.
    artifact.check_equal("plan/bundles", plan["predict"]["bundles"] == [str(run.run_dir / "bundle")], True)
    artifact.check_equal("plan/records", plan["cohort"]["records"] == str(tmp_path / "slides.csv"), True)
    artifact.check_equal("plan/output_dir", plan["predict"]["output_dir"] == str(output_dir), True)
    artifact.check_equal("plan/tile_size_px", plan["extraction"]["tiling"]["requested_tile_size_px"], ROI_PX)
    artifact.check_equal("plan/spacing_um", plan["extraction"]["tiling"]["requested_spacing_um"], SPACING_UM)
    artifact.check_at_least("tiles_encoded", sum(encoded_images), 1)
    index = pd.read_parquet(output_dir / "index.parquet")
    artifact.check_equal("index/sample_ids", list(index.sample_id), ["slide0", "slide1"])
    for row in index.itertuples():
        path = output_dir / row.path
        with tifffile.TiffFile(path) as tiff:
            label_map = tiff.pages[0].asarray()
            metadata = json.loads(tiff.pages[0].description)
        truth = truths[row.sample_id]
        artifact.check_equal(f"{row.sample_id}/shape", list(label_map.shape), list(truth.shape))
        artifact.check_equal(f"{row.sample_id}/not_predicted", int((label_map == 0).sum()), 0)
        artifact.check_at_least(f"{row.sample_id}/mean_dice", mean_dice(label_map.astype(int) - 1, truth), 0.8)
        artifact.check_equal(
            f"{row.sample_id}/values",
            metadata["values"],
            {"0": "not predicted", "1": "class_0", "2": "class_1", "3": "class_2"},
        )
    record = json.loads((output_dir / index.path[0]).with_suffix(".json").read_text())
    artifact.check_equal("record/num_tiles", record["num_tiles"], 6 * 8)
    provenance = json.loads((output_dir / "provenance.json").read_text())
    artifact.check_equal("provenance/bundles", len(provenance["bundles"]), 1)

    # The plan replays: every slide is done, so nothing is encoded and no map changes.
    # (The maps' metadata names the bundle's absolute path, so their digests are not
    # recorded: they differ from one session's temporary directory to the next.)
    maps = {row.sample_id: sha256(output_dir / row.path) for row in index.itertuples()}
    encoded_images.clear()
    soma_main(["predict", "predictions/plan.yaml"])
    artifact.check_equal("replay/tiles_encoded", sum(encoded_images), 0)
    replayed = {row.sample_id: sha256(output_dir / row.path) for row in index.itertuples()}
    artifact.check_equal("replay/maps_unchanged", replayed == maps, True)
    artifact.assert_passed()
