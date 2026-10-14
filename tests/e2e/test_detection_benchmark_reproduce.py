"""``soma reproduce detection/midog --from-run-dir`` on a run over synthetic tiled ROIs.

The ROIs are cut into overlapping supervision tiles by the benchmark's own tiler
(:func:`soma.benchmark.tiles.tile_detection_manifest`), a run on the MIDOG recipe shape
(dense grids, ``lightweight_conv``, single-class Hungarian detection head) trains through
``soma run``, and the registered ``detection/midog`` benchmark then scores that run dir
itself: thresholds frozen on the stitched tune ROIs, test decoded and stitched back to
ROIs, ``predictions.json`` written, the native MIDOG F1 reported. A multi-fold run is
scored per fold and averaged.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soma.benchmark.detection import read_cell_predictions, score_dataset_points
from soma.benchmark.tiles import tile_detection_manifest
from soma.cli import main as soma_main
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, learner_section, run_soma, trainer_section
from tests.e2e.synthetic import BACKGROUND, CLASS_COLOURS, noisy

#: MIDOG's nominal frame and matching margin; ROIs of 96 px cut into 64 px tiles that
#: overlap by 32 px (edge-flush walk), so every ROI yields four tiles.
SPACING_UM = 0.25
MATCH_UM = 7.5
ROI_PX = 96
TILE_PX = 64
OVERLAP_PX = 16
N_ROIS = 8


def write_roi_manifest(root: Path, *, num_folds: int, seed: int = 0) -> Path:
    """Flat PNG ROIs with mitosis-like squares (one class), and a ``num_folds`` split.

    Fold ``k`` tests ROIs ``2k, 2k+1``, tunes the next two and trains on the rest.
    """
    rng = np.random.default_rng(seed)
    (root / "images").mkdir(parents=True)
    (root / "points").mkdir()
    rows, splits = [], []
    sample_ids = [f"roi{i}" for i in range(N_ROIS)]
    for sample_id in sample_ids:
        image = np.empty((ROI_PX, ROI_PX, 3), dtype=np.int16)
        image[:] = BACKGROUND
        points = []
        # Jittered 2 x 2 layout: cells stay > 30 px (the 7.5 µm margin) apart.
        for cx in (22, 72):
            for cy in (22, 72):
                if rng.random() < 0.2:
                    continue
                x, y = (int(v) for v in (cx + rng.integers(-4, 5), cy + rng.integers(-4, 5)))
                image[y - 5 : y + 6, x - 5 : x + 6] = CLASS_COLOURS[2]
                points.append((x, y, 0))
        Image.fromarray(noisy(image, rng, 10).astype(np.uint8)).save(root / "images" / f"{sample_id}.png")
        pd.DataFrame(points, columns=["x", "y", "class"]).to_csv(
            root / "points" / f"{sample_id}.csv", index=False
        )
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(root / "images" / f"{sample_id}.png"),
                "points_path": str(root / "points" / f"{sample_id}.csv"),
                "spacing_at_level_0": SPACING_UM,
            }
        )
    for fold in range(num_folds):
        test = {sample_ids[(2 * fold + i) % N_ROIS] for i in (0, 1)}
        tune = {sample_ids[(2 * fold + i) % N_ROIS] for i in (2, 3)}
        for sample_id in sample_ids:
            split = "test" if sample_id in test else "tune" if sample_id in tune else "train"
            splits.append({"sample_id": sample_id, "split": split, "fold": fold})
    pd.DataFrame(rows).to_csv(root / "dataset.csv", index=False)
    pd.DataFrame(splits).to_csv(root / "splits.csv", index=False)
    return root


def midog_shaped_config(tmp_path: Path, tiled: Path, encoder_name: str) -> dict:
    """The committed MIDOG recipe's shape at CPU size: dense grids, single-class Hungarian head."""
    return {
        "run": {"output_dir": str(tmp_path / "seed_0"), "seed": 0},
        "cohort": {"records": str(tiled / "dataset.csv"), "splits": str(tiled / "splits.csv")},
        "features": {
            "kind": "cached",
            "encoder": {"name": encoder_name},
            "dense": {"spacing_um": SPACING_UM, "target_size": TILE_PX, "tolerance": 0.10},
            "execution": {**CPU_EXECUTION, "batch_size": 8},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": learner_section(
            "dense",
            {
                "decoder": {"name": "lightweight_conv"},
                "head": {
                    "name": "detection",
                    "params": {
                        "num_classes": 1,
                        "match_distance": MATCH_UM,
                        "sigma": 2.5,
                        "matching": "hungarian",
                        "foreground_weight": 10.0,
                    },
                },
            },
            learning_rate=3e-3,
            normalization={"method": "zscore"},
        ),
        "trainer": trainer_section(30, monitor={"monitor": "mean_f1", "mode": "max"}),
        "evaluation": {"metrics": ["mean_f1", "f1_per_class"]},
    }


def reproduce_from_run_dir(seed_root: Path, capsys) -> tuple[int, float]:
    """Run ``soma reproduce detection/midog --from-run-dir`` and parse the measured F1."""
    capsys.readouterr()
    code = 0
    try:
        soma_main(["reproduce", "detection/midog", "--from-run-dir", str(seed_root)])
    except SystemExit as exc:
        code = int(exc.code or 0)
    out = capsys.readouterr().out
    match = re.search(r"detection/midog f1 = (-?\d+\.\d+)", out)
    assert match, f"no measured f1 in reproduce output:\n{out}"
    return code, float(match.group(1))


def reproduce_failure(seed_root: Path) -> str | None:
    """The exception type ``soma reproduce detection/midog --from-run-dir`` raised, if any."""
    try:
        soma_main(["reproduce", "detection/midog", "--from-run-dir", str(seed_root)])
    except (FileNotFoundError, ValueError) as exc:
        return type(exc).__name__
    except SystemExit as exc:
        return f"exit {exc.code}" if exc.code else None
    return None


@pytest.mark.parametrize("num_folds", [1, 3])
def test_reproduce_detection_midog_scores_its_own_run(
    num_folds, tmp_path, capsys, encoder_name, new_artifact
):
    artifact = new_artifact(f"reproduce_detection_midog_{num_folds}_fold")
    rois = write_roi_manifest(tmp_path / "roi", num_folds=num_folds)
    tiled = tmp_path / "tiled"
    tile_detection_manifest(rois, tiled, tile_size=TILE_PX, overlap=OVERLAP_PX, target_spacing=SPACING_UM)
    config = midog_shaped_config(tmp_path, tiled, encoder_name)
    run = run_soma(config, tmp_path / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)

    code, reported = reproduce_from_run_dir(tmp_path / "seed_0", capsys)
    artifact.check_equal("exit_code", code, 0)

    score_dir = run.run_dir / "detection_score"
    fold_dirs = [score_dir] if num_folds == 1 else [score_dir / f"fold_{k}" for k in range(num_folds)]
    per_fold = []
    for k, directory in enumerate(fold_dirs):
        label = "" if num_folds == 1 else f"fold_{k}/"
        written = (directory / "predictions.json").is_file()
        artifact.check_equal(f"{label}predictions_json_written", written, True)
        if not written:
            continue
        predictions = read_cell_predictions(directory / "predictions.json")
        # Scored per ROI: the test tiles were stitched back to their two parent ROIs.
        artifact.check_equal(
            f"{label}test_rois",
            sorted(s.sample_id for s in predictions.samples),
            sorted(f"roi{(2 * k + i) % N_ROIS}" for i in (0, 1)),
        )
        per_fold.append(score_dataset_points("midog", predictions.samples)["f1"])
        metrics = json.loads((directory / "metrics.json").read_text())
        artifact.check_equal(f"{label}metrics_json_test_f1", metrics["test"]["f1"], per_fold[-1])
        artifact.check_equal(f"{label}threshold_frame", metrics["score_threshold_selection_frame"], "stitched_roi")
    expected = float(np.mean(per_fold)) if per_fold else float("nan")
    artifact.check_equal("reported_f1_is_the_native_score", round(reported, 4), round(expected, 4))
    artifact.check_at_least("reported_f1", reported, 0.5)
    # A fold without its test predictions (an interrupted launch, ``holdout_test``) is
    # refused before any score file is written: it never caches an empty-test score.
    test_parquet = (run.run_dir if num_folds == 1 else run.run_dir / "fold_0") / "predictions_test.parquet"
    test_parquet.rename(test_parquet.with_suffix(".bak"))
    score_dir.rename(score_dir.with_name("detection_score.bak"))
    artifact.check_equal("missing_test/raises", reproduce_failure(tmp_path / "seed_0"), "FileNotFoundError")
    artifact.check_equal("missing_test/no_cache", (score_dir / "predictions.json").exists(), False)
    shutil.rmtree(score_dir, ignore_errors=True)
    score_dir.with_name("detection_score.bak").rename(score_dir)
    test_parquet.with_suffix(".bak").rename(test_parquet)
    # A scored run re-aggregates off its predictions.json cache: the parquet files are no
    # longer read.
    for parquet in run.run_dir.glob("**/predictions_*.parquet"):
        parquet.unlink()
    code, rescored = reproduce_from_run_dir(tmp_path / "seed_0", capsys)
    artifact.check_equal("rescore/exit_code", code, 0)
    artifact.check_equal("rescore/reported_f1", rescored, reported)
    if num_folds > 1:
        summary = json.loads((score_dir / "metrics.json").read_text())
        artifact.check_equal("aggregate/n_folds", summary["n_replicates"], num_folds)
        artifact.check_equal("aggregate/mean_f1", round(summary["mean"]["f1"], 4), round(expected, 4))
        artifact.check_equal("aggregate/std_f1", round(summary["std"]["f1"], 6), round(float(np.std(per_fold)), 6))
        # A run missing one of its declared folds (still training, or a dead launch) is
        # refused: it never reports the mean of fewer folds.
        shutil.rmtree(score_dir)
        last = run.run_dir / f"fold_{num_folds - 1}"
        last.rename(last.with_name("pending_fold"))
        artifact.check_equal("missing_fold/raises", reproduce_failure(tmp_path / "seed_0"), "FileNotFoundError")
        artifact.check_equal("missing_fold/no_score_dir", score_dir.exists(), False)
    artifact.assert_passed()
