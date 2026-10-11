"""Scoring grid predictions that soma did not produce: label maps and points.

A user brings a label-map directory (PNGs + ``index.parquet``) or a points parquet and
the cohort's ``dataset.csv`` (with ``label_mask_path`` / ``points_path``) and
``splits.csv``; ``soma evaluate`` scores them with no run directory. The expected numbers
are worked by hand from the values below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import soma
from soma.cli import main as soma_main
from soma.data import AnnotationManifest, PointSource
from soma.evaluation import LabelMapPredictions, PointPredictions

#: Two 4x4 ground-truth label maps and a model's label maps. ``s0``'s right half is class
#: 1 and the model calls one of those 8 pixels class 0; ``s1`` is all class 0, predicted
#: exactly. Per-image Dice: s0 = (16/17 + 14/15) / 2, s1 = 1 (class 1 is undefined there);
#: IoU: s0 = (8/9 + 7/8) / 2, s1 = 1.
IDS = ["s0", "s1"]
_HALF = np.array([[0, 0, 1, 1]] * 4, dtype=np.uint8)
TRUE_MAPS = {"s0": _HALF, "s1": np.zeros((4, 4), dtype=np.uint8)}
PREDICTED_MAPS = {"s0": _HALF.copy(), "s1": np.zeros((4, 4), dtype=np.uint8)}
PREDICTED_MAPS["s0"][0, 2] = 0
#: Annotated points (level-0 x, y, class) and detections (x, y, class, score). Within the
#: 2 px tolerance s0's first detection matches its class-0 point; its second, class 0 at
#: (3, 3), sits on a class-1 point (one FP for class 0, one FN for class 1); s1's point is
#: missed. Class 0: TP 1, FP 1, FN 1 -> F1 0.5; class 1: FN 1 -> F1 0; mean F1 0.25.
TRUE_POINTS = {"s0": [(1, 1, 0), (3, 3, 1)], "s1": [(2, 2, 0)]}
DETECTIONS = [("s0", 1.5, 1.0, 0, 0.9), ("s0", 3.0, 3.0, 0, 0.8)]


def _write_cohort(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample_id in IDS:
        mask = root / f"{sample_id}_mask.png"
        Image.fromarray(TRUE_MAPS[sample_id]).save(mask)
        points = root / f"{sample_id}_points.csv"
        pd.DataFrame(TRUE_POINTS[sample_id], columns=["x", "y", "class"]).to_csv(points, index=False)
        rows.append({"sample_id": sample_id, "label_mask_path": str(mask), "points_path": str(points)})
    pd.DataFrame(rows).to_csv(root / "dataset.csv", index=False)
    pd.DataFrame({"sample_id": IDS, "split": ["test", "test"]}).to_csv(root / "splits.csv", index=False)
    return root / "dataset.csv", root / "splits.csv"


def test_external_grid_predictions_are_scored_through_the_cli(tmp_path, new_artifact):
    artifact = new_artifact("evaluate_external_grid_predictions")
    dataset_csv, splits_csv = _write_cohort(tmp_path / "cohort")

    label_maps = tmp_path / "external" / "label_maps"
    LabelMapPredictions(PREDICTED_MAPS, classes=["background", "tumour"]).save(label_maps)
    maps_json = tmp_path / "label_maps.json"
    soma_main(
        [
            "evaluate", str(label_maps), "--cohort", str(dataset_csv), "--splits", str(splits_csv),
            "--split", "test", "--metrics", "mean_dice", "mean_iou", "dice_per_class",
            "--out", str(maps_json),
        ]
    )
    scored = json.loads(maps_json.read_text())
    artifact.check_equal("label_maps/task_family", scored["task_family"], "segmentation")
    artifact.check_equal("label_maps/num_samples", scored["num_samples"], 2)
    dice = ((16 / 17 + 14 / 15) / 2 + 1.0) / 2
    iou = ((8 / 9 + 7 / 8) / 2 + 1.0) / 2
    artifact.check_equal("label_maps/mean_dice", round(scored["metrics"]["mean_dice"], 6), round(dice, 6))
    artifact.check_equal("label_maps/mean_iou", round(scored["metrics"]["mean_iou"], 6), round(iou, 6))
    # Dataset-global per-class Dice: class 1 has 7 hits over 7 predicted + 8 true pixels.
    artifact.check_equal(
        "label_maps/dice_class_1", round(scored["metrics"]["dice_class_1"], 6), round(14 / 15, 6)
    )

    frame = pd.DataFrame(DETECTIONS, columns=["sample_id", "x", "y", "class", "score"])
    points_path = tmp_path / "external" / "points.parquet"
    PointPredictions(frame, classes=["a", "b"], sample_ids=IDS, match_distance_px=2.0).to_parquet(
        points_path
    )
    points_json = tmp_path / "points.json"
    soma_main(
        [
            "evaluate", str(points_path), "--cohort", str(dataset_csv), "--splits", str(splits_csv),
            "--metrics", "mean_f1", "f1_per_class", "--out", str(points_json),
        ]
    )
    scored = json.loads(points_json.read_text())
    artifact.check_equal("points/task_family", scored["task_family"], "detection")
    artifact.check_equal("points/mean_f1", scored["metrics"]["mean_f1"], 0.25)
    artifact.check_equal("points/f1_class_0", scored["metrics"]["f1_class_0"], 0.5)
    artifact.check_equal("points/f1_class_1", scored["metrics"]["f1_class_1"], 0.0)

    # The same scores through Python, with greedy matching as a metric-leaf param.
    targets = PointSource.from_manifest(AnnotationManifest.from_csv(dataset_csv))
    greedy = soma.evaluate(
        PointPredictions.from_parquet(points_path),
        targets,
        metrics=[{"name": "mean_f1", "params": {"matching": "greedy"}}],
    )
    artifact.check_equal("points/greedy_mean_f1", greedy.metrics["mean_f1"], 0.25)
    # A threshold is a frame operation: dropping the 0.8 detection removes the FP, so class
    # 0 scores TP 1, FN 1 -> F1 2/3 and the mean is 1/3.
    above = soma.evaluate(PointPredictions.from_parquet(points_path).with_threshold(0.85), targets)
    artifact.check_equal("points/thresholded_mean_f1", round(above.metrics["mean_f1"], 6), round(1 / 3, 6))
    artifact.assert_passed()
