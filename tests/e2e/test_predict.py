"""Whole-slide prediction: a segmentation bundle trained on ROIs, then ``soma predict``
and ``soma evaluate``; a detection bundle trained on ROIs, then ``soma predict``.

The segmentation run trains on the dense segmentation ROIs (discs of two colours on
background) over the weight-free ``fake`` encoder's cached grids. ``soma predict`` then
cuts larger slides with the same discs into hs2p's slide grid, runs the bundle over every
tile behind its frozen encoder stage, and writes one label TIFF per slide. ``soma
evaluate`` scores the maps against the slides' annotation rasters.

The detection run trains on the dense detection ROIs (small squares of two colours, their
centres the points). ``soma predict`` writes one points parquet per slide.

The runs read their ROIs with OpenSlide, and their bundles record that reader. The
prediction configs name no slide reader, so ``soma predict`` reads the slides with
OpenSlide too.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tifffile
import yaml

from soma.cli import main as soma_main
from soma.evaluation.detection import match_points
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, learner_section, run_soma, sha256, trainer_section
from tests.e2e.synthetic import BACKGROUND, CLASS_COLOURS, SPACING_UM, make_dense_dataset, noisy, write_tiff

ROI_PX = 96
#: The rows each annotation raster leaves out (255, the bundle's ignore value).
IGNORED_ROWS = 16


def dense_config(tmp_path, encoder_name, kind, *, head, metrics, epochs):
    """A run over the ``kind`` dense ROIs, cached through the ``fake`` encoder."""
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", kind, size=ROI_PX)
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
            {"decoder": {"name": "lightweight_conv"}, "head": head},
            learning_rate=3e-3,
            normalization={"method": "zscore"},
        ),
        "trainer": trainer_section(epochs),
        "evaluation": {"metrics": metrics},
    }


def segmentation_config(tmp_path, encoder_name):
    return dense_config(
        tmp_path,
        encoder_name,
        "segmentation",
        head={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice"],
        epochs=20,
    )


def write_disc_slide(root, sample_id, *, height, width, seed):
    """A slide of discs of both classes on background, a mask of tissue everywhere, an
    annotation raster, and the slide's class of each pixel (0 background, 1 and 2 the
    discs). The annotation raster holds the classes, except in its first rows (ignored)."""
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
    annotation = truth.copy()
    annotation[:IGNORED_ROWS] = 255
    resolution = (1e4 / SPACING_UM, 1e4 / SPACING_UM)
    for name, raster in (("mask", np.ones((height, width), dtype=np.uint8)), ("annotation", annotation)):
        tifffile.imwrite(
            root / f"{sample_id}_{name}.tif", raster, tile=(16, 16), resolution=resolution, resolutionunit="CENTIMETER"
        )
    row = {
        "sample_id": sample_id,
        "image_path": f"slides/{sample_id}.tif",
        "mask_path": f"slides/{sample_id}_mask.tif",
        "label_mask_path": f"slides/{sample_id}_annotation.tif",
    }
    return row, truth


def class_counts(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Each class's (pixels predicted and true, predicted, true)."""
    return np.array(
        [[((predicted == c) & (truth == c)).sum(), (predicted == c).sum(), (truth == c).sum()] for c in (0, 1, 2)]
    )


def dice(counts: np.ndarray) -> np.ndarray:
    return 2 * counts[:, 0] / (counts[:, 1] + counts[:, 2])


def mean_dice(predicted: np.ndarray, truth: np.ndarray) -> float:
    return float(dice(class_counts(predicted, truth)).mean())


def readers_of(opened, paths) -> list[str]:
    """The names of the hs2p readers that opened any of ``paths``."""
    wanted = {path.resolve() for path in paths}
    return sorted({reader for path, reader in opened if path in wanted})


def test_predict_writes_a_label_tiff_per_slide_from_a_trained_bundle(
    tmp_path, monkeypatch, encoder_name, encoded_images, opened_files, new_artifact
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
        # No slide reader: the bundle's. The masks are read with tifffile.
        "extraction": {"tiling": {"overlap": 0.5, "mask_backend": "tifffile"}},
    }
    (tmp_path / "predict.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.chdir(tmp_path)  # the config's relative paths resolve against the working directory
    slides = [tmp_path / "slides" / f"{row['sample_id']}.tif" for row in rows]
    annotations = [tmp_path / row["label_mask_path"] for row in rows]
    encoded_images.clear()
    opened_files.clear()

    soma_main(["predict", "predict.yaml"])

    output_dir = tmp_path / "predictions"
    plan = yaml.safe_load((output_dir / "plan.yaml").read_text())
    # The run directory stands for its bundle, and every path is absolute.
    artifact.check_equal("plan/bundles", plan["predict"]["bundles"] == [str(run.run_dir / "bundle")], True)
    artifact.check_equal("plan/records", plan["cohort"]["records"] == str(tmp_path / "slides.csv"), True)
    artifact.check_equal("plan/output_dir", plan["predict"]["output_dir"] == str(output_dir), True)
    artifact.check_equal("plan/tile_size_px", plan["extraction"]["tiling"]["requested_tile_size_px"], ROI_PX)
    artifact.check_equal("plan/spacing_um", plan["extraction"]["tiling"]["requested_spacing_um"], SPACING_UM)
    # The bundle's slide reader is planned, and every slide is read with it.
    artifact.check_equal("plan/backend", plan["extraction"]["tiling"]["backend"], "openslide")
    artifact.check_equal("predict/slide_readers", readers_of(opened_files, slides), ["openslide"])
    artifact.check_at_least("tiles_encoded", sum(encoded_images), 1)
    index = pd.read_parquet(output_dir / "index.parquet")
    artifact.check_equal("index/sample_ids", list(index.sample_id), ["slide0", "slide1"])
    label_maps = {}
    for row in index.itertuples():
        path = output_dir / row.path
        with tifffile.TiffFile(path) as tiff:
            label_map = label_maps[row.sample_id] = tiff.pages[0].asarray()
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

    # soma evaluate on the output directory: the maps, streamed against the annotation
    # rasters, score what the arrays give by hand once the ignored rows are left out. The
    # slides and the rasters are read with the readers of plan.yaml.
    opened_files.clear()
    soma_main(
        ["evaluate", "predictions", "--cohort", "slides.csv", "--metrics", "mean_dice",
         "dataset_global_mean_dice", "--out", "evaluation.json"]
    )
    artifact.check_equal("evaluate/slide_readers", readers_of(opened_files, slides), ["openslide"])
    artifact.check_equal("evaluate/annotation_readers", readers_of(opened_files, annotations), ["tifffile"])
    scored = json.loads((tmp_path / "evaluation.json").read_text())
    counts = {
        sid: class_counts(label_maps[sid][IGNORED_ROWS:].astype(int) - 1, truths[sid][IGNORED_ROWS:])
        for sid in truths
    }
    per_slide = float(np.mean([dice(c).mean() for c in counts.values()]))
    pooled = float(dice(sum(counts.values())).mean())
    artifact.check_equal("evaluate/num_samples", scored["num_samples"], 2)
    artifact.check_equal("evaluate/mean_dice", round(scored["metrics"]["mean_dice"], 6), round(per_slide, 6))
    artifact.check_equal(
        "evaluate/dataset_global_mean_dice", round(scored["metrics"]["dataset_global_mean_dice"], 6), round(pooled, 6)
    )
    for row in scored["per_sample"]:
        height, width = truths[row["sample_id"]].shape
        artifact.check_equal(f"evaluate/{row['sample_id']}/annotated_px", row["annotated_px"], (height - IGNORED_ROWS) * width)
        artifact.check_equal(f"evaluate/{row['sample_id']}/not_predicted_fraction", row["not_predicted_fraction"], 0.0)
        artifact.check_equal(
            f"evaluate/{row['sample_id']}/tp_class_1", row["tp_class_1"], int(counts[row["sample_id"]][1, 0])
        )

    # --backend and --mask-backend replace the readers of plan.yaml. Both readers decode
    # the same pixels here, so the scores do not change.
    opened_files.clear()
    soma_main(
        ["evaluate", "predictions", "--cohort", "slides.csv", "--metrics", "mean_dice",
         "dataset_global_mean_dice", "--backend", "tifffile", "--mask-backend", "pil", "--out", "readers.json"]
    )
    artifact.check_equal("evaluate_readers/slide_readers", readers_of(opened_files, slides), ["tifffile"])
    artifact.check_equal("evaluate_readers/annotation_readers", readers_of(opened_files, annotations), ["pil"])
    rescored = json.loads((tmp_path / "readers.json").read_text())
    artifact.check_equal("evaluate_readers/metrics", rescored["metrics"], scored["metrics"])
    artifact.check_equal("evaluate_readers/per_sample", rescored["per_sample"] == scored["per_sample"], True)
    artifact.assert_passed()


# --- detection --------------------------------------------------------------------------- #

#: The detection head's matching distance in um; its NMS distance is the same.
MATCH_DISTANCE_UM = 3.0
#: The run's classes: the ROIs' points hold id 0 for the light cells and 1 for the dark.
CLASSES = {"light": [0], "dark": [1]}
#: The id of the one point each ROI's points file also holds, on the background: dropped.
DROPPED = 2


def detection_config(tmp_path, encoder_name):
    config = dense_config(
        tmp_path,
        encoder_name,
        "detection",
        head={
            "name": "detection",
            "params": {"classes": CLASSES, "drop": [DROPPED], "match_distance": MATCH_DISTANCE_UM, "sigma": 1.5},
        },
        metrics=["mean_f1", "f1_per_class"],
        epochs=30,
    )
    for points_path in pd.read_csv(config["cohort"]["records"]).points_path:
        points = pd.read_csv(points_path)
        dropped = pd.DataFrame([{"x": 2, "y": 2, "class": DROPPED}])
        pd.concat([points, dropped], ignore_index=True).to_csv(points_path, index=False)
    return config


def write_cell_slide(root, sample_id, *, height, width, seed):
    """A slide of 11 px cells of both classes on background, one near each node of a
    32 px lattice (some on the seams of a 96 px grid at overlap 0.5), and a mask of tissue
    everywhere. Returns its row and the cells' ``(x, y, class)``, the class from 0."""
    rng = np.random.default_rng(seed)
    image = np.empty((height, width, 3), dtype=np.int16)
    image[:] = BACKGROUND
    cells = []
    for y0 in range(16, height - 15, 32):
        for x0 in range(16, width - 15, 32):
            x, y = (int(v) for v in np.array([x0, y0]) + rng.integers(-6, 7, 2))
            x, y = min(max(x, 6), width - 7), min(max(y, 6), height - 7)
            cls = int(rng.integers(1, 3))
            image[y - 5 : y + 6, x - 5 : x + 6] = CLASS_COLOURS[cls]
            cells.append((x, y, cls - 1))
    write_tiff(root / f"{sample_id}.tif", noisy(image, rng, 10))
    resolution = (1e4 / SPACING_UM, 1e4 / SPACING_UM)
    tifffile.imwrite(
        root / f"{sample_id}_mask.tif",
        np.ones((height, width), dtype=np.uint8),
        tile=(16, 16),
        resolution=resolution,
        resolutionunit="CENTIMETER",
    )
    row = {"sample_id": sample_id, "image_path": f"slides/{sample_id}.tif", "mask_path": f"slides/{sample_id}_mask.tif"}
    return row, np.asarray(cells, dtype=np.float64)


def closest_same_class_px(points: pd.DataFrame) -> float:
    """The smallest distance between two points of one class (inf without a pair)."""
    closest = np.inf
    for _, group in points.groupby("class"):
        xy = group[["x", "y"]].to_numpy()
        if len(xy) > 1:
            distances = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
            closest = min(closest, float(distances[np.triu_indices(len(xy), 1)].min()))
    return closest


def test_predict_writes_a_points_parquet_per_slide_from_a_trained_bundle(
    tmp_path, monkeypatch, encoder_name, encoded_images, new_artifact
):
    artifact = new_artifact("predict_detection_slides")
    run = run_soma(detection_config(tmp_path, encoder_name), tmp_path / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)

    (tmp_path / "slides").mkdir()
    row, cells = write_cell_slide(tmp_path / "slides", "slide0", height=192, width=288, seed=0)
    pd.DataFrame([row]).to_csv(tmp_path / "slides.csv", index=False)
    config = {
        "predict": {"output_dir": "predictions", "bundles": str(run.run_dir.relative_to(tmp_path)), "batch_size": 8},
        "cohort": {"records": "slides.csv"},
        # A 24 px (12 um) margin, wider than the 3 um NMS distance.
        "extraction": {"tiling": {"overlap": 0.5, "mask_backend": "tifffile"}},
    }
    (tmp_path / "predict.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.chdir(tmp_path)
    encoded_images.clear()

    soma_main(["predict", "predict.yaml"])

    output_dir = tmp_path / "predictions"
    plan = yaml.safe_load((output_dir / "plan.yaml").read_text())
    artifact.check_equal("plan/bundles", plan["predict"]["bundles"] == [str(run.run_dir / "bundle")], True)
    artifact.check_equal("plan/backend", plan["extraction"]["tiling"]["backend"], "openslide")
    artifact.check_at_least("tiles_encoded", sum(encoded_images), 1)
    index = pd.read_parquet(output_dir / "index.parquet")
    # One parquet per slide, in a folder named after the bundle's run.
    artifact.check_equal("index/paths", list(index.path) == [f"{run.run_dir.name}/slide0.parquet"], True)
    artifact.check_equal("index/num_tiles", int(index.num_tiles[0]), 4 * 6)  # the 48 px lattice over 192 x 288
    path = output_dir / index.path[0]
    points = pd.read_parquet(path)
    artifact.check_equal("parquet/columns", list(points.columns), ["x", "y", "class", "score"])
    artifact.check_equal("index/num_points", int(index.num_points[0]), len(points))
    metadata = json.loads(pq.read_schema(path).metadata[b"soma"])
    preprocessing = json.loads((run.run_dir / "bundle" / "preprocessing.json").read_text())
    recorded = preprocessing["score_threshold"]
    # The bundle records the run's class scheme, and the parquet copies it in class order.
    artifact.check_equal("bundle/class_scheme", preprocessing["class_scheme"], {"classes": CLASSES, "drop": [DROPPED]})
    artifact.check_equal("metadata/kind", metadata["kind"], "slide_points")
    artifact.check_equal("metadata/classes", metadata["classes"], CLASSES)
    artifact.check_equal("metadata/class_order", list(metadata["classes"]), list(CLASSES))
    artifact.check_equal("metadata/drop", metadata["drop"], [DROPPED])
    artifact.check_equal("metadata/match_distance_um", metadata["match_distance_um"], MATCH_DISTANCE_UM)
    artifact.check_equal("metadata/nms_distance_um", metadata["nms_distance_um"], MATCH_DISTANCE_UM)
    artifact.check_equal("metadata/level0_spacing_um", metadata["level0_spacing_um"], SPACING_UM)
    artifact.check_equal("metadata/threshold", metadata["threshold"] == recorded, True)
    # The seam NMS leaves no two points of one class closer than the NMS distance.
    artifact.check_at_least("points/closest_same_class_px", closest_same_class_px(points), MATCH_DISTANCE_UM / SPACING_UM)
    # No threshold is applied to the rows: the parquet keeps the peaks below the bundle's.
    names = list(metadata["classes"])
    thresholds = recorded if isinstance(recorded, dict) else {name: recorded for name in names}
    cut = points["class"].map(lambda c: thresholds[names[c]])
    artifact.check_at_least("points/below_threshold", int((points.score < cut).sum()), 1)
    # The points land on the drawn cells in level-0 pixels: every cell is found by a point
    # of its class, and some score threshold scores the slide well. (The bundle's own
    # threshold, picked on two tune ROIs, is not the check.)
    f1 = {}
    for threshold in [0.0, *sorted(set(points.score))]:
        kept = points[points.score >= threshold]
        tp, fp, fn = match_points(
            kept[["x", "y"]].to_numpy(),
            kept["class"].to_numpy(),
            kept.score.to_numpy(),
            cells[:, :2],
            cells[:, 2].astype(np.int64),
            num_classes=2,
            delta=MATCH_DISTANCE_UM / SPACING_UM,
        ).sum(axis=0)
        f1[threshold] = (int(fn), round(float(2 * tp / (2 * tp + fp + fn)), 6))
    artifact.check_equal("slide0/cells_missed_at_min_score", f1[0.0][0], 0)
    artifact.check_at_least("slide0/best_f1", max(score for _, score in f1.values()), 0.8)

    # The plan replays: the slide is done, so nothing is encoded and the parquet is the same.
    before = sha256(path)
    encoded_images.clear()
    soma_main(["predict", "predictions/plan.yaml"])
    artifact.check_equal("replay/tiles_encoded", sum(encoded_images), 0)
    artifact.check_equal("replay/points_unchanged", sha256(path) == before, True)
    artifact.assert_passed()
