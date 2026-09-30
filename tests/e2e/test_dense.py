"""Dense prediction: ROIs -> dense token grids -> decoder -> segmentation / detection."""

from __future__ import annotations

from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset


def _config(tmp_path, encoder_name, kind, *, roi_px, task, metrics, epochs):
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", kind, size=roi_px)
    return {
        "run": {"output_root": str(tmp_path / "runs"), "seed": 0},
        "data": {
            "dataset_csv": str(dataset_csv),
            "splits_csv": str(splits_csv),
            "dataset_type": kind,
        },
        "preprocessing": {
            "backend": "openslide",
            "requested_tile_size_px": roi_px,
            "requested_spacing_um": SPACING_UM,
        },
        "execution": CPU_EXECUTION,
        "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        "encoder": {"name": encoder_name, "precision": "fp32", "batch_size": 8},
        "decoder": {"name": "lightweight_conv"},
        "task": task,
        "evaluation": {"metrics": metrics},
        "normalization": {"method": "zscore"},
        "training": {
            "epochs": epochs,
            "learning_rate": 3e-3,
            "batch_size": 4,
            "patience": epochs,
            **CPU_LOADER,
        },
    }


def test_segmentation_is_learned(tmp_path, encoder_name, encoded_images, new_artifact):
    artifact = new_artifact("dense_segmentation")
    config = _config(
        tmp_path,
        encoder_name,
        "segmentation",
        roi_px=96,
        task={"name": "segmentation", "params": {"num_classes": 3}},
        metrics=["mean_dice", "mean_iou"],
        epochs=20,
    )
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("rois_encoded", sum(encoded_images), 12)
    artifact.check_at_least("test/mean_dice", run.summary["test/mean_dice"], 0.8)
    artifact.check_at_least("test/mean_iou", run.summary["test/mean_iou"], 0.7)
    artifact.assert_passed()


def test_detection_is_learned(tmp_path, encoder_name, encoded_images, new_artifact):
    artifact = new_artifact("dense_detection")
    config = _config(
        tmp_path,
        encoder_name,
        "detection",
        roi_px=96,
        task={
            "name": "detection",
            "params": {"num_classes": 2, "match_distance": 3.0, "sigma": 1.5},
        },
        metrics=["mean_f1", "f1_per_class"],
        epochs=30,
    )
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("rois_encoded", sum(encoded_images), 12)
    artifact.check_at_least("test/mean_f1", run.summary["test/mean_f1"], 0.6)
    artifact.assert_passed()
