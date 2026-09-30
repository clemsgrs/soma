"""Tile-level classification: patch images -> extraction -> head -> evaluate (no MIL bag)."""

from __future__ import annotations

from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma
from tests.e2e.synthetic import make_tile_dataset


def test_tile_classification_is_learned(tmp_path, encoder_name, encoded_images, new_artifact):
    artifact = new_artifact("tile_binary_classification")
    dataset_csv, splits_csv = make_tile_dataset(tmp_path / "data")
    config = {
        "run": {"output_root": str(tmp_path / "runs"), "seed": 0},
        "data": {
            "dataset_csv": str(dataset_csv),
            "splits_csv": str(splits_csv),
            "dataset_type": "tile",
        },
        "execution": CPU_EXECUTION,
        "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        "encoder": {"name": encoder_name, "precision": "fp32", "batch_size": 16},
        "task": {"name": "binary_classification"},
        "evaluation": {"metrics": ["auroc", "balanced_accuracy"]},
        "training": {
            "epochs": 30,
            "learning_rate": 5e-3,
            "batch_size": 4,
            "patience": 30,
            **CPU_LOADER,
        },
    }
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("tiles_encoded", sum(encoded_images), 32)
    artifact.check_equal("test/num_samples", run.summary["test/num_samples"], 8.0)
    artifact.check_at_least("test/auroc", run.summary["test/auroc"], 0.95)
    artifact.check_at_least("test/balanced_accuracy", run.summary["test/balanced_accuracy"], 0.85)
    artifact.assert_passed()
