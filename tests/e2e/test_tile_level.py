"""Tile-level classification: patch images -> extraction -> head -> evaluate (no MIL bag)."""

from __future__ import annotations

from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma
from tests.e2e.synthetic import make_tile_dataset

#: The tile feature cache key of the tile scenario (pinned in slice 6c).
PINNED_TILE_CACHE_KEYS = ["image/84e4e51bb8b8651d"]


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
    # Cache keys derive from the features config alone; a config-layout change must not move them.
    cache_keys = sorted(p.relative_to(tmp_path / "cache").as_posix() for p in (tmp_path / "cache").glob("*/*"))
    artifact.check_equal("cache_keys", cache_keys, PINNED_TILE_CACHE_KEYS)
    artifact.check_equal("test/num_samples", run.summary["test/num_samples"], 8.0)
    artifact.check_at_least("test/auroc", run.summary["test/auroc"], 0.95)
    artifact.check_at_least("test/balanced_accuracy", run.summary["test/balanced_accuracy"], 0.85)
    artifact.assert_passed()


def test_tile_multi_output_regression_is_learned(tmp_path, encoder_name, new_artifact):
    """``task.params.num_outputs`` makes the cohort carry ``value_<k>`` targets: two
    outputs, each a function of the tile's colour, scored by ``mean_pearson``. An unused
    ``value_2`` manifest column stays metadata: neither the head nor the baseline that
    fills feature-less samples reads it as a third output."""
    import numpy as np
    import pandas as pd
    from PIL import Image

    artifact = new_artifact("tile_multi_output_regression")
    dataset_csv, splits_csv = make_tile_dataset(tmp_path / "data")
    frame = pd.read_csv(dataset_csv).drop(columns="label")
    colour = np.stack([np.asarray(Image.open(path), dtype=np.float64).mean(axis=(0, 1)) for path in frame["image_path"]])
    frame["value_0"] = np.round(colour[:, 0] / 25.5, 4)
    frame["value_1"] = np.round(10.0 - colour[:, 1] / 25.5, 4)
    frame["value_2"] = 1.0
    frame.to_csv(dataset_csv, index=False)
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
        "task": {"name": "regression", "params": {"num_outputs": 2}},
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
    predictions = pd.read_parquet(run.run_dir / "predictions_test.parquet")
    artifact.check_equal(
        "predictions/columns", [c for c in predictions.columns if c.startswith("value")], ["value_0", "value_1"]
    )
    artifact.check_at_least("test/mean_pearson", run.summary["test/mean_pearson"], 0.8)
    artifact.assert_passed()
