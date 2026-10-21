"""Tile-level classification: patch images -> extraction -> head -> evaluate (no MIL bag)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from tests.e2e.harness import (
    CPU_EXECUTION,
    CPU_LOADER,
    PROBE_SCENARIO_LR,
    init_config,
    learner_section,
    run_preset,
    run_soma,
    trainer_section,
)
from tests.e2e.synthetic import make_tile_dataset

#: The tile feature cache key of the tile scenario (pinned in slice 6c; moved once in
#: slice 8, when the scenarios took soma.testing's ``fake`` encoder).
PINNED_TILE_CACHE_KEYS = ["image/445e17ca75d972e4"]


def test_tile_classification_is_learned(tmp_path, encoder_name, encoded_images, new_artifact):
    artifact = new_artifact("tile_binary_classification")
    dataset_csv, splits_csv = make_tile_dataset(tmp_path / "data")
    # A tile encoder without an aggregator: each image is one sample, embedded whole.
    config = {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "encoder": {"name": encoder_name},
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 16},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": learner_section("mil", {"head": {"name": "binary_classification"}}, learning_rate=5e-3),
        "trainer": trainer_section(30),
        "evaluation": {"metrics": ["auroc", "balanced_accuracy"]},
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
    """The head's ``num_outputs`` makes the cohort carry ``value_<k>`` targets: two
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
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "encoder": {"name": encoder_name},
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 16},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": learner_section(
            "mil", {"head": {"name": "regression", "params": {"num_outputs": 2}}}, learning_rate=5e-3
        ),
        "trainer": trainer_section(30),
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


def test_the_tile_regression_preset_runs_from_the_command_line(tmp_path, encoder_name, encoded_images, new_artifact):
    """``soma run --preset tile_regression`` with the ``fake`` encoder: each image is one
    sample, embedded whole, and a ``regression`` head learns its ``value`` (a function of
    the image's colour) with no aggregator, in batches of 256. Every test image is scored.

    The preset's probe recipe (AdamW at lr 3e-4) takes one step per epoch on the 16
    training images, which barely moves a head over three features in 50 epochs. So the
    same preset, printed by ``soma init`` with only the learning rate raised, runs next: it
    reads the first run's features from the cache (zero images encoded), its loss falls
    and its metric thresholds hold.
    """
    artifact = new_artifact("tile_preset_regression")
    dataset_csv, splits_csv = make_tile_dataset(tmp_path / "data")
    frame = pd.read_csv(dataset_csv).drop(columns="label")
    red = np.array([np.asarray(Image.open(path), dtype=np.float64)[..., 0].mean() for path in frame["image_path"]])
    frame["value"] = np.round((red - 155.0) / 30.0, 4)  # about -2 (dark) to 2 (light)
    frame.to_csv(dataset_csv, index=False)
    out = tmp_path / "runs"

    shipped = run_preset("tile_regression", ["--encoder", encoder_name], dataset_csv, splits_csv, out)
    artifact.record_run("preset", shipped)
    artifact.check_equal("preset/tiles_encoded", sum(encoded_images), 32)
    plan = yaml.safe_load((shipped.run_dir / "plan.yaml").read_text())
    model = plan["learner"]["params"]
    artifact.check_equal("preset/plan/aggregator", model.get("aggregator"), None)
    artifact.check_equal("preset/plan/batch_size", plan["loaders"]["batch_size"], 256)
    artifact.check_equal("preset/plan/head", model["head"]["name"], "regression")
    artifact.check_equal("preset/test/num_samples", shipped.summary["test/num_samples"], 8.0)

    encoded_images.clear()
    config = init_config("tile_regression", "--encoder", encoder_name)
    config["cohort"].update(records=str(dataset_csv), splits=str(splits_csv))
    config["run"]["output_dir"] = str(out)
    config["learner"]["params"]["optimizer"]["params"]["lr"] = PROBE_SCENARIO_LR
    run = run_soma(config, tmp_path / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_equal("run/tiles_encoded", sum(encoded_images), 0)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("run/test/r2", run.summary["test/r2"], 0.8)
    artifact.assert_passed()
