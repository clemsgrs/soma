"""Resolve one fixed MIL recipe against tile- and slide-level encoders.

A slide-level benchmark fixes one MIL recipe and varies the encoder axis.
:func:`soma.encoders.resolve_aggregator` (soma 1.12.0) decides, from encoder
registry metadata alone, what that recipe means for each encoder:

* a **tile** encoder produces bags of tile vectors → the fixed MIL recipe
  applies unchanged;
* a **slide** encoder produces one slide vector directly → no aggregator
  (``None``);
* a **patient** encoder cannot serve a slide-level benchmark → a clear error.

No encoder weights are downloaded or loaded at any point — resolution and
config validation read only registry metadata.

Run with no arguments:

    python examples/aggregator_resolution.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from soma import Component, ExperimentConfig, Plan
from soma.config import (
    CohortConfig,
    EvaluationConfig,
    FeaturesConfig,
    LearnerConfig,
    RunConfig,
    TrainerConfig,
)
from soma.encoders import resolve_aggregator
from soma.pipeline.layout import build_experiment_spec
from soma.plan import check

# --- Scaffolding (not soma API): manifest CSVs only -------------------------------
# Config validation never opens the images, so the manifests are enough.

WORK = Path(tempfile.mkdtemp(prefix="soma-aggregator-resolution-"))
sample_ids = [f"s{i:02d}" for i in range(4)]
dataset_csv = WORK / "dataset.csv"
splits_csv = WORK / "splits.csv"
pd.DataFrame(
    {
        "sample_id": sample_ids,
        "image_path": [str(WORK / "slides" / f"{s}.tif") for s in sample_ids],
        "label": [0, 1, 0, 1],
    }
).to_csv(dataset_csv, index=False)
pd.DataFrame(
    {"sample_id": sample_ids, "split": ["train", "train", "tune", "tune"]}
).to_csv(splits_csv, index=False)

# --- 1. One fixed MIL recipe ------------------------------------------------------
RECIPE = Component("abmil", {"hidden_dim": 128})
print(f"fixed MIL recipe: {RECIPE.name} params={RECIPE.params}")


def build_validated_config(encoder_name: str) -> ExperimentConfig:
    """Resolve the recipe for one encoder and build a checked ExperimentConfig.

    :func:`soma.plan.check` validates the component combination; no encoder weights
    are loaded.
    """
    aggregator = resolve_aggregator(encoder_name, RECIPE)
    params = {"head": "binary_classification"}
    if aggregator is not None:
        params["aggregator"] = aggregator
    config = ExperimentConfig(
        run=RunConfig(output_dir=WORK / "outputs" / encoder_name, seed=0),
        cohort=CohortConfig(records=dataset_csv, splits=splits_csv),
        # Tile size and spacing resolve from the encoder's preset when the run extracts.
        features=FeaturesConfig(encoder=encoder_name),
        learner=LearnerConfig("mil", params),
        trainer=TrainerConfig(max_epochs=5),
        evaluation=EvaluationConfig(metrics=["balanced_accuracy"]),
    )
    check(config)
    return config


# --- 2. Tile encoder: the recipe applies unchanged --------------------------------
tile_config = build_validated_config("phikon")
tile_spec = build_experiment_spec(Plan(tile_config))
tile_aggregator = tile_config.learner.params["aggregator"]
print()
print("phikon (tile encoder):")
print(f"  resolved aggregator: {tile_aggregator.name} params={tile_aggregator.params}")
print(f"  config checked -> experiment slug: {tile_spec.slug}")

# --- 3. Slide encoder: no aggregator ----------------------------------------------
slide_config = build_validated_config("titan")
slide_spec = build_experiment_spec(Plan(slide_config))
print()
print("titan (slide encoder):")
print(f"  resolved aggregator: {slide_config.learner.params.get('aggregator')}")
print(f"  config checked -> experiment slug: {slide_spec.slug}")

# --- 4. Patient encoder: refused with a clear error -------------------------------
print()
print("moozy (patient encoder):")
try:
    resolve_aggregator("moozy", RECIPE)
except ValueError as error:
    print(f"  refused as expected: {error}")

assert tile_aggregator == RECIPE
assert "aggregator" not in slide_config.learner.params
