"""``cohort.train_subset`` on a cohort of plain samples: ``soma.Pipeline`` trains each fold
on the listed ids of its train split, and refuses a list that does not fit every fold
before the run starts. The ROI cohort of whole slides is the end-to-end scenario's
(``tests/e2e/test_dense.py``)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from soma.data import CachedSetSource
from soma.pipeline import Pipeline
from tests.experiment_configs import experiment
from tests.pooled_features import setup_multifold_data, setup_synthetic_data


def _config(tmp_path: Path, subset: str, *, folds: int = 1):
    data = tmp_path / "data"
    data.mkdir()
    write = setup_multifold_data if folds == 2 else setup_synthetic_data
    dataset_csv, splits_csv, feature_dir = write(data)
    subset_file = tmp_path / "subset.txt"
    subset_file.write_text(subset)
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / "outputs",
        aggregator="mean_pool",
        cohort={"train_subset": str(subset_file)},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 2},
        normalization={"method": "zscore"},
    )
    return config, CachedSetSource(feature_dir)


def test_a_cohort_of_slides_trains_on_the_listed_slides(tmp_path: Path):
    # Six train slides (s0-s5), tune s6, test s7.
    config, source = _config(tmp_path, "s3\ns0\n")

    result = Pipeline(config, source=source).run()

    adaptor = json.loads((result.run_dir / "feature_adapter.json").read_text())
    assert adaptor["n_support_samples"] == 2
    tested = pd.read_parquet(result.run_dir / "predictions_test.parquet")
    assert list(tested["sample_id"]) == ["s7"]


def test_a_train_subset_outside_a_folds_train_split_fails_before_the_run_starts(tmp_path: Path):
    # Fold 0 trains s0-s5 and fold 1 s2-s7: s0 is fold 1's tune slide.
    config, source = _config(tmp_path, "s2\ns0\n", folds=2)

    with pytest.raises(ValueError, match=r"'s0'.*fold 1.*'tune'"):
        Pipeline(config, source=source)
    assert not (tmp_path / "outputs").exists()
