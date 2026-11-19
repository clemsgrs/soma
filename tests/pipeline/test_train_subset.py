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


def test_the_plan_is_inferred_from_the_whole_cohort(tmp_path: Path):
    # Both listed slides are tumor; the label map still holds both classes.
    config, source = _config(tmp_path, "s0\ns2\n")

    head = Pipeline(config, source=source).plan.learner.params["head"]

    assert head.params["label_map"] == {"normal": 0, "tumor": 1}


def _survival_config(tmp_path: Path, train_times: list[float], subset: str):
    """A ``train_quantiles`` run with three bins over the uncensored ``train_times``,
    then one tune and one test sample."""
    ids = [f"s{i}" for i in range(len(train_times) + 2)]
    data = tmp_path / "data"
    data.mkdir()
    records, splits = data / "dataset.csv", data / "splits.csv"
    pd.DataFrame(
        {"sample_id": ids, "image_path": [f"/slides/{s}.svs" for s in ids], "time": [*train_times, 1.0, 2.0], "event": 1}
    ).to_csv(records, index=False)
    pd.DataFrame({"sample_id": ids, "split": ["train"] * len(train_times) + ["tune", "test"]}).to_csv(splits, index=False)
    subset_file = tmp_path / "subset.txt"
    subset_file.write_text(subset)
    return experiment(
        records,
        splits,
        output_dir=tmp_path / "outputs",
        head="discrete_survival",
        head_params={"binning": "train_quantiles", "num_bins": 3},
        aggregator="mean_pool",
        cohort={"train_subset": str(subset_file)},
    )


def test_the_survival_bins_are_checked_on_the_listed_training_samples(tmp_path: Path):
    # The whole train split [1, 2, 2, 2, 2, 2, 3] ties the 1/3 and 2/3 cut points at 2;
    # the listed times [1, 2, 3] do not.
    config = _survival_config(tmp_path, [1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0], "s0\ns1\ns6\n")

    head = Pipeline(config).plan.learner.params["head"]

    assert head.params["num_bins"] == 3 and head.params["bin_edges"] is None


def test_a_subset_too_small_for_the_survival_bins_fails_before_the_run_starts(tmp_path: Path):
    # Seven distinct times form three bins; the two listed do not.
    config = _survival_config(tmp_path, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], "s0\ns1\n")

    with pytest.raises(ValueError, match=r"^Fold 0: .*num_bins=3 .* 2 uncensored training time\(s\)"):
        Pipeline(config)
    assert not (tmp_path / "outputs").exists()
