"""Per-epoch subgroup monitoring: a cross-validated Cox run that holds its test split
out, logs each centre's tune values at every evaluated epoch and selects its checkpoint
on the mean C-index over centres. The same run with ``trainer.eval_train`` also logs the
train split, pooled and per centre, scored in eval mode, and changes no tune value."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import soma
from soma.data import CachedSetSource, Cohort
from soma.learn import loaders
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma

FOLDS = 3
BLOCK = 12
EPOCHS = 6
MONITOR = "tune/mean_over_centre/c_index"


def _write_cohort(root: Path) -> tuple[Path, Path, Path]:
    """36 samples in three blocks of 12, with pooled feature bags. Fold ``f`` tunes on
    block ``f``, tests on block ``f + 1`` and trains on the third. Centres alternate, so
    each tune split holds six samples of each. A sample's tiles sit along one direction
    by its risk, and a higher risk gives a shorter time; one sample in four is censored."""
    import torch

    rng = np.random.default_rng(0)
    root.mkdir(parents=True)
    features = root / "features"
    features.mkdir()
    ids = [f"s{i:02d}" for i in range(FOLDS * BLOCK)]
    risk = rng.uniform(-1.0, 1.0, size=len(ids))
    direction = rng.normal(size=8)
    for index, sample_id in enumerate(ids):
        bag = risk[index] * direction + rng.normal(scale=0.5, size=(4 + index % 5, 8))
        torch.save(torch.from_numpy(bag.astype(np.float32)), features / f"{sample_id}.pt")
    pd.DataFrame(
        {
            "sample_id": ids,
            "image_path": [f"/slides/{sid}.tif" for sid in ids],
            "time": np.round(10.0 * np.exp(-risk) + rng.uniform(0.0, 1.0, size=len(ids)), 3),
            "event": [0 if index % 4 == 3 else 1 for index in range(len(ids))],
            "centre": ["A" if index % 2 == 0 else "B" for index in range(len(ids))],
        }
    ).to_csv(root / "dataset.csv", index=False)
    roles = ("tune", "test", "train")
    pd.DataFrame(
        [
            {"fold": fold, "sample_id": sid, "split": roles[(index // BLOCK - fold) % FOLDS]}
            for fold in range(FOLDS)
            for index, sid in enumerate(ids)
        ]
    ).to_csv(root / "splits.csv", index=False)
    return root / "dataset.csv", root / "splits.csv", features


def test_a_held_out_cv_run_monitors_each_centre_and_selects_on_their_mean(tmp_path, new_artifact, capsys):
    artifact = new_artifact("slide_mil_cv_subgroup_monitoring")
    dataset_csv, splits_csv, feature_dir = _write_cohort(tmp_path / "data")
    monitor = {"monitor": MONITOR, "mode": "max"}
    config = {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {"execution": CPU_EXECUTION},
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": {
            "name": "mil",
            "params": {
                "aggregator": {"name": "mean_pool"},
                "head": {"name": "cox_survival"},
                "optimizer": {"name": "adam", "params": {"lr": 5e-2, "weight_decay": 1e-5}},
            },
        },
        "trainer": {
            "max_epochs": EPOCHS,
            "val_every_n_epochs": 2,
            "callbacks": [
                {"name": "checkpoint", "params": monitor},
                {"name": "early_stopping", "params": {**monitor, "patience": EPOCHS}},
            ],
        },
        "evaluation": {"metrics": ["c_index"], "subgroups": ["centre"], "holdout_test": True},
    }
    run = run_soma(config, tmp_path / "config.yaml", "--feature-dir", str(feature_dir))
    console = capsys.readouterr()
    artifact.record_run("cv", run)
    artifact.check_training_reduced_loss("cv", run)

    group_keys = [
        f"tune/centre={group}/{name}" for group in ("A", "B") for name in ("loss", "c_index")
    ] + [f"tune/mean_over_centre/{name}" for name in ("loss", "loss_groups", "c_index", "c_index_groups")]
    evaluated = [epoch % 2 == 1 for epoch in range(EPOCHS)]
    for fold, epochs in sorted(run.histories().items()):
        artifact.check_equal(f"{fold}/epochs", len(epochs), EPOCHS)
        for key in group_keys:
            logged = [row.get(key) is not None for row in epochs]
            artifact.check_equal(f"{fold}/{key}/logged_on_evaluated_epochs", logged, evaluated)
        means = [row[MONITOR] for row in epochs if row[MONITOR] is not None]
        artifact.check_equal(f"{fold}/{MONITOR}/finite", all(math.isfinite(m) for m in means), True)
        best = max(means)
        provenance = json.loads((run.run_dir / fold / "bundle" / "provenance.json").read_text())
        selected = provenance["selected_epoch"]
        artifact.check_equal(f"{fold}/selection_monitor", provenance["selection_monitor"], [MONITOR, best])
        artifact.check_equal(f"{fold}/selected_epoch_has_the_best_mean", epochs[selected][MONITOR], best)
        first_best = next(row["epoch"] for row in epochs if row[MONITOR] == best)
        artifact.check_equal(f"{fold}/selected_epoch_is_the_first_best", selected, first_best)
        # Cox: each centre's partial likelihood runs over its own risk sets, so the mean
        # over centres is a stratified loss, not the pooled one.
        last = epochs[-1]
        artifact.check_equal(
            f"{fold}/mean_over_centre_loss_differs_from_the_pooled_loss",
            not math.isclose(last["tune/mean_over_centre/loss"], last["tune/loss"], rel_tol=1e-6),
            True,
        )
        written = (run.run_dir / fold / "subgroup_metrics_tune.json").exists()
        artifact.check_equal(f"{fold}/subgroup_metrics_tune_written", written, True)
    for group in ("A", "B"):
        artifact.check_equal(
            f"summary/tune/centre={group}/c_index_mean", f"tune/centre={group}/c_index_mean" in run.summary, True
        )
    mean_c_index = run.summary["tune/mean_over_centre/c_index_mean"]
    artifact.check_at_least("summary/tune/mean_over_centre/c_index_mean", mean_c_index, 0.6)
    # The progress lines name the monitor, but no single centre.
    artifact.check_equal("console/shows_the_monitor", f"selected {MONITOR}=" in console.out, True)
    artifact.check_equal("console/no_group_key", "centre=" in console.out + console.err, False)

    # The same run with the eval-mode train pass.
    eval_train = {
        **config,
        "run": {**config["run"], "output_dir": str(tmp_path / "runs_eval_train")},
        "trainer": {**config["trainer"], "eval_train": True},
    }
    on = run_soma(eval_train, tmp_path / "config_eval_train.yaml", "--feature-dir", str(feature_dir))
    console = capsys.readouterr()
    artifact.record_run("cv_eval_train", on)
    train_keys = ["train/loss", "train/c_index"] + [key.replace("tune/", "train/", 1) for key in group_keys]
    cohort = Cohort.from_csv(dataset_csv, splits_csv, targets=["time", "event"])
    source = CachedSetSource(feature_dir)
    off_histories = run.histories()
    for fold, epochs in sorted(on.histories().items()):
        for key in train_keys:
            logged = [row.get(key) is not None for row in epochs]
            artifact.check_equal(f"eval_train/{fold}/{key}/logged_on_evaluated_epochs", logged, evaluated)
        # The pass draws nothing from the training RNG: tune and the selection are unchanged.
        tune_values = [{k: v for k, v in row.items() if k.startswith("tune/")} for row in epochs]
        off_tune_values = [{k: v for k, v in row.items() if k.startswith("tune/")} for row in off_histories[fold]]
        artifact.check_equal(f"eval_train/{fold}/tune_history_unchanged", tune_values == off_tune_values, True)
        selected = json.loads((on.run_dir / fold / "bundle" / "provenance.json").read_text())["selected_epoch"]
        off_selected = json.loads((run.run_dir / fold / "bundle" / "provenance.json").read_text())["selected_epoch"]
        artifact.check_equal(f"eval_train/{fold}/selected_epoch_unchanged", selected, off_selected)
        # The bundle holds the weights the pass scored at the selected epoch: rescoring every
        # train sample with it gives the logged values. Cox: the loss is the partial
        # likelihood over the whole train split, not over the event-balanced training batches.
        index = int(fold.removeprefix("fold_"))
        bundle = soma.load(on.run_dir / fold / "bundle")
        rescored = soma.predict(bundle.model, loaders(source, cohort, head=bundle.model.task_head, fold=index).train)
        targets = cohort.targets("train", fold=index)
        c_index = soma.evaluate(rescored, targets, metrics=["c_index"]).metrics["c_index"]
        row = epochs[selected]
        artifact.check_equal(
            f"eval_train/{fold}/train_c_index_is_the_rescored_train_split",
            math.isclose(row["train/c_index"], c_index, abs_tol=1e-9),
            True,
        )
        risk = rescored.frame.set_index("sample_id")["risk_score"].reindex(targets.index)
        loss = _partial_likelihood(risk.to_numpy(), targets["time"].to_numpy(), targets["event"].to_numpy())
        artifact.check_equal(
            f"eval_train/{fold}/train_loss_is_the_partial_likelihood_over_the_train_split",
            math.isclose(row["train/loss"], loss, rel_tol=1e-5),
            True,
        )
    # The progress lines show the pooled train loss, but no single centre.
    artifact.check_equal("eval_train/console/shows_the_train_loss", "| train/loss " in console.out, True)
    artifact.check_equal("eval_train/console/no_group_key", "centre=" in console.out + console.err, False)
    artifact.assert_passed()


def _partial_likelihood(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Breslow's negative partial log-likelihood, averaged over the events."""
    terms = [
        -(risk[i] - math.log(np.exp(risk[time >= time[i]]).sum())) for i in range(len(risk)) if event[i]
    ]
    return float(np.mean(terms))
