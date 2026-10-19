"""Runtime callbacks: ``Pipeline(config, callback_factory=...)`` attaches observers to the
Trainer of each fold it trains.

The factory is a runtime input, like ``source``: ``run()`` calls it once per fold it
trains, with the fold's :class:`~soma.pipeline.FoldContext`, and appends the callbacks it
returns after the plan's. An observer reads each epoch while the fold trains, before
anything of the fold is written.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import yaml

from soma.data import CachedSetSource
from soma.pipeline import FoldContext, Pipeline
from tests.experiment_configs import experiment
from tests.pooled_features import setup_multifold_data, setup_synthetic_data

SEED = 7


def _inputs(tmp_path: Path, *, folds: int = 1, trainer: dict[str, Any] | None = None, run: dict | None = None):
    """A mean-pool classifier over pooled bags (one fold, or two), and the bags' source."""
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    write = setup_multifold_data if folds == 2 else setup_synthetic_data
    dataset_csv, splits_csv, feature_dir = write(data)
    config = experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / "outputs",
        seed=SEED,
        aggregator="mean_pool",
        loaders={"batch_size": 2},
        trainer={"max_epochs": 3, **(trainer or {})},
        run=run,
    )
    return config, CachedSetSource(feature_dir)


def _with_run(config, **run):
    return replace(config, run=replace(config.run, **run))


def _fold_bytes(run_dir: Path, fold: int) -> dict[str, bytes]:
    """The fold's predictions and weights, the files a changed random state would change."""
    fold_dir = run_dir / f"fold_{fold}"
    return {
        name: (fold_dir / name).read_bytes()
        for name in ("predictions_test.parquet", "bundle/weights.safetensors")
    }


class Observers:
    """A callback factory that keeps every context it is given and what its observers read."""

    def __init__(self) -> None:
        self.contexts: list[FoldContext] = []
        #: ``(hook, fold, epoch)`` in call order, over every observer.
        self.events: list[tuple[str, int, int]] = []
        #: What an observer read in each ``on_epoch_end``, over every observer.
        self.epochs: list[dict[str, Any]] = []

    def __call__(self, context: FoldContext) -> list[Any]:
        self.contexts.append(context)
        return [EpochReader(context, self)]


class EpochReader:
    """A read-only observer: it reads the documented Trainer attributes at each epoch end."""

    def __init__(self, context: FoldContext, log: Observers) -> None:
        self.context, self.log = context, log

    def on_epoch_start(self, trainer, learner) -> None:
        self.log.events.append(("epoch_start", self.context.fold, trainer.epoch))

    def on_epoch_end(self, trainer, learner) -> None:
        self.log.events.append(("epoch_end", self.context.fold, trainer.epoch))
        self.log.epochs.append(
            {
                "fold": self.context.fold,
                "epoch": trainer.epoch,
                "total_epochs": trainer.total_epochs,
                "train_loss": trainer.train_loss,
                "lr": trainer.lr,
                "optimizer_steps": trainer.optimizer_steps,
                "summary": None if trainer.summary is None else dict(trainer.summary),
                "row": dict(trainer.history.rows[-1]),
                "fold_files": sorted(p.name for p in self.context.fold_dir.glob("*.json")),
            }
        )


def _run_record(run_dir: Path) -> dict[str, Any]:
    return yaml.safe_load((run_dir / "run.yaml").read_text())


def _history(fold_dir: Path) -> list[dict[str, Any]]:
    return json.loads((fold_dir / "training_history.json").read_text())["epochs"]


def _as_written(row: dict[str, Any]) -> dict[str, Any]:
    """``row`` as ``training_history.json`` writes it: a NaN (an AUROC on a one-class
    tune split) is ``null``."""
    return {key: None if isinstance(value, float) and math.isnan(value) else value for key, value in row.items()}


def test_the_factory_gets_the_context_of_each_fold_it_trains(tmp_path: Path):
    config, source = _inputs(tmp_path, folds=2)
    observers = Observers()

    result = Pipeline(config, source=source, callback_factory=observers).run()

    run_dir = result.run_dir
    assert observers.contexts == [
        FoldContext(
            run_id=run_dir.name,
            run_dir=run_dir,
            fold_dir=run_dir / f"fold_{fold}",
            fold=fold,
            num_folds=2,
            seed=SEED,
        )
        for fold in (0, 1)
    ]


def test_a_single_fold_trains_in_the_run_dir(tmp_path: Path):
    config, source = _inputs(tmp_path)
    observers = Observers()

    result = Pipeline(config, source=source, callback_factory=observers).run()

    [context] = observers.contexts
    assert (context.fold, context.num_folds) == (0, 1)
    assert context.fold_dir == context.run_dir == result.run_dir
    with pytest.raises(FrozenInstanceError):
        context.fold = 1


def test_an_observer_reads_each_epoch_while_the_fold_trains(tmp_path: Path):
    config, source = _inputs(tmp_path, trainer={"max_epochs": 3})
    observers = Observers()

    result = Pipeline(config, source=source, callback_factory=observers).run()

    # Epoch k is read before epoch k + 1 starts…
    assert observers.events == [
        (hook, 0, epoch) for epoch in range(3) for hook in ("epoch_start", "epoch_end")
    ]
    # …and before the fold writes its metrics or its history.
    assert all("metrics.json" not in seen["fold_files"] for seen in observers.epochs)
    assert all("training_history.json" not in seen["fold_files"] for seen in observers.epochs)
    # The row read at epoch k is epoch k's row of training_history.json.
    history = _history(result.run_dir)
    assert [_as_written(seen["row"]) for seen in observers.epochs] == history
    for seen, row in zip(observers.epochs, history):
        assert (seen["epoch"], seen["total_epochs"]) == (row["epoch"], 3)
        assert (seen["train_loss"], seen["lr"]) == (row["train_loss"], row["lr"])
        assert _as_written(seen["summary"]) == {key: row[key] for key in seen["summary"]}
        assert "loss" in seen["summary"]
    # Six train bags in batches of two: three optimizer steps per epoch.
    assert [seen["optimizer_steps"] for seen in observers.epochs] == [3, 6, 9]


def test_an_epoch_without_tune_evaluation_has_no_summary_and_no_tune_values(tmp_path: Path):
    config, source = _inputs(tmp_path, trainer={"max_epochs": 4, "val_every_n_epochs": 2})
    observers = Observers()

    result = Pipeline(config, source=source, callback_factory=observers).run()

    history = _history(result.run_dir)
    train_side = {"epoch", "train_loss", "lr", "elapsed_seconds"}
    assert [seen["summary"] is None for seen in observers.epochs] == [True, False, True, False]
    for seen, row in zip(observers.epochs, history):
        if seen["summary"] is None:
            assert set(seen["row"]) == train_side
            assert {key: row[key] for key in train_side} == seen["row"]
            assert all(row[key] is None for key in set(row) - train_side)
        else:
            assert _as_written(seen["row"]) == row
            assert "loss" in seen["summary"]


def test_a_resume_builds_observers_only_for_the_folds_it_trains(tmp_path: Path):
    config, source = _inputs(tmp_path, folds=2, run={"run_id": "shared-run"})
    first = Observers()
    run_dir = Pipeline(config, source=source, callback_factory=first).run().run_dir
    assert [context.fold for context in first.contexts] == [0, 1]
    # A crash between fold 1's metrics and its bundle: fold 1 is not complete.
    shutil.rmtree(run_dir / "fold_1" / "bundle")

    resumed = Observers()
    Pipeline(config, source=source, callback_factory=resumed).run()

    # The complete fold 0 is skipped without observers; fold 1 restarts from epoch 0.
    assert [context.fold for context in resumed.contexts] == [1]
    assert [(seen["fold"], seen["epoch"]) for seen in resumed.epochs] == [(1, 0), (1, 1), (1, 2)]


def test_a_fold_outside_run_folds_gets_no_observers(tmp_path: Path):
    config, source = _inputs(tmp_path, folds=2, run={"run_id": "shared-run", "folds": (1,)})
    observers = Observers()

    Pipeline(config, source=source, callback_factory=observers).run()

    assert [context.fold for context in observers.contexts] == [1]
    assert {seen["fold"] for seen in observers.epochs} == {1}


def test_building_observers_does_not_move_the_fold_random_state(tmp_path: Path):
    """The factory runs before the fold is seeded: even one that draws from every global
    generator leaves each fold's weights and predictions byte-identical."""
    config, source = _inputs(tmp_path, folds=2, run={"run_id": "without-factory"})
    plain = Pipeline(config, source=source).run().run_dir

    def drawing(context: FoldContext) -> list[Any]:
        random.random()
        np.random.rand()
        torch.rand(1)
        return []

    drawn = Pipeline(_with_run(config, run_id="drawing-factory"), source=source, callback_factory=drawing).run()

    for fold in (0, 1):
        assert _fold_bytes(drawn.run_dir, fold) == _fold_bytes(plain, fold)


def test_a_factory_leaves_the_plan_the_experiment_and_the_component_record_unchanged(tmp_path: Path):
    config, source = _inputs(tmp_path)

    plain = Pipeline(config, source=source).run().run_dir
    observed = Pipeline(config, source=source, callback_factory=Observers()).run().run_dir

    assert observed != plain
    assert (observed / "config.yaml").read_bytes() == (plain / "config.yaml").read_bytes()
    assert (observed / "plan.yaml").read_bytes() == (plain / "plan.yaml").read_bytes()
    assert _run_record(observed)["experiment_id"] == _run_record(plain)["experiment_id"]
    assert observed.parent == plain.parent  # one experiment dir
    assert _run_record(observed)["components"] == _run_record(plain)["components"]


@pytest.mark.parametrize("started_with_factory", [True, False], ids=["drop-the-factory", "add-a-factory"])
def test_a_resume_may_drop_or_add_the_factory(tmp_path: Path, started_with_factory: bool):
    config, source = _inputs(tmp_path, folds=2, run={"run_id": "shared-run"})

    def launch(with_factory: bool):
        return Pipeline(config, source=source, callback_factory=Observers() if with_factory else None).run()

    run_dir = launch(started_with_factory).run_dir
    shutil.rmtree(run_dir / "fold_1" / "bundle")

    resumed = launch(not started_with_factory)

    assert [fold.fold for fold in resumed.fold_results] == [1]
    assert _run_record(run_dir)["status"] == "completed"
    assert (run_dir / "summary.json").is_file()


class Exploding:
    def on_epoch_end(self, trainer, learner) -> None:
        raise RuntimeError("the observer broke at epoch end")


class ExceptionWitness:
    def __init__(self) -> None:
        self.seen: list[BaseException] = []

    def on_exception(self, trainer, learner, exc: BaseException) -> None:
        self.seen.append(exc)


def test_an_observer_that_raises_fails_the_run(tmp_path: Path):
    config, source = _inputs(tmp_path)
    contexts: list[FoldContext] = []
    witness = ExceptionWitness()

    def factory(context: FoldContext) -> list[Any]:
        contexts.append(context)
        return [Exploding(), witness]

    with pytest.raises(RuntimeError, match="the observer broke at epoch end"):
        Pipeline(config, source=source, callback_factory=factory).run()

    # Every callback hears of the error, even one after the observer that raised.
    assert [str(exc) for exc in witness.seen] == ["the observer broke at epoch end"]
    record = _run_record(contexts[0].run_dir)
    assert record["status"] == "failed"
    assert "the observer broke at epoch end" in record["error"]
    assert not (contexts[0].fold_dir / "metrics.json").exists()


def test_a_factory_that_raises_fails_the_run(tmp_path: Path):
    config, source = _inputs(tmp_path)
    contexts: list[FoldContext] = []

    def factory(context: FoldContext) -> list[Any]:
        contexts.append(context)
        raise ValueError("no observer for this fold")

    with pytest.raises(ValueError, match="no observer for this fold"):
        Pipeline(config, source=source, callback_factory=factory).run()

    record = _run_record(contexts[0].run_dir)
    assert record["status"] == "failed"
    assert "no observer for this fold" in record["error"]
    assert not (contexts[0].fold_dir / "bundle").exists()
