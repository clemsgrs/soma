"""A run's folds train side by side with ``devices``, one fresh process per fold.

The launch extracts the features and fills the feature cache, then hands each pending
fold to a ``spawn`` worker on a free device; the worker trains it as a ``run.folds``
launch of the same run would. The launch then writes ``summary.json``. The result must
be the one the folds give trained one after another. The callback factories live in
``tests.pipeline.spawn_pipelines``: a worker imports them by name, so it sees no
monkeypatch or test-local function.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from pathlib import Path

import pandas as pd
import pytest
import yaml

from soma.pipeline import Pipeline
from tests.benchmark.spawn_benchmarks import cache_snapshot
from tests.e2e.synthetic import make_tile_cv_splits, make_tile_dataset
from tests.pipeline.spawn_pipelines import CACHE_SNAPSHOT, PID_FILE, FoldRecorder, tile_cv_config

CPUS = ["cpu", "cpu"]
FOLDS = 3
#: The packed feature matrix a cached source writes beside the features it packs.
PACKED_FILENAME = "packed_features.pt"


@pytest.fixture(scope="module")
def tile_cv(tmp_path_factory) -> tuple[Path, Path, Path]:
    """The synthetic tile cohort: its records, its one-fold splits and 3-fold CV splits."""
    dataset_csv, splits_csv = make_tile_dataset(tmp_path_factory.mktemp("tiles"))
    return dataset_csv, splits_csv, make_tile_cv_splits(dataset_csv, FOLDS)


@pytest.fixture(scope="module")
def fold_runs(tmp_path_factory, tile_cv):
    """The 3-fold run with its folds in turn, then over two CPU workers, each from an
    empty output root of its own."""
    dataset_csv, _splits_csv, cv_csv = tile_cv
    root = tmp_path_factory.mktemp("fold-runs")

    def run(name, devices):
        output_root = root / name
        recorder = FoldRecorder(cache_root=output_root / "feature_cache")
        config = tile_cv_config(dataset_csv, cv_csv, output_root)
        return Pipeline(config, callback_factory=recorder, devices=devices).run()

    return run("in_turn", None), run("parallel", CPUS)


def _fold_dirs(run_dir: Path) -> list[Path]:
    return [run_dir / f"fold_{fold}" for fold in range(FOLDS)]


def test_folds_trained_side_by_side_give_the_bit_identical_result(fold_runs):
    in_turn, parallel = fold_runs

    for sequential_dir, parallel_dir in zip(_fold_dirs(in_turn.run_dir), _fold_dirs(parallel.run_dir)):
        assert (parallel_dir / "metrics.json").read_bytes() == (sequential_dir / "metrics.json").read_bytes()
        pd.testing.assert_frame_equal(
            pd.read_parquet(parallel_dir / "predictions_test.parquet"),
            pd.read_parquet(sequential_dir / "predictions_test.parquet"),
            check_exact=True,
        )
    assert (parallel.run_dir / "summary.json").read_bytes() == (in_turn.run_dir / "summary.json").read_bytes()
    assert parallel.summary == in_turn.summary
    assert parallel.pending_folds == () == in_turn.pending_folds
    # The folds trained in other processes, so this launch holds none of them.
    assert parallel.fold_results == []
    assert [result.fold for result in in_turn.fold_results] == [0, 1, 2]


def test_each_fold_trains_in_its_own_process_and_logs_beside_its_artifacts(fold_runs):
    in_turn, parallel = fold_runs

    pids = [int((directory / PID_FILE).read_text()) for directory in _fold_dirs(parallel.run_dir)]
    assert len(set(pids)) == FOLDS and os.getpid() not in pids
    for directory in _fold_dirs(parallel.run_dir):
        log = (directory / "worker.log").read_text()
        assert f"{directory.name.replace('_', ' ')} on cpu" in log
    # Without devices every fold trains here, and no worker log is written.
    assert {int((d / PID_FILE).read_text()) for d in _fold_dirs(in_turn.run_dir)} == {os.getpid()}
    assert list(in_turn.run_dir.rglob("worker.log")) == []
    assert multiprocessing.active_children() == []


def test_the_launch_completes_the_run_with_its_own_config(fold_runs):
    _in_turn, parallel = fold_runs

    assert yaml.safe_load((parallel.run_dir / "run.yaml").read_text())["status"] == "completed"
    # Each worker records its launch as partial, also the last one to finish: only the
    # launch that started them completes the run.
    index = pd.read_csv(parallel.run_dir.parents[3] / "indexes" / "runs.csv")
    statuses = index.loc[index["run_id"] == parallel.run_dir.name, "status"].tolist()
    assert statuses[0] == "running" and statuses[-1] == "completed"
    assert sorted(statuses[1:-1]) == ["partial"] * FOLDS + ["running"] * FOLDS
    # The workers record a config with their run.folds and run.run_id; the run keeps the
    # one it was started with, so its config.yaml and plan.yaml replay the whole run.
    for name in ("config.yaml", "plan.yaml"):
        run = yaml.safe_load((parallel.run_dir / name).read_text())["run"]
        assert run["folds"] is None and run["run_id"] is None, name
    assert list(parallel.run_dir.glob(".*")) == []


def _extracted(snapshot: dict[str, list[int]]) -> dict[str, list[int]]:
    """A cache snapshot without the packed matrix, which the first fold to read the cache
    writes beside the features it packs."""
    return {name: entry for name, entry in snapshot.items() if Path(name).name != PACKED_FILENAME}


def test_the_workers_read_the_features_the_launch_extracted(fold_runs):
    _in_turn, parallel = fold_runs
    cache_root = parallel.run_dir.parents[3] / "feature_cache"

    # As each fold starts, every extracted file is the one the launch wrote: same inode,
    # size and modification time as at the end, so no worker extracted again.
    final = cache_snapshot(cache_root)
    snapshots = [json.loads((directory / CACHE_SNAPSHOT).read_text()) for directory in _fold_dirs(parallel.run_dir)]
    assert _extracted(final)
    assert [_extracted(snapshot) for snapshot in snapshots] == [_extracted(final)] * FOLDS
    assert [Path(name).name for name in set(final) - set(_extracted(final))] == [PACKED_FILENAME]
    assert list(cache_root.rglob("*.tmp")) == []


# --- the process boundary -------------------------------------------------------------


class _Unpicklable:
    """A feature source stand-in that cannot be pickled: it holds a lock."""

    def __init__(self) -> None:
        self.lock = threading.Lock()


@pytest.mark.parametrize("name", ["source", "callback_factory"])
def test_an_input_that_does_not_pickle_is_refused_before_any_extraction(tmp_path, tile_cv, name):
    dataset_csv, _splits_csv, cv_csv = tile_cv
    config = tile_cv_config(dataset_csv, cv_csv, tmp_path / "out")
    inputs = {"source": _Unpicklable()} if name == "source" else {"callback_factory": lambda context: []}

    with pytest.raises(TypeError, match=rf"{name}.*does not pickle.*module level") as raised:
        Pipeline(config, devices=CPUS, **inputs).run()

    assert str(raised.value).count("does not pickle") == 1
    assert not (tmp_path / "out").exists()


def test_devices_need_the_feature_cache_the_workers_read(tmp_path, tile_cv):
    dataset_csv, _splits_csv, cv_csv = tile_cv
    config = tile_cv_config(dataset_csv, cv_csv, tmp_path / "out")
    config.features.cache.enabled = False

    with pytest.raises(ValueError, match="features.cache.enabled"):
        Pipeline(config, devices=CPUS).run()

    assert not (tmp_path / "out").exists()


def test_a_device_that_is_not_one_is_refused_when_the_pipeline_is_built(tmp_path, tile_cv):
    dataset_csv, _splits_csv, cv_csv = tile_cv

    with pytest.raises(ValueError, match="'gpu0' is not a device"):
        Pipeline(tile_cv_config(dataset_csv, cv_csv, tmp_path / "out"), devices=["gpu0"])


def test_a_one_fold_run_warns_and_trains_here(tmp_path, tile_cv):
    dataset_csv, splits_csv, _cv_csv = tile_cv
    contexts = []

    def factory(context):  # a local function: a one-fold run never pickles it
        contexts.append(context.fold)
        return []

    config = tile_cv_config(dataset_csv, splits_csv, tmp_path / "out")
    with pytest.warns(UserWarning, match=r"devices.*ignored.*one fold.*on cpu.*CUDA_VISIBLE_DEVICES"):
        result = Pipeline(config, callback_factory=factory, devices=CPUS).run()

    assert contexts == [0]
    assert [fold.fold for fold in result.fold_results] == [0]
    assert (result.run_dir / "summary.json").is_file()
    assert list(result.run_dir.rglob("worker.log")) == []


# --- run.folds, resume and failures ---------------------------------------------------


def _started(output: str) -> list[str]:
    return [line.split(" started on")[0] for line in output.splitlines() if " started on " in line]


def test_run_folds_spread_their_subset_and_a_resume_dispatches_only_the_pending_folds(tmp_path, tile_cv, capsys):
    dataset_csv, _splits_csv, cv_csv = tile_cv
    config = tile_cv_config(dataset_csv, cv_csv, tmp_path / "out", run_id="shared")
    subset = tile_cv_config(dataset_csv, cv_csv, tmp_path / "out", run_id="shared", folds=(0, 2))

    first = Pipeline(subset, callback_factory=FoldRecorder(), devices=CPUS).run()

    run_dir = first.run_dir
    assert sorted(_started(capsys.readouterr().out)) == ["fold 0", "fold 2"]
    assert first.pending_folds == (1,) and first.summary == {}
    assert not (run_dir / "fold_1").exists()
    assert not (run_dir / "summary.json").exists()
    assert yaml.safe_load((run_dir / "run.yaml").read_text())["status"] == "partial"
    kept = {fold: (run_dir / f"fold_{fold}" / "metrics.json").stat().st_mtime_ns for fold in (0, 2)}

    # The resume trains the one pending fold, on a single device, and completes the run.
    second = Pipeline(config, callback_factory=FoldRecorder(), devices=["cpu"]).run()

    assert _started(capsys.readouterr().out) == ["fold 1"]
    assert {fold: (run_dir / f"fold_{fold}" / "metrics.json").stat().st_mtime_ns for fold in (0, 2)} == kept
    assert int((run_dir / "fold_1" / PID_FILE).read_text()) != os.getpid()
    assert second.pending_folds == ()
    assert yaml.safe_load((run_dir / "run.yaml").read_text())["status"] == "completed"
    summary = (run_dir / "summary.json").read_bytes()

    # With every fold complete, a relaunch starts no worker and only summarizes.
    (run_dir / "summary.json").unlink()
    third = Pipeline(config, devices=CPUS).run()

    assert _started(capsys.readouterr().out) == []
    assert (run_dir / "summary.json").read_bytes() == summary
    assert third.summary == second.summary


def test_a_failing_fold_stops_new_folds_lets_the_running_one_finish_and_a_rerun_resumes(tmp_path, tile_cv):
    dataset_csv, _splits_csv, cv_csv = tile_cv
    config = tile_cv_config(dataset_csv, cv_csv, tmp_path / "out", run_id="failing")

    # Fold 0 fails while fold 1 still runs: fold 0's device is free, but fold 2 must not
    # start on it, and fold 1 completes before the error is raised.
    with pytest.raises(ValueError, match="fold 0 failed") as raised:
        Pipeline(config, callback_factory=FoldRecorder(actions={0: "raise", 1: "slow"}), devices=CPUS).run()

    (run_dir,) = (tmp_path / "out" / "experiments").glob("*/runs/failing")
    log = run_dir / "fold_0" / "worker.log"
    assert any("fold 0" in note and str(log) in note for note in raised.value.__notes__)
    assert "ValueError: fold 0 failed" in log.read_text()
    assert (run_dir / "fold_1" / "metrics.json").is_file()
    assert not (run_dir / "fold_2" / PID_FILE).exists()
    assert not (run_dir / "summary.json").exists()
    assert yaml.safe_load((run_dir / "run.yaml").read_text())["status"] == "failed"
    assert multiprocessing.active_children() == []
    kept = (run_dir / "fold_1" / "metrics.json").stat().st_mtime_ns

    resumed = Pipeline(config, callback_factory=FoldRecorder(), devices=CPUS).run()

    assert (run_dir / "fold_1" / "metrics.json").stat().st_mtime_ns == kept
    assert resumed.pending_folds == ()
    assert (run_dir / "summary.json").is_file()
    assert set(json.loads((run_dir / "summary.json").read_text())) == set(resumed.summary)
