"""Resumable multi-fold CV runs (issue #244).

Covers the four moving parts of resume:

* **run-id resolution** — ``resume`` reuses the latest run dir, ``run_id`` pins one,
  neither mints a fresh timestamped id (unchanged default);
* **disk aggregation** — ``summary.json`` is rebuilt from every fold's ``metrics.json``
  on disk, so folds skipped this session still count (the correctness fix);
* **fold-skip guard + end-to-end resume** — a relaunch into the same run dir skips
  folds that already wrote ``metrics.json`` and the resumed summary aggregates all folds;
* **config-drift guard** — resuming into a run dir whose saved config differs is refused.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
    save_config,
)
from soma.output_layout import (
    latest_existing_run_id,
    resolve_managed_output_paths,
)
from soma.pipeline import (
    _aggregate_fold_metrics,
    _aggregate_fold_metrics_from_disk,
    _guard_resume_config_drift,
)


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def _make_config(tmp_path: Path, **overrides) -> PipelineConfig:
    dataset_csv = _write_csv(
        tmp_path / "dataset.csv",
        "sample_id,image_path,label\ns0,/slides/s0.svs,tumor\n",
    )
    splits_csv = _write_csv(
        tmp_path / "splits.csv",
        "fold,sample_id,split\n0,s0,train\n",
    )
    defaults = dict(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "outputs",
        dataset_type="slide",
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=0.5),
        cache=CacheConfig(),
        encoder=EncoderConfig(name="uni2"),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 128}),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(seed=7, epochs=10, learning_rate=1e-4),
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _write_metrics(fold_dir: Path, *, tune: dict, test: dict) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "metrics.json").write_text(json.dumps({"tune": tune, "test": test}))


# --------------------------------------------------------------------------- #
# run-id resolution
# --------------------------------------------------------------------------- #


def test_default_mints_fresh_timestamped_run_id(tmp_path: Path):
    layout = resolve_managed_output_paths(_make_config(tmp_path))
    assert layout.run_id.endswith("__local")
    assert layout.run_dir == layout.experiment_dir / "runs" / layout.run_id


def test_explicit_run_id_config_pins_that_run(tmp_path: Path):
    layout = resolve_managed_output_paths(_make_config(tmp_path, run_id="pinned-run"))
    assert layout.run_id == "pinned-run"
    assert layout.run_dir.name == "pinned-run"


def test_resume_reuses_latest_existing_run(tmp_path: Path):
    cfg = _make_config(tmp_path)
    first = resolve_managed_output_paths(cfg)
    first.run_dir.mkdir(parents=True)
    # A newer run dir under the same experiment — resume must land on the latest.
    (first.experiment_dir / "runs" / "2999-01-01_00-00-00__local").mkdir()

    resumed = resolve_managed_output_paths(replace(cfg, resume=True))
    assert resumed.run_id == "2999-01-01_00-00-00__local"
    assert resumed.experiment_dir == first.experiment_dir


def test_resume_without_prior_run_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no prior run"):
        resolve_managed_output_paths(_make_config(tmp_path, resume=True))


def test_explicit_run_id_arg_still_wins_over_config(tmp_path: Path):
    # Back-compat: callers passing run_id= (tests, leaderboard replay) are unaffected.
    layout = resolve_managed_output_paths(
        _make_config(tmp_path, resume=True), run_id="2020-01-01_00-00-00__local"
    )
    assert layout.run_id == "2020-01-01_00-00-00__local"


def test_latest_existing_run_id_orders_lexically(tmp_path: Path):
    runs = tmp_path / "runs"
    assert latest_existing_run_id(tmp_path) is None
    for name in ["2026-01-01_09-00-00__local", "2026-01-02_08-00-00__local"]:
        (runs / name).mkdir(parents=True)
    assert latest_existing_run_id(tmp_path) == "2026-01-02_08-00-00__local"


# --------------------------------------------------------------------------- #
# disk aggregation — the correctness fix
# --------------------------------------------------------------------------- #


def test_from_disk_aggregates_every_fold_present(tmp_path: Path):
    # Two completed folds on disk; only one would be "in memory" on a resume.
    _write_metrics(tmp_path / "fold_0", tune={"auroc": 0.6}, test={"auroc": 0.8})
    _write_metrics(tmp_path / "fold_1", tune={"auroc": 0.7}, test={"auroc": 0.6})

    summary = _aggregate_fold_metrics_from_disk(tmp_path, single_fold=False, include_tune=False)

    # Mean over BOTH folds (0.8, 0.6) — not just the one that ran this session.
    assert summary["test/auroc_mean"] == pytest.approx(0.7)
    assert summary["test/auroc_std"] == pytest.approx(np.std([0.8, 0.6]))
    assert "tune/auroc_mean" not in summary  # tune excluded unless include_tune


def test_from_disk_single_fold_reads_run_dir_metrics(tmp_path: Path):
    _write_metrics(tmp_path, tune={"auroc": 0.9}, test={"auroc": 0.85})
    summary = _aggregate_fold_metrics_from_disk(tmp_path, single_fold=True, include_tune=False)
    assert summary == {"test/auroc": 0.85}


def test_from_disk_include_tune_surfaces_tune(tmp_path: Path):
    _write_metrics(tmp_path, tune={"auroc": 0.9}, test={"auroc": 0.85})
    summary = _aggregate_fold_metrics_from_disk(tmp_path, single_fold=True, include_tune=True)
    assert summary == {"tune/auroc": 0.9, "test/auroc": 0.85}


def test_from_disk_matches_in_memory_aggregation(tmp_path: Path):
    from soma.evaluation import EvaluationReport
    from soma.pipeline import FoldResult

    per_fold = [
        {"tune": {"auroc": 0.6}, "test": {"auroc": 0.8}},
        {"tune": {"auroc": 0.7}, "test": {"auroc": 0.6}},
    ]
    for i, m in enumerate(per_fold):
        _write_metrics(tmp_path / f"fold_{i}", tune=m["tune"], test=m["test"])
    fold_results = [
        FoldResult(
            fold=i,
            train_result=None,
            tune_report=EvaluationReport(split="tune", metrics=m["tune"], predictions=[]),
            test_reports={"test": EvaluationReport(split="test", metrics=m["test"], predictions=[])},
        )
        for i, m in enumerate(per_fold)
    ]
    disk = _aggregate_fold_metrics_from_disk(tmp_path, single_fold=False, include_tune=False)
    in_memory = _aggregate_fold_metrics(fold_results, include_tune=False)
    assert disk == in_memory


# --------------------------------------------------------------------------- #
# config-drift guard
# --------------------------------------------------------------------------- #


def test_drift_guard_noop_when_not_resuming(tmp_path: Path):
    run_dir = tmp_path / "run"
    save_config(_make_config(tmp_path), run_dir / "config.yaml")
    # Different config, but resume/run_id unset → guard must not fire.
    changed = _make_config(tmp_path, training=TrainingConfig(seed=999))
    _guard_resume_config_drift(run_dir, changed)  # no raise


def test_drift_guard_passes_on_identical_config(tmp_path: Path):
    run_dir = tmp_path / "run"
    cfg = _make_config(tmp_path)
    save_config(cfg, run_dir / "config.yaml")
    # Adding the resume directive alone is not drift (it is not serialized).
    _guard_resume_config_drift(run_dir, replace(cfg, resume=True))  # no raise


def test_drift_guard_refuses_changed_config(tmp_path: Path):
    run_dir = tmp_path / "run"
    save_config(_make_config(tmp_path, training=TrainingConfig(seed=7)), run_dir / "config.yaml")
    changed = _make_config(tmp_path, training=TrainingConfig(seed=42), resume=True)
    with pytest.raises(ValueError, match="Refusing to resume"):
        _guard_resume_config_drift(run_dir, changed)


def test_drift_guard_noop_when_no_saved_config(tmp_path: Path):
    # A pinned run id naming a not-yet-created dir has nothing to compare.
    _guard_resume_config_drift(tmp_path / "fresh", _make_config(tmp_path, run_id="fresh"))


def test_drift_guard_allows_changing_operational_mirror_root(tmp_path: Path):
    run_dir = tmp_path / "run"
    original = _make_config(tmp_path, mirror_root=tmp_path / "shared-a")
    save_config(original, run_dir / "config.yaml")

    resumed = replace(
        original,
        mirror_root=tmp_path / "shared-b",
        resume=True,
    )

    _guard_resume_config_drift(run_dir, resumed)


# --------------------------------------------------------------------------- #
# end-to-end resume (CPU synthetic pipeline)
# --------------------------------------------------------------------------- #


def test_resume_skips_completed_fold_and_summary_covers_all_folds(tmp_path: Path):
    pytest.importorskip("torch")
    from soma.dataset import Dataset, Splits
    from soma.features import FeatureStore
    from soma.pipeline import train
    from tests.test_pipeline import _setup_multifold_data

    dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
    dataset = Dataset(dataset_csv)
    splits = Splits(splits_csv, dataset)
    store = FeatureStore(feature_dir)
    run_dir = tmp_path / "output"
    # test_digest + run_id mirror the production Pipeline.run() call so the resume also
    # exercises the #247 test-clobber guard (a resume must keep scoring its own test set).
    kwargs = dict(
        feature_store=store,
        dataset=dataset,
        splits=splits,
        aggregator=AggregatorConfig(name="mean_pool"),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=2, patience=10, batch_size=2),
        run_dir=run_dir,
        test_digest="test-identity-abc",
        run_id="2026-07-08_10-00-00__local",
    )

    # Single-shot 2-fold run, then simulate an interruption after fold 0.
    train(**kwargs)
    assert (run_dir / "fold_0" / "metrics.json").exists()
    assert (run_dir / "fold_1" / "metrics.json").exists()
    fold0_metrics = run_dir / "fold_0" / "metrics.json"
    fold0_mtime = fold0_metrics.stat().st_mtime_ns
    shutil.rmtree(run_dir / "fold_1")
    (run_dir / "summary.json").unlink()

    # Resume into the same run dir.
    resumed = train(**kwargs)

    # Fold 0 was skipped (its metrics.json is byte-for-byte untouched)…
    assert fold0_metrics.stat().st_mtime_ns == fold0_mtime
    # …only the missing fold retrained, so the in-memory result holds fold 1 alone…
    assert [fr.fold for fr in resumed.fold_results] == [1]
    # …yet fold 1 is back on disk, WITH test metrics (the clobber guard didn't skip
    # scoring on resume)…
    fold1_metrics = json.loads((run_dir / "fold_1" / "metrics.json").read_text())
    assert "test" in fold1_metrics and "auroc" in fold1_metrics["test"]
    # …and the summary aggregates BOTH folds.
    summary = json.loads((run_dir / "summary.json").read_text())
    disk_aurocs = [
        json.loads((run_dir / f"fold_{i}" / "metrics.json").read_text())["test"]["auroc"]
        for i in (0, 1)
    ]
    assert summary["test/auroc_mean"] == pytest.approx(float(np.mean(disk_aurocs)))
