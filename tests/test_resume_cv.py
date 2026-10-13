"""Resumable multi-fold CV runs (issue #244).

Covers the four moving parts of resume:

* **run-id resolution** — ``resume`` reuses the latest run dir, ``run_id`` pins one,
  neither mints a fresh timestamped id (unchanged default);
* **disk aggregation** — ``summary.json`` is rebuilt from every fold's ``metrics.json``
  on disk, so folds skipped this session still count (the correctness fix);
* **fold-skip guard + end-to-end resume** — a relaunch into the same run dir skips
  complete folds (``metrics.json`` and ``bundle/``, ADR 0016), restarts the others from
  epoch 0, and the resumed summary aggregates all folds;
* **config-drift guard** — resuming into a run dir whose saved config differs is refused.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from slide2vec import ExecutionOptions

from tests.extraction_configs import extraction
from soma.config import (
    AggregatorConfig,
    CacheConfig,
    DecoderConfig,
    FeaturesConfig,
    EvalConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
    load_config,
    save_config,
)
from soma.pipeline.layout import (
    latest_existing_run_id,
    resolve_managed_output_paths,
)
from soma.pipeline.aggregate import aggregate_fold_metrics_from_disk
from soma.pipeline.layout import guard_resume_config_drift


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
        features=FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224)),
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

    summary = aggregate_fold_metrics_from_disk(tmp_path, single_fold=False, include_tune=False, num_folds=2)

    # Mean over BOTH folds (0.8, 0.6) — not just the one that ran this session.
    assert summary["test/auroc_mean"] == pytest.approx(0.7)
    assert summary["test/auroc_std"] == pytest.approx(np.std([0.8, 0.6], ddof=1))
    assert "tune/auroc_mean" not in summary  # tune excluded unless include_tune


def test_from_disk_single_fold_reads_run_dir_metrics(tmp_path: Path):
    _write_metrics(tmp_path, tune={"auroc": 0.9}, test={"auroc": 0.85})
    summary = aggregate_fold_metrics_from_disk(tmp_path, single_fold=True, include_tune=False, num_folds=1)
    assert summary == {"test/auroc": 0.85}


def test_from_disk_include_tune_surfaces_tune(tmp_path: Path):
    _write_metrics(tmp_path, tune={"auroc": 0.9}, test={"auroc": 0.85})
    summary = aggregate_fold_metrics_from_disk(tmp_path, single_fold=True, include_tune=True, num_folds=1)
    assert summary == {"tune/auroc": 0.9, "test/auroc": 0.85}


# --------------------------------------------------------------------------- #
# config-drift guard
# --------------------------------------------------------------------------- #


def test_drift_guard_noop_when_not_resuming(tmp_path: Path):
    run_dir = tmp_path / "run"
    save_config(_make_config(tmp_path), run_dir / "config.yaml")
    # Different config, but resume/run_id unset → guard must not fire.
    changed = _make_config(tmp_path, training=TrainingConfig(seed=999))
    guard_resume_config_drift(run_dir, changed)  # no raise


def test_drift_guard_passes_on_identical_config(tmp_path: Path):
    run_dir = tmp_path / "run"
    cfg = _make_config(tmp_path)
    save_config(cfg, run_dir / "config.yaml")
    # Adding the resume directive alone is not drift (it is not serialized).
    guard_resume_config_drift(run_dir, replace(cfg, resume=True))  # no raise


_SEGMENTATION = dict(
    dataset_type="segmentation",
    aggregator=None,
    decoder=DecoderConfig(name="lightweight_conv", params={"hidden_dim": 64}),
    task=TaskConfig(name="segmentation", params={"num_classes": 2}),
)


@pytest.mark.parametrize("overrides", [{}, _SEGMENTATION], ids=["slide", "segmentation"])
def test_drift_guard_passes_resuming_from_the_saved_config_of_a_preset_tiling(tmp_path: Path, overrides):
    """A Python config that leaves the tiling (and a grid run's dense options) to the
    encoder preset saves what the loader resolves, so resuming from the config.yaml it
    saved is not drift."""
    run_dir = tmp_path / "run"
    cfg = _make_config(tmp_path, features=FeaturesConfig(encoder="uni2"), **overrides)
    save_config(cfg, run_dir / "config.yaml")

    reloaded = load_config(run_dir / "config.yaml")

    guard_resume_config_drift(run_dir, replace(reloaded, resume=True))  # no raise


def test_drift_guard_refuses_changed_config(tmp_path: Path):
    run_dir = tmp_path / "run"
    save_config(_make_config(tmp_path, training=TrainingConfig(seed=7)), run_dir / "config.yaml")
    changed = _make_config(tmp_path, training=TrainingConfig(seed=42), resume=True)
    with pytest.raises(ValueError, match="Refusing to resume"):
        guard_resume_config_drift(run_dir, changed)


def test_drift_guard_ignores_the_machine_resolved_execution_resources(tmp_path: Path):
    """slide2vec resolves the GPU count and the preprocessing worker count against the
    machine when its ExecutionOptions are built, so the saved config records what the
    original machine had; resuming on another machine is not drift."""
    run_dir = tmp_path / "run"
    features = FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224))
    saved = _make_config(
        tmp_path,
        features=replace(
            features, execution=ExecutionOptions(num_gpus=4, num_preprocessing_workers=32)
        ),
    )
    save_config(saved, run_dir / "config.yaml")
    resumed = _make_config(
        tmp_path,
        features=replace(
            features, execution=ExecutionOptions(num_gpus=1, num_preprocessing_workers=8)
        ),
        resume=True,
    )
    guard_resume_config_drift(run_dir, resumed)  # no raise


def test_drift_guard_refuses_a_changed_execution_precision(tmp_path: Path):
    run_dir = tmp_path / "run"
    features = FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224))
    save_config(
        _make_config(tmp_path, features=replace(features, execution=ExecutionOptions(precision="fp16"))),
        run_dir / "config.yaml",
    )
    changed = _make_config(
        tmp_path,
        features=replace(features, execution=ExecutionOptions(precision="fp32")),
        resume=True,
    )
    with pytest.raises(ValueError, match="Refusing to resume"):
        guard_resume_config_drift(run_dir, changed)


def test_drift_guard_noop_when_no_saved_config(tmp_path: Path):
    # A pinned run id naming a not-yet-created dir has nothing to compare.
    guard_resume_config_drift(tmp_path / "fresh", _make_config(tmp_path, run_id="fresh"))


def test_drift_guard_allows_changing_operational_mirror_root(tmp_path: Path):
    run_dir = tmp_path / "run"
    original = _make_config(tmp_path, mirror_root=tmp_path / "shared-a")
    save_config(original, run_dir / "config.yaml")

    resumed = replace(
        original,
        mirror_root=tmp_path / "shared-b",
        resume=True,
    )

    guard_resume_config_drift(run_dir, resumed)


def test_drift_guard_refuses_unknown_evaluation_settings(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    original = _make_config(tmp_path)
    save_config(original, run_dir / "config.yaml")
    saved_path = run_dir / "config.yaml"
    saved = yaml.safe_load(saved_path.read_text())
    saved["evaluation"]["patient_oof"] = {"arm": "legacy"}
    saved_path.write_text(yaml.safe_dump(saved))
    resumed = replace(original, resume=True)

    with pytest.raises(ValueError, match="evaluation"):
        guard_resume_config_drift(run_dir, resumed)


# --------------------------------------------------------------------------- #
# end-to-end resume (CPU synthetic pipeline)
# --------------------------------------------------------------------------- #


def _multifold_pipeline_config(tmp_path: Path, **overrides) -> PipelineConfig:
    from tests.test_pipeline import _setup_multifold_data

    dataset_csv, splits_csv, feature_dir = _setup_multifold_data(tmp_path)
    (tmp_path / "feature_dir.txt").write_text(str(feature_dir))
    defaults = dict(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "output",
        dataset_type="slide",
        aggregator=AggregatorConfig(name="mean_pool"),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=2, patience=10, batch_size=2, pin_memory=False),
        run_id="2026-07-08_10-00-00__local",
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _run(tmp_path: Path, config: PipelineConfig):
    from soma.data import CachedSetSource
    from soma.pipeline import Pipeline

    feature_dir = Path((tmp_path / "feature_dir.txt").read_text())
    return Pipeline(config, source=CachedSetSource(feature_dir)).run()


def test_resume_skips_completed_folds_and_restarts_a_fold_without_its_bundle(tmp_path: Path):
    pytest.importorskip("torch")
    config = _multifold_pipeline_config(tmp_path)
    run_dir = _run(tmp_path, config).run_dir
    for fold in (0, 1):
        assert (run_dir / f"fold_{fold}" / "metrics.json").is_file()
        assert (run_dir / f"fold_{fold}" / "bundle").is_dir()
    fold0_metrics = run_dir / "fold_0" / "metrics.json"
    fold0_mtime = fold0_metrics.stat().st_mtime_ns
    # A crash between fold 1's metrics and its bundle: the fold is not complete.
    shutil.rmtree(run_dir / "fold_1" / "bundle")
    (run_dir / "summary.json").unlink()

    resumed = _run(tmp_path, replace(config, resume=True))

    # Fold 0 was skipped (its metrics.json is byte-for-byte untouched)…
    assert fold0_metrics.stat().st_mtime_ns == fold0_mtime
    # …only fold 1 retrained, from epoch 0, so the in-memory result holds it alone…
    assert [fr.fold for fr in resumed.fold_results] == [1]
    assert (run_dir / "fold_1" / "bundle" / "weights.safetensors").is_file()
    # …WITH test metrics (the #247 clobber guard does not skip scoring on resume)…
    fold1_metrics = json.loads((run_dir / "fold_1" / "metrics.json").read_text())
    assert "test" in fold1_metrics and "auroc" in fold1_metrics["test"]
    # …and the summary aggregates BOTH folds.
    summary = json.loads((run_dir / "summary.json").read_text())
    disk_aurocs = [
        json.loads((run_dir / f"fold_{i}" / "metrics.json").read_text())["test"]["auroc"]
        for i in (0, 1)
    ]
    assert summary["test/auroc_mean"] == pytest.approx(float(np.nanmean(disk_aurocs)), nan_ok=True)


def test_from_disk_ignores_nothing_but_rejects_undeclared_fold_dirs(tmp_path: Path):
    _write_metrics(tmp_path / "fold_0", tune={"auroc": 0.9}, test={"auroc": 0.8})
    _write_metrics(tmp_path / "fold_1", tune={"auroc": 0.7}, test={"auroc": 0.6})
    # A leftover fold from a 3-fold launch under a now-2-fold split file.
    _write_metrics(tmp_path / "fold_2", tune={"auroc": 0.5}, test={"auroc": 0.4})

    with pytest.raises(ValueError, match=r"fold_2.*do not declare"):
        aggregate_fold_metrics_from_disk(
            tmp_path, single_fold=False, include_tune=False, num_folds=2
        )


def test_from_disk_reads_only_declared_folds_in_order(tmp_path: Path):
    _write_metrics(tmp_path / "fold_0", tune={"auroc": 0.9}, test={"auroc": 0.8})
    _write_metrics(tmp_path / "fold_1", tune={"auroc": 0.7}, test={"auroc": 0.6})
    summary = aggregate_fold_metrics_from_disk(
        tmp_path, single_fold=False, include_tune=False, num_folds=3
    )
    # fold_2 is declared but not yet on disk (pending): mean over the two present.
    assert summary["test/auroc_mean"] == pytest.approx(0.7)


def test_drift_guard_refuses_changed_manifest_content_under_same_paths(tmp_path: Path):
    run_dir = tmp_path / "run"
    cfg = _make_config(tmp_path)
    save_config(cfg, run_dir / "config.yaml")
    # The run recorded the identity minted from the original train/tune content.
    from soma.pipeline.layout import build_experiment_spec

    (run_dir / "run.yaml").write_text(
        yaml.safe_dump({"experiment_id": build_experiment_spec(cfg).experiment_id})
    )
    guard_resume_config_drift(run_dir, replace(cfg, resume=True))  # unchanged: passes

    # Same paths, different label -> different train/tune content.
    Path(cfg.dataset_csv).write_text("sample_id,image_path,label\ns0,/slides/s0.svs,normal\n")
    with pytest.raises(ValueError, match="train/tune content"):
        guard_resume_config_drift(run_dir, replace(cfg, resume=True))


# --- run.folds: launches that split the folds of one run between them ----------------


def test_folds_directive_loads_from_yaml_and_stays_out_of_the_saved_config(tmp_path: Path):
    from soma.config import config_yaml_dict, load_config

    config = _make_config(tmp_path, run_id="shared-run")
    save_config(config, tmp_path / "config.yaml")
    loaded = load_config(
        tmp_path / "config.yaml", overrides={"run": {"run_id": "shared-run", "folds": [1, 3]}}
    )
    assert loaded.folds == (1, 3)
    assert config_yaml_dict(loaded) == config_yaml_dict(config)


@pytest.mark.parametrize("folds", [(), (1, 1), (-1,)])
def test_folds_directive_rejects_malformed_selections(tmp_path: Path, folds):
    with pytest.raises(ValueError, match="run.folds"):
        _make_config(tmp_path, run_id="shared-run", folds=folds)


def test_folds_directive_needs_a_shared_run_dir(tmp_path: Path):
    with pytest.raises(ValueError, match="run.run_id"):
        _make_config(tmp_path, folds=(0,))


def test_launches_that_split_the_folds_match_a_single_launch(tmp_path: Path):
    pytest.importorskip("torch")
    config = _multifold_pipeline_config(tmp_path, run_id="shared-run")

    # The launch given fold 1 trains it alone and writes no summary: fold 0 is pending.
    first = _run(tmp_path, replace(config, folds=(1,)))
    run_dir = first.run_dir
    assert [fr.fold for fr in first.fold_results] == [1]
    assert first.pending_folds == (0,) and first.summary == {}
    assert not (run_dir / "fold_0").exists()
    assert not (run_dir / "summary.json").exists()
    fold1_mtime = (run_dir / "fold_1" / "metrics.json").stat().st_mtime_ns

    # The launch given fold 0 finds every fold complete afterwards, so it summarizes both.
    second = _run(tmp_path, replace(config, folds=(0,)))
    assert [fr.fold for fr in second.fold_results] == [0]
    assert second.pending_folds == ()
    assert (run_dir / "fold_1" / "metrics.json").stat().st_mtime_ns == fold1_mtime
    split_summary = json.loads((run_dir / "summary.json").read_text())

    # Folds seed independently, so splitting them changes nothing about the result.
    single = _run(tmp_path, replace(config, run_id="single-run"))
    assert split_summary == pytest.approx(single.summary, nan_ok=True)


def test_folds_outside_the_splits_are_refused(tmp_path: Path):
    pytest.importorskip("torch")
    config = _multifold_pipeline_config(tmp_path, run_id="shared-run", folds=(2,))
    with pytest.raises(ValueError, match="run.folds"):
        _run(tmp_path, config)
