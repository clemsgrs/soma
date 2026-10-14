"""Resumable multi-fold CV runs (issue #244).

Covers the four moving parts of resume:

* **run-id resolution** — ``resume`` reuses the latest run dir, ``run_id`` pins one,
  neither mints a fresh timestamped id (unchanged default);
* **disk aggregation** — ``summary.json`` is rebuilt from every fold's ``metrics.json``
  on disk, so folds skipped this session still count (the correctness fix);
* **fold-skip guard + end-to-end resume** — a relaunch into the same run dir skips
  complete folds (``metrics.json`` and ``bundle/``, ADR 0016), restarts the others from
  epoch 0, and the resumed summary aggregates all folds;
* **plan-drift guard** — resuming into a run dir whose saved plan differs is refused.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml
from slide2vec import ExecutionOptions

from tests.extraction_configs import extraction
from soma.config import ExperimentConfig, FeaturesConfig
from soma.pipeline.layout import (
    latest_existing_run_id,
    resolve_managed_output_paths,
)
from soma.pipeline.aggregate import aggregate_fold_metrics_from_disk
from soma.pipeline.layout import guard_resume_plan_drift, record_plan
from soma.plan import Plan, load_cohort, resolve
from tests.experiment_configs import experiment


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


_DATASET = "sample_id,image_path,label\n" + "".join(
    f"s{i},/slides/s{i}.svs,{label}\n" for i, label in enumerate(("tumor", "normal") * 2)
)


def _make_config(tmp_path: Path, **overrides) -> ExperimentConfig:
    """A slide-level AB-MIL classifier; ``overrides`` are :func:`experiment` fields and
    ``run`` keywords (``run_id``, ``resume``, ``folds``, ``mirror_root``)."""
    dataset_csv = _write_csv(tmp_path / "dataset.csv", _DATASET)
    splits_csv = _write_csv(
        tmp_path / "splits.csv",
        "fold,sample_id,split\n0,s0,train\n0,s1,train\n0,s2,tune\n0,s3,test\n",
    )
    run = {key: overrides.pop(key) for key in ("run_id", "resume", "folds", "mirror_root") if key in overrides}
    fields = dict(
        output_dir=tmp_path / "outputs",
        seed=7,
        features=FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224)),
        aggregator={"name": "abmil", "params": {"hidden_dim": 128}},
        optimizer={"name": "adam", "params": {"lr": 1e-4}},
        trainer={"max_epochs": 10},
        run=run,
    )
    fields.update(overrides)
    return experiment(dataset_csv, splits_csv, **fields)


def _with_run(config: ExperimentConfig, **run) -> ExperimentConfig:
    return replace(config, run=replace(config.run, **run))


def _plan(config: ExperimentConfig) -> Plan:
    return resolve(config, load_cohort(config))


def _record(run_dir: Path, config: ExperimentConfig) -> Plan:
    """Record ``config`` and its plan in ``run_dir`` the way a run start does."""
    run_dir.mkdir(parents=True, exist_ok=True)
    plan = _plan(config)
    record_plan(run_dir, plan, config=config)
    return plan


def _write_metrics(fold_dir: Path, *, tune: dict, test: dict) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "metrics.json").write_text(json.dumps({"tune": tune, "test": test}))


# --------------------------------------------------------------------------- #
# run-id resolution
# --------------------------------------------------------------------------- #


def test_default_mints_fresh_timestamped_run_id(tmp_path: Path):
    layout = resolve_managed_output_paths(Plan(_make_config(tmp_path)))
    assert layout.run_id.endswith("__local")
    assert layout.run_dir == layout.experiment_dir / "runs" / layout.run_id


def test_explicit_run_id_config_pins_that_run(tmp_path: Path):
    layout = resolve_managed_output_paths(Plan(_make_config(tmp_path, run_id="pinned-run")))
    assert layout.run_id == "pinned-run"
    assert layout.run_dir.name == "pinned-run"


def test_resume_reuses_latest_existing_run(tmp_path: Path):
    cfg = _make_config(tmp_path)
    first = resolve_managed_output_paths(Plan(cfg))
    first.run_dir.mkdir(parents=True)
    # A newer run dir under the same experiment — resume must land on the latest.
    (first.experiment_dir / "runs" / "2999-01-01_00-00-00__local").mkdir()

    resumed = resolve_managed_output_paths(Plan(_with_run(cfg, resume=True)))
    assert resumed.run_id == "2999-01-01_00-00-00__local"
    assert resumed.experiment_dir == first.experiment_dir


def test_resume_without_prior_run_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no prior run"):
        resolve_managed_output_paths(Plan(_make_config(tmp_path, resume=True)))


def test_explicit_run_id_arg_still_wins_over_config(tmp_path: Path):
    # Back-compat: callers passing run_id= (tests, leaderboard replay) are unaffected.
    layout = resolve_managed_output_paths(
        Plan(_make_config(tmp_path, resume=True)), run_id="2020-01-01_00-00-00__local"
    )
    assert layout.run_id == "2020-01-01_00-00-00__local"


#: A launch second: two runs started within it would otherwise share one run id (#609).
_SAME_SECOND = datetime(2026, 10, 13, 13, 27, 38, tzinfo=timezone.utc)


def test_fresh_launches_in_the_same_second_get_their_own_run_dirs(tmp_path: Path):
    plan = Plan(_make_config(tmp_path))
    run_dirs = []
    for _ in range(3):
        layout = resolve_managed_output_paths(plan, when=_SAME_SECOND)
        layout.run_dir.mkdir(parents=True)  # what the launch does next
        run_dirs.append(layout.run_dir)

    first = run_dirs[0].name
    assert first.endswith("__local")
    assert [path.name for path in run_dirs] == [first, f"{first}-2", f"{first}-3"]
    assert len({path.parent for path in run_dirs}) == 1


def test_a_bare_resume_picks_the_last_of_the_same_second_launches(tmp_path: Path):
    cfg = _make_config(tmp_path)
    launched = []
    for _ in range(11):  # past -9, where the counters stop sorting as text
        layout = resolve_managed_output_paths(Plan(cfg), when=_SAME_SECOND)
        layout.run_dir.mkdir(parents=True)
        launched.append(layout.run_id)

    resumed = resolve_managed_output_paths(Plan(_with_run(cfg, resume=True)))
    assert resumed.run_id == launched[-1] == f"{launched[0]}-11"


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
    _record(run_dir, _make_config(tmp_path))
    # Different plan, but resume/run_id unset → guard must not fire.
    guard_resume_plan_drift(run_dir, _plan(_make_config(tmp_path, seed=999)))  # no raise


def test_drift_guard_passes_on_identical_plan(tmp_path: Path):
    run_dir = tmp_path / "run"
    cfg = _make_config(tmp_path)
    _record(run_dir, cfg)
    # Adding the resume directive alone is not drift (a launch-only field).
    guard_resume_plan_drift(run_dir, _plan(_with_run(cfg, resume=True)))  # no raise


_SEGMENTATION = dict(
    aggregator=None,
    decoder={"name": "lightweight_conv", "params": {"hidden_dim": 64}},
    head="segmentation",
    head_params={"num_classes": 2},
)


@pytest.mark.parametrize("overrides", [{}, _SEGMENTATION], ids=["slide", "segmentation"])
def test_drift_guard_passes_resuming_from_the_saved_config_of_a_preset_tiling(tmp_path: Path, overrides):
    """A Python config that leaves the tiling (and a grid run's dense options) to the
    encoder preset resolves to the same plan as the config.yaml it saved, so resuming
    from that file is not drift."""
    run_dir = tmp_path / "run"
    cfg = _make_config(tmp_path, features=FeaturesConfig(encoder="uni2"), **overrides)
    _record(run_dir, cfg)

    reloaded = ExperimentConfig.from_yaml(run_dir / "config.yaml")

    guard_resume_plan_drift(run_dir, _plan(_with_run(reloaded, resume=True)))  # no raise


def test_drift_guard_refuses_a_changed_seed(tmp_path: Path):
    run_dir = tmp_path / "run"
    _record(run_dir, _make_config(tmp_path, seed=7))
    changed = _make_config(tmp_path, seed=42, resume=True)
    with pytest.raises(ValueError, match="Refusing to resume.*run"):
        guard_resume_plan_drift(run_dir, _plan(changed))


def test_drift_guard_refuses_a_changed_head(tmp_path: Path):
    run_dir = tmp_path / "run"
    _record(run_dir, _make_config(tmp_path))
    changed = _make_config(tmp_path, head_params={"dropout": 0.5}, resume=True)
    with pytest.raises(ValueError, match="Refusing to resume.*learner"):
        guard_resume_plan_drift(run_dir, _plan(changed))


def test_a_run_resumes_on_a_machine_with_other_gpu_and_cpu_counts(tmp_path: Path, monkeypatch):
    """The plan keeps the execution resources as given (unset: slide2vec resolves them
    where the run uses them), so the saved plan does not record the first machine."""
    import os

    import torch

    def machine(cpus: int, gpus: int) -> None:
        monkeypatch.setattr(os, "cpu_count", lambda: cpus)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: gpus)

    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        monkeypatch.delenv(env_name, raising=False)
    run_dir = tmp_path / "run"
    machine(cpus=32, gpus=4)
    _record(run_dir, _make_config(tmp_path))
    machine(cpus=8, gpus=1)
    guard_resume_plan_drift(run_dir, _plan(_make_config(tmp_path, resume=True)))  # no raise


def test_drift_guard_refuses_a_changed_gpu_count(tmp_path: Path):
    # A count the config states is part of the plan, as the precision is.
    run_dir = tmp_path / "run"
    features = FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224))
    _record(run_dir, _make_config(tmp_path, features=replace(features, execution=ExecutionOptions(num_gpus=4))))
    changed = _make_config(
        tmp_path,
        features=replace(features, execution=ExecutionOptions(num_gpus=1)),
        resume=True,
    )
    with pytest.raises(ValueError, match="Refusing to resume.*features"):
        guard_resume_plan_drift(run_dir, _plan(changed))


def test_drift_guard_refuses_a_changed_execution_precision(tmp_path: Path):
    run_dir = tmp_path / "run"
    features = FeaturesConfig(encoder="uni2", extraction=extraction(0.5, 224))
    _record(run_dir, _make_config(tmp_path, features=replace(features, execution=ExecutionOptions(precision="fp16"))))
    changed = _make_config(
        tmp_path,
        features=replace(features, execution=ExecutionOptions(precision="fp32")),
        resume=True,
    )
    with pytest.raises(ValueError, match="Refusing to resume"):
        guard_resume_plan_drift(run_dir, _plan(changed))


def test_drift_guard_noop_when_no_saved_plan(tmp_path: Path):
    # A pinned run id naming a not-yet-created dir has nothing to compare.
    guard_resume_plan_drift(tmp_path / "fresh", _plan(_make_config(tmp_path, run_id="fresh")))


def test_drift_guard_allows_changing_operational_mirror_root(tmp_path: Path):
    run_dir = tmp_path / "run"
    original = _make_config(tmp_path, mirror_root=tmp_path / "shared-a")
    _record(run_dir, original)

    resumed = _with_run(original, mirror_root=tmp_path / "shared-b", resume=True)

    guard_resume_plan_drift(run_dir, _plan(resumed))


def test_drift_guard_refuses_unknown_evaluation_settings(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    original = _make_config(tmp_path)
    _record(run_dir, original)
    saved_path = run_dir / "plan.yaml"
    saved = yaml.safe_load(saved_path.read_text())
    saved["evaluation"]["patient_oof"] = {"arm": "legacy"}
    saved_path.write_text(yaml.safe_dump(saved))

    with pytest.raises(ValueError, match="evaluation"):
        guard_resume_plan_drift(run_dir, _plan(_with_run(original, resume=True)))


# --------------------------------------------------------------------------- #
# end-to-end resume (CPU synthetic pipeline)
# --------------------------------------------------------------------------- #


def _multifold_pipeline_config(tmp_path: Path, **run) -> ExperimentConfig:
    from tests.pooled_features import setup_multifold_data

    dataset_csv, splits_csv, feature_dir = setup_multifold_data(tmp_path)
    (tmp_path / "feature_dir.txt").write_text(str(feature_dir))
    return experiment(
        dataset_csv,
        splits_csv,
        output_dir=tmp_path / "output",
        aggregator="mean_pool",
        loaders={"batch_size": 2},
        trainer={"max_epochs": 2},
        scheduler="cosine",
        run={"run_id": "2026-07-08_10-00-00__local", **run},
    )


def _run(tmp_path: Path, config: ExperimentConfig):
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

    resumed = _run(tmp_path, _with_run(config, resume=True))

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
    plan = _record(run_dir, cfg)
    # The run recorded the identity minted from the original train/tune content.
    from soma.pipeline.layout import build_experiment_spec

    (run_dir / "run.yaml").write_text(
        yaml.safe_dump({"experiment_id": build_experiment_spec(plan).experiment_id})
    )
    guard_resume_plan_drift(run_dir, _plan(_with_run(cfg, resume=True)))  # unchanged: passes

    # Same paths, different label -> different train/tune content (the plan is unchanged).
    Path(cfg.cohort.records).write_text(_DATASET.replace("s0,/slides/s0.svs,tumor", "s0,/slides/s0.svs,normal"))
    with pytest.raises(ValueError, match="train/tune content"):
        guard_resume_plan_drift(run_dir, _plan(_with_run(cfg, resume=True)))


# --- run.folds: launches that split the folds of one run between them ----------------


def test_folds_directive_loads_from_yaml_and_is_launch_scoped(tmp_path: Path):
    config = _make_config(tmp_path, run_id="shared-run")
    config.to_yaml(tmp_path / "config.yaml")
    loaded = ExperimentConfig.from_yaml(
        tmp_path / "config.yaml", overrides={"run": {"folds": [1, 3]}}
    )
    assert loaded.run.folds == (1, 3)
    # Which folds a launch takes on is not what the run is: its plan compares equal.
    assert _plan(loaded).comparable() == _plan(config).comparable()


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
    first = _run(tmp_path, _with_run(config, folds=(1,)))
    run_dir = first.run_dir
    assert [fr.fold for fr in first.fold_results] == [1]
    assert first.pending_folds == (0,) and first.summary == {}
    assert not (run_dir / "fold_0").exists()
    assert not (run_dir / "summary.json").exists()
    fold1_mtime = (run_dir / "fold_1" / "metrics.json").stat().st_mtime_ns

    # The launch given fold 0 finds every fold complete afterwards, so it summarizes both.
    second = _run(tmp_path, _with_run(config, folds=(0,)))
    assert [fr.fold for fr in second.fold_results] == [0]
    assert second.pending_folds == ()
    assert (run_dir / "fold_1" / "metrics.json").stat().st_mtime_ns == fold1_mtime
    split_summary = json.loads((run_dir / "summary.json").read_text())

    # Folds seed independently, so splitting them changes nothing about the result.
    single = _run(tmp_path, _with_run(config, run_id="single-run"))
    assert split_summary == pytest.approx(single.summary, nan_ok=True)


def test_folds_outside_the_splits_are_refused(tmp_path: Path):
    pytest.importorskip("torch")
    config = _multifold_pipeline_config(tmp_path, run_id="shared-run", folds=(2,))
    with pytest.raises(ValueError, match="run.folds"):
        _run(tmp_path, config)
