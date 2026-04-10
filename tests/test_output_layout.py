from __future__ import annotations

import csv
from pathlib import Path

import yaml

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.output_layout import (
    ExperimentSpec,
    build_experiment_spec,
    canonical_experiment_payload,
    create_run_metadata,
    resolve_managed_output_paths,
    update_experiment_index,
    update_run_index,
)


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _make_pipeline_config(tmp_path: Path, **overrides) -> PipelineConfig:
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
        preprocessing=PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(),
        encoder=EncoderConfig(name="uni2"),
        aggregator=AggregatorConfig(name="abmil", params={"hidden_dim": 128}),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(seed=7, epochs=10, learning_rate=1e-4),
        tags=["baseline"],
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_canonical_experiment_payload_omits_seed(tmp_path: Path):
    cfg_a = _make_pipeline_config(tmp_path, training=TrainingConfig(seed=1, epochs=10, learning_rate=1e-4))
    cfg_b = _make_pipeline_config(tmp_path, training=TrainingConfig(seed=999, epochs=10, learning_rate=1e-4))

    payload_a = canonical_experiment_payload(cfg_a)
    payload_b = canonical_experiment_payload(cfg_b)

    assert payload_a == payload_b
    assert payload_a["training"]["epochs"] == 10
    assert "seed" not in payload_a["training"]


def test_build_experiment_spec_uses_slug_and_short_hash(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)

    spec = build_experiment_spec(config)

    assert isinstance(spec, ExperimentSpec)
    assert spec.slug.startswith("dataset-uni2-abmil-binary-classification_")
    assert spec.experiment_dirname == spec.slug
    assert len(spec.short_hash) == 12
    assert spec.dataset_checksum
    assert spec.splits_checksum


def test_resolve_managed_output_paths_groups_same_experiment_and_changes_run_dir(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)

    first = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")
    second = resolve_managed_output_paths(config, run_id="2026-04-09_16-23-10__local")

    assert first.experiment_dir == second.experiment_dir
    assert first.run_dir != second.run_dir
    assert first.run_dir.parent == first.experiment_dir / "runs"
    assert first.index_dir == Path(config.output_root) / "indexes"


def test_create_run_metadata_records_status_and_seed(tmp_path: Path):
    config = _make_pipeline_config(tmp_path, training=TrainingConfig(seed=123, epochs=10))
    layout = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")

    metadata = create_run_metadata(
        config=config,
        experiment=layout.experiment,
        run_dir=layout.run_dir,
        run_id=layout.run_id,
        status="running",
    )

    assert metadata.seed == 123
    assert metadata.status == "running"
    assert metadata.experiment_id == layout.experiment.experiment_id
    assert metadata.resolved_output_dir == layout.run_dir.resolve()


def test_run_and_experiment_indexes_upsert_rows(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)
    layout = resolve_managed_output_paths(config, run_id="2026-04-09_16-22-10__local")
    run = create_run_metadata(
        config=config,
        experiment=layout.experiment,
        run_dir=layout.run_dir,
        run_id=layout.run_id,
        status="running",
    )

    update_experiment_index(layout.index_dir / "experiments.csv", layout.experiment, num_runs=1, latest_run_id=run.run_id, latest_status=run.status)
    update_run_index(layout.index_dir / "runs.csv", run)

    completed = run.with_updates(status="completed")
    update_run_index(layout.index_dir / "runs.csv", completed)
    update_experiment_index(
        layout.index_dir / "experiments.csv",
        layout.experiment,
        num_runs=1,
        latest_run_id=completed.run_id,
        latest_status=completed.status,
    )

    with (layout.index_dir / "experiments.csv").open(newline="", encoding="utf-8") as handle:
        experiment_rows = list(csv.DictReader(handle))
    with (layout.index_dir / "runs.csv").open(newline="", encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))

    assert len(experiment_rows) == 1
    assert experiment_rows[0]["latest_status"] == "completed"
    assert len(run_rows) == 1
    assert run_rows[0]["status"] == "completed"


def test_experiment_spec_roundtrips_through_yaml(tmp_path: Path):
    config = _make_pipeline_config(tmp_path)
    spec = build_experiment_spec(config)
    path = tmp_path / "experiment.yaml"

    path.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert loaded["experiment_id"] == spec.experiment_id
    assert loaded["slug"] == spec.slug
    assert loaded["canonical_spec"]["task"]["name"] == "binary_classification"
