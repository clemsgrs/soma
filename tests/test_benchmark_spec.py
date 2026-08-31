"""External benchmark-spec execution through Soma's public object API."""

from __future__ import annotations

from pathlib import Path

import pytest

import soma.benchmarks.run as run_mod
from soma.benchmarks import (
    BenchmarkRunResult,
    BenchmarkSpec,
    MetricResult,
    get_benchmark,
    get_ranking_metrics,
    get_reported_metrics,
    run_benchmark_spec,
)


def _build_config(**kwargs):
    return kwargs


def _score(run_dir: str | Path) -> dict[str, float]:
    return {"test/accuracy": 0.75}


def test_benchmark_spec_is_public_and_validates_metrics_without_registration():
    import soma.benchmarks

    spec = BenchmarkSpec(
        name="fmtf/example",
        canonical_seeds=(7, 3, 11),
        primary_metric="test/accuracy",
        reported_metrics=("test/accuracy", "test/f1"),
        ranking_metrics=("test/accuracy",),
        build_config=_build_config,
        score=_score,
    )

    assert get_reported_metrics(spec) == ("test/accuracy", "test/f1")
    assert get_ranking_metrics(spec) == ("test/accuracy",)
    assert "BenchmarkSpec" in soma.benchmarks.__all__
    with pytest.raises(KeyError, match="Unknown benchmark 'fmtf/example'"):
        get_benchmark(spec.name)


def test_benchmark_spec_rejects_invalid_metric_declaration_on_construction():
    with pytest.raises(ValueError, match="Reported metrics must be unique"):
        BenchmarkSpec(
            name="fmtf/invalid",
            canonical_seeds=(7,),
            primary_metric="test/accuracy",
            reported_metrics=("test/accuracy", "test/accuracy"),
            build_config=_build_config,
            score=_score,
        )


def test_benchmark_spec_defaults_optional_metric_declarations_to_primary():
    spec = BenchmarkSpec(
        name="fmtf/default-metrics",
        canonical_seeds=(7,),
        primary_metric="test/accuracy",
        build_config=_build_config,
        score=_score,
    )

    assert spec.reported_metrics == ("test/accuracy",)
    assert spec.ranking_metrics == ("test/accuracy",)


def test_benchmark_spec_rejects_invalid_ranking_declaration_on_construction():
    with pytest.raises(ValueError, match="not Reported: 'test/auc'"):
        BenchmarkSpec(
            name="fmtf/invalid-ranking",
            canonical_seeds=(7,),
            primary_metric="test/accuracy",
            reported_metrics=("test/accuracy", "test/f1"),
            ranking_metrics=("test/accuracy", "test/auc"),
            build_config=_build_config,
            score=_score,
        )


@pytest.mark.parametrize("canonical_seeds", [(), (7, 7)])
def test_benchmark_spec_rejects_empty_or_duplicate_canonical_seeds(canonical_seeds):
    with pytest.raises(ValueError, match="canonical_seeds must be non-empty and unique"):
        BenchmarkSpec(
            name="fmtf/invalid-seeds",
            canonical_seeds=canonical_seeds,
            primary_metric="test/accuracy",
            build_config=_build_config,
            score=_score,
        )


def test_run_benchmark_spec_runs_canonical_seeds_with_shared_cache_and_aggregates(
    tmp_path, monkeypatch
):
    built: list[dict] = []
    ran: list[dict] = []

    def build_config(**kwargs):
        built.append(kwargs)
        return kwargs

    scores = {
        "seed_7": {"metric/a": 1.0, "metric/b": 4.0},
        "seed_3": {"metric/a": 2.0, "metric/b": 6.0},
        "seed_11": {"metric/a": 3.0, "metric/b": 8.0},
    }
    spec = BenchmarkSpec(
        name="fmtf/example",
        canonical_seeds=(7, 3, 11),
        primary_metric="metric/a",
        reported_metrics=("metric/a", "metric/b"),
        build_config=build_config,
        score=lambda run_dir: scores[Path(run_dir).name],
    )

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            ran.append(self.config)

    monkeypatch.setattr(run_mod, "_pipeline_cls", lambda: FakePipeline)
    output_root = tmp_path / "evidence"
    dataset_csv = tmp_path / "dataset.csv"
    splits_csv = tmp_path / "splits.csv"

    result = run_benchmark_spec(
        spec,
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        encoder="fmtf-encoder",
        output_root=output_root,
    )

    seed_roots = tuple(output_root / f"seed_{seed}" for seed in (7, 3, 11))
    assert result == BenchmarkRunResult(
        status=0,
        metrics=(
            MetricResult("metric/a", 2.0, 1.0, 3),
            MetricResult("metric/b", 6.0, 2.0, 3),
        ),
        seed_roots=seed_roots,
    )
    assert ran == built
    assert [config["seed"] for config in built] == [7, 3, 11]
    assert [config["output_root"] for config in built] == list(seed_roots)
    assert {config["dataset_csv"] for config in built} == {dataset_csv}
    assert {config["splits_csv"] for config in built} == {splits_csv}
    assert {config["encoder"] for config in built} == {"fmtf-encoder"}
    assert {config["overrides"]["cache"]["root_dir"] for config in built} == {
        str(output_root / "feature_cache")
    }


def test_run_benchmark_spec_uses_exact_explicit_seed_and_cache_root(tmp_path, monkeypatch):
    built: list[dict] = []

    def build_config(**kwargs):
        built.append(kwargs)
        return kwargs

    spec = BenchmarkSpec(
        name="fmtf/example",
        canonical_seeds=(7, 3, 11),
        primary_metric="metric/a",
        build_config=build_config,
        score=lambda run_dir: {"metric/a": 5.0},
    )

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(run_mod, "_pipeline_cls", lambda: FakePipeline)
    output_root = tmp_path / "evidence"
    cache_root = tmp_path / "shared-cache"

    result = run_benchmark_spec(
        spec,
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        encoder="fmtf-encoder",
        output_root=output_root,
        cache_root=cache_root,
        seeds=(42,),
    )

    assert result == BenchmarkRunResult(
        status=0,
        metrics=(MetricResult("metric/a", 5.0, 0.0, 1),),
        seed_roots=(output_root / "seed_42",),
    )
    assert [config["seed"] for config in built] == [42]
    assert built[0]["overrides"] == {
        "cache": {"enabled": True, "root_dir": str(cache_root)}
    }


@pytest.mark.parametrize("seeds", [3, (), (7, 7)])
def test_run_benchmark_spec_rejects_invalid_explicit_seeds_before_training(tmp_path, seeds):
    spec = BenchmarkSpec(
        name="fmtf/example",
        canonical_seeds=(7, 3, 11),
        primary_metric="metric/a",
        build_config=lambda **kwargs: pytest.fail("invalid seeds reached build_config"),
        score=lambda run_dir: {"metric/a": 5.0},
    )

    with pytest.raises(
        ValueError, match="seeds must be a non-empty sequence of unique integers"
    ):
        run_benchmark_spec(
            spec,
            dataset_csv=tmp_path / "dataset.csv",
            splits_csv=tmp_path / "splits.csv",
            encoder="fmtf-encoder",
            output_root=tmp_path / "evidence",
            seeds=seeds,
        )


def test_run_benchmark_spec_missing_reported_metric_raises_value_error(
    tmp_path, monkeypatch
):
    spec = BenchmarkSpec(
        name="fmtf/example",
        canonical_seeds=(7,),
        primary_metric="metric/a",
        reported_metrics=("metric/a", "metric/b"),
        build_config=lambda **kwargs: kwargs,
        score=lambda run_dir: {"metric/a": 1.0},
    )

    class FakePipeline:
        def __init__(self, config):
            pass

        def run(self):
            pass

    monkeypatch.setattr(run_mod, "_pipeline_cls", lambda: FakePipeline)

    with pytest.raises(ValueError, match="metric/b"):
        run_benchmark_spec(
            spec,
            dataset_csv=tmp_path / "dataset.csv",
            splits_csv=tmp_path / "splits.csv",
            encoder="fmtf-encoder",
            output_root=tmp_path / "evidence",
        )
