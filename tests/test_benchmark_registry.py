"""Benchmark registry mechanics + a fixture Benchmark (protocol-as-code, ADR 0002).

Exercises the parts that do not need OCELOT data or a GPU: register/lookup/list, the
structural :class:`Benchmark` protocol, ``reference/<name>.csv`` parsing with a per-row
tolerance, axis matching (broad banner vs keyed rows), the default ``summary.json`` scorer,
and the :class:`Facet`. A fixture Benchmark that uses the DEFAULT scorer proves the
registry works independently of OCELOT's greedy override.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from soma.benchmarks import (
    Benchmark,
    Facet,
    ReferenceRow,
    expected_rows,
    get_benchmark,
    list_benchmarks,
    load_reference,
    register_benchmark,
    score_from_summary,
)
from soma.benchmarks import registry as registry_mod
from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.curation.manifest import CuratedManifest


class _FixtureBenchmark:
    """A minimal benchmark that leans on the DEFAULT (summary.json) scorer."""

    name = "fixture_bench"
    facet = Facet(fixed={"decoder": "linear"}, varied=("encoder",))
    canonical_seeds = (0, 1)
    primary_metric = "accuracy"
    reference_environment: dict[str, str] = {}

    def curate(self, raw_root, out_dir) -> CuratedManifest:
        return CuratedManifest(dataset_csv=Path(out_dir) / "dataset.csv", splits_csv=Path(out_dir) / "splits.csv")

    def build_config(self, **axes) -> PipelineConfig:
        return PipelineConfig(
            dataset_csv="d.csv",
            splits_csv="s.csv",
            output_root="out",
            dataset_type="tile",
            cache=CacheConfig(),
            encoder=EncoderConfig(name=axes.get("encoder", "uni2")),
            aggregator=AggregatorConfig(name="abmil"),
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=1),
        )

    def expected(self, **axes):
        return [ReferenceRow(key={}, metric="accuracy", expected=0.9, tolerance=0.05, source="fixture")]

    def score(self, run_dir) -> dict[str, float]:
        return score_from_summary(run_dir)


@pytest.fixture()
def fixture_benchmark():
    bench = _FixtureBenchmark()
    register_benchmark(bench)
    try:
        yield bench
    finally:
        registry_mod._REGISTRY.pop(bench.name, None)


def test_register_lookup_and_list(fixture_benchmark):
    assert get_benchmark("fixture_bench") is fixture_benchmark
    assert "fixture_bench" in list_benchmarks()
    # list is sorted
    assert list_benchmarks() == sorted(list_benchmarks())


def test_get_benchmark_unknown_is_keyerror():
    with pytest.raises(KeyError):
        get_benchmark("does_not_exist")


def test_fixture_conforms_to_benchmark_protocol(fixture_benchmark):
    assert isinstance(fixture_benchmark, Benchmark)


def test_reference_row_axis_matching():
    banner = ReferenceRow(key={}, metric="m", expected=0.7, tolerance=0.02, source="")
    keyed = ReferenceRow(key={"encoder": "virchow2"}, metric="m", expected=0.7, tolerance=0.02, source="")
    # Empty key = broad banner: matches any axes (or none).
    assert banner.matches({})
    assert banner.matches({"encoder": "uni2"})
    # Keyed row matches only when the populated cell equals the axis.
    assert keyed.matches({"encoder": "virchow2"})
    assert not keyed.matches({"encoder": "uni2"})
    assert not keyed.matches({})


def test_reference_row_within_tolerance():
    row = ReferenceRow(key={}, metric="m", expected=0.70, tolerance=0.02, source="")
    assert row.within_tolerance(0.685)
    assert row.within_tolerance(0.715)
    assert not row.within_tolerance(0.75)


def test_load_reference_parses_ocelot_band():
    rows = load_reference("ocelot")
    assert rows, "ocelot reference must have at least one row"
    banner = rows[0]
    # Broad, config-agnostic banner: no populated key cells.
    assert banner.key == {}
    assert banner.metric == "mean_f1"
    assert banner.expected == pytest.approx(0.6995, abs=1e-6)
    assert banner.tolerance == pytest.approx(0.02, abs=1e-6)  # per-row tolerance column
    assert banner.source  # a non-empty provenance string


def test_load_reference_requires_value_columns(tmp_path, monkeypatch):
    # A table missing the fixed value columns fails fast.
    import soma.benchmarks.registry as reg

    class _Fake:
        def joinpath(self, _name):
            return self

        def open(self, newline=""):
            import io

            return io.StringIO("dataset,expected\n,0.5\n")

    monkeypatch.setattr(reg.resources, "files", lambda _pkg: _Fake())
    with pytest.raises(ValueError):
        reg.load_reference("whatever")


def test_expected_rows_filters_by_axes_and_metric():
    rows = expected_rows("ocelot", metric="mean_f1", encoder="virchow2", spacing=0.2)
    assert len(rows) == 1
    assert rows[0].metric == "mean_f1"
    # A metric that isn't tabulated yields nothing.
    assert expected_rows("ocelot", metric="dice") == []


def test_score_from_summary_reads_metrics(tmp_path: Path):
    summary = {"test/accuracy_mean": 0.83, "test/auroc_mean": 0.91}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    scored = score_from_summary(tmp_path)
    assert scored == pytest.approx(summary)


def test_score_from_summary_finds_nested_summary(tmp_path: Path):
    nested = tmp_path / "experiments" / "exp" / "runs" / "r0"
    nested.mkdir(parents=True)
    (nested / "summary.json").write_text(json.dumps({"test/dice_mean": 0.55}))
    scored = score_from_summary(tmp_path)
    assert scored["test/dice_mean"] == pytest.approx(0.55)


def test_facet_records_fixed_and_varied():
    facet = Facet(fixed={"decoder": "conv"}, varied=("encoder", "spacing"))
    assert facet.fixed == {"decoder": "conv"}
    assert facet.varied == ("encoder", "spacing")
