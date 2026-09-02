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
from types import SimpleNamespace

import pytest

from soma.benchmarks import (
    Benchmark,
    Facet,
    ReferenceRow,
    expected_rows,
    get_benchmark,
    get_ranking_metrics,
    get_reported_metrics,
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


def test_metric_accessors_preserve_declared_order_and_legacy_fallback():
    legacy = SimpleNamespace(name="legacy", primary_metric="primary")
    multi = SimpleNamespace(
        name="multi",
        primary_metric="median",
        reported_metrics=("median", "f0", "ltm10"),
        ranking_metrics=("median", "ltm10"),
    )

    assert get_reported_metrics(legacy) == ("primary",)
    assert get_ranking_metrics(legacy) == ("primary",)
    assert get_reported_metrics(multi) == ("median", "f0", "ltm10")
    assert get_ranking_metrics(multi) == ("median", "ltm10")


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ((), "Reported metrics must be non-empty"),
        (("primary", "primary"), "Reported metrics must be unique"),
    ],
)
def test_reported_metrics_reject_empty_or_duplicate_declarations(declared, message):
    benchmark = SimpleNamespace(
        name="invalid", primary_metric="primary", reported_metrics=declared
    )

    with pytest.raises(ValueError, match=message):
        get_reported_metrics(benchmark)


@pytest.mark.parametrize(
    "declared",
    [("secondary",), ("primary", "secondary", "primary")],
)
def test_reported_metrics_require_primary_exactly_once(declared):
    benchmark = SimpleNamespace(
        name="invalid", primary_metric="primary", reported_metrics=declared
    )

    with pytest.raises(ValueError, match="primary_metric 'primary' exactly once"):
        get_reported_metrics(benchmark)


def test_reported_metrics_reject_unordered_declarations():
    benchmark = SimpleNamespace(
        name="invalid", primary_metric="primary", reported_metrics={"primary", "secondary"}
    )

    with pytest.raises(ValueError, match="ordered sequence"):
        get_reported_metrics(benchmark)


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        ((), "Ranking metrics must be non-empty"),
        (("primary", "other", "other"), "Ranking metrics must be unique"),
    ],
)
def test_ranking_metrics_reject_empty_or_duplicate_declarations(declared, message):
    benchmark = SimpleNamespace(
        name="invalid",
        primary_metric="primary",
        reported_metrics=("primary", "other"),
        ranking_metrics=declared,
    )

    with pytest.raises(ValueError, match=message):
        get_ranking_metrics(benchmark)


def test_ranking_metrics_must_be_a_subset_of_reported_metrics():
    benchmark = SimpleNamespace(
        name="invalid",
        primary_metric="primary",
        reported_metrics=("primary", "diagnostic"),
        ranking_metrics=("primary", "unreported"),
    )

    with pytest.raises(ValueError, match="not Reported: 'unreported'"):
        get_ranking_metrics(benchmark)


@pytest.mark.parametrize(
    "declared",
    [("other",), ("primary", "other", "primary")],
)
def test_ranking_metrics_require_primary_exactly_once(declared):
    benchmark = SimpleNamespace(
        name="invalid",
        primary_metric="primary",
        reported_metrics=("primary", "other"),
        ranking_metrics=declared,
    )

    with pytest.raises(ValueError, match="primary_metric 'primary' exactly once"):
        get_ranking_metrics(benchmark)


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
    assert not row.relative  # absolute by default
    assert row.tolerance_band() == pytest.approx(0.02)
    assert row.within_tolerance(0.685)
    assert row.within_tolerance(0.715)
    assert not row.within_tolerance(0.75)


def test_reference_row_relative_tolerance_scales_with_expected():
    # relative=True reinterprets tolerance as a fraction of expected: a ±2% band on 0.90 is
    # ±0.018, so 0.883 (Δ -0.017) passes but 0.870 (Δ -0.030) fails.
    row = ReferenceRow(
        key={}, metric="m", expected=0.90, tolerance=0.02, source="", relative=True
    )
    assert row.tolerance_band() == pytest.approx(0.018)
    assert row.within_tolerance(0.883)
    assert not row.within_tolerance(0.870)
    # The SAME 0.02 tolerance is a different absolute band at a different expected value.
    low = ReferenceRow(
        key={}, metric="m", expected=0.50, tolerance=0.02, source="", relative=True
    )
    assert low.tolerance_band() == pytest.approx(0.010)


def test_load_reference_parses_ocelot_band():
    rows = load_reference("ocelot")
    assert rows, "ocelot reference must have at least one row"
    gate = rows[0]
    # The gate row is pinned to the anchor it was measured on: every key cell populated.
    assert gate.key == {"encoder": "virchow2"}
    assert gate.metric == "mean_f1"
    assert gate.expected == pytest.approx(0.6995, abs=1e-6)
    assert gate.tolerance == pytest.approx(0.02, abs=1e-6)  # per-row tolerance column
    assert gate.source  # a non-empty provenance string


def test_ocelot_carries_external_guidance_anchors():
    # OCELOT ships ≥1 non-gating external anchor (official/best-reported, snapshotted from
    # histoboard) with a human label + a clickable URL; the "~0.70-0.73" prose is promoted
    # OUT of the gate row's source into these structured rows.
    rows = load_reference("ocelot")
    gate = [r for r in rows if r.metric == "mean_f1" and not r.is_external]
    external = [r for r in rows if r.metric == "mean_f1" and r.is_external]

    assert len(gate) == 1, "OCELOT keeps exactly one gate row for mean_f1"
    assert len(external) >= 1, "OCELOT must carry at least one external guidance anchor"
    # Multiple independently-labelled/linked anchors are supported (official + best-reported).
    assert len(external) >= 2
    for row in external:
        assert row.label, "an external anchor carries a human label"
        assert row.url.startswith("http"), "an external anchor carries a linkable URL"
        assert row.tolerance == pytest.approx(0.0)  # never gates

    # The promoted figure lives on the external rows, not the gate row's source.
    assert "0.70-0.73" not in gate[0].source
    assert any(0.70 <= r.expected <= 0.73 for r in external)


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


def _fake_reference_csv(monkeypatch, text: str) -> None:
    """Point ``load_reference`` at an in-memory CSV (mirrors the missing-column test)."""
    import soma.benchmarks.registry as reg

    class _Fake:
        def joinpath(self, _name):
            return self

        def open(self, newline=""):
            import io

            return io.StringIO(text)

    monkeypatch.setattr(reg.resources, "files", lambda _pkg: _Fake())


def test_load_reference_parses_kind_label_url_columns(monkeypatch):
    # An external (non-gating) guidance row carries a kind marker, a human label, and a
    # linkable URL; a gate row (or an absent/blank kind cell) defaults to kind="gate".
    _fake_reference_csv(
        monkeypatch,
        "metric,expected,tolerance,kind,label,url,source\n"
        "m,0.70,0.02,gate,,,soma reproduced anchor\n"
        "m,0.73,,external,best reported,https://example.org/board,captured 2026-07-03\n",
    )
    rows = load_reference("whatever")
    gate, external = rows[0], rows[1]
    assert gate.kind == "gate"
    assert external.kind == "external"
    assert external.label == "best reported"
    assert external.url == "https://example.org/board"
    assert external.expected == pytest.approx(0.73)
    # A blank tolerance on an external row is tolerated (it never gates).
    assert external.tolerance == pytest.approx(0.0)


def test_load_reference_parses_relative_tolerance_mode(monkeypatch):
    # tolerance_mode=relative reinterprets the tolerance as a fraction of expected; an
    # absent/blank cell defaults to absolute.
    _fake_reference_csv(
        monkeypatch,
        "metric,expected,tolerance,tolerance_mode,source\n"
        "m,0.80,0.02,relative,two percent band\n"
        "m,0.80,0.02,,absolute default\n",
    )
    rel, absolute = load_reference("whatever")
    assert rel.relative and rel.tolerance_band() == pytest.approx(0.016)
    assert not absolute.relative and absolute.tolerance_band() == pytest.approx(0.02)


def test_load_reference_rejects_unknown_tolerance_mode(monkeypatch):
    _fake_reference_csv(
        monkeypatch,
        "metric,expected,tolerance,tolerance_mode,source\nm,0.80,0.02,percent,bad\n",
    )
    with pytest.raises(ValueError, match="tolerance_mode"):
        load_reference("whatever")


def test_load_reference_defaults_kind_to_gate_when_column_absent(monkeypatch):
    # A legacy CSV with no kind/label/url columns still parses; every row is a gate.
    _fake_reference_csv(
        monkeypatch,
        "metric,expected,tolerance,source\nm,0.70,0.02,anchor\n",
    )
    (row,) = load_reference("whatever")
    assert row.kind == "gate"
    assert row.label == "" and row.url == ""


def test_expected_rows_filters_by_axes_and_metric():
    # The gate row is keyed to the anchor; external guidance rows share the metric with an
    # empty key (match anything), so filter by kind to isolate the gate.
    rows = expected_rows("ocelot", metric="mean_f1", encoder="virchow2", spacing=0.2)
    gate = [r for r in rows if r.kind == "gate"]
    assert len(gate) == 1
    assert gate[0].metric == "mean_f1"
    # Another encoder has no gate (only the non-gating anchors match).
    other = expected_rows("ocelot", metric="mean_f1", encoder="uni2", spacing=0.2)
    assert all(r.is_external for r in other)
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


def test_relative_band_uses_absolute_expected_for_negative_values():
    row = ReferenceRow(
        key={}, metric="delta", expected=-0.5, tolerance=0.1, source="", relative=True
    )
    assert row.tolerance_band() == pytest.approx(0.05)
    assert row.within_tolerance(-0.52) and not row.within_tolerance(-0.6)


def test_load_reference_rejects_placeholder_and_banner_gate_rows(tmp_path, monkeypatch):
    def _with(text: str):
        target = tmp_path / "toy.csv"
        target.write_text(text)

        class _Files:
            def joinpath(self, _name):
                return target

        monkeypatch.setattr(registry_mod.resources, "files", lambda _pkg: _Files())
        return load_reference("toy")

    header = "dataset,encoder,metric,expected,tolerance,kind,source\n"
    with pytest.raises(ValueError, match="expected 0.0"):
        _with(header + "d,e,f1,0.0,0.02,gate,placeholder\n")
    with pytest.raises(ValueError, match=r"key column\(s\) \['encoder'\] blank"):
        _with(header + "d,,f1,0.7,0.02,gate,banner\n")
    # External anchors are exempt from both rules.
    rows = _with(header + ",,f1,0.7,,external,anchor\n")
    assert rows[0].is_external and rows[0].key == {}


def test_packaged_reference_gate_rows_are_fully_keyed_and_non_placeholder():
    for name in ("ocelot", "midog", "monkey", "eva", "hest", "croma"):
        for row in load_reference(name):
            if row.is_external:
                continue
            assert row.expected != 0.0, (name, row)
            assert row.key, (name, row)


def test_key_columns_are_everything_left_of_metric():
    assert registry_mod._key_columns(["dataset", "encoder", "metric", "expected", "kind"]) == [
        "dataset", "encoder"
    ]
    assert registry_mod._key_columns(["metric", "measured"]) == []
