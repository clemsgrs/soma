"""Importable ``run_benchmark()`` orchestration extracted from the reproduce CLI (#370).

``soma.benchmarks.run_benchmark`` must carry everything ``soma reproduce`` provides —
the canonical-seed loop, reference-row resolution and tolerance status, provenance
stamping, and the results-ledger append — with a parameterizable ``results_root`` so an
external harness (the FMTF benchmark leaderboard) can host its own committed ledger.
The CLI stays byte-identical: it is a thin caller of the same function.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import soma.benchmarks.run as run_mod
from soma.benchmarks import (
    MeasuredRow,
    append_result,
    load_results,
    reproduced_rows,
    run_benchmark,
)
from soma.benchmarks import registry as registry_mod
from soma.benchmarks.registry import Facet, ReferenceRow, register_benchmark


class _LedgerBenchmark:
    """A registered fixture benchmark with a gate reference keyed by encoder."""

    name = "run_api_fixture"
    facet = Facet(fixed={}, varied=("encoder",))
    canonical_seeds = (0, 1, 2)
    primary_metric = "test/accuracy"
    reference_environment: dict[str, str] = {}

    def __init__(self):
        self.built: list[dict] = []
        self.curate_calls = 0

    def curate(self, raw_root, out_dir):
        from soma.curation.manifest import CuratedManifest

        self.curate_calls += 1
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return CuratedManifest(
            dataset_csv=out_dir / "dataset.csv", splits_csv=out_dir / "splits.csv"
        )

    def build_config(self, **kwargs):
        self.built.append(kwargs)
        return object()

    def expected(self, **axes):
        return [
            ReferenceRow(
                key={"encoder": "fixture-encoder"},
                metric="test/accuracy",
                expected=0.70,
                tolerance=0.05,
                source="fixture",
            )
        ]

    def score(self, run_dir):
        return {"test/accuracy": 0.71}


@pytest.fixture()
def ledger_benchmark():
    bench = _LedgerBenchmark()
    register_benchmark(bench)
    try:
        yield bench
    finally:
        registry_mod._REGISTRY.pop(bench.name, None)


@pytest.fixture()
def stub_pipeline(monkeypatch):
    """Replace the training Pipeline with a no-op for full-mode orchestration tests."""

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(run_mod, "_pipeline_cls", lambda: FakePipeline)
    return FakePipeline


def _run_dir_with_summary(tmp_path, accuracy: float) -> Path:
    (tmp_path / "summary.json").write_text(json.dumps({"test/accuracy": accuracy}))
    return tmp_path


# --- public API surface ---------------------------------------------------------------


def test_run_benchmark_is_public_api():
    import soma.benchmarks

    assert "run_benchmark" in soma.benchmarks.__all__
    assert soma.benchmarks.run_benchmark is run_benchmark


def test_run_benchmark_unknown_name_fails_fast_with_known_names():
    with pytest.raises(KeyError, match="Unknown benchmark"):
        run_benchmark("no_such_benchmark")


# --- byte-identical CLI behavior -------------------------------------------------------


def test_run_benchmark_from_run_dir_output_is_byte_identical_to_cli(
    capsys, tmp_path, ledger_benchmark
):
    from soma.cli import main

    run_dir = _run_dir_with_summary(tmp_path, 0.71)

    code = run_benchmark(
        ledger_benchmark.name, encoder="fixture-encoder", from_run_dir=run_dir
    )
    api_out = capsys.readouterr().out

    try:
        main(
            [
                "reproduce",
                ledger_benchmark.name,
                "--encoder",
                "fixture-encoder",
                "--from-run-dir",
                str(run_dir),
            ]
        )
    except SystemExit as exc:
        cli_code = int(exc.code or 0)
    cli_out = capsys.readouterr().out

    assert code == cli_code == 0
    assert api_out == cli_out
    assert "[REFERENCE OK]" in api_out


# --- canonical-seed loop ---------------------------------------------------------------


def test_run_benchmark_runs_canonical_seed_loop_by_default(
    tmp_path, ledger_benchmark, stub_pipeline
):
    code = run_benchmark(
        ledger_benchmark.name,
        encoder="fixture-encoder",
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "out",
    )

    assert code == 0
    assert ledger_benchmark.curate_calls == 1
    assert [kw["seed"] for kw in ledger_benchmark.built] == [0, 1, 2]
    # Every seed shares ONE feature-cache root (extraction runs once per encoder).
    roots = {kw["overrides"]["cache"]["root_dir"] for kw in ledger_benchmark.built}
    assert roots == {str(tmp_path / "out" / "feature_cache")}


def test_run_benchmark_seeds_runs_the_first_n_canonical_style_seeds(
    tmp_path, ledger_benchmark, stub_pipeline
):
    code = run_benchmark(
        ledger_benchmark.name,
        encoder="fixture-encoder",
        seeds=1,
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "out",
    )

    assert code == 0
    assert [kw["seed"] for kw in ledger_benchmark.built] == [0]


# --- record + results_root: the external-ledger seam -----------------------------------


def test_run_benchmark_record_appends_to_external_results_root(
    capsys, tmp_path, monkeypatch, ledger_benchmark
):
    # The in-package ledger must stay untouched when an external root is given.
    sentinel = tmp_path / "in-package" / f"{ledger_benchmark.name}.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: sentinel)
    monkeypatch.setattr(run_mod, "_provenance", lambda: ("2026-08-20", "abc123", "5.7.0"))
    results_root = tmp_path / "external-ledger"
    run_dir = _run_dir_with_summary(tmp_path, 0.71)

    code = run_benchmark(
        ledger_benchmark.name,
        encoder="fixture-encoder",
        from_run_dir=run_dir,
        record=True,
        results_root=results_root,
    )

    assert code == 0
    assert not sentinel.exists()
    ledger = results_root / f"{ledger_benchmark.name}.csv"
    assert ledger.is_file()
    assert "recorded" in capsys.readouterr().out

    rows = load_results(ledger_benchmark.name, results_root=results_root)
    assert len(rows) == 1
    row = rows[0]
    assert row.key == {"encoder": "fixture-encoder"}
    assert row.metric == "test/accuracy"
    assert row.measured == pytest.approx(0.71)
    assert row.n_seeds == 1 and row.std is None
    # Provenance stamped exactly as the CLI stamps it.
    assert row.date == "2026-08-20"
    assert row.soma_commit == "abc123"
    assert row.slide2vec_version == "5.7.0"
    assert row.source == "soma reproduce --record"


def test_run_benchmark_full_mode_records_seed_spread_to_external_root(
    tmp_path, monkeypatch, ledger_benchmark, stub_pipeline
):
    sentinel = tmp_path / "in-package" / f"{ledger_benchmark.name}.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: sentinel)
    results_root = tmp_path / "external-ledger"

    code = run_benchmark(
        ledger_benchmark.name,
        encoder="fixture-encoder",
        raw_root=tmp_path / "raw",
        output_root=tmp_path / "out",
        record=True,
        results_root=results_root,
    )

    assert code == 0
    assert not sentinel.exists()
    rows = reproduced_rows(
        ledger_benchmark.name, results_root=results_root, encoder="fixture-encoder"
    )
    assert len(rows) == 1
    assert rows[0].n_seeds == 3
    assert rows[0].std == pytest.approx(0.0)
    assert rows[0].date and rows[0].soma_commit


def test_run_benchmark_without_results_root_uses_the_package_ledger(
    tmp_path, monkeypatch, ledger_benchmark
):
    ledger = tmp_path / "package" / f"{ledger_benchmark.name}.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: ledger)
    run_dir = _run_dir_with_summary(tmp_path, 0.71)

    code = run_benchmark(
        ledger_benchmark.name,
        encoder="fixture-encoder",
        from_run_dir=run_dir,
        record=True,
    )

    assert code == 0
    assert ledger.is_file()
    assert len(load_results(ledger_benchmark.name)) == 1


# --- registry: append_result / load_results / reproduced_rows accept a results root ----


def test_append_result_accepts_results_root_outside_the_checkout(tmp_path):
    results_root = tmp_path / "somewhere" / "else"
    row = MeasuredRow(
        key={"encoder": "uni2"},
        metric="test/accuracy",
        measured=0.9,
        n_seeds=1,
        date="2026-08-20",
        soma_commit="abc123",
        slide2vec_version="5.7.0",
        source="external harness",
    )

    path = append_result("toy", row, key_order=["encoder"], results_root=results_root)

    assert path == results_root / "toy.csv"
    header = path.read_text().splitlines()[0]
    assert header == (
        "encoder,metric,measured,std,n_seeds,date,soma_commit,"
        "slide2vec_version,croma_version,source"
    )

    # Append-only, and readable back through the same root.
    append_result("toy", row, results_root=results_root)
    assert len(load_results("toy", results_root=results_root)) == 2
    latest = reproduced_rows("toy", results_root=results_root, encoder="uni2")[-1]
    assert latest.measured == pytest.approx(0.9)
