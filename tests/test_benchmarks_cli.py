"""CLI wiring for ``soma list benchmarks`` and ``soma reproduce`` (issue #213).

The full curate→train→score reproduction needs OCELOT data + a GPU, so the reproduction is
verified via ``--from-run-dir`` with the greedy scorer's data/GPU-bound inner seam stubbed:
this exercises registry lookup, the fast-path log, the greedy ``score`` override, and the
informational reference status.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import soma.cli as cli
from soma.cli import main
from soma.benchmarks import registry as registry_mod
from soma.benchmarks.registry import Facet, ReferenceRow, register_benchmark


def _run_cli(argv: list[str]) -> int:
    """Invoke the CLI, returning the exit code (main raises SystemExit for reproduce)."""
    try:
        main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_list_benchmarks_shows_ocelot(capsys):
    main(["list", "benchmarks"])
    out = capsys.readouterr().out
    assert "ocelot" in out
    assert "Benchmarks" in out


def test_reproduce_from_run_dir_passes_within_tolerance(monkeypatch, capsys, tmp_path):
    # mean_f1 = 0.70 is within 0.6995 +/- 0.02 -> REFERENCE OK. Stub the greedy inner
    # seam so no data/GPU is touched.
    monkeypatch.setattr(
        "soma.benchmarks.ocelot._greedy_report_for_run",
        lambda run_dir, matching="greedy": {
            "matching": "greedy",
            "tune": {"mean_f1": 0.71},
            "test": {"headline": {"metrics": {"mean_f1": 0.70}}},
        },
    )
    code = _run_cli(["reproduce", "ocelot", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert out == (
        "Reproducing benchmark 'ocelot' — canonical seeds [0], running [0].\n"
        "  Fast paths: --seeds 1 (single-seed smoke) | --from-run-dir <dir> "
        "(re-score an existing run, no training).\n"
        "  Cache-aware: feature extraction is cached and shared across seeds and repeat "
        "runs, so it runs once per encoder.\n"
        "[REFERENCE OK] ocelot mean_f1 = 0.7000  "
        "(reference 0.6995, Δ +0.0005, tolerance ±0.0200)\n"
    )


def test_reproduce_from_run_dir_reports_potential_drift_without_failing(monkeypatch, capsys, tmp_path):
    # mean_f1 = 0.50 is far outside the reference band. The comparison is diagnostic: it
    # highlights potential drift but does not turn a successfully scored run into a failure.
    monkeypatch.setattr(
        "soma.benchmarks.ocelot._greedy_report_for_run",
        lambda run_dir, matching="greedy": {
            "matching": "greedy",
            "tune": {"mean_f1": 0.71},
            "test": {"headline": {"metrics": {"mean_f1": 0.50}}},
        },
    )
    code = _run_cli(["reproduce", "ocelot", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "POTENTIAL DRIFT" in out
    assert "FAIL" not in out


def test_reproduce_logs_fast_path_hint_at_start(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(
        "soma.benchmarks.ocelot._greedy_report_for_run",
        lambda run_dir, matching="greedy": {
            "matching": "greedy",
            "tune": {"mean_f1": 0.71},
            "test": {"headline": {"metrics": {"mean_f1": 0.70}}},
        },
    )
    _run_cli(["reproduce", "ocelot", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    # The AC: log that a fast path exists so the full-seed cost isn't a surprise.
    assert "--seeds 1" in out
    assert "--from-run-dir" in out
    assert "canonical seeds" in out


def test_reproduce_default_runs_canonical_seed_set(monkeypatch, capsys, tmp_path):
    # Without --seeds, reproduce announces the benchmark's canonical seed set (OCELOT: [0]).
    monkeypatch.setattr(
        "soma.benchmarks.ocelot._greedy_report_for_run",
        lambda run_dir, matching="greedy": {
            "matching": "greedy",
            "tune": {"mean_f1": 0.71},
            "test": {"headline": {"metrics": {"mean_f1": 0.70}}},
        },
    )
    _run_cli(["reproduce", "ocelot", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "canonical seeds [0]" in out


# --- external (non-gating) guidance anchors excluded from the gate (issue #226) --------


class _ExternalOnlyBenchmark:
    """A benchmark whose primary metric carries ONLY external guidance rows (no gate)."""

    name = "ext_only_fixture"
    facet = Facet(fixed={}, varied=())
    canonical_seeds = (0,)
    primary_metric = "mean_f1"
    reference_environment: dict[str, str] = {}

    def curate(self, raw_root, out_dir):  # pragma: no cover - unused
        raise NotImplementedError

    def build_config(self, **axes):  # pragma: no cover - unused
        raise NotImplementedError

    def expected(self, **axes):
        return [
            ReferenceRow(
                key={},
                metric="mean_f1",
                expected=0.73,
                tolerance=0.0,
                source="captured 2026-07-03",
                kind="external",
                label="best reported",
                url="https://example.org/board",
            )
        ]

    def score(self, run_dir):
        # External-only benchmarks ARE scored so the Measured value can be rendered beside
        # the external Reference (#260) — they are just never tolerance-checked.
        return {"mean_f1": 0.60}


class _MultiMetricBenchmark:
    name = "multi_metric_fixture"
    facet = Facet(fixed={}, varied=())
    canonical_seeds = (0, 1)
    primary_metric = "median"
    reported_metrics = ("median", "f0", "ltm10")
    ranking_metrics = ("median", "ltm10")
    reference_environment: dict[str, str] = {}

    def __init__(self, scores=None):
        self.scores = scores or {"median": 0.50, "f0": 0.25, "ltm10": 0.75}
        self.scores_by_seed = None

    def curate(self, raw_root, out_dir):
        from pathlib import Path

        from soma.curation.manifest import CuratedManifest

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return CuratedManifest(
            dataset_csv=Path(out_dir) / "dataset.csv",
            splits_csv=Path(out_dir) / "splits.csv",
        )

    def build_config(self, **axes):
        return object()

    def expected(self, **axes):
        return [
            ReferenceRow(
                key={"encoder": "fixture"},
                metric="median",
                expected=0.48,
                tolerance=0.05,
                source="fixture",
            ),
            ReferenceRow(
                key={"encoder": "fixture"},
                metric="f0",
                expected=0.20,
                tolerance=0.0,
                source="fixture",
                kind="external",
                label="published diagnostic",
                url="https://example.org/diagnostic",
            ),
        ]

    def score(self, run_dir):
        if self.scores_by_seed is not None:
            return dict(self.scores_by_seed[Path(run_dir).name])
        return dict(self.scores)


@pytest.fixture()
def multi_metric_benchmark():
    bench = _MultiMetricBenchmark()
    register_benchmark(bench)
    try:
        yield bench
    finally:
        registry_mod._REGISTRY.pop(bench.name, None)


@pytest.fixture()
def external_only_benchmark():
    bench = _ExternalOnlyBenchmark()
    register_benchmark(bench)
    try:
        yield bench
    finally:
        registry_mod._REGISTRY.pop(bench.name, None)


def test_reproduce_external_only_renders_measured_beside_reference(capsys, tmp_path, external_only_benchmark):
    # A primary metric with only external guidance rows must NOT silently gate on guidance,
    # but it is no longer an error (#260): reproduce renders the Measured value beside the
    # external Reference (0.73), tolerance-checks NOTHING, and exits 0.
    code = _run_cli(["reproduce", "ext_only_fixture", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert out == (
        "Reproducing benchmark 'ext_only_fixture' — canonical seeds [0], running [0].\n"
        "  Fast paths: --seeds 1 (single-seed smoke) | --from-run-dir <dir> "
        "(re-score an existing run, no training).\n"
        "  Cache-aware: feature extraction is cached and shared across seeds and repeat "
        "runs, so it runs once per encoder.\n"
        "[MEASURED] ext_only_fixture mean_f1 = 0.6000\n"
        "  reference [best reported]: mean_f1 = 0.7300  "
        "(Δ -0.1300, external — context only)  <https://example.org/board>\n"
    )


def test_reproduce_from_run_dir_names_every_missing_reported_metric(
    capsys, tmp_path, multi_metric_benchmark
):
    multi_metric_benchmark.scores = {"median": 0.50}

    code = _run_cli(
        ["reproduce", multi_metric_benchmark.name, "--from-run-dir", str(tmp_path)]
    )
    captured = capsys.readouterr()

    assert code == 2
    assert multi_metric_benchmark.name in captured.err
    assert "f0" in captured.err
    assert "ltm10" in captured.err


def test_reproduce_renders_reported_metrics_in_order_and_only_primary_gates(
    capsys, tmp_path, multi_metric_benchmark
):
    code = _run_cli(
        ["reproduce", multi_metric_benchmark.name, "--from-run-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out

    assert code == 0
    primary_at = out.index("[REFERENCE OK] multi_metric_fixture median = 0.5000")
    external_at = out.index("[MEASURED] multi_metric_fixture f0 = 0.2500")
    diagnostic_at = out.index("[MEASURED] multi_metric_fixture ltm10 = 0.7500")
    assert primary_at < external_at < diagnostic_at
    assert "reference [published diagnostic]: f0 = 0.2000  (Δ +0.0500" in out
    assert "multi_metric_fixture ltm10 — no reference matches axes={}" in out
    assert out.count("REFERENCE OK") == 1
    assert "POTENTIAL DRIFT" not in out


def test_reproduce_full_mode_averages_each_reported_metric_independently(
    monkeypatch, capsys, tmp_path, multi_metric_benchmark
):
    import argparse
    import types

    multi_metric_benchmark.scores_by_seed = {
        "seed_0": {"median": 0.40, "f0": 0.20, "ltm10": 0.60},
        "seed_1": {"median": 0.60, "f0": 0.40, "ltm10": 0.80},
    }
    monkeypatch.setattr(cli, "Pipeline", lambda config: types.SimpleNamespace(run=lambda: None))
    args = argparse.Namespace(
        encoder=None,
        from_run_dir=None,
        seeds=None,
        raw_root=str(tmp_path / "raw"),
        curated_dir=None,
        out_dir=None,
        output_root=str(tmp_path / "out"),
        cache_root=None,
        record=False,
    )

    code = cli._reproduce_one(multi_metric_benchmark, args)
    out = capsys.readouterr().out

    assert code == 0
    assert "median = 0.5000" in out
    assert "f0 = 0.3000" in out
    assert "ltm10 = 0.7000" in out


def test_reproduce_record_writes_every_reported_metric_from_primary_anchor(
    capsys, tmp_path, monkeypatch, multi_metric_benchmark
):
    ledger = tmp_path / "ledger" / "multi.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: ledger)
    monkeypatch.setattr(cli, "_provenance", lambda: ("2026-08-11", "abc123", "5.7.0"))

    code = _run_cli(
        [
            "reproduce",
            multi_metric_benchmark.name,
            "--from-run-dir",
            str(tmp_path),
            "--record",
        ]
    )

    assert code == 0
    rows = registry_mod.load_results("multi_metric_fixture")
    assert [row.metric for row in rows] == ["median", "f0", "ltm10"]
    assert [row.measured for row in rows] == pytest.approx([0.50, 0.25, 0.75])
    assert {tuple(row.key.items()) for row in rows} == {
        (("encoder", "fixture"),)
    }
    assert all(row.n_seeds == 1 and row.std is None for row in rows)
    assert all(row.date and row.soma_commit and row.slide2vec_version for row in rows)
    assert "recorded" in capsys.readouterr().out


def test_reproduce_record_skips_every_metric_without_unambiguous_primary_anchor(
    capsys, tmp_path, monkeypatch, multi_metric_benchmark
):
    ledger = tmp_path / "ledger" / "multi.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: ledger)
    multi_metric_benchmark.expected = lambda **axes: [
        ReferenceRow(
            key={"encoder": encoder},
            metric="median",
            expected=0.48,
            tolerance=0.0,
            source="fixture",
            kind="external",
        )
        for encoder in ("one", "two")
    ]

    code = _run_cli(
        [
            "reproduce",
            multi_metric_benchmark.name,
            "--from-run-dir",
            str(tmp_path),
            "--record",
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert not ledger.exists()
    assert "no reference row to key --record on" in out
    assert "nothing recorded" in out


def test_reproduce_full_mode_names_every_missing_reported_metric(
    monkeypatch, capsys, tmp_path, multi_metric_benchmark
):
    import argparse
    import types

    multi_metric_benchmark.scores_by_seed = {"seed_0": {"median": 0.40}}
    monkeypatch.setattr(cli, "Pipeline", lambda config: types.SimpleNamespace(run=lambda: None))
    args = argparse.Namespace(
        encoder=None,
        from_run_dir=None,
        seeds=1,
        raw_root=str(tmp_path / "raw"),
        curated_dir=None,
        out_dir=None,
        output_root=str(tmp_path / "out"),
        cache_root=None,
        record=False,
    )

    with pytest.raises(SystemExit) as exc:
        cli._reproduce_one(multi_metric_benchmark, args)
    err = capsys.readouterr().err

    assert exc.value.code == 2
    assert multi_metric_benchmark.name in err
    assert "f0" in err
    assert "ltm10" in err


def test_reproduce_full_mode_records_independent_standard_deviations(
    monkeypatch, tmp_path, multi_metric_benchmark
):
    import argparse
    import types

    ledger = tmp_path / "ledger" / "multi.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: ledger)
    monkeypatch.setattr(cli, "_provenance", lambda: ("2026-08-11", "abc123", "5.7.0"))
    monkeypatch.setattr(cli, "Pipeline", lambda config: types.SimpleNamespace(run=lambda: None))
    multi_metric_benchmark.scores_by_seed = {
        "seed_0": {"median": 0.40, "f0": 0.20, "ltm10": 0.60},
        "seed_1": {"median": 0.60, "f0": 0.50, "ltm10": 1.00},
    }
    args = argparse.Namespace(
        encoder=None,
        from_run_dir=None,
        seeds=None,
        raw_root=str(tmp_path / "raw"),
        curated_dir=None,
        out_dir=None,
        output_root=str(tmp_path / "out"),
        cache_root=None,
        record=True,
    )

    assert cli._reproduce_one(multi_metric_benchmark, args) == 0

    rows = registry_mod.load_results("multi_metric_fixture")
    assert [row.measured for row in rows] == pytest.approx([0.50, 0.35, 0.80])
    assert [row.std for row in rows] == pytest.approx(
        [0.1414, 0.2121, 0.2828], abs=5e-5
    )
    assert [row.n_seeds for row in rows] == [2, 2, 2]


def test_reproduce_external_anchors_do_not_override_comparable_reference(
    monkeypatch, capsys, tmp_path
):
    # OCELOT's CSV carries external anchors (0.70, 0.73) OUTSIDE the gate band alongside the
    # comparable row (0.6995 ± 0.02). A measured 0.70 matches it; external anchors must not
    # turn the result into a drift warning or a "multiple rows" error.
    monkeypatch.setattr(
        "soma.benchmarks.ocelot._greedy_report_for_run",
        lambda run_dir, matching="greedy": {
            "matching": "greedy",
            "tune": {"mean_f1": 0.71},
            "test": {"headline": {"metrics": {"mean_f1": 0.70}}},
        },
    )
    code = _run_cli(["reproduce", "ocelot", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "REFERENCE OK" in out


def test_reproduce_unknown_benchmark_exits_nonzero(capsys):
    code = _run_cli(["reproduce", "nope"])
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown benchmark" in err


def test_reproduce_full_mode_requires_raw_root(capsys):
    code = _run_cli(["reproduce", "ocelot"])
    err = capsys.readouterr().err
    assert code == 2
    assert "raw-root" in err


def test_reproduce_rejects_spacing_flag(capsys):
    # Reproducing a benchmark fixes the whole protocol except the encoder; spacing is no
    # longer a reproduce axis (vary it via custom configs + a leaderboard instead).
    code = _run_cli(["reproduce", "ocelot", "--encoder", "uni2", "--spacing", "0.25"])
    err = capsys.readouterr().err
    assert code == 2
    assert "unrecognized arguments" in err and "--spacing" in err


# --- full-mode shared feature cache across seeds --------------------------------------


class _CaptureCacheBenchmark:
    """Fake benchmark that records the overrides ``_reproduce_one`` passes per seed."""

    name = "cache_share_fixture"
    facet = Facet(fixed={}, varied=())
    canonical_seeds = (0, 1, 2)
    primary_metric = "score"
    reference_environment: dict = {}

    def __init__(self):
        self.captured: list[dict] = []
        self.curate_calls = 0

    def expected(self, **axes):
        return [
            ReferenceRow(
                key={},
                metric="score",
                expected=0.5,
                tolerance=1.0,
                source="fixture",
                kind="gate",
                label=None,
                url=None,
            )
        ]

    def curate(self, raw_root, out_dir):
        from pathlib import Path

        from soma.curation.manifest import CuratedManifest

        self.curate_calls += 1
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return CuratedManifest(
            dataset_csv=Path(out_dir) / "dataset.csv",
            splits_csv=Path(out_dir) / "splits.csv",
        )

    def build_config(self, **kwargs):
        self.captured.append(kwargs)
        return object()

    def score(self, run_dir):
        return {"score": 0.5}


def _reproduce_full(monkeypatch, tmp_path, *, cache_root):
    import argparse
    import types

    monkeypatch.setattr(cli, "Pipeline", lambda config: types.SimpleNamespace(run=lambda: None))
    bench = _CaptureCacheBenchmark()
    args = argparse.Namespace(
        encoder=None,
        spacing=None,
        from_run_dir=None,
        seeds=None,
        raw_root=str(tmp_path / "raw"),
        out_dir=None,
        output_root=str(tmp_path / "out"),
        cache_root=cache_root,
    )
    code = cli._reproduce_one(bench, args)
    roots = [kw["overrides"]["cache"]["root_dir"] for kw in bench.captured]
    return code, bench, roots


def test_reproduce_full_mode_shares_one_feature_cache_across_seeds(monkeypatch, tmp_path):
    from pathlib import Path

    code, bench, roots = _reproduce_full(monkeypatch, tmp_path, cache_root=None)

    assert code == 0
    # One config built per canonical seed, all pointing at ONE cache root (extraction once).
    assert len(roots) == len(bench.canonical_seeds)
    assert len(set(roots)) == 1
    # Default root lives beside the run outputs, NOT under a per-seed dir.
    assert roots[0] == str(Path(tmp_path / "out" / "feature_cache"))
    assert "seed_" not in roots[0]
    assert all(kw["overrides"]["cache"]["enabled"] is True for kw in bench.captured)
    # Each seed still gets its own run output dir under the shared root's parent.
    seed_outputs = [str(kw["output_root"]) for kw in bench.captured]
    assert len(set(seed_outputs)) == len(bench.canonical_seeds)


def test_reproduce_full_mode_cache_root_flag_relocates_shared_cache(monkeypatch, tmp_path):
    custom = str(tmp_path / "fast" / "cache")
    code, _bench, roots = _reproduce_full(monkeypatch, tmp_path, cache_root=custom)

    assert code == 0
    # --cache-root overrides the location but keeps the single-shared-root guarantee.
    assert set(roots) == {custom}


# --- --curated-dir fast path (skip re-curation) --------------------------------------


def _reproduce_curated(monkeypatch, curated_dir, *, raw_root=None):
    import argparse
    import types

    monkeypatch.setattr(cli, "Pipeline", lambda config: types.SimpleNamespace(run=lambda: None))
    bench = _CaptureCacheBenchmark()
    args = argparse.Namespace(
        encoder=None,
        spacing=None,
        from_run_dir=None,
        seeds=None,
        raw_root=raw_root,
        curated_dir=str(curated_dir),
        out_dir=None,
        output_root=None,
        cache_root=None,
    )
    code = cli._reproduce_one(bench, args)
    return code, bench


def test_reproduce_curated_dir_skips_curation(monkeypatch, capsys, tmp_path):
    curated = tmp_path / "curated"
    curated.mkdir()
    (curated / "dataset.csv").write_text("sample_id,image_path\n")
    (curated / "splits.csv").write_text("sample_id,split,fold\n")

    code, bench = _reproduce_curated(monkeypatch, curated)
    out = capsys.readouterr().out

    assert code == 0
    # curate() is never called; the pipeline is pointed straight at the provided manifest.
    assert bench.curate_calls == 0
    assert "skipping curation" in out
    built = bench.captured[0]
    assert str(built["dataset_csv"]) == str(curated / "dataset.csv")
    assert str(built["splits_csv"]) == str(curated / "splits.csv")


def test_reproduce_curated_dir_missing_manifest_errors(monkeypatch, tmp_path):
    empty = tmp_path / "not_a_manifest"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="dataset.csv"):
        _reproduce_curated(monkeypatch, empty)


def test_reproduce_needs_a_manifest_source(capsys):
    # Neither --raw-root, --curated-dir, nor --from-run-dir: error lists all three modes.
    code = _run_cli(["reproduce", "ocelot"])
    err = capsys.readouterr().err
    assert code == 2
    assert "--curated-dir" in err


# --- EVA family (issue #219) ----------------------------------------------------------


def test_list_benchmarks_shows_eva_family(capsys):
    main(["list", "benchmarks"])
    out = capsys.readouterr().out
    assert "eva/bach" in out
    assert "eva/patch_camelyon" in out


def _write_summary(tmp_path, balanced_accuracy: float):
    (tmp_path / "summary.json").write_text(
        json.dumps({"test/balanced_accuracy": balanced_accuracy})
    )


def test_reproduce_eva_from_run_dir_passes_within_tolerance(capsys, tmp_path):
    # bach/uni2 reference is 0.915; a run reporting 0.915 -> REFERENCE OK.
    _write_summary(tmp_path, 0.915)
    code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "REFERENCE OK" in out


def test_reproduce_eva_encoder_narrows_to_one_backbone(capsys, tmp_path):
    # A run reporting 0.883 warns against the default (uni2, 0.915), while --encoder
    # virchow2 narrows the reference to virchow2's matching 0.883. Both runs succeed.
    _write_summary(tmp_path, 0.883)
    default_code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(tmp_path)])
    default_out = capsys.readouterr().out
    assert default_code == 0
    assert "POTENTIAL DRIFT" in default_out
    narrowed_code = _run_cli(
        ["reproduce", "eva/bach", "--encoder", "virchow2", "--from-run-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert narrowed_code == 0
    assert "REFERENCE OK" in out


def test_reproduce_eva_from_run_dir_reports_drift_without_failing(capsys, tmp_path):
    _write_summary(tmp_path, 0.60)  # far below bach/uni2's 0.915
    code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "POTENTIAL DRIFT" in out


def test_reproduce_record_appends_measured_row_with_provenance(capsys, tmp_path, monkeypatch):
    # --record writes to the results ledger; route it to a temp file, not the shipped CSV.
    ledger = tmp_path / "ledger" / "eva.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: ledger)
    _write_summary(tmp_path, 0.914)

    code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(tmp_path), "--record"])
    out = capsys.readouterr().out
    assert code == 0
    assert "recorded" in out

    from soma.benchmarks import reproduced_rows

    rows = reproduced_rows("eva", dataset="bach", encoder="uni2")
    assert len(rows) == 1
    row = rows[0]
    assert row.key == {"dataset": "bach", "encoder": "uni2"}
    assert row.measured == pytest.approx(0.914)
    # A re-scored run has no seed *spread* (std stays empty) but it is still one seed;
    # an empty n_seeds would read as "unknown" rather than "one".
    assert row.n_seeds == 1
    assert row.std is None
    assert row.date and row.soma_commit  # provenance captured at run time
    assert row.source == "soma reproduce --record"


def test_reproduce_record_keys_broad_reference_row_by_encoder(capsys, tmp_path, monkeypatch):
    """A broad reference's empty key must not yield an unattributable ledger row.

    OCELOT's reference is one config-agnostic band, so its key states no encoder. Copying
    that key verbatim appended a row that ``latest_measurements`` silently drops (it skips
    rows whose encoder is ``None``) — recorded in appearance only. The encoder comes from
    the axes instead, so the row joins the encoder-keyed rows already in the ledger.
    """
    ledger = tmp_path / "ledger" / "ocelot.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: ledger)
    monkeypatch.setattr(
        "soma.benchmarks.ocelot._greedy_report_for_run",
        lambda run_dir, matching="greedy": {
            "matching": "greedy",
            "tune": {"mean_f1": 0.71},
            "test": {"headline": {"metrics": {"mean_f1": 0.70}}},
        },
    )

    code = _run_cli(
        ["reproduce", "ocelot", "--encoder", "virchow2", "--from-run-dir", str(tmp_path), "--record"]
    )
    assert code == 0
    assert "recorded" in capsys.readouterr().out

    from soma.benchmarks import reproduced_rows

    rows = reproduced_rows("ocelot", encoder="virchow2")
    assert len(rows) == 1
    assert rows[0].key == {"encoder": "virchow2"}
    assert rows[0].measured == pytest.approx(0.70)


def test_record_axes_prefers_explicit_axis_over_the_run_dir(tmp_path):
    """An explicit ``--encoder`` wins; the run dir only fills what the CLI left implicit."""
    benchmark = registry_mod.get_benchmark("ocelot")
    run_dir = _write_run(
        tmp_path,
        encoder="virchow2",
        balanced_accuracy=0.70,
    )

    assert cli._record_axes(benchmark, {}, run_dir) == {"encoder": "virchow2"}
    assert cli._record_axes(benchmark, {"encoder": "uni2"}, run_dir) == {"encoder": "uni2"}
    # Nothing explicit and nothing resolvable on disk -> omitted, never guessed.
    assert cli._record_axes(benchmark, {}, None) == {}


def test_record_result_never_overwrites_a_key_the_reference_states(tmp_path, monkeypatch):
    """A keyed reference (EVA/HEST) owns its cell; axes only fill blanks.

    Otherwise a stale or mis-parsed axis could silently repoint a measurement at a
    different reference cell than the one it was just compared against.
    """
    ledger = tmp_path / "ledger" / "eva.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: ledger)
    benchmark = registry_mod.get_benchmark("eva/bach")
    row = ReferenceRow(
        key={"dataset": "bach", "encoder": "uni2"},
        metric="balanced_accuracy",
        expected=0.915,
        tolerance=0.02,
        source="test",
    )

    cli._record_result(
        benchmark, row, 0.9, std=None, n_seeds=1, key_axes={"encoder": "somethingelse"}
    )

    from soma.benchmarks import reproduced_rows

    assert reproduced_rows("eva", dataset="bach", encoder="uni2")[0].key["encoder"] == "uni2"


def test_git_commit_dirty_ignores_untracked_scratch(tmp_path, monkeypatch):
    """Provenance pins *code* state: untracked run outputs/notes must not mark it dirty.

    ``soma reproduce`` leaves scratch (``soma_reproduce/``, design notes) in the checkout, so
    a bare ``git status --porcelain`` would spuriously tag every ``--record`` row ``-dirty``.
    Only tracked modifications should.
    """
    import subprocess

    repo = tmp_path / "repo"
    pkg = repo / "soma"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("add", "-A")
    git("commit", "-qm", "init")

    import soma
    monkeypatch.setattr(soma, "__file__", str(pkg / "__init__.py"))

    clean = cli._git_commit()
    assert clean != "unknown" and not clean.endswith("-dirty")

    # Untracked scratch (like the run's own output dir) must NOT taint provenance.
    (repo / "soma_reproduce").mkdir()
    (repo / "soma_reproduce" / "out.json").write_text("{}")
    assert cli._git_commit() == clean

    # A tracked code modification DOES mark it dirty.
    (pkg / "__init__.py").write_text("# edited\n")
    assert cli._git_commit() == f"{clean}-dirty"


def _write_run(tmp_path, *, encoder: str, balanced_accuracy: float):
    """A minimal self-describing run dir (experiment.json + run.yaml + summary.json).

    ``load_run_record`` reads the encoder axis from the experiment's ``canonical_spec`` two
    levels above the run dir; this fixture is what lets ``--from-run-dir`` key the reference
    row on the run's OWN encoder. Returns the concrete run dir.
    """
    exp = tmp_path / "experiments" / "exp"
    run_dir = exp / "runs" / "run"
    run_dir.mkdir(parents=True)
    (exp / "experiment.json").write_text(
        json.dumps(
            {
                "experiment_id": "e",
                "dataset_checksum": "d",
                "splits_checksum": "s",
                "canonical_spec": {
                    "encoder": {"name": encoder},
                    "task": {"name": "multiclass_classification"},
                },
            }
        )
    )
    (run_dir / "run.yaml").write_text("seed: 0\nexperiment_id: e\nstatus: completed\n")
    (run_dir / "summary.json").write_text(
        json.dumps({"test/balanced_accuracy": balanced_accuracy})
    )
    return run_dir


def test_reproduce_from_run_dir_keys_reference_on_run_encoder(capsys, tmp_path):
    # The run recorded encoder=virchow2, so --from-run-dir must select virchow2's 0.883
    # reference on its own — NOT the benchmark default (uni2, 0.915) — with no --encoder.
    run_dir = _write_run(tmp_path, encoder="virchow2", balanced_accuracy=0.883)
    code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(run_dir)])
    out = capsys.readouterr().out
    assert code == 0
    assert "REFERENCE OK" in out


def test_reproduce_from_run_dir_encoder_without_reference_row_skips_comparison(capsys, tmp_path):
    # The run used encoder=uni (no eva/bach reference row). It is still a valid execution of
    # the packaged protocol: reproduce reports the measurement, makes the skipped comparison
    # explicit, and succeeds without silently borrowing another encoder's reference.
    run_dir = _write_run(tmp_path, encoder="uni", balanced_accuracy=0.767)
    code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(run_dir)])
    captured = capsys.readouterr()
    assert code == 0
    assert "[MEASURED] eva/bach test/balanced_accuracy = 0.7670" in captured.out
    assert "[REFERENCE SKIPPED]" in captured.out
    assert "'encoder': 'uni'" in captured.out
    assert captured.err == ""


def test_reproduce_from_run_dir_cli_encoder_overrides_run_encoder(capsys, tmp_path):
    # An explicit --encoder still wins over the run's recorded axis (setdefault semantics):
    # the uni run is force-compared against virchow2's 0.883 and matches it.
    run_dir = _write_run(tmp_path, encoder="uni", balanced_accuracy=0.883)
    code = _run_cli(
        ["reproduce", "eva/bach", "--encoder", "virchow2", "--from-run-dir", str(run_dir)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "REFERENCE OK" in out


def test_reproduce_eva_family_fans_out_over_members(monkeypatch, capsys, tmp_path):
    # `soma reproduce eva` fans out over the whole eva/<dataset> family.
    seen: list[str] = []
    monkeypatch.setattr(
        cli, "_reproduce_one", lambda bench, args, **kwargs: (seen.append(bench.name) or 0)
    )
    code = _run_cli(["reproduce", "eva", "--raw-root", str(tmp_path)])
    assert code == 0
    assert set(seen) == {
        "eva/bach",
        "eva/breakhis",
        "eva/crc",
        "eva/mhist",
        "eva/gleason_arvaniti",
        "eva/patch_camelyon",
    }


def test_reproduce_eva_family_from_run_dir_needs_single_subbenchmark(capsys, tmp_path):
    code = _run_cli(["reproduce", "eva", "--from-run-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2
    assert "sub-benchmark" in err


def test_reproduce_unknown_eva_family_member_exits_nonzero(capsys):
    code = _run_cli(["reproduce", "eva/nope", "--from-run-dir", "/x"])
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown benchmark" in err
