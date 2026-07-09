"""CLI wiring for ``soma list benchmarks`` and ``soma reproduce`` (issue #213).

The full curate→train→score reproduction needs OCELOT data + a GPU, so the reproduction is
verified via ``--from-run-dir`` with the greedy scorer's data/GPU-bound inner seam stubbed:
this exercises registry lookup, the fast-path log, the greedy ``score`` override, the
per-row tolerance check, and the non-zero exit on failure.
"""

from __future__ import annotations

import json

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
    # mean_f1 = 0.70 is within 0.6995 +/- 0.02 -> PASS -> exit 0. Stub the greedy inner
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
    assert "PASS" in out


def test_reproduce_from_run_dir_exits_nonzero_on_failure(monkeypatch, capsys, tmp_path):
    # mean_f1 = 0.50 is far outside the band -> FAIL -> non-zero exit.
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
    assert code != 0
    assert "FAIL" in out


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

    def score(self, run_dir):  # pragma: no cover - must never run without a gate row
        raise AssertionError("score() must not be called when there is no gate reference row")


@pytest.fixture()
def external_only_benchmark():
    bench = _ExternalOnlyBenchmark()
    register_benchmark(bench)
    try:
        yield bench
    finally:
        registry_mod._REGISTRY.pop(bench.name, None)


def test_reproduce_external_only_metric_errors_no_gate_row(capsys, tmp_path, external_only_benchmark):
    # A primary metric with only external guidance rows must NOT silently gate on guidance:
    # it errors "no gate reference row" and never scores the run.
    code = _run_cli(["reproduce", "ext_only_fixture", "--from-run-dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2
    assert "no gate reference row" in err


def test_reproduce_external_anchors_never_flip_the_gate_verdict(monkeypatch, capsys, tmp_path):
    # OCELOT's CSV carries external anchors (0.70, 0.73) OUTSIDE the gate band alongside the
    # gate (0.6995 ± 0.02). A measured 0.70 passes the gate; the external anchors must not
    # turn it into a FAIL or a "multiple rows" error.
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
    assert "PASS" in out


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
    # bach/uni2 reference is 0.915; a run reporting 0.915 -> PASS -> exit 0.
    _write_summary(tmp_path, 0.915)
    code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS" in out


def test_reproduce_eva_encoder_narrows_to_one_backbone(capsys, tmp_path):
    # A run reporting 0.883: fails against the default (uni2, 0.915) but passes once
    # --encoder virchow2 narrows the reference to virchow2's 0.883.
    _write_summary(tmp_path, 0.883)
    default_code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(tmp_path)])
    assert default_code == 1
    narrowed_code = _run_cli(
        ["reproduce", "eva/bach", "--encoder", "virchow2", "--from-run-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert narrowed_code == 0
    assert "PASS" in out


def test_reproduce_eva_from_run_dir_exits_nonzero_on_failure(capsys, tmp_path):
    _write_summary(tmp_path, 0.60)  # far below bach/uni2's 0.915
    code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code != 0
    assert "FAIL" in out


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
    assert row.n_seeds is None  # a re-scored single run has no seed spread
    assert row.date and row.soma_commit  # provenance captured at run time
    assert row.source == "soma reproduce --record"


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
    assert "PASS" in out


def test_reproduce_from_run_dir_encoder_without_reference_row_errors_honestly(capsys, tmp_path):
    # The run used encoder=uni (no eva/bach reference row); rather than silently comparing
    # against the uni2 default, --from-run-dir surfaces the missing row and exits 2.
    run_dir = _write_run(tmp_path, encoder="uni", balanced_accuracy=0.767)
    code = _run_cli(["reproduce", "eva/bach", "--from-run-dir", str(run_dir)])
    err = capsys.readouterr().err
    assert code == 2
    assert "no gate reference row" in err
    assert "'encoder': 'uni'" in err


def test_reproduce_from_run_dir_cli_encoder_overrides_run_encoder(capsys, tmp_path):
    # An explicit --encoder still wins over the run's recorded axis (setdefault semantics):
    # the uni run is force-compared against virchow2's 0.883 and passes.
    run_dir = _write_run(tmp_path, encoder="uni", balanced_accuracy=0.883)
    code = _run_cli(
        ["reproduce", "eva/bach", "--encoder", "virchow2", "--from-run-dir", str(run_dir)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "PASS" in out


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
