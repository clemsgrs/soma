"""CLI wiring for ``soma list benchmarks`` and ``soma reproduce`` (issue #213).

The full curate→train→score reproduction needs OCELOT data + a GPU, so the reproduction is
verified via ``--from-run-dir`` with the greedy scorer's data/GPU-bound inner seam stubbed:
this exercises registry lookup, the fast-path log, the greedy ``score`` override, the
per-row tolerance check, and the non-zero exit on failure.
"""

from __future__ import annotations

import pytest

from soma.cli import main


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
