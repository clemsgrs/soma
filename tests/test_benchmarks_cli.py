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
    assert "no reference row" in err
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
