"""The registered ``midog`` benchmark: its seeds and its reference gate."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

import soma.cli as cli
from soma.benchmark import get_benchmark, load_results
from soma.benchmark import registry as registry_mod
from soma.benchmark.curation import CuratedManifest
from soma.benchmark.detection import CellPredictions, SamplePrediction, write_cell_predictions


def test_midog_gate_is_the_three_seed_mean_the_default_run_measures():
    # The default MIDOG run averages its three seeds before the check, so the gate is the
    # three-seed mean of the packaged measurement, not one seed's score.
    bench = get_benchmark("midog")
    (gate,) = [row for row in bench.expected(encoder="virchow2") if not row.is_external]
    provenance = json.loads(
        resources.files("soma.benchmark.midog").joinpath("provenance.json").read_text(encoding="utf-8")
    )
    (measured,) = [
        row for row in provenance["transferred_protocol"]["aggregates"] if row["encoder"] == "virchow2"
    ]

    assert len(measured["per_seed"]) == len(bench.canonical_seeds) == 3
    assert gate.expected == pytest.approx(0.7619)
    assert gate.expected == pytest.approx(measured["mean"], abs=5e-5)
    assert gate.tolerance == pytest.approx(0.02)
    assert "three-seed mean" in gate.source
    assert "0.7626 / 0.7620 / 0.7609" in gate.source
    assert "seed 0" not in gate.source


def _curate(raw_root, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True)
    (out / "dataset.csv").write_text("sample_id,image_path\n", encoding="utf-8")
    (out / "splits.csv").write_text("sample_id,split\ns0,test\n", encoding="utf-8")
    return CuratedManifest(dataset_csv=out / "dataset.csv", splits_csv=out / "splits.csv")


class _FakePipeline:
    def __init__(self, config):
        self.config = config

    def run(self):
        pass


def _reproduce(*argv: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["reproduce", "midog", *argv])
    return exit_info.value.code


@pytest.fixture()
def ledger(monkeypatch, tmp_path: Path) -> Path:
    """Route ``--record`` to a temporary ledger, never the packaged one."""
    path = tmp_path / "ledger" / "midog.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: path)
    return path


def test_reproduce_midog_checks_the_three_seed_mean_against_the_gate(monkeypatch, capsys, tmp_path: Path):
    # A default run trains seeds 0-2 and checks their mean F1 against the Virchow2 gate.
    bench = get_benchmark("midog")
    per_seed = iter([0.7626, 0.7620, 0.7609])
    monkeypatch.setattr(bench, "curate", _curate)
    monkeypatch.setattr(bench, "score", lambda run_dir: {"f1": next(per_seed)})
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    code = _reproduce("--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"))

    assert code == 0
    assert "[REFERENCE OK] midog f1 = 0.7618  (reference 0.7619," in capsys.readouterr().out


def test_reproduce_midog_records_the_default_three_seed_run(monkeypatch, tmp_path: Path, ledger: Path):
    bench = get_benchmark("midog")
    per_seed = iter([0.7626, 0.7620, 0.7609])
    monkeypatch.setattr(bench, "curate", _curate)
    monkeypatch.setattr(bench, "score", lambda run_dir: {"f1": next(per_seed)})
    monkeypatch.setattr(cli, "Pipeline", _FakePipeline)

    code = _reproduce(
        "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"), "--record"
    )

    assert code == 0
    (row,) = load_results("midog")
    assert row.key == {"dataset": "midog", "encoder": "virchow2", "spacing": "0.25"}
    assert (row.n_seeds, row.measured) == (3, pytest.approx(0.761833, abs=1e-6))
    assert row.std == pytest.approx(0.000862, abs=1e-6)  # sample SD of the three seeds


def test_reproduce_midog_refuses_to_record_a_partial_seed_set(monkeypatch, capsys, tmp_path: Path, ledger: Path):
    # --seeds 1 is a smoke test: its one-seed score would land in a three-seed ledger and be
    # shown against a three-seed gate, so --record refuses it before any work starts.
    bench = get_benchmark("midog")
    monkeypatch.setattr(bench, "curate", lambda *args: pytest.fail("curated a refused run"))
    monkeypatch.setattr(cli, "Pipeline", lambda config: pytest.fail("trained a refused run"))

    code = _reproduce(
        "--seeds", "1",
        "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"), "--record",
    )

    assert code != 0
    assert not ledger.exists()
    err = capsys.readouterr().err
    assert "canonical seeds [0, 1, 2]" in err
    assert "without --seeds" in err and "--from-run-dir <output root>" in err


def test_reproduce_midog_refuses_to_record_a_rescore_of_one_seed(monkeypatch, capsys, tmp_path: Path, ledger: Path):
    # A seed_0 rescore is a spot check of one seed; --record refuses it before scoring.
    bench = get_benchmark("midog")
    seed_root = tmp_path / "out" / "seed_0"
    seed_root.mkdir(parents=True)
    monkeypatch.setattr(bench, "score", lambda run_dir: pytest.fail("scored a refused rescore"))

    code = _reproduce("--from-run-dir", str(seed_root), "--record")

    assert code != 0
    assert not ledger.exists()
    err = capsys.readouterr().err
    assert "canonical seeds [0, 1, 2]" in err and "seeds [0]" in err


def _seed_outputs(tmp_path: Path, monkeypatch, per_seed: dict[str, float]) -> tuple[Path, list[Path]]:
    """An output root holding one dir per seed, scored as ``per_seed``; returns the dirs scored."""
    out = tmp_path / "out"
    for name in per_seed:
        (out / name).mkdir(parents=True)
    scored: list[Path] = []

    def score(run_dir):
        scored.append(Path(run_dir))
        return {"f1": per_seed[Path(run_dir).name]}

    monkeypatch.setattr(get_benchmark("midog"), "score", score)
    return out, scored


_THREE_SEEDS = {"seed_0": 0.7626, "seed_1": 0.7620, "seed_2": 0.7609}


def test_rescoring_a_whole_midog_output_checks_the_three_seed_mean(monkeypatch, capsys, tmp_path: Path):
    # Pointed at the output root, a rescore scores every seed run there, as the default
    # run did, and checks their mean against the three-seed gate.
    out, scored = _seed_outputs(tmp_path, monkeypatch, _THREE_SEEDS)

    code = _reproduce("--from-run-dir", str(out))

    assert code == 0
    assert scored == [out / "seed_0", out / "seed_1", out / "seed_2"]
    assert "[REFERENCE OK] midog f1 = 0.7618  (reference 0.7619," in capsys.readouterr().out


def test_rescoring_one_midog_seed_still_scores_that_seed_alone(monkeypatch, capsys, tmp_path: Path):
    out, scored = _seed_outputs(tmp_path, monkeypatch, _THREE_SEEDS)

    code = _reproduce("--from-run-dir", str(out / "seed_1"))

    assert code == 0
    assert scored == [out / "seed_1"]
    assert "[REFERENCE OK] midog f1 = 0.7620  (reference 0.7619," in capsys.readouterr().out


def test_recording_a_whole_midog_output_appends_one_three_seed_row(monkeypatch, tmp_path: Path, ledger: Path):
    out, _ = _seed_outputs(tmp_path, monkeypatch, _THREE_SEEDS)

    code = _reproduce("--from-run-dir", str(out), "--record")

    assert code == 0
    (row,) = load_results("midog")
    assert row.key == {"dataset": "midog", "encoder": "virchow2", "spacing": "0.25"}
    assert (row.n_seeds, row.measured) == (3, pytest.approx(0.761833, abs=1e-6))
    assert row.std == pytest.approx(0.000862, abs=1e-6)  # sample SD of the three seeds


def test_reproduce_midog_refuses_to_record_a_partial_output(monkeypatch, capsys, tmp_path: Path, ledger: Path):
    # Seeds 0-1 of 0-2 can be rescored and checked, but not recorded.
    out, scored = _seed_outputs(tmp_path, monkeypatch, {"seed_0": 0.7626, "seed_1": 0.7620})

    code = _reproduce("--from-run-dir", str(out), "--record")

    assert code != 0
    assert scored == [] and not ledger.exists()
    assert "seeds [0, 1]" in capsys.readouterr().err


def test_rescoring_a_whole_output_refuses_seed_runs_of_different_encoders(capsys, tmp_path: Path):
    # A mean over seed runs of different encoders measures no one cell.
    out = tmp_path / "out"
    for seed, encoder in enumerate(["virchow2", "uni2", "virchow2"]):
        write_cell_predictions(
            out / f"seed_{seed}" / "predictions.json",
            CellPredictions(encoder, "midog", seed, "f1", 0.25,
                            [SamplePrediction("s0", [[10, 10]], [0.9], [0], [[10, 11]], [0])]),
        )

    code = _reproduce("--from-run-dir", str(out))

    assert code != 0
    captured = capsys.readouterr()
    assert "REFERENCE" not in captured.out
    assert "seed_1" in captured.err and "uni2" in captured.err and "virchow2" in captured.err


def test_rescoring_a_whole_output_names_a_seed_run_that_is_not_finished(tmp_path: Path):
    # seed_2 has not written its predictions yet: the rescore stops there and says so,
    # rather than reporting the mean of two seeds as the three-seed result.
    out = tmp_path / "out"
    for seed in (0, 1):
        write_cell_predictions(
            out / f"seed_{seed}" / "predictions.json",
            CellPredictions("virchow2", "midog", seed, "f1", 0.25,
                            [SamplePrediction("s0", [[10, 10]], [0.9], [0], [[10, 11]], [0])]),
        )
    (out / "seed_2").mkdir()

    with pytest.raises(FileNotFoundError, match=r"seed run .*seed_2 is not finished"):
        cli.main(["reproduce", "midog", "--from-run-dir", str(out)])
