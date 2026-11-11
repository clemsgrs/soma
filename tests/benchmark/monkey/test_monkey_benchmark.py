"""The registered ``monkey`` benchmark: what ``soma reproduce`` runs by default."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import soma.cli as cli
from soma.benchmark import get_benchmark, load_results
from soma.benchmark import registry as registry_mod
from soma.benchmark.curation import CuratedManifest


def _curate(raw_root, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True)
    (out / "dataset.csv").write_text("sample_id,image_path\n", encoding="utf-8")
    (out / "splits.csv").write_text(
        "sample_id,split,fold\n" + "".join(f"roi{k},test,{k}\n" for k in range(5)),
        encoding="utf-8",
    )
    return CuratedManifest(dataset_csv=out / "dataset.csv", splits_csv=out / "splits.csv")


def _reproduce(*argv: str) -> int:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["reproduce", "monkey", *argv])
    return exit_info.value.code


@pytest.fixture()
def ledger(monkeypatch, tmp_path: Path) -> Path:
    """Route ``--record`` to a temporary ledger, never the packaged one."""
    path = tmp_path / "ledger" / "monkey.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda name: path)
    return path


def test_reproduce_monkey_runs_one_seed_across_the_five_folds_by_default(monkeypatch, capsys, tmp_path: Path):
    # The MONKEY protocol is seed 0 across five patient folds: without --seeds, reproduce
    # trains that one seed (one Pipeline over all five folds), not three.
    bench = get_benchmark("monkey")
    runs = []

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            folds = pd.read_csv(self.config.cohort.splits)["fold"].nunique()
            runs.append((self.config.run.seed, folds))

    monkeypatch.setattr(bench, "curate", _curate)
    monkeypatch.setattr(bench, "score", lambda run_dir: {"froc_inflammatory": 0.29})
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    code = _reproduce("--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"))

    assert code == 0
    assert runs == [(0, 5)]
    out = capsys.readouterr().out
    assert "Reproducing benchmark 'monkey' — canonical seeds [0], running [0]." in out
    # The view supplies its dataset and spacing, so the Virchow2 gate applies.
    assert "[REFERENCE OK] monkey froc_inflammatory = 0.2900  (reference 0.2929," in out


def test_reproduce_monkey_records_its_one_seed_run(monkeypatch, tmp_path: Path, ledger: Path):
    bench = get_benchmark("monkey")

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    monkeypatch.setattr(bench, "curate", _curate)
    monkeypatch.setattr(bench, "score", lambda run_dir: {"froc_inflammatory": 0.29})
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    code = _reproduce(
        "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"), "--record"
    )

    assert code == 0
    (row,) = load_results("monkey")
    assert row.key == {"dataset": "monkey", "encoder": "virchow2", "spacing": "0.242"}
    assert (row.measured, row.n_seeds) == (pytest.approx(0.29), 1)


def test_reproduce_monkey_refuses_to_record_extra_seeds(monkeypatch, capsys, tmp_path: Path, ledger: Path):
    # MONKEY's replicates are its five folds of seed 0; a three-seed mean is another
    # measurement than the one its ledger and gate hold, so --record refuses it up front.
    bench = get_benchmark("monkey")
    monkeypatch.setattr(bench, "curate", lambda *args: pytest.fail("curated a refused run"))
    monkeypatch.setattr(cli, "Pipeline", lambda config: pytest.fail("trained a refused run"))

    code = _reproduce(
        "--seeds", "3",
        "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"), "--record",
    )

    assert code != 0
    assert not ledger.exists()
    assert "canonical seeds [0]" in capsys.readouterr().err
