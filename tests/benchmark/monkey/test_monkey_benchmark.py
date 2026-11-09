"""The registered ``monkey`` benchmark: what ``soma reproduce`` runs by default."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import soma.cli as cli
from soma.benchmark import get_benchmark
from soma.benchmark.curation import CuratedManifest


def test_reproduce_monkey_runs_one_seed_across_the_five_folds_by_default(monkeypatch, capsys, tmp_path: Path):
    # The MONKEY protocol is seed 0 across five patient folds: without --seeds, reproduce
    # trains that one seed (one Pipeline over all five folds), not three.
    bench = get_benchmark("monkey")

    def curate(raw_root, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True)
        (out / "dataset.csv").write_text("sample_id,image_path\n", encoding="utf-8")
        (out / "splits.csv").write_text(
            "sample_id,split,fold\n" + "".join(f"roi{k},test,{k}\n" for k in range(5)),
            encoding="utf-8",
        )
        return CuratedManifest(dataset_csv=out / "dataset.csv", splits_csv=out / "splits.csv")

    runs = []

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            folds = pd.read_csv(self.config.cohort.splits)["fold"].nunique()
            runs.append((self.config.run.seed, folds))

    monkeypatch.setattr(bench, "curate", curate)
    monkeypatch.setattr(bench, "score", lambda run_dir: {"froc_inflammatory": 0.29})
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    with pytest.raises(SystemExit) as exit_info:
        cli.main([
            "reproduce", "monkey",
            "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"),
        ])

    assert exit_info.value.code == 0
    assert runs == [(0, 5)]
    out = capsys.readouterr().out
    assert "Reproducing benchmark 'monkey' — canonical seeds [0], running [0]." in out
    # The view supplies its dataset and spacing, so the Virchow2 gate applies.
    assert "[REFERENCE OK] monkey froc_inflammatory = 0.2900  (reference 0.2929," in out
