"""The registered ``midog`` benchmark: its seeds and its reference gate."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest

import soma.cli as cli
from soma.benchmark import get_benchmark
from soma.benchmark.curation import CuratedManifest


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


def test_reproduce_midog_checks_the_three_seed_mean_against_the_gate(monkeypatch, capsys, tmp_path: Path):
    # A default run trains seeds 0-2 and checks their mean F1 against the Virchow2 gate.
    bench = get_benchmark("midog")

    def curate(raw_root, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True)
        (out / "dataset.csv").write_text("sample_id,image_path\n", encoding="utf-8")
        (out / "splits.csv").write_text("sample_id,split\ns0,test\n", encoding="utf-8")
        return CuratedManifest(dataset_csv=out / "dataset.csv", splits_csv=out / "splits.csv")

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            pass

    per_seed = iter([0.7626, 0.7620, 0.7609])
    monkeypatch.setattr(bench, "curate", curate)
    monkeypatch.setattr(bench, "score", lambda run_dir: {"f1": next(per_seed)})
    monkeypatch.setattr(cli, "Pipeline", FakePipeline)

    with pytest.raises(SystemExit) as exit_info:
        cli.main([
            "reproduce", "midog",
            "--raw-root", str(tmp_path / "raw"), "--output-root", str(tmp_path / "out"),
        ])

    assert exit_info.value.code == 0
    assert "[REFERENCE OK] midog f1 = 0.7618  (reference 0.7619," in capsys.readouterr().out
