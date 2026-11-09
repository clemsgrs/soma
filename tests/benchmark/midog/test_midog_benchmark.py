"""The registered ``midog`` benchmark: its seeds and its reference gate."""

from __future__ import annotations

import json
from importlib import resources

import pytest

from soma.benchmark import expected_rows, get_benchmark


def test_midog_gate_is_the_three_seed_mean_the_default_run_measures():
    # The default MIDOG run averages its three seeds before the check, so the gate is the
    # three-seed mean of the packaged measurement, not one seed's score.
    bench = get_benchmark("midog")
    (gate,) = [
        row
        for row in expected_rows("midog", metric="f1", dataset="midog", encoder="virchow2", spacing="0.25")
        if not row.is_external
    ]
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
