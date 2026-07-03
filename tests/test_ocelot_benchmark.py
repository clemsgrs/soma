"""OCELOT registered as a first-class Benchmark (ADR 0002, issue #213).

No OCELOT raw data or GPU here: ``build_config`` loading a committed YAML, ``curate``
delegating to the curator, ``expected`` reading the reference band, the greedy ``score``
override's metric extraction (fed a fixture greedy report), and the reference-table shape
are all verifiable offline. The live greedy re-score (``_greedy_report_for_run``) needs the
cached dense grids + a torch runtime and is covered by the ``--from-run-dir`` CLI test with
that seam stubbed.
"""

from __future__ import annotations

import csv
from importlib import resources
from pathlib import Path

import pytest

from soma.benchmarks import get_benchmark
from soma.benchmarks.ocelot import OCELOT, extract_test_metrics


def _greedy_report(mean_f1: float) -> dict:
    return {
        "matching": "greedy",
        "score_threshold_per_class": [0.53, 0.49],
        "tune": {"mean_f1": 0.7146},
        "test": {
            "headline": {
                "note": "reported result — thresholds frozen from tune",
                "score_threshold_per_class": [0.53, 0.49],
                "metrics": {
                    "mean_f1": mean_f1,
                    "f1_class_0": 0.6739,
                    "f1_class_1": 0.7250,
                    "mean_f1_per_image": 0.6006,
                },
            }
        },
    }


def test_ocelot_registered_under_name():
    assert get_benchmark("ocelot") is OCELOT
    assert OCELOT.name == "ocelot"
    assert OCELOT.primary_metric == "mean_f1"
    assert OCELOT.canonical_seeds == (0,)


def test_ocelot_declares_canonical_facet():
    facet = OCELOT.facet
    # Recipe backbone fixed; the campaign varies encoder x spacing.
    assert facet.varied == ("encoder", "spacing")
    assert facet.fixed["decoder"] == "lightweight_conv"
    assert facet.fixed["task"] == "detection"


def test_ocelot_reference_environment_is_recorded():
    env = OCELOT.reference_environment
    assert env["torch"] and env["cuda"]  # a small reference env shown alongside a run


def test_build_config_loads_committed_anchor_yaml():
    cfg = OCELOT.build_config(
        dataset_csv="/x/dataset.csv", splits_csv="/x/splits.csv", output_root="/out", seed=3
    )
    assert cfg.dataset_type == "detection"
    assert cfg.encoder.name == "virchow2"
    assert cfg.decoder.name == "lightweight_conv"
    assert str(cfg.dataset_csv) == "/x/dataset.csv"
    assert str(cfg.splits_csv) == "/x/splits.csv"
    assert str(cfg.output_root) == "/out"
    assert cfg.training.seed == 3


def test_build_config_selects_config_by_axes():
    cfg = OCELOT.build_config(encoder="uni2", spacing=0.25)
    assert cfg.encoder.name == "uni2"


def test_build_config_unknown_axes_is_keyerror():
    with pytest.raises(KeyError):
        OCELOT.build_config(encoder="phikon", spacing=0.2)


def test_expected_returns_broad_band():
    rows = OCELOT.expected()
    assert len(rows) == 1
    assert rows[0].metric == "mean_f1"
    assert rows[0].expected == pytest.approx(0.6995, abs=1e-6)
    # The broad banner also matches when axes are supplied.
    assert OCELOT.expected(encoder="virchow2", spacing=0.2)[0].expected == pytest.approx(0.6995)


def test_extract_test_metrics_reads_test_headline():
    metrics = extract_test_metrics(_greedy_report(0.70))
    assert metrics["mean_f1"] == pytest.approx(0.70)
    assert metrics["f1_class_1"] == pytest.approx(0.7250)
    # tune metrics are merged under a tune_ prefix.
    assert metrics["tune_mean_f1"] == pytest.approx(0.7146)


def test_extract_test_metrics_without_headline_raises():
    with pytest.raises(ValueError):
        extract_test_metrics({"matching": "greedy", "tune": {"mean_f1": 0.7}})


def test_score_override_uses_greedy_matcher(monkeypatch, tmp_path: Path):
    # The score OVERRIDE runs the greedy matcher; stub the GPU/data-bound inner seam so we
    # verify score() plumbs the greedy report into the primary metric offline.
    monkeypatch.setattr(
        "soma.benchmarks.ocelot._greedy_report_for_run",
        lambda run_dir, matching="greedy": _greedy_report(0.6991),
    )
    metrics = OCELOT.score(tmp_path)
    assert metrics["mean_f1"] == pytest.approx(0.6991)


def test_reference_csv_has_key_metric_expected_tolerance_source_columns():
    with resources.files("soma.benchmarks.reference").joinpath("ocelot.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames
        rows = list(reader)
    # Columns: key… then the fixed value columns, per-row tolerance present.
    assert columns[-4:] == ["metric", "expected", "tolerance", "source"]
    key_columns = columns[:-4]
    assert key_columns, "reference table declares key columns left of `metric`"
    # The single OCELOT row is the broad banner: all key cells empty (config-agnostic).
    assert len(rows) == 1
    assert all(not (rows[0][c] or "").strip() for c in key_columns)
    assert float(rows[0]["tolerance"]) == pytest.approx(0.02)
