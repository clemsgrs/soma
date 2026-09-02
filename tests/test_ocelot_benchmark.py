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
    # The canonical comparison fixes physical spacing and varies only the encoder.
    assert facet.varied == ("encoder",)
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
    # No spacing given (as reproduce now calls it): the protocol fixes it at the anchor.
    assert cfg.preprocessing.requested_spacing_um == pytest.approx(0.2)
    assert str(cfg.dataset_csv) == "/x/dataset.csv"
    assert str(cfg.splits_csv) == "/x/splits.csv"
    assert str(cfg.output_root) == "/out"
    assert cfg.training.seed == 3


@pytest.mark.parametrize(
    ("encoder", "spacing", "target_size"),
    [
        ("virchow2", 0.2, 1024),
        ("virchow2", 0.25, 819),
        ("virchow2", 0.5, 410),
        ("uni2", 0.25, 819),
        ("uni2", 0.5, 410),
    ],
)
def test_build_config_selects_one_native_manifest_for_every_protocol(
    encoder: str, spacing: float, target_size: int
):
    cfg = OCELOT.build_config(encoder=encoder, spacing=spacing)

    assert cfg.encoder.name == encoder
    assert cfg.preprocessing.requested_spacing_um == pytest.approx(spacing)
    assert cfg.preprocessing.requested_tile_size_px == target_size
    assert Path(cfg.dataset_csv).parts[-3:] == ("ocelot", "curated", "dataset.csv")
    assert Path(cfg.splits_csv).parts[-3:] == ("ocelot", "curated", "splits.csv")


def test_build_config_accepts_unreferenced_encoder_at_packaged_spacing():
    cfg = OCELOT.build_config(encoder="phikon", spacing=0.2)
    assert cfg.encoder.name == "phikon"
    assert cfg.preprocessing.requested_spacing_um == pytest.approx(0.2)
    assert cfg.encoder.allow_non_recommended_settings is True


def test_expected_returns_broad_band():
    # `expected()` returns the gate band plus the non-gating external anchors (issue #226);
    # the gate is the single config-agnostic tolerance row.
    gate = [r for r in OCELOT.expected() if not r.is_external]
    assert len(gate) == 1
    assert gate[0].metric == "mean_f1"
    assert gate[0].expected == pytest.approx(0.6995, abs=1e-6)
    # The broad banner also matches when axes are supplied.
    gate_axed = [
        r for r in OCELOT.expected(encoder="virchow2", spacing=0.2) if not r.is_external
    ]
    assert gate_axed[0].expected == pytest.approx(0.6995)


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
    # The required value columns are present, plus the external-anchor markers (issue #226).
    for col in ("metric", "expected", "tolerance", "source", "kind", "label", "url"):
        assert col in columns
    key_columns = columns[: columns.index("metric")]
    assert key_columns, "reference table declares key columns left of `metric`"
    gate_rows = [r for r in rows if (r.get("kind") or "gate").strip() != "external"]
    external_rows = [r for r in rows if (r.get("kind") or "").strip() == "external"]
    # One gate row, pinned to the anchor it was measured on, + the external guidance anchors.
    assert len(gate_rows) == 1
    assert external_rows, "the CSV carries non-gating external guidance anchors"
    gate = gate_rows[0]
    assert {c: gate[c] for c in key_columns} == {"encoder": "virchow2"}
    assert float(gate["tolerance"]) == pytest.approx(0.02)
    # External anchors carry a label + linkable URL and are not tolerance-checked.
    for row in external_rows:
        assert (row["label"] or "").strip()
        assert (row["url"] or "").strip().startswith("http")


def test_greedy_rescore_passes_nms_distance_through():
    """The re-scorer resolves ``task.params.nms_distance`` like the pipeline instead of
    hard-coding the NMS radius to the matching distance (source-level pin: the greedy
    re-scorer needs a trained run + dense cache to execute)."""
    import inspect

    import soma.benchmarks.ocelot as ocelot_mod

    source = inspect.getsource(ocelot_mod._greedy_report_for_run)
    assert "nms_distance_px=nms_px" in source
    assert 'p.get("nms_distance")' in source
