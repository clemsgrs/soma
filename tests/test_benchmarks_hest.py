"""HestBenchmark registered as ``hest/IDC`` (issue #259) — offline conformance + wiring.

No HEST data or GPU here: the registry lookup, the Benchmark-Protocol conformance, the
``build_config`` recipe (dataset_type=spatial_expression, probe method, PCA latent dim 256,
encoder axis defaulting to uni2), the external-only ``reference/hest.csv`` parse, the
default ``summary.json`` scorer, and ``curate`` delegation are all verifiable offline. The
live curate→probe→score reproduction (HEST download + slide2vec weights + GPU) is a manual
vertical-slice reproduction, not a CI unit test.
"""

from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path

import pytest

from soma.benchmarks import Benchmark, get_benchmark, list_benchmarks
from soma.benchmarks import hest


# --- registration ---------------------------------------------------------------------


def test_hest_idc_is_registered_and_conforms_to_protocol():
    names = list_benchmarks()
    assert "hest/IDC" in names
    bench = get_benchmark("hest/IDC")
    assert bench.name == "hest/IDC"
    assert bench.primary_metric == "test/mean_pearson_mean"
    assert bench.canonical_seeds == (0,)
    assert isinstance(bench, Benchmark)  # structural protocol conformance


def test_only_hest_idc_is_registered_no_other_hest_name():
    hest_names = [n for n in list_benchmarks() if n.split("/", 1)[0] == "hest"]
    assert hest_names == ["hest/IDC"]


def test_hest_facet_fixes_task_and_varies_encoder():
    facet = get_benchmark("hest/IDC").facet
    assert facet.varied == ("encoder",)
    assert facet.fixed["dataset"] == "IDC"
    assert facet.fixed["task"] == "regression"


# --- build_config recipe fidelity -----------------------------------------------------


def test_build_config_encodes_spatial_expression_probe(tmp_path):
    config = get_benchmark("hest/IDC").build_config(
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
        seed=3,
    )
    assert config.dataset_type == "spatial_expression"
    assert config.task.name == "regression"
    assert config.training.method == "ridge_pca_probe"
    assert config.task.params["pca_components"] == 256  # PCA latent dim
    assert config.encoder.name == "uni2"
    assert config.encoder.output_variant is None  # uni2 uses the slide2vec default
    assert config.evaluation.metrics == ["pearson"]
    assert config.training.seed == 3


def test_build_config_defaults_to_uni2(tmp_path):
    config = get_benchmark("hest/IDC").build_config(
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
    )
    assert config.encoder.name == hest.DEFAULT_ENCODER == "uni2"


def test_build_config_pins_virchow2_cls_variant(tmp_path):
    config = get_benchmark("hest/IDC").build_config(
        encoder="virchow2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
    )
    # HEST's virchow2 is CLS-only (1280-d); the 2560-d concat default would not match.
    assert config.encoder.output_variant == "cls"


def test_build_config_applies_cache_overrides(tmp_path):
    config = get_benchmark("hest/IDC").build_config(
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
        overrides={"cache": {"enabled": True, "root_dir": str(tmp_path / "cache")}},
    )
    assert config.cache.enabled is True
    assert str(config.cache.root_dir) == str(tmp_path / "cache")


def test_build_config_rejects_unknown_encoder(tmp_path):
    with pytest.raises((ValueError, KeyError)):
        get_benchmark("hest/IDC").build_config(
            encoder="definitely_not_an_encoder",
            dataset_csv=tmp_path / "dataset.csv",
            splits_csv=tmp_path / "splits.csv",
            output_root=tmp_path / "runs",
        )


# --- external-only reference table (issue #260) ---------------------------------------


def test_reference_csv_has_external_schema_header():
    with resources.files("soma.benchmarks.reference").joinpath("hest.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames
        rows = list(reader)
    assert columns == [
        "dataset", "encoder", "metric", "expected", "tolerance", "kind", "label", "url", "source",
    ]
    # Every populated row is an external (non-gating) IDC anchor with a label + url; no gate
    # row exists (the point of #260: render Measured beside Reference, never tolerance-check).
    assert rows, "reference/hest.csv must carry the published external IDC rows"
    for row in rows:
        assert row["dataset"] == "IDC"
        assert row["metric"] == "test/mean_pearson_mean"
        assert row["kind"] == "external"
        assert row["tolerance"].strip() == ""  # external rows never gate
        assert row["label"] and row["url"].startswith("http")


def test_reference_csv_carries_vertical_slice_encoder_numbers():
    by_encoder = {}
    with resources.files("soma.benchmarks.reference").joinpath("hest.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            by_encoder[row["encoder"]] = row
    # The two vertical-slice encoders carry their confirmed HEST-leaderboard IDC Pearson.
    assert float(by_encoder["uni2"]["expected"]) == pytest.approx(0.5898)
    assert float(by_encoder["virchow2"]["expected"]) == pytest.approx(0.5971)


def test_expected_returns_external_rows_for_resolved_encoder():
    bench = get_benchmark("hest/IDC")
    # Default axis resolves to uni2; --encoder narrows to that encoder's own row.
    uni2_rows = bench.expected(encoder="uni2")
    virchow2_rows = bench.expected(encoder="virchow2")
    assert len(uni2_rows) == 1 and len(virchow2_rows) == 1
    assert bench.expected() == uni2_rows  # DEFAULT_ENCODER == uni2
    (uni2,) = uni2_rows
    (virchow2,) = virchow2_rows
    assert uni2.expected == pytest.approx(0.5898)
    assert virchow2.expected == pytest.approx(0.5971)
    # External (non-gating) anchors with a human label + linkable url; never a gate.
    for row in (uni2, virchow2):
        assert row.is_external and row.kind == "external"
        assert row.metric == "test/mean_pearson_mean"
        assert row.label and row.url.startswith("http")
        assert row.tolerance == 0.0  # blank tolerance -> non-gating


def test_expected_unknown_encoder_returns_no_rows():
    # An encoder with no published HEST number yields no reference row (accuracy over coverage).
    assert get_benchmark("hest/IDC").expected(encoder="prism") == []


# --- default scorer + curation delegation ---------------------------------------------


def test_score_uses_default_summary_scorer(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"test/mean_pearson_mean": 0.42}))
    scored = get_benchmark("hest/IDC").score(tmp_path)
    assert scored["test/mean_pearson_mean"] == pytest.approx(0.42)


def test_curate_delegates_to_curate_hest(monkeypatch, tmp_path):
    calls = {}

    def _fake_curate(raw_root, output_dir, *, task):
        calls["args"] = (str(raw_root), str(output_dir), task)
        from soma.curation.manifest import CuratedManifest

        return CuratedManifest(
            dataset_csv=Path(output_dir) / "dataset.csv",
            splits_csv=Path(output_dir) / "splits.csv",
        )

    monkeypatch.setattr(hest, "curate_hest", _fake_curate)
    manifest = get_benchmark("hest/IDC").curate(tmp_path / "raw", tmp_path / "out")
    assert calls["args"] == (str(tmp_path / "raw"), str(tmp_path / "out"), "IDC")
    assert manifest.dataset_csv == tmp_path / "out" / "dataset.csv"


# --- reproduce CLI: external-only renders Measured beside Reference, never gates (#260) --


def _run_cli(argv: list[str]) -> int:
    """Invoke the CLI, returning the exit code (reproduce raises SystemExit)."""
    from soma.cli import main

    try:
        main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_reproduce_hest_renders_measured_beside_external_reference(capsys, tmp_path):
    # `soma reproduce hest/IDC --from-run-dir` re-scores the run's summary and renders the
    # Measured row next to HEST's published external Reference (default encoder uni2). It
    # must COMPLETE (exit 0) and tolerance-check NOTHING — the delta is guidance, not a gate.
    (tmp_path / "summary.json").write_text(json.dumps({"test/mean_pearson_mean": 0.51}))
    code = _run_cli(["reproduce", "hest/IDC", "--from-run-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "MEASURED" in out
    assert "0.5100" in out  # soma's Measured value
    assert "0.5898" in out  # HEST's published uni2 Reference, rendered beside it
    assert "not gated" in out  # explicitly non-gating
    assert "PASS" not in out and "FAIL" not in out  # nothing is tolerance-checked


def test_reproduce_hest_keeps_encoder_reference_beside_measured(capsys, tmp_path):
    # --encoder virchow2 renders the Measured value beside virchow2's own HEST number (0.5971),
    # not uni2's — still non-gating, still exit 0.
    (tmp_path / "summary.json").write_text(json.dumps({"test/mean_pearson_mean": 0.55}))
    code = _run_cli(
        ["reproduce", "hest/IDC", "--encoder", "virchow2", "--from-run-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "0.5971" in out  # virchow2's HEST Reference
    assert "0.5898" not in out  # NOT uni2's
    assert "PASS" not in out and "FAIL" not in out
