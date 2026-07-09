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


# --- external-only reference table ----------------------------------------------------


def test_reference_csv_has_external_schema_header():
    with resources.files("soma.benchmarks.reference").joinpath("hest.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames
        rows = list(reader)
    # The header carries the external-anchor schema so a later issue can append rows.
    assert columns == [
        "dataset", "encoder", "metric", "expected", "tolerance", "kind", "label", "url", "source",
    ]
    # No reference rows are populated yet (external rows arrive in a later issue).
    assert rows == []


def test_expected_returns_no_rows_yet():
    # expected() may return no rows: the external Reference rows arrive in a later issue.
    assert get_benchmark("hest/IDC").expected() == []
    assert get_benchmark("hest/IDC").expected(encoder="uni2") == []


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


# --- reproduce CLI is external-only: no gate row, so it does not silently gate ----------


def test_reproduce_hest_errors_no_gate_row(capsys):
    # Consistent with issue #226: a benchmark with only external (or no) rows must NOT
    # silently gate on guidance; `soma reproduce hest/IDC` errors until a gate row exists.
    # The manual reproduction runs the Pipeline directly (see the PR body).
    from soma.cli import main

    try:
        main(["reproduce", "hest/IDC", "--from-run-dir", "/nonexistent"])
    except SystemExit as exc:
        code = int(exc.code or 0)
    else:
        code = 0
    err = capsys.readouterr().err
    assert code == 2
    assert "no gate reference row" in err
