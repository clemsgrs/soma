"""HestBenchmark family ``hest/<task>`` (all 9 HEST-Benchmark tasks) — offline conformance.

No HEST data or GPU here: the registry lookup (all 9 tasks), the Benchmark-Protocol
conformance, the ``build_config`` recipe (dataset_type=spatial_expression, probe method, PCA
latent dim 256, encoder axis defaulting to uni2, the virchow2 CLS variant / h-optimus-1
default), the external-only ``reference/hest.csv`` parse (9 tasks × the reproduction encoders),
the default ``summary.json`` scorer, ``curate`` delegation, and the external-only ``--record``
path are all verifiable offline. The live curate→probe→score reproduction (HEST download +
slide2vec weights + GPU) is a manual campaign, not a CI unit test.
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


def test_all_nine_hest_benchmark_tasks_are_registered():
    # HEST-Benchmark has 9 scored tasks; each is registered as hest/<task> (like eva/<dataset>).
    hest_names = sorted(n for n in list_benchmarks() if n.split("/", 1)[0] == "hest")
    assert hest_names == sorted(f"hest/{t}" for t in hest.HEST_TASKS)
    assert len(hest.HEST_TASKS) == 9
    # HCC ships data on the HF hub but is NOT a scored HEST-Benchmark task -> never registered.
    assert "HCC" not in hest.HEST_TASKS
    assert "hest/HCC" not in list_benchmarks()


def test_hest_facet_fixes_task_and_varies_encoder():
    facet = get_benchmark("hest/PAAD").facet
    assert facet.varied == ("encoder",)
    assert facet.fixed["dataset"] == "PAAD"
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
    # Issue #261: virchow2 is the second encoder of the vertical slice. build_config must
    # pin the CLS-only output variant (leaderboard-relevant; 1280-d, not slide2vec's 2560-d
    # CLS+mean concat default) and otherwise emit the same valid spatial-expression probe
    # recipe as uni2. "cls" is the exact token slide2vec's virchow2 encoder accepts for the
    # CLS-only variant (slide2vec.encoders.models.virchow: output_variants={"cls": 1280, ...}).
    config = get_benchmark("hest/IDC").build_config(
        encoder="virchow2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
        seed=3,
    )
    assert config.encoder.name == "virchow2"
    # HEST's virchow2 is CLS-only (1280-d); the 2560-d concat default would not match.
    assert config.encoder.output_variant == "cls"
    # Everything else stays the fixed HEST probe recipe (identical to the uni2 build).
    assert config.dataset_type == "spatial_expression"
    assert config.task.name == "regression"
    assert config.training.method == "ridge_pca_probe"
    assert config.task.params["pca_components"] == 256  # PCA latent dim
    assert config.evaluation.metrics == ["pearson"]
    assert config.training.seed == 3


def test_build_config_h_optimus_1_uses_slide2vec_default_variant(tmp_path):
    # h-optimus-1 is the third reproduction encoder. Its only slide2vec output is the 1536-d
    # "default" CLS token — exactly what TRIDENT extracts — so it needs NO variant override
    # (unlike virchow2). build_config must leave output_variant=None and otherwise emit the
    # same spatial-expression probe recipe.
    config = get_benchmark("hest/IDC").build_config(
        encoder="h-optimus-1",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
    )
    assert config.encoder.name == "h-optimus-1"
    assert config.encoder.output_variant is None  # slide2vec default (CLS, 1536-d)
    assert config.dataset_type == "spatial_expression"
    assert config.training.method == "ridge_pca_probe"


def test_output_variants_maps_only_virchow2_to_cls():
    # The variant map is a targeted override: only virchow2 needs a non-default variant
    # (CLS-only); uni2 and h-optimus-1 fall through to slide2vec's default (plain CLS token).
    assert hest.OUTPUT_VARIANTS == {"virchow2": "cls"}
    assert hest.OUTPUT_VARIANTS.get("uni2") is None
    assert hest.OUTPUT_VARIANTS.get("h-optimus-1") is None


def test_build_config_pins_hest_tile_geometry(tmp_path):
    # The tile scale is a property of the BENCHMARK, not of the encoder: HEST predicts from a
    # 112x112 µm tile rendered at 224x224 px, i.e. 112/224 = 0.5 µm/px. build_config must pin
    # both so the encoder axis varies the encoder and nothing else — an encoder-chosen scale
    # would silently turn the comparison into one of magnifications.
    assert hest.TILE_SIZE_PX == 224
    assert hest.SPACING_UM == pytest.approx(112 / 224)
    config = get_benchmark("hest/IDC").build_config(
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
    )
    assert config.preprocessing.requested_tile_size_px == 224
    assert config.preprocessing.requested_spacing_um == pytest.approx(0.5)


@pytest.mark.parametrize("encoder", ["uni2", "virchow2", "h-optimus-1"])
def test_build_config_resolves_preprocessing_for_every_reproduction_encoder(encoder, tmp_path):
    # Regression: virchow2 declares supported_spacing_um=[0.25, 0.5, 1.0, 2.0], so soma's
    # validator (rightly) refuses to pick one and raised, making hest/<task> unrunnable on it.
    # uni2 / h-optimus-1 declare a scalar 0.5 and so auto-resolved, which hid the gap behind
    # the uni2-only vertical slice. With the geometry pinned, all three resolve to the SAME
    # 0.5 µm/px @ 224 px — and the pin is a no-op for the two that already worked, so numbers
    # recorded before it stay comparable.
    from soma.encoders.validation import resolve_preprocessing_config

    config = get_benchmark("hest/PAAD").build_config(
        encoder=encoder,
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
    )
    resolved = resolve_preprocessing_config(config.encoder, config.preprocessing)
    assert resolved.requested_spacing_um == pytest.approx(0.5)
    assert resolved.requested_tile_size_px == 224


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
    # Every populated row is an external (non-gating) anchor with a label + url; no gate row
    # exists (#260: render Measured beside Reference, never tolerance-check). Datasets are the
    # 9 HEST-Benchmark tasks (IDC carries the full ~18-encoder leaderboard; the other tasks
    # carry the reproduction encoders).
    assert rows, "reference/hest.csv must carry the published external rows"
    for row in rows:
        assert row["dataset"] in hest.HEST_TASKS
        assert row["metric"] == "test/mean_pearson_mean"
        assert row["kind"] == "external"
        assert row["tolerance"].strip() == ""  # external rows never gate
        assert row["label"] and row["url"].startswith("http")


def test_reference_csv_carries_reproduction_encoder_numbers_across_tasks():
    by_cell = {}
    with resources.files("soma.benchmarks.reference").joinpath("hest.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            by_cell[(row["dataset"], row["encoder"])] = float(row["expected"])
    # The three reproduction encoders carry their confirmed HEST-leaderboard IDC Pearson ...
    assert by_cell[("IDC", "uni2")] == pytest.approx(0.5898)
    assert by_cell[("IDC", "virchow2")] == pytest.approx(0.5971)
    assert by_cell[("IDC", "h-optimus-1")] == pytest.approx(0.6024)
    # ... and a sample of the per-task numbers that drive the reproduction proof (PAAD reorders
    # the encoders vs IDC: uni2 > h-optimus-1 > virchow2 — a genuine rank test, not a foregone one).
    assert by_cell[("PAAD", "uni2")] == pytest.approx(0.5001)
    assert by_cell[("PAAD", "h-optimus-1")] == pytest.approx(0.4964)
    assert by_cell[("PAAD", "virchow2")] == pytest.approx(0.4779)
    # All 9 tasks × 3 reproduction encoders are present (27 cells).
    repro = {c for c in by_cell if c[1] in ("uni2", "virchow2", "h-optimus-1")}
    assert len({d for d, _ in repro}) == 9 and len(repro) == 27


def test_expected_returns_external_row_for_non_idc_task():
    # A non-IDC task resolves its own external reference row (proves the family, not just IDC).
    (row,) = get_benchmark("hest/PAAD").expected(encoder="h-optimus-1")
    assert row.key == {"dataset": "PAAD", "encoder": "h-optimus-1"}
    assert row.expected == pytest.approx(0.4964)
    assert row.is_external and row.tolerance == 0.0


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
    assert "context only" in out
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


def test_reproduce_record_appends_external_only_cell_to_ledger(tmp_path, monkeypatch, capsys):
    # The external-only --record fix: HEST has no gate row, so before the fix --record was a
    # silent no-op (it keyed off a gate row that never exists). Now it falls back to the single
    # matching EXTERNAL row, so a reproduced Pearson still lands in results/hest.csv keyed to
    # join the HEST reference — which is what the docs' A/B/C reproduction proof reads.
    from soma.benchmarks import load_results, registry

    ledger = tmp_path / "hest_results.csv"
    monkeypatch.setattr(registry, "_results_file", lambda name: ledger)

    (tmp_path / "summary.json").write_text(json.dumps({"test/mean_pearson_mean": 0.5914}))
    code = _run_cli(
        ["reproduce", "hest/IDC", "--encoder", "uni2", "--from-run-dir", str(tmp_path), "--record"]
    )
    assert code == 0
    assert "recorded" in capsys.readouterr().out.lower()

    (row,) = load_results("hest")  # reads the redirected ledger
    assert row.key == {"dataset": "IDC", "encoder": "uni2"}  # keyed off the external row
    assert row.metric == "test/mean_pearson_mean"
    assert row.measured == pytest.approx(0.5914)
    # A single re-scored run has no seed *spread* (std stays empty), but it is one seed.
    assert row.std is None and row.n_seeds == 1


def test_reproduce_record_writes_no_orphan_row_when_encoder_has_no_reference(
    tmp_path, monkeypatch, capsys
):
    # An encoder with no published HEST number still runs and scores. It cannot be keyed into
    # the reference-indexed ledger, so --record explains the skip and leaves no orphan row.
    from soma.benchmarks import load_results, registry

    ledger = tmp_path / "hest_results.csv"
    monkeypatch.setattr(registry, "_results_file", lambda name: ledger)

    (tmp_path / "summary.json").write_text(json.dumps({"test/mean_pearson_mean": 0.40}))
    code = _run_cli(
        ["reproduce", "hest/IDC", "--encoder", "prism", "--from-run-dir", str(tmp_path), "--record"]
    )
    assert code == 0
    out = capsys.readouterr().out.lower()
    assert "[reference skipped]" in out
    assert "no reference row to key --record on" in out
    assert load_results("hest") == []  # no orphan row written
    assert not ledger.exists()


def test_record_reference_row_declines_when_external_rows_are_ambiguous():
    # _record_reference_row keys a --record entry on the gate row, falling back to the single
    # matching external row for an external-only benchmark. If the axes leave MORE than one
    # external row matching the primary metric there is no unambiguous cell to key on, so it
    # declines (returns None) rather than silently picking the first — that is the branch the
    # CLI turns into "nothing recorded".
    from soma.benchmarks.registry import ReferenceRow
    from soma.cli import _record_reference_row

    def _ext(encoder: str) -> ReferenceRow:
        return ReferenceRow(
            key={"dataset": "IDC", "encoder": encoder},
            metric="test/mean_pearson_mean",
            expected=0.59,
            tolerance=None,
            source="test fixture",
            kind="external",
        )

    class _Stub:
        primary_metric = "test/mean_pearson_mean"

        def __init__(self, rows):
            self._rows = rows

        def expected(self, **_axes):
            return self._rows

    one, two = _ext("uni2"), _ext("virchow2")
    assert _record_reference_row(_Stub([one]), {}, None) is one  # unambiguous → key on it
    assert _record_reference_row(_Stub([one, two]), {}, None) is None  # ambiguous → decline
    assert _record_reference_row(_Stub([]), {}, None) is None  # nothing → decline
