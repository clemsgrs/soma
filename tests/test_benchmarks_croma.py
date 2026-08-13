"""Offline conformance tests for the registered CRoMa benchmark family."""

import csv
import json
from importlib import resources
from pathlib import Path

import pytest

import soma.cli as cli
from soma.benchmarks import (
    Benchmark,
    MeasuredRow,
    ReferenceRow,
    get_benchmark,
    get_ranking_metrics,
    get_reported_metrics,
    list_benchmarks,
    load_results,
    reproduction_report,
)
from soma.benchmarks import registry as registry_mod
from soma.benchmarks import reproduction as reproduction_mod
from soma.config import RepresentationConfig
from soma.benchmarks.croma import CROMA_0_3_ENCODER_PANEL
from soma.curation.manifest import CuratedManifest


def test_croma_family_registers_all_three_cohorts():
    names = [name for name in list_benchmarks() if name.startswith("croma/")]

    assert names == [
        "croma/camelyon",
        "croma/tcga-4x4",
        "croma/tolkach-esca",
    ]
    assert all(isinstance(get_benchmark(name), Benchmark) for name in names)


def test_croma_family_appears_in_cli_benchmark_listing(capsys):
    cli.main(["list", "benchmarks"])

    out = capsys.readouterr().out
    assert "croma/camelyon" in out
    assert "croma/tcga-4x4" in out
    assert "croma/tolkach-esca" in out


def test_croma_config_materializes_the_fixed_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "soma.benchmarks.croma.validate_croma_0_3_encoder_panel", lambda: None
    )
    benchmark = get_benchmark("croma/camelyon")

    config = benchmark.build_config(
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "runs",
    )

    assert config.dataset_type == "tile"
    assert config.task is None
    assert config.representation == RepresentationConfig(
        kind="croma",
        confounder_column="medical_center",
        split="test",
        evaluation_design="all",
        m=5,
        alpha=0.10,
    )
    assert config.training.seed == 0
    assert config.encoder.name == "uni2"
    assert config.encoder.output_variant == "default"


def test_croma_facet_materializes_the_fixed_protocol():
    facet = get_benchmark("croma/camelyon").facet

    assert facet.fixed == {
        "dataset": "camelyon",
        "dataset_type": "tile",
        "representation.kind": "croma",
        "representation.confounder_column": "medical_center",
        "representation.split": "test",
        "representation.evaluation_design": "all",
        "representation.m": 5,
        "representation.alpha": 0.10,
    }
    assert facet.varied == ("encoder",)


def test_croma_declares_reported_and_ranking_metric_roles_in_order():
    benchmark = get_benchmark("croma/camelyon")

    assert get_reported_metrics(benchmark) == (
        "test/croma_median",
        "test/croma_f0",
        "test/croma_ltm10",
    )
    assert get_ranking_metrics(benchmark) == (
        "test/croma_median",
        "test/croma_ltm10",
    )


def test_croma_validates_panel_pins(tmp_path, monkeypatch):
    validations = []
    monkeypatch.setattr(
        "soma.benchmarks.croma.validate_croma_0_3_encoder_panel",
        lambda: validations.append("validated"),
    )
    benchmark = get_benchmark("croma/camelyon")

    panel = benchmark.build_config(encoder="uni2", output_root=tmp_path / "panel")

    assert validations == ["validated"]
    assert panel.encoder.output_variant == "default"


def test_croma_keeps_encoders_outside_the_panel_runnable(tmp_path, monkeypatch):
    validations = []
    monkeypatch.setattr(
        "soma.benchmarks.croma.validate_croma_0_3_encoder_panel",
        lambda: validations.append("validated"),
    )

    outside = get_benchmark("croma/camelyon").build_config(
        encoder="isight", output_root=tmp_path / "outside"
    )

    assert validations == []
    assert outside.encoder.name == "isight"
    assert outside.encoder.output_variant is None


def test_croma_reference_has_one_external_row_per_cohort_encoder_metric():
    path = resources.files("soma.benchmarks.reference").joinpath("croma.csv")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    keys = {(row["dataset"], row["encoder"], row["metric"]) for row in rows}
    assert len(rows) == len(keys) == 234
    assert {row["dataset"] for row in rows} == {
        "camelyon",
        "tcga-4x4",
        "tolkach-esca",
    }
    assert {row["encoder"] for row in rows} == {
        spec.soma_encoder for spec in CROMA_0_3_ENCODER_PANEL.values()
    }
    assert {row["metric"] for row in rows} == {
        "test/croma_median",
        "test/croma_f0",
        "test/croma_ltm10",
    }
    assert all(row["kind"] == "external" for row in rows)
    assert all(row["label"] and row["url"] for row in rows)
    assert all(row["tolerance"] == "" for row in rows)
    by_key = {
        (row["dataset"], row["encoder"], row["metric"]): float(row["expected"])
        for row in rows
    }
    assert by_key[("camelyon", "rudolfv2-s", "test/croma_median")] == 0.32404
    assert by_key[("tcga-4x4", "midnight", "test/croma_f0")] == 0.140104
    assert by_key[("tolkach-esca", "mascaret", "test/croma_ltm10")] == 0.009418


def test_croma_expected_returns_three_ordered_external_rows_for_the_encoder():
    benchmark = get_benchmark("croma/camelyon")

    rows = benchmark.expected(encoder="rudolfv2-s")

    assert [row.metric for row in rows] == [
        "test/croma_median",
        "test/croma_f0",
        "test/croma_ltm10",
    ]
    assert [row.expected for row in rows] == [0.32404, 0.047108, -0.024488]
    assert all(row.is_external and row.tolerance == 0.0 for row in rows)
    assert benchmark.expected(encoder="isight") == []


def test_dinov2_b_is_the_only_published_ranking_ineligible_control():
    benchmark = get_benchmark("croma/camelyon")

    assert benchmark.is_ranking_eligible(encoder="dinov2-vitb14") is False
    assert all(
        benchmark.is_ranking_eligible(encoder=spec.soma_encoder)
        for model, spec in CROMA_0_3_ENCODER_PANEL.items()
        if model != "DINOv2-B"
    )


def _mock_reproduction_rows(monkeypatch, metric, encoders, references, measured):
    monkeypatch.setattr(
        reproduction_mod,
        "load_reference",
        lambda _name: [
            ReferenceRow(
                key={"dataset": "camelyon", "encoder": encoder},
                metric=metric,
                expected=value,
                tolerance=0.0,
                source="fixture",
                kind="external",
            )
            for encoder, value in zip(encoders, references)
        ],
    )
    monkeypatch.setattr(
        reproduction_mod,
        "load_results",
        lambda _name: [
            MeasuredRow(
                key={"dataset": "camelyon", "encoder": encoder},
                metric=metric,
                measured=value,
            )
            for encoder, value in zip(encoders, measured)
        ],
    )


def test_rank_agreement_keeps_dino_visible_but_excludes_control_pairs(monkeypatch):
    encoders = ("uni2", "conch", "dinov2-vitb14")
    _mock_reproduction_rows(
        monkeypatch,
        "test/croma_median",
        encoders,
        references=(0.3, 0.2, 0.9),
        measured=(0.4, 0.1, 0.99),
    )

    report = reproduction_report("croma")

    assert [cell.encoder for cell in report.cells] == list(encoders)
    assert [(pair.encoder_high, pair.encoder_low) for pair in report.pairs] == [
        ("uni2", "conch")
    ]


def test_reported_only_f0_produces_no_rank_agreement(monkeypatch):
    _mock_reproduction_rows(
        monkeypatch,
        "test/croma_f0",
        encoders=("uni2", "conch"),
        references=(0.3, 0.2),
        measured=(0.4, 0.1),
    )

    report = reproduction_report("croma", metric="test/croma_f0")

    assert len(report.cells) == 2
    assert report.pairs == []
    assert report.spearman_by_dataset == {"camelyon": None}


def test_croma_curates_its_cohort_from_the_prepared_family_root(
    tmp_path, monkeypatch
):
    calls = []

    def fake_curate(raw_root, out_dir, *, cohort):
        calls.append((Path(raw_root), Path(out_dir), cohort))
        return CuratedManifest(
            Path(out_dir) / "dataset.csv", Path(out_dir) / "splits.csv"
        )

    monkeypatch.setattr("soma.benchmarks.croma.curate_croma_view", fake_curate)
    benchmark = get_benchmark("croma/tcga-4x4")

    manifest = benchmark.curate(tmp_path / "prepared", tmp_path / "curated")

    assert calls == [(tmp_path / "prepared", tmp_path / "curated", "tcga-4x4")]
    assert manifest.dataset_csv == tmp_path / "curated" / "dataset.csv"


def test_croma_scores_all_three_reported_metrics_from_summary(tmp_path):
    expected = {
        "test/croma_median": 0.31,
        "test/croma_f0": 0.12,
        "test/croma_ltm10": -0.04,
    }
    (tmp_path / "summary.json").write_text(json.dumps(expected), encoding="utf-8")

    assert get_benchmark("croma/camelyon").score(tmp_path) == expected


def test_croma_rescore_renders_three_external_deltas_without_a_gate(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "test/croma_median": 0.31,
                "test/croma_f0": 0.12,
                "test/croma_ltm10": -0.04,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "Pipeline",
        lambda _config: pytest.fail("--from-run-dir must not construct a pipeline"),
    )

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "reproduce",
                "croma/camelyon",
                "--encoder",
                "conch",
                "--from-run-dir",
                str(tmp_path),
            ]
        )

    out = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert out.count("[MEASURED] croma/camelyon test/croma_") == 3
    assert out.count("external — context only") == 3
    assert out.count("Δ ") == 3
    assert "REFERENCE OK" not in out
    assert "POTENTIAL DRIFT" not in out


def test_croma_reference_provenance_pins_release_producer_protocol_and_sources():
    path = resources.files("soma.benchmarks.reference").joinpath(
        "croma.provenance.json"
    )
    provenance = json.loads(path.read_text(encoding="utf-8"))

    assert provenance == {
        "source_release": "Croma 0.3.0",
        "artifact_producer": {"croma_version": "0.2.0"},
        "exported": "2026-08-10",
        "protocol": {
            "name": "median-k",
            "m": 5,
            "alpha": 0.10,
            "evaluation_design": "all",
        },
        "files": {
            "results/camelyon.csv": "bb0a0dd0e62db64cd6d240fc98cfc4ae06794d46b06d5eae80e9bd3b6535d4d7",
            "results/tcga-4x4.csv": "d30c1ae9305a83fe5bd0a9e4afa90cfbf7b09663cb41e66571b1abb18eaa231b",
            "results/tolkach-esca.csv": "43fb47f602c35a3a12cd473b3ef08336739763972dd42e8fd0b8778a89b4cd3a",
        },
        "paper": "https://arxiv.org/abs/2607.25497",
    }


def _record_croma_result(tmp_path, monkeypatch, *, run_croma_version=None):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "test/croma_median": 0.123456789,
                "test/croma_f0": 0.234567891,
                "test/croma_ltm10": -0.345678912,
            }
        ),
        encoding="utf-8",
    )
    if run_croma_version is not None:
        (run_dir / "run.yaml").write_text(
            f"representation_provenance:\n  croma: {run_croma_version}\n",
            encoding="utf-8",
        )
    ledger = tmp_path / "results" / "croma.csv"
    monkeypatch.setattr(registry_mod, "_results_file", lambda _name: ledger)
    monkeypatch.setattr(cli, "_provenance", lambda: ("2026-08-11", "abc123", "5.7.0"))
    monkeypatch.setattr(cli, "_runtime_croma_version", lambda: "0.3.0")

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "reproduce",
                "croma/camelyon",
                "--encoder",
                "uni2",
                "--from-run-dir",
                str(run_dir),
                "--record",
            ]
        )

    assert exit_info.value.code == 0
    return load_results("croma")


def test_croma_record_uses_the_run_croma_version(tmp_path, monkeypatch):
    rows = _record_croma_result(tmp_path, monkeypatch, run_croma_version="0.2.9")

    assert all(row.croma_version == "0.2.9" for row in rows)


def test_croma_record_marks_missing_historical_croma_provenance_unknown(
    tmp_path, monkeypatch
):
    rows = _record_croma_result(tmp_path, monkeypatch)

    assert all(row.croma_version == "unknown" for row in rows)


def test_croma_record_preserves_rank_precision(tmp_path, monkeypatch):
    rows = _record_croma_result(tmp_path, monkeypatch)

    assert [row.measured for row in rows] == [0.123456789, 0.234567891, -0.345678912]


def _run_croma_family(tmp_path, monkeypatch):
    curated_calls = []
    config_calls = []

    def fake_curate(self, raw_root, out_dir):
        curated_calls.append((self.cohort, Path(raw_root), Path(out_dir)))
        return CuratedManifest(
            Path(out_dir) / "dataset.csv", Path(out_dir) / "splits.csv"
        )

    def fake_build_config(self, **kwargs):
        config_calls.append((self.cohort, kwargs))
        return object()

    monkeypatch.setattr(
        "soma.benchmarks.croma.CromaBenchmark.curate", fake_curate
    )
    monkeypatch.setattr(
        "soma.benchmarks.croma.CromaBenchmark.build_config", fake_build_config
    )
    monkeypatch.setattr(
        "soma.benchmarks.croma.CromaBenchmark.score",
        lambda _self, _run_dir: {
            "test/croma_median": 0.31,
            "test/croma_f0": 0.12,
            "test/croma_ltm10": -0.04,
        },
    )
    monkeypatch.setattr(
        cli, "Pipeline", lambda _config: type("P", (), {"run": lambda _s: None})()
    )
    prepared = tmp_path / "prepared"
    common_cache = tmp_path / "feature-cache"

    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            [
                "reproduce",
                "croma",
                "--raw-root",
                str(prepared),
                "--output-root",
                str(tmp_path / "runs"),
                "--cache-root",
                str(common_cache),
            ]
        )

    assert exit_info.value.code == 0
    return prepared, common_cache, curated_calls, config_calls


def test_croma_family_uses_the_prepared_family_root(tmp_path, monkeypatch):
    prepared, _common_cache, curated_calls, _config_calls = _run_croma_family(
        tmp_path, monkeypatch
    )

    assert curated_calls == [
        ("camelyon", prepared, prepared / "curated" / "camelyon"),
        ("tcga-4x4", prepared, prepared / "curated" / "tcga-4x4"),
        ("tolkach-esca", prepared, prepared / "curated" / "tolkach-esca"),
    ]


def test_croma_family_passes_one_explicit_cache_root_to_every_cohort(
    tmp_path, monkeypatch
):
    _prepared, common_cache, _curated_calls, config_calls = _run_croma_family(
        tmp_path, monkeypatch
    )

    assert [cohort for cohort, _kwargs in config_calls] == [
        "camelyon",
        "tcga-4x4",
        "tolkach-esca",
    ]
    assert {
        kwargs["overrides"]["cache"]["root_dir"] for _cohort, kwargs in config_calls
    } == {str(common_cache)}
