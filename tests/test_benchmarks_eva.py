"""EVA promoted into the Benchmark registry as per-dataset sub-benchmarks (issue #219).

No EVA raw data or GPU here: the registry lookup, the per-dataset ``build_config`` recipe
fidelity (epochs mapping, lr/wd/batch, balanced-accuracy metric, tune_is_test split,
virchow2 CLS-only), the keyed ``reference/eva.csv`` per-row tolerance parse + axis
selection, the default ``summary.json`` scorer, and ``curate`` delegation are all
verifiable offline. The live curate→train→score reproduction (needs data + GPU) is what
the ``--from-run-dir`` CLI test exercises with a fixture summary.
"""

from __future__ import annotations

import csv
import json
import math
from importlib import resources
from pathlib import Path

import pandas as pd
import pytest

from soma.benchmarks import Benchmark, get_benchmark, list_benchmarks
from soma.benchmarks import eva
from soma.dataset import Dataset, Splits


EVA_DATASETS = ("bach", "breakhis", "crc", "mhist", "gleason_arvaniti", "patch_camelyon")


def _write_splits(tmp_path: Path, n_train: int, n_test: int = 3) -> Path:
    rows = [{"sample_id": f"tr{i}", "split": "train", "fold": 0} for i in range(n_train)]
    rows += [{"sample_id": f"te{i}", "split": "test", "fold": 0} for i in range(n_test)]
    path = tmp_path / "splits.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# --- registration ---------------------------------------------------------------------


def test_eva_subbenchmarks_registered():
    names = list_benchmarks()
    for dataset in EVA_DATASETS:
        name = f"eva/{dataset}"
        assert name in names, f"{name} not registered"
        bench = get_benchmark(name)
        assert bench.name == name
        assert bench.primary_metric == "test/balanced_accuracy"
        assert bench.canonical_seeds == (0, 1, 2, 3, 4)
        assert isinstance(bench, Benchmark)  # structural protocol conformance


def test_eva_facet_fixes_dataset_and_varies_encoder():
    facet = get_benchmark("eva/bach").facet
    assert facet.varied == ("encoder",)
    assert facet.fixed["dataset"] == "bach"


# --- epochs mapping (eva step budget) -------------------------------------------------


@pytest.mark.parametrize(
    "n_train, expected_epochs",
    [
        (268, 6250),  # bach
        (1132, 2500),  # breakhis
        (100_000, 32),  # crc
        (262_144, 13),  # patch_camelyon
    ],
)
def test_epochs_for_train_size_matches_eva_step_budget(n_train, expected_epochs):
    assert eva.epochs_for_train_size(n_train) == expected_epochs
    spe = math.ceil(n_train / eva.HEAD_BATCH_SIZE)
    assert eva.epochs_for_train_size(n_train) == max(1, math.ceil(eva.MAX_STEPS / spe))


def test_epochs_for_train_size_is_at_least_one_for_huge_datasets():
    assert eva.epochs_for_train_size(10_000_000) == 1


def test_epochs_for_train_size_rejects_nonpositive():
    with pytest.raises(ValueError):
        eva.epochs_for_train_size(0)


# --- build_config recipe fidelity -----------------------------------------------------


def test_build_config_encodes_eva_protocol(tmp_path):
    splits = _write_splits(tmp_path, n_train=268)
    config = get_benchmark("eva/bach").build_config(
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
        seed=3,
    )
    assert config.dataset_type == "tile"
    assert config.encoder.name == "uni2"
    assert config.encoder.output_variant is None  # uni2 uses the slide2vec default (1536-d CLS)
    assert config.task.name == "multiclass_classification"
    assert "balanced_accuracy" in config.evaluation.metrics

    training = config.training
    assert training.learning_rate == eva.LEARNING_RATE  # 3e-4
    assert training.weight_decay == eva.WEIGHT_DECAY  # 0.01, not 0.0
    assert training.optimizer == "adamw"
    assert training.scheduler == "none"
    assert training.batch_size == eva.HEAD_BATCH_SIZE  # 256
    assert training.monitor == "balanced_accuracy"
    assert training.monitor_mode == "max"
    assert training.seed == 3
    assert training.tune_is_test is True  # bach reports on the validation split
    assert training.epochs == 6250  # computed from n_train=268
    assert training.patience == 1250  # eva's per-dataset value


def test_build_config_pins_virchow2_cls_variant(tmp_path):
    splits = _write_splits(tmp_path, n_train=512)
    config = get_benchmark("eva/patch_camelyon").build_config(
        encoder="virchow2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
    )
    # eva's paige_virchow2 is CLS-only (1280-d); the 2560-d default would not match.
    assert config.encoder.output_variant == "cls"
    assert config.training.tune_is_test is False  # pcam has a real val + test split


def test_build_config_defaults_to_default_encoder(tmp_path):
    splits = _write_splits(tmp_path, n_train=268)
    config = get_benchmark("eva/bach").build_config(
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
    )
    assert config.encoder.name == eva.DEFAULT_ENCODER


def test_build_config_honours_smoke_overrides(tmp_path):
    splits = _write_splits(tmp_path, n_train=268)
    config = get_benchmark("eva/bach").build_config(
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
        epochs=1,
        patience=1,
        encoder_batch_size=8,
    )
    assert config.training.epochs == 1
    assert config.training.patience == 1
    assert config.encoder.batch_size == 8


def test_build_config_applies_cache_overrides(tmp_path):
    splits = _write_splits(tmp_path, n_train=268)
    config = get_benchmark("eva/bach").build_config(
        encoder="uni2",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
        epochs=1,
        overrides={"cache": {"enabled": True, "root_dir": str(tmp_path / "cache")}},
    )
    assert config.cache.enabled is True
    assert str(config.cache.root_dir) == str(tmp_path / "cache")


def test_build_config_accepts_encoder_without_packaged_reference(tmp_path):
    splits = _write_splits(tmp_path, n_train=10)
    config = get_benchmark("eva/bach").build_config(
        encoder="phikon",
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=splits,
        output_root=tmp_path / "runs",
        epochs=1,
    )
    assert config.encoder.name == "phikon"
    assert config.encoder.output_variant is None


# --- keyed reference table ------------------------------------------------------------


def test_reference_csv_is_keyed_schema():
    with resources.files("soma.benchmarks.reference").joinpath("eva.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames
        rows = list(reader)
    assert columns == [
        "dataset",
        "encoder",
        "metric",
        "expected",
        "tolerance",
        "tolerance_mode",
        "source",
    ]
    # Every row is keyed (dataset + encoder populated) with a per-row ±2% relative band.
    for row in rows:
        assert row["dataset"].strip()
        assert row["encoder"].strip()
        assert float(row["tolerance"]) > 0
        assert row["tolerance_mode"] == "relative"
    # Spot-check the known bach numbers (validated live in PR #87).
    by_key = {(r["dataset"], r["encoder"], r["metric"]): float(r["expected"]) for r in rows}
    assert by_key[("bach", "uni2", "test/balanced_accuracy")] == pytest.approx(0.915)
    assert by_key[("bach", "virchow2", "test/balanced_accuracy")] == pytest.approx(0.883)
    assert by_key[("crc", "uni2", "test/balanced_accuracy")] == pytest.approx(0.965)


def test_expected_selects_keyed_row_by_encoder():
    bach = get_benchmark("eva/bach")
    uni2 = bach.expected(encoder="uni2")
    assert len(uni2) == 1
    assert uni2[0].metric == "test/balanced_accuracy"
    assert uni2[0].expected == pytest.approx(0.915)
    assert uni2[0].tolerance > 0
    virchow2 = bach.expected(encoder="virchow2")
    assert virchow2[0].expected == pytest.approx(0.883)


def test_expected_defaults_to_default_encoder():
    # No encoder axis -> the benchmark's default encoder row (uni2 for bach).
    rows = get_benchmark("eva/bach").expected()
    assert len(rows) == 1
    assert rows[0].expected == pytest.approx(0.915)


def test_expected_patch_camelyon_reports_val_and_test():
    rows = get_benchmark("eva/patch_camelyon").expected(encoder="uni2")
    by_metric = {r.metric: r.expected for r in rows}
    assert by_metric["test/balanced_accuracy"] == pytest.approx(0.95)  # eva test column
    assert by_metric["tune/balanced_accuracy"] == pytest.approx(0.944)  # eva val column


# --- default scorer + curation delegation ---------------------------------------------


def test_score_uses_default_summary_scorer(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"test/balanced_accuracy": 0.91}))
    scored = get_benchmark("eva/bach").score(tmp_path)
    assert scored["test/balanced_accuracy"] == pytest.approx(0.91)


def test_curate_delegates_to_eva_curator(monkeypatch, tmp_path):
    calls = {}

    def _fake_curate(name, raw_root, output_dir, *, tune_fraction):
        calls["args"] = (name, str(raw_root), str(output_dir), tune_fraction)
        from soma.curation.manifest import CuratedManifest

        return CuratedManifest(
            dataset_csv=Path(output_dir) / "dataset.csv",
            splits_csv=Path(output_dir) / "splits.csv",
        )

    monkeypatch.setattr(eva, "curate_eva_patch_dataset", _fake_curate)
    manifest = get_benchmark("eva/bach").curate(tmp_path / "raw", tmp_path / "out")
    # Delegates with the dataset name and the tune-is-test curation fraction (0.0).
    assert calls["args"] == ("bach", str(tmp_path / "raw"), str(tmp_path / "out"), 0.0)
    assert manifest.dataset_csv == tmp_path / "out" / "dataset.csv"


def test_gleason_arvaniti_curation_is_compatible_with_its_tune_is_test_config(tmp_path):
    """Regression: the gleason manifest must load under the benchmark's own tune_is_test.

    Previously the curator emitted both a tune (ZT76 val) and a test (test_patches_750)
    split, which ``Splits`` rejects under ``tune_is_test=True`` — so the never-run
    ``eva/gleason_arvaniti`` reproduction would have failed at fold construction.
    """
    raw = tmp_path / "gleason_raw"
    for array_id, class_idx in (("ZT111", 0), ("ZT199", 1), ("ZT76", 2)):
        image = (
            raw
            / "train_validation_patches_750"
            / f"{array_id}_01_A_1_1"
            / f"{array_id}_01_A_1_1_patch_1_class_{class_idx}.jpg"
        )
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"")

    bench = get_benchmark("eva/gleason_arvaniti")
    manifest = bench.curate(raw, tmp_path / "curated")
    config = bench.build_config(
        dataset_csv=manifest.dataset_csv,
        splits_csv=manifest.splits_csv,
        output_root=tmp_path / "runs",
    )
    assert config.training.tune_is_test is True

    dataset = Dataset(manifest.dataset_csv)
    # Must not raise: single held-out split (EVA val ZT76) under tune_is_test.
    Splits(manifest.splits_csv, dataset, tune_is_test=config.training.tune_is_test)
