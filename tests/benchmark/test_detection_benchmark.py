"""Unit tests for the ``detection-benchmark`` encoder-ranking harness (issue #246).

All cold on synthetic fixtures — no real data, no GPU, no model downloads (same pattern as
the curator / scorer tests). The GPU orchestration (extract, train, live per-sample decode)
in ``examples/detection_benchmark/campaign.py`` is not exercised here; its pure seams
(replicate resolution, cell planning, skip guards, aggregate-from-disk) are.
"""

from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from soma.benchmark import get_benchmark
from soma.benchmark.registry import Benchmark
from soma.benchmark import detection
from soma.benchmark.detection import (
    DETECTION_BENCHMARK,
    DetectionDatasetBenchmark,
    Cell,
    CellPredictions,
    DatasetSpec,
    DetectionBenchmark,
    RosterEntry,
    SamplePrediction,
    aggregate_cell,
    aggregate_rank,
    bootstrap_rank_stability,
    build_ranking_report,
    dataset_spec,
    rank_consistency,
    rank_encoders,
    read_cell_predictions,
    register_dataset,
    registered_datasets,
    replicate_plan,
    resolve_stratify_maps,
    score_dataset_points,
    select_subsets,
    stratify_robustness,
    write_cell_predictions,
)


# --- registration + build_config -----------------------------------------------------


def test_registered_per_dataset_not_as_the_harness():
    # The multi-dataset harness object is not a registry benchmark (its curate takes a
    # dataset argument); the per-dataset views are, with encoder as the only varied axis.
    with pytest.raises(KeyError):
        get_benchmark("detection-benchmark")
    for dataset, metric in (("midog", "f1"), ("monkey", "froc_inflammatory")):
        bench = get_benchmark(f"detection/{dataset}")
        assert isinstance(bench, DetectionDatasetBenchmark)
        assert isinstance(bench, Benchmark)  # runtime protocol conformance
        assert bench.primary_metric == metric
        assert bench.facet.varied == ("encoder",)
        assert bench.facet.fixed["dataset"] == dataset
        assert bench.facet.fixed["decoder"] == "lightweight_conv"
    assert isinstance(DETECTION_BENCHMARK, DetectionBenchmark)
    assert DETECTION_BENCHMARK.datasets == ("ocelot", "midog", "monkey")


def test_the_harness_names_no_dataset_and_imports_no_scorer():
    # Each detection protocol registers its own DatasetSpec; the harness keeps no table of
    # datasets and imports none of the protocol packages (their scorers and curators).
    import ast

    tree = ast.parse(Path(detection.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    protocols = ("eva", "hest", "ocelot", "midog", "monkey", "croma")
    assert not {m for m in imported if m and m.startswith(tuple(f"soma.benchmark.{p}" for p in protocols))}
    assert not hasattr(detection, "DATASETS")
    assert not hasattr(detection, "DATASET_ORDER")


def test_each_detection_protocol_registers_its_dataset_with_its_own_scorer():
    assert registered_datasets() == ("ocelot", "midog", "monkey")
    for dataset in registered_datasets():
        spec = dataset_spec(dataset)
        assert spec.name == dataset
        assert spec.score.__module__.startswith(f"soma.benchmark.{dataset}")
        assert spec.curate.__module__.startswith(f"soma.benchmark.{dataset}")
        assert spec.config.is_file()
        assert spec.config.is_relative_to(Path(detection.__file__).parent / dataset)


def _toy_spec(**changes) -> DatasetSpec:
    fields = dict(
        name="toy",
        spacing_um=0.5,
        metric_name="toy_score",
        num_classes=1,
        match_um=(4.0,),
        match_method="greedy",
        tolerance=None,
        test_source="local_holdout",
        config=Path("toy.yaml"),
        score=lambda samples, spec: {"toy_score": float(len(samples)) * spec.spacing_um},
        curate=lambda raw_root, out_dir, spec: ("curated", str(raw_root), str(out_dir), spec.name),
    )
    return DatasetSpec(**{**fields, **changes})


def test_a_registered_dataset_is_scored_and_curated_by_its_spec(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(detection, "_DATASETS", dict(detection._DATASETS))
    register_dataset(_toy_spec())

    assert registered_datasets()[-1] == "toy"
    samples = [SamplePrediction("s0", [], [], [], [], []), SamplePrediction("s1", [], [], [], [], [])]
    assert score_dataset_points("toy", samples) == {"toy_score": 1.0}
    assert DETECTION_BENCHMARK.curate("raw", tmp_path, dataset="toy") == ("curated", "raw", str(tmp_path), "toy")
    assert DETECTION_BENCHMARK.metric_for("toy") == "toy_score"


def test_a_dataset_registers_once(monkeypatch):
    monkeypatch.setattr(detection, "_DATASETS", dict(detection._DATASETS))

    with pytest.raises(ValueError, match="already registered"):
        register_dataset(_toy_spec(name="midog"))


def test_per_dataset_view_delegates_to_the_harness(tmp_path: Path):
    bench = get_benchmark("detection/midog")
    cfg = bench.build_config(encoder="uni2", dataset_csv="/x/d.csv", output_root="/out", seed=3)
    assert cfg.features.encoder.name == "uni2" and cfg.learner.params["head"].params["num_classes"] == 1
    assert cfg.run.seed == 3 and str(cfg.cohort.records) == "/x/d.csv"
    # expected() reads midog/reference.csv (external anchors only, so any encoder matches).
    rows = bench.expected(encoder="uni2")
    assert rows and all(r.is_external for r in rows)
    write_cell_predictions(
        tmp_path / "predictions.json",
        CellPredictions("uni2", "midog", 0, "f1", 0.25,
                        [SamplePrediction("s0", [[10, 10]], [0.9], [0], [[10, 11]], [0])]),
    )
    assert bench.score(tmp_path)["f1"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "dataset,num_classes,metric",
    [("ocelot", 2, "mean_f1"), ("midog", 1, "f1"), ("monkey", 2, "froc_inflammatory")],
)
def test_build_config_resolves_encoder_dataset(dataset, num_classes, metric):
    bench = DETECTION_BENCHMARK
    cfg = bench.build_config(
        encoder="uni2", dataset=dataset,
        dataset_csv="/x/d.csv", splits_csv="/x/s.csv", output_root="/out", seed=5,
    )
    params = cfg.learner.params
    assert cfg.learner.name == "dense" and params["head"].name == "detection"
    assert cfg.features.encoder.name == "uni2"  # roster encoder swapped into the committed base
    assert params["decoder"].name == "lightweight_conv"
    assert params["head"].params["num_classes"] == num_classes
    assert str(cfg.cohort.records) == "/x/d.csv"
    assert str(cfg.run.output_dir) == "/out"
    assert cfg.run.seed == 5
    assert bench.metric_for(dataset) == metric


def test_build_config_is_roster_size_agnostic():
    # An encoder not in DEFAULT_ROSTER still resolves (extraction is per-encoder additive).
    cfg = DETECTION_BENCHMARK.build_config(encoder="phikon", dataset="ocelot")
    assert cfg.features.encoder.name == "phikon"


def test_midog_config_carries_relaxed_tolerance():
    cfg = DETECTION_BENCHMARK.build_config(encoder="virchow2", dataset="midog")
    from soma.pipeline.extraction import resolve_pipeline_preprocessing

    assert resolve_pipeline_preprocessing(cfg).tiling.tolerance == pytest.approx(0.10)
    assert cfg.features.dense.tolerance == pytest.approx(0.10)


@pytest.mark.parametrize("dataset", ["midog", "monkey"])
def test_tiled_datasets_store_the_dense_cache_in_fp16(dataset):
    # The tiled datasets' dense grids are large: fp16 halves the cache for the big encoders.
    cfg = DETECTION_BENCHMARK.build_config(encoder="virchow2", dataset=dataset)
    assert cfg.features.cache.dtype == "fp16"


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        DETECTION_BENCHMARK.build_config(encoder="uni2", dataset="pannuke")


def test_midog_benchmark_curate_emits_nominal_tiled_manifest(tmp_path: Path):
    raw = tmp_path / "raw"
    images = raw / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (1920, 1024), (127, 127, 127)).save(images / "001.png")
    (raw / "MIDOG2022_training_enriched.json").write_text(
        json.dumps(
            {
                "categories": [{"id": 1, "name": "mitotic figure"}],
                "images": [
                    {
                        "id": 1,
                        "file_name": "001.png",
                        "width": 1920,
                        "height": 1024,
                        "patient_id": "p1",
                        "tumortype": "breast",
                        "spacing": 0.23,
                    }
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [925, 75, 975, 125]}
                ],
            }
        )
    )
    out = tmp_path / "curated"

    manifest = DetectionBenchmark().curate(raw, out, dataset="midog")

    assert manifest.dataset_csv == out / "dataset.csv"
    assert manifest.splits_csv == out / "splits.csv"
    assert manifest.summary_json == out / "summary.json"
    assert all(
        (out / "roi" / name).is_file()
        for name in ("dataset.csv", "splits.csv", "summary.json")
    )
    roi = pd.read_csv(out / "roi" / "dataset.csv")
    assert roi["spacing_at_level_0"].tolist() == [0.25]

    tiled = pd.read_csv(manifest.dataset_csv).sort_values("tile_x")
    assert tiled[["tile_x", "tile_y"]].astype(int).values.tolist() == [[0, 0], [896, 0]]
    assert tiled["source_wsi"].tolist() == ["midog_001", "midog_001"]
    assert tiled[["roi_width", "roi_height"]].astype(int).values.tolist() == [
        [1920, 1024], [1920, 1024]
    ]
    assert tiled["spacing_at_level_0"].tolist() == [0.25, 0.25]
    assert [Image.open(path).size for path in tiled["image_path"]] == [(1024, 1024), (1024, 1024)]
    locals_ = [pd.read_csv(path).values.tolist() for path in tiled["points_path"]]
    assert locals_ == [[[950.0, 100.0, 0.0]], [[54.0, 100.0, 0.0]]]

    summary = json.loads(manifest.summary_json.read_text())
    assert {k: summary[k] for k in ("tile_size", "overlap", "spacing_at_level_0")} == {
        "tile_size": 1024,
        "overlap": 128,
        "spacing_at_level_0": 0.25,
    }


# --- replicate abstraction -----------------------------------------------------------


def test_replicate_plan_multifold_uses_folds():
    axis, ids = replicate_plan(5)
    assert axis == "folds"
    assert ids == [0, 1, 2, 3, 4]


def test_replicate_plan_singlefold_uses_seeds():
    axis, ids = replicate_plan(1, seeds=[0, 1, 2])
    assert axis == "seeds"
    assert ids == [0, 1, 2]


def test_replicate_plan_rejects_zero_folds_and_no_seeds():
    with pytest.raises(ValueError):
        replicate_plan(0)
    with pytest.raises(ValueError):
        replicate_plan(1, seeds=[])


# --- cell aggregation ----------------------------------------------------------------


def test_aggregate_cell_mean_std_and_tune():
    cell = aggregate_cell(
        "virchow2", "ocelot", [0.70, 0.72, 0.74],
        replicate_axis="seeds", tune_per_replicate=[0.71, 0.73, 0.75],
    )
    assert cell.metric_name == "mean_f1"
    assert cell.mean == pytest.approx(0.72)
    assert cell.std == pytest.approx(0.016329931618554516)
    assert cell.n_replicates == 3
    assert cell.replicate_axis == "seeds"
    assert cell.test_source == "local_holdout"
    assert cell.tune_mean == pytest.approx(0.73)
    d = cell.as_dict()
    assert d["metric_name"] == "mean_f1" and d["per_replicate"] == [0.70, 0.72, 0.74]


def test_aggregate_cell_single_replicate_zero_std():
    cell = aggregate_cell("uni2", "midog", [0.66], replicate_axis="folds")
    assert cell.std == 0.0 and cell.n_replicates == 1
    assert "tune_mean" not in cell.as_dict()  # no tune values → key omitted


# --- ranking + consistency -----------------------------------------------------------


def test_rank_encoders_orders_best_first_with_int_ranks():
    ranking = rank_encoders({"a": 0.7, "b": 0.5, "c": 0.6})
    assert [r["encoder"] for r in ranking] == ["a", "c", "b"]
    assert [r["rank"] for r in ranking] == [1, 2, 3]
    assert all(isinstance(r["rank"], int) for r in ranking)


def test_rank_encoders_ties_share_competition_rank():
    ranking = rank_encoders({"a": 0.7, "b": 0.7, "c": 0.5})
    ranks = {r["encoder"]: r["rank"] for r in ranking}
    assert ranks["a"] == 1 and ranks["b"] == 1  # tie → both rank 1
    assert ranks["c"] == 3  # competition ("min") ranking skips rank 2


def test_rank_consistency_spearman_kendall_and_pairs():
    per_dataset = {
        "ocelot": {"a": 1, "b": 2, "c": 3},
        "midog": {"a": 1, "b": 2, "c": 3},  # identical → perfect agreement
        "monkey": {"a": 3, "b": 2, "c": 1},  # reversed
    }
    out = rank_consistency(per_dataset, ["a", "b", "c"])
    assert out["n_encoders"] == 3
    assert out["pairs"]["ocelot|midog"]["spearman"] == pytest.approx(1.0)
    assert out["pairs"]["ocelot|monkey"]["spearman"] == pytest.approx(-1.0)
    assert -1.0 <= out["spearman"] <= 1.0


def test_rank_consistency_insufficient_overlap_is_none():
    out = rank_consistency({"ocelot": {"a": 1}, "midog": {"b": 1}}, ["a", "b"])
    assert out["pairs"]["ocelot|midog"]["spearman"] is None
    assert out["spearman"] is None


def test_aggregate_rank_averages_over_present_datasets():
    agg = aggregate_rank({"ocelot": {"a": 1, "b": 2}, "midog": {"a": 3}}, ["a", "b"])
    assert agg["a"] == pytest.approx(2.0)  # (1 + 3) / 2
    assert agg["b"] == pytest.approx(2.0)  # only present in ocelot


# --- frozen selections ---------------------------------------------------------------


def _six_roster():
    return (
        RosterEntry("a"),
        RosterEntry("b"),
        RosterEntry("c", is_compact=True),
        RosterEntry("d", is_compact=True, is_control=True),
        RosterEntry("e"),
        RosterEntry("f"),
    )


def test_select_subsets_rule():
    # aggregate order (best→worst): e, a, b, f, c, d
    per_dataset = {
        "ocelot": {"a": 2, "b": 3, "c": 5, "d": 6, "e": 1, "f": 4},
        "midog": {"a": 2, "b": 3, "c": 5, "d": 6, "e": 1, "f": 4},
    }
    sel = select_subsets(per_dataset, _six_roster(), backbone_top_k=3, efficiency_top_n=2)
    # backbone = top3 (e,a,b) ∪ compact (c,d)
    assert sel["backbone_subset"] == ["e", "a", "b", "c", "d"]
    # efficiency = top2 (e,a) ∪ mid-third (b,f) ∪ compact (c,d) ∪ control (d)
    assert sel["efficiency_subset"] == ["e", "a", "b", "f", "c", "d"]
    # compact + control always present in both.
    for name in ("c", "d"):
        assert name in sel["backbone_subset"] and name in sel["efficiency_subset"]
    assert "d" in sel["efficiency_subset"]  # control


def test_select_subsets_recomputes_for_smaller_roster():
    roster = (RosterEntry("a"), RosterEntry("b"), RosterEntry("c", is_compact=True))
    per_dataset = {"ocelot": {"a": 1, "b": 2, "c": 3}}
    sel = select_subsets(per_dataset, roster, backbone_top_k=1, efficiency_top_n=1)
    assert sel["backbone_subset"] == ["a", "c"]  # top1 ∪ compact
    assert "c" in sel["efficiency_subset"]


# --- per-sample prediction cache + native re-aggregation -----------------------------


def test_sample_prediction_roundtrip(tmp_path: Path):
    preds = CellPredictions(
        encoder="uni2", dataset="ocelot", replicate=1, metric_name="mean_f1", spacing_um=0.2,
        samples=[
            SamplePrediction("s0", [[10.0, 10.0]], [0.9], [0], [[10.2, 10.0]], [0], [True], 1.5),
            SamplePrediction("s1", [[20.0, 20.0]], [0.4], [1], [], [], [False], 1.5),
        ],
    )
    path = write_cell_predictions(tmp_path / "predictions.json", preds)
    loaded = read_cell_predictions(path)
    assert loaded.encoder == "uni2" and loaded.dataset == "ocelot" and loaded.replicate == 1
    assert [s.sample_id for s in loaded.samples] == ["s0", "s1"]
    assert loaded.samples[0].matched == [True]
    assert loaded.samples[0].area_mm2 == pytest.approx(1.5)


def test_score_dataset_points_ocelot():
    spec = dataset_spec("ocelot")
    # Two classes, both predictions land within δ=3µm/0.2 = 15px of their GT → perfect.
    samples = [
        SamplePrediction("s0", [[10, 10], [50, 50]], [0.9, 0.8], [0, 1],
                         [[10, 11], [50, 51]], [0, 1]),
    ]
    out = score_dataset_points("ocelot", samples, spec=spec)
    assert out["mean_f1"] == pytest.approx(1.0)


def test_score_dataset_points_midog():
    samples = [
        SamplePrediction("s0", [[10, 10]], [0.9], [0], [[10, 11]], [0]),  # within 7.5µm/0.25
        SamplePrediction("s1", [[200, 200]], [0.9], [0], [[10, 10]], [0]),  # far → FP + FN
    ]
    out = score_dataset_points("midog", samples)
    assert 0.0 <= out["f1"] <= 1.0
    assert out["f1"] == pytest.approx(0.5)  # tp=1, fp=1, fn=1 → 2·1/(2·1+1+1)


def test_score_dataset_points_monkey_returns_every_leaderboard_froc():
    # Final-leaderboard columns: the merged inflammatory-cell FROC (the headline) and the
    # two per-class FROCs, plus the class average kept for the Track-2 reference row.
    samples = [
        SamplePrediction("s0", [[10, 10], [30, 30]], [0.9, 0.8], [0, 1],
                         [[10, 11], [30, 31]], [0, 1], area_mm2=0.05),
    ]
    out = score_dataset_points("monkey", samples)
    assert set(out) == {"froc_inflammatory", "froc_lymphocytes", "froc_monocytes", "mean_froc"}
    assert all(0.0 <= v <= 1.0 for v in out.values())
    assert out["mean_froc"] == pytest.approx((out["froc_lymphocytes"] + out["froc_monocytes"]) / 2)
    assert dataset_spec("monkey").metric_name == "froc_inflammatory"


def test_score_dataset_points_monkey_headline_is_the_merged_class():
    # One lymphocyte found as a monocyte: wrong class, so both per-class FROCs miss it, but
    # the merged inflammatory-cell FROC (class-agnostic) counts it.
    samples = [
        SamplePrediction("s0", [[10, 10]], [0.9], [1], [[10, 11]], [0], area_mm2=0.05),
    ]
    out = score_dataset_points("monkey", samples)
    assert out["froc_inflammatory"] == pytest.approx(1.0)
    assert out["froc_lymphocytes"] == pytest.approx(0.0)


def test_reaggregation_off_cache_with_no_training(tmp_path: Path):
    """A subset re-aggregation (the deferred robustness stratification / bootstrap) runs off
    the persisted cache alone — no model, no retrain, no re-extract."""
    preds = CellPredictions(
        encoder="uni2", dataset="midog", replicate=0, metric_name="f1", spacing_um=0.25,
        samples=[
            SamplePrediction("hit", [[10, 10]], [0.9], [0], [[10, 11]], [0]),   # TP
            SamplePrediction("miss", [[500, 500]], [0.9], [0], [[10, 10]], [0]),  # FP + FN
        ],
    )
    path = write_cell_predictions(tmp_path / "predictions.json", preds)
    loaded = read_cell_predictions(path)
    # Stratify to the "hit" sample only → perfect F1, purely from the cache.
    stratum = [s for s in loaded.samples if s.sample_id == "hit"]
    assert score_dataset_points("midog", stratum)["f1"] == pytest.approx(1.0)
    # The full set is worse — proving the re-aggregation is real, not a constant.
    assert score_dataset_points("midog", loaded.samples)["f1"] < 1.0


def test_benchmark_score_reads_persisted_predictions(tmp_path: Path):
    write_cell_predictions(
        tmp_path / "predictions.json",
        CellPredictions("uni2", "ocelot", 0, "mean_f1", 0.2,
                        [SamplePrediction("s0", [[10, 10]], [0.9], [0], [[10, 11]], [0])]),
    )
    metrics = DETECTION_BENCHMARK.score(tmp_path)
    assert metrics["mean_f1"] == pytest.approx(1.0)


# --- bootstrap stability -------------------------------------------------------------


def test_bootstrap_rank_stability_paired_ci():
    # Encoder "good" beats "bad" on every sample → good is rank 1 in every resample.
    good = [SamplePrediction(f"s{i}", [[i, 0]], [0.9], [0], [[i, 0.1]], [0]) for i in range(6)]
    bad = [SamplePrediction(f"s{i}", [[i, 0]], [0.9], [0], [[i + 100, 0]], [0]) for i in range(6)]
    out = bootstrap_rank_stability(
        "midog", {"good": good, "bad": bad}, n_boot=200, seed=0
    )
    assert out["good"]["rank_median"] == 1.0
    assert out["good"]["rank_ci"] == [1.0, 1.0]
    assert out["bad"]["rank_median"] == 2.0
    assert out["good"]["n_boot"] == 200


def test_bootstrap_is_deterministic_for_a_seed():
    samples = {
        "a": [SamplePrediction(f"s{i}", [[i, 0]], [0.9], [0], [[i, 0.1]], [0]) for i in range(5)],
        "b": [SamplePrediction(f"s{i}", [[i, 0]], [0.5], [0], [[i, 5]], [0]) for i in range(5)],
    }
    a = bootstrap_rank_stability("midog", samples, n_boot=50, seed=7)
    b = bootstrap_rank_stability("midog", samples, n_boot=50, seed=7)
    assert a == b


# --- full ranking report -------------------------------------------------------------


def _report_cells():
    roster = _six_roster()
    means = {
        "ocelot": {"a": 0.70, "b": 0.66, "c": 0.55, "d": 0.50, "e": 0.72, "f": 0.60},
        "midog": {"a": 0.65, "b": 0.63, "c": 0.45, "d": 0.40, "e": 0.68, "f": 0.55},
        "monkey": {"a": 0.52, "b": 0.58, "c": 0.48, "d": 0.35, "e": 0.70, "f": 0.62},
    }
    cells = []
    for ds, per in means.items():
        for enc, val in per.items():
            cells.append(
                aggregate_cell(enc, ds, [val, val + 0.01], replicate_axis="seeds",
                               tune_per_replicate=[val + 0.02])
            )
    return roster, cells


def test_build_ranking_report_schema():
    roster, cells = _report_cells()
    report = build_ranking_report(cells, roster=roster, git_sha="deadbeef")
    # Top-level schema keys.
    assert set(report) == {
        "config", "cells", "ranking", "robustness", "reference_bands", "selections"
    }
    assert report["config"]["decoder"] == "lightweight_conv"
    assert report["config"]["git_sha"] == "deadbeef"
    assert report["config"]["encoder_list"] == [e.name for e in roster]
    assert set(report["config"]["per_dataset_spacing"]) == {"ocelot", "midog", "monkey"}
    # robustness deferred to #248 — key present, empty.
    assert report["robustness"] == {}
    # Every cell carries its OWN metric_name (no pooled cross-dataset scalar anywhere).
    metric_by_ds = {c["dataset"]: c["metric_name"] for c in report["cells"]}
    assert metric_by_ds == {"ocelot": "mean_f1", "midog": "f1", "monkey": "froc_inflammatory"}
    # No pooled scalar leaked into the report.
    assert "overall" not in report and "pooled" not in report


def test_report_emits_test_and_tune_ranks():
    roster, cells = _report_cells()
    report = build_ranking_report(cells, roster=roster)
    per_dataset = report["ranking"]["per_dataset"]
    for ds in ("ocelot", "midog", "monkey"):
        assert "test" in per_dataset[ds] and "tune" in per_dataset[ds]
        assert [r["rank"] for r in per_dataset[ds]["test"]] == [1, 2, 3, 4, 5, 6]


def test_report_rank_consistency_and_selections_present():
    roster, cells = _report_cells()
    report = build_ranking_report(cells, roster=roster)
    rc = report["ranking"]["rank_consistency"]
    assert rc["n_encoders"] == 6
    assert rc["spearman"] is not None and rc["kendall"] is not None
    assert set(rc["pairs"]) == {"ocelot|midog", "ocelot|monkey", "midog|monkey"}
    sel = report["selections"]
    assert set(sel) == {"backbone_subset", "efficiency_subset"}
    assert "d" in sel["efficiency_subset"]  # control always kept


def test_report_stability_populated_when_samples_given():
    roster, cells = _report_cells()
    stability_samples = {
        "midog": {
            "e": [SamplePrediction(f"s{i}", [[i, 0]], [0.9], [0], [[i, 0.1]], [0]) for i in range(4)],
            "d": [SamplePrediction(f"s{i}", [[i, 0]], [0.9], [0], [[i + 90, 0]], [0]) for i in range(4)],
        }
    }
    report = build_ranking_report(cells, roster=roster, stability_samples=stability_samples, n_boot=50)
    stab = report["ranking"]["stability"]["midog"]
    assert stab["e"]["rank_median"] == 1.0 and stab["d"]["rank_median"] == 2.0


def test_report_reference_bands_scaffolds():
    roster, cells = _report_cells()
    report = build_ranking_report(cells, roster=roster)
    bands = report["reference_bands"]
    assert set(bands) == {"ocelot", "midog", "monkey"}
    for ds in ("midog", "monkey"):
        kinds = {row["kind"] for row in bands[ds]}
        assert "external" in kinds  # non-gating guidance anchor present


# --- reference CSV scaffolds ---------------------------------------------------------


@pytest.mark.parametrize(
    "dataset,metrics,gated",
    [("midog", ("f1",), True), ("monkey", ("froc_inflammatory", "froc_lymphocytes", "froc_monocytes", "mean_froc"), False)],
)
def test_reference_scaffold_shape(dataset, metrics, gated):
    with resources.files(f"soma.benchmark.{dataset}").joinpath("reference.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames
        rows = list(reader)
    for col in ("metric", "expected", "tolerance", "kind", "label", "url", "source"):
        assert col in columns
    external = [r for r in rows if (r.get("kind") or "").strip() == "external"]
    gate = [r for r in rows if (r.get("kind") or "gate").strip() != "external"]
    if gated:
        # The measured local headline: fully keyed (dataset/encoder/spacing), a real
        # number with a real tolerance (load_reference enforces the same invariants).
        assert gate
        for row in gate:
            assert all((row.get(c) or "").strip() for c in ("dataset", "encoder", "spacing"))
            assert float(row["expected"]) > 0 and float(row["tolerance"]) > 0
    else:
        # No placeholder gate row: a gate is added only once the local sweep has measured
        # a headline (load_reference rejects expected-0 / blank-key gate rows).
        assert not gate
    assert external
    # Every row names a metric the native scorer emits, and the headline has a row.
    assert {r["metric"] for r in rows} <= set(metrics)
    assert any(r["metric"] == dataset_spec(dataset).metric_name for r in rows)
    for row in external:
        assert row["url"].strip().startswith("http")
        # Filled non-gating guidance: a real published number with a snapshot date.
        assert float(row["expected"]) > 0 and "TODO" not in row["source"]
        assert "Snapshotted" in row["source"]


# --- robustness stratification (#248) ------------------------------------------------


def _strat_sample(sid: str, matched: bool = True) -> SamplePrediction:
    """One-GT sample: a matching prediction (TP -> per-sample F1 1.0) or none (FN -> 0.0)."""
    if matched:
        return SamplePrediction(sid, [[100.0, 100.0]], [0.9], [0], [[101.0, 100.0]], [0], [True])
    return SamplePrediction(sid, [], [], [], [[101.0, 100.0]], [0], [])


def test_stratify_robustness_per_group_means_and_spread():
    groups = {"s0": "scannerA", "s1": "scannerA", "s2": "scannerB", "s3": "scannerB"}
    robust = [[_strat_sample(s) for s in groups] for _ in range(2)]  # perfect on both groups
    brittle = [  # perfect on scannerA, all-FN on scannerB
        [_strat_sample(s, matched=groups[s] == "scannerA") for s in groups] for _ in range(2)
    ]
    block = stratify_robustness(
        "midog", {"robust": robust, "brittle": brittle}, groups, stratify_by="scanner"
    )
    assert block["stratify_by"] == "scanner" and block["metric_name"] == "f1"
    assert block["group_sizes"] == {"scannerA": 2, "scannerB": 2}
    r = block["per_encoder"]["robust"]
    assert r["per_group"]["scannerA"]["mean"] == pytest.approx(1.0)
    assert r["per_group"]["scannerB"]["mean"] == pytest.approx(1.0)
    assert r["spread"] == pytest.approx(0.0)
    b = block["per_encoder"]["brittle"]
    assert b["per_group"]["scannerA"]["mean"] == pytest.approx(1.0)
    assert b["per_group"]["scannerB"]["mean"] == pytest.approx(0.0)
    assert b["spread"] == pytest.approx(1.0)
    assert b["best_group"] == "scannerA" and b["worst_group"] == "scannerB"
    assert b["per_group"]["scannerA"]["per_replicate"] == [1.0, 1.0]


def test_stratify_robustness_unmapped_sample_raises():
    samples = [[_strat_sample("s0"), _strat_sample("s9")]]
    with pytest.raises(ValueError, match="no 'scanner' group"):
        stratify_robustness("midog", {"e": samples}, {"s0": "A"}, stratify_by="scanner")


def test_resolve_stratify_maps_prefers_dedicated_axes(tmp_path: Path):
    p = tmp_path / "dataset.csv"
    p.write_text("sample_id,scanner,tumor_type,domain\nr1,S1,T1,T1\nr2,S2,T2,T2\n")
    axes = resolve_stratify_maps(p)
    # Both dedicated axes usable -> scanner headline + tumor_type secondary, no derived domain.
    assert set(axes) == {"scanner", "tumor_type"}
    assert axes["scanner"] == {"r1": "S1", "r2": "S2"}


def test_resolve_stratify_maps_falls_back_to_domain(tmp_path: Path):
    p = tmp_path / "dataset.csv"
    # scanner single-valued and tumor_type blank on one row -> both unusable; domain works.
    p.write_text("sample_id,scanner,tumor_type,domain\nr1,S1,T1,d1\nr2,S1,,d2\n")
    axes = resolve_stratify_maps(p)
    assert set(axes) == {"domain"}
    assert axes["domain"] == {"r1": "d1", "r2": "d2"}


def test_resolve_stratify_maps_single_group_yields_nothing(tmp_path: Path):
    p = tmp_path / "dataset.csv"
    p.write_text("sample_id,scanner,domain\nr1,S1,d1\nr2,S1,d1\n")
    assert resolve_stratify_maps(p) == {}


def test_resolve_stratify_maps_keys_tiled_rows_by_parent_roi(tmp_path: Path):
    p = tmp_path / "dataset.csv"
    p.write_text("sample_id,source_wsi,scanner\nr1_t0,r1,S1\nr1_t1,r1,S1\nr2_t0,r2,S2\n")
    assert resolve_stratify_maps(p)["scanner"] == {"r1": "S1", "r2": "S2"}


def test_resolve_stratify_maps_rejects_inconsistent_roi(tmp_path: Path):
    p = tmp_path / "dataset.csv"
    p.write_text("sample_id,source_wsi,scanner\nr1_t0,r1,S1\nr1_t1,r1,S2\nr2_t0,r2,S2\n")
    with pytest.raises(ValueError, match="inconsistent 'scanner'"):
        resolve_stratify_maps(p)


def test_report_carries_robustness_block():
    cell = Cell(
        encoder="a", dataset="midog", metric_name="f1", per_replicate=(0.5,), mean=0.5,
        std=0.0, n_replicates=1, replicate_axis="seeds", test_source="local_holdout",
    )
    block = {"midog": {"scanner": {"stratify_by": "scanner"}}}
    report = build_ranking_report([cell], roster=(RosterEntry("a"),), robustness=block)
    assert report["robustness"] == block
    # Omitting it keeps the empty-block schema.
    assert build_ranking_report([cell], roster=(RosterEntry("a"),))["robustness"] == {}


