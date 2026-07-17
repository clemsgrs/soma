"""Unit tests for the ``detection-benchmark`` encoder-ranking harness (issue #246).

All cold on synthetic fixtures — no real data, no GPU, no model downloads (same pattern as
the curator / scorer tests). The GPU orchestration (extract, train, live per-sample decode)
in ``examples/detection_benchmark/campaign.py`` is not exercised here; its pure seams
(replicate resolution, cell planning, skip guards, aggregate-from-disk) are.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from soma.benchmarks import get_benchmark
from soma.benchmarks.detection_benchmark import (
    DATASET_ORDER,
    DEFAULT_ROSTER,
    Cell,
    CellPredictions,
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
    replicate_plan,
    score_dataset_points,
    select_subsets,
    write_cell_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "examples" / "detection_benchmark" / "campaign.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("db_campaign", DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# --- registration + build_config -----------------------------------------------------


def test_registered_under_name():
    bench = get_benchmark("detection-benchmark")
    assert isinstance(bench, DetectionBenchmark)
    assert bench.name == "detection-benchmark"
    assert bench.datasets == ("ocelot", "midog", "monkey")
    assert bench.facet.varied == ("encoder", "dataset")
    assert bench.facet.fixed["decoder"] == "lightweight_conv"


@pytest.mark.parametrize(
    "dataset,num_classes,metric",
    [("ocelot", 2, "mean_f1"), ("midog", 1, "f1"), ("monkey", 2, "mean_froc")],
)
def test_build_config_resolves_encoder_dataset(dataset, num_classes, metric):
    bench = get_benchmark("detection-benchmark")
    cfg = bench.build_config(
        encoder="uni2", dataset=dataset,
        dataset_csv="/x/d.csv", splits_csv="/x/s.csv", output_root="/out", seed=5,
    )
    assert cfg.dataset_type == "detection"
    assert cfg.encoder.name == "uni2"  # roster encoder swapped into the committed base
    assert cfg.decoder.name == "lightweight_conv"
    assert cfg.task.params["num_classes"] == num_classes
    assert str(cfg.dataset_csv) == "/x/d.csv"
    assert str(cfg.output_root) == "/out"
    assert cfg.training.seed == 5
    assert bench.metric_for(dataset) == metric


def test_build_config_is_roster_size_agnostic():
    # An encoder not in DEFAULT_ROSTER still resolves (extraction is per-encoder additive).
    cfg = get_benchmark("detection-benchmark").build_config(encoder="phikon", dataset="ocelot")
    assert cfg.encoder.name == "phikon"


def test_midog_config_carries_relaxed_tolerance():
    cfg = get_benchmark("detection-benchmark").build_config(encoder="virchow2", dataset="midog")
    assert cfg.preprocessing.tolerance == pytest.approx(0.10)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        get_benchmark("detection-benchmark").build_config(encoder="uni2", dataset="pannuke")


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
    assert roi["level0_spacing"].tolist() == [0.25]

    tiled = pd.read_csv(manifest.dataset_csv).sort_values("tile_x")
    assert tiled[["tile_x", "tile_y"]].astype(int).values.tolist() == [[0, 0], [896, 0]]
    assert tiled["source_wsi"].tolist() == ["midog_001", "midog_001"]
    assert tiled[["roi_width", "roi_height"]].astype(int).values.tolist() == [
        [1920, 1024], [1920, 1024]
    ]
    assert tiled["level0_spacing"].tolist() == [0.25, 0.25]
    assert [Image.open(path).size for path in tiled["image_path"]] == [(1024, 1024), (1024, 1024)]
    locals_ = [pd.read_csv(path).values.tolist() for path in tiled["points_path"]]
    assert locals_ == [[[950.0, 100.0, 0.0]], [[54.0, 100.0, 0.0]]]

    summary = json.loads(manifest.summary_json.read_text())
    assert {k: summary[k] for k in ("tile_size", "overlap", "level0_spacing")} == {
        "tile_size": 1024,
        "overlap": 128,
        "level0_spacing": 0.25,
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


def test_score_dataset_points_monkey_returns_mean_froc():
    samples = [
        SamplePrediction("s0", [[10, 10], [30, 30]], [0.9, 0.8], [0, 1],
                         [[10, 11], [30, 31]], [0, 1], area_mm2=0.05),
    ]
    out = score_dataset_points("monkey", samples)
    assert "mean_froc" in out and 0.0 <= out["mean_froc"] <= 1.0


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
    metrics = get_benchmark("detection-benchmark").score(tmp_path)
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
    assert metric_by_ds == {"ocelot": "mean_f1", "midog": "f1", "monkey": "mean_froc"}
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


@pytest.mark.parametrize("dataset,metric", [("midog", "f1"), ("monkey", "mean_froc")])
def test_reference_scaffold_shape(dataset, metric):
    with resources.files("soma.benchmarks.reference").joinpath(f"{dataset}.csv").open(newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames
        rows = list(reader)
    for col in ("metric", "expected", "tolerance", "kind", "label", "url", "source"):
        assert col in columns
    external = [r for r in rows if (r.get("kind") or "").strip() == "external"]
    gate = [r for r in rows if (r.get("kind") or "gate").strip() != "external"]
    assert len(gate) == 1 and external
    assert all(r["metric"] == metric for r in rows)
    for row in external:
        assert row["url"].strip().startswith("http")
        assert "TODO" in row["source"]  # clearly-marked placeholder, no invented numbers
    assert "TODO" in gate[0]["source"]


# --- driver: replicate resolution, planning, skip guards, aggregate-from-disk --------


def _write_splits(path: Path, folds: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if folds == 1:
        df = pd.DataFrame(
            {"sample_id": ["s0", "s1", "s2"], "split": ["train", "tune", "test"], "fold": [0, 0, 0]}
        )
    else:
        rows = []
        for f in range(folds):
            rows += [
                {"sample_id": "s0", "split": "train", "fold": f},
                {"sample_id": "s1", "split": "tune", "fold": f},
                {"sample_id": "s2", "split": "test", "fold": f},
            ]
        df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_driver_replicate_ids_single_and_multifold(tmp_path: Path):
    m = _load_driver()
    _write_splits(tmp_path / "single" / "curated" / "splits.csv", folds=1)
    _write_splits(tmp_path / "multi" / "curated" / "splits.csv", folds=4)
    assert m.dataset_replicate_ids(tmp_path / "single" / "curated" / "splits.csv", seeds=[0, 1, 2]) == (
        "seeds", [0, 1, 2],
    )
    assert m.dataset_replicate_ids(tmp_path / "multi" / "curated" / "splits.csv") == ("folds", [0, 1, 2, 3])


def test_driver_plan_cells_is_roster_size_agnostic(tmp_path: Path):
    m = _load_driver()
    data_root = tmp_path / "data"
    _write_splits(data_root / "ocelot" / "curated" / "splits.csv", folds=1)  # seeds
    _write_splits(data_root / "monkey" / "curated" / "splits.csv", folds=2)  # folds
    roster = (RosterEntry("virchow2"), RosterEntry("uni2"))
    plan = m.plan_cells(roster, ["ocelot", "monkey"], data_root, seeds=[0, 1, 2])
    # 2 encoders × (3 ocelot seeds + 2 monkey folds) = 10 cells.
    assert len(plan) == 10
    axes = {(c["dataset"], c["replicate_axis"]) for c in plan}
    assert ("ocelot", "seeds") in axes and ("monkey", "folds") in axes


def test_driver_skip_guards(tmp_path: Path):
    m = _load_driver()
    out = tmp_path / "out"
    assert not m.training_done(out, "ocelot", "uni2", 0)
    assert not m.metrics_exists(out, "ocelot", "uni2", 0)
    cd = m.cell_dir(out, "ocelot", "uni2", 0)
    cd.mkdir(parents=True)
    (cd / "metrics.json").write_text("{}")
    assert m.metrics_exists(out, "ocelot", "uni2", 0)

    # soma writes runs to <output_root>/experiments/<key>/runs/<ts>/ — NOT to output_root
    # itself. A guard that probes cell_dir/best_model.pt never fires against the real layout.
    run = cd / "experiments" / "dataset-uni2-slide-detection_abc123" / "runs" / "2026-01-01__local"
    run.mkdir(parents=True)
    run.joinpath("best_model.pt").write_text("x")
    # best_model.pt is rewritten on every epoch improvement, so it is present mid-training:
    # a half-trained (e.g. crashed) cell must NOT count as done, or phase 2 skips training it
    # and scores a partial model.
    assert not m.training_done(out, "ocelot", "uni2", 0)

    # summary.json is written once, at the end of the run — the honest completion marker.
    run.joinpath("summary.json").write_text("{}")
    assert m.training_done(out, "ocelot", "uni2", 0)


def test_roi_threshold_sweep_uses_stitched_predictions_and_restores_head(monkeypatch):
    m = _load_driver()
    head = SimpleNamespace(
        score_threshold=[0.25],
        num_classes=1,
        delta_px=30.0,
        nms_distance_px=30.0,
        matching="hungarian",
        level0_spacing=0.25,
    )
    manifest = SimpleNamespace(
        samples={
            "roi_t0": SimpleNamespace(
                metadata={
                    "source_wsi": "roi", "tile_x": 0, "tile_y": 0,
                    "roi_width": 512, "roi_height": 512,
                }
            ),
            "roi_t1": SimpleNamespace(
                metadata={
                    "source_wsi": "roi", "tile_x": 100, "tile_y": 0,
                    "roi_width": 512, "roi_height": 512,
                }
            ),
        }
    )
    tiles = [
        SamplePrediction(
            "roi_t0",
            pred_xy=[[100.0, 100.0], [300.0, 300.0]], pred_score=[0.9, 0.8], pred_class=[0, 0],
            gt_xy=[[100.0, 100.0]], gt_class=[0], matched=[True, False],
        ),
        SamplePrediction(
            "roi_t1",
            pred_xy=[[0.0, 100.0]], pred_score=[0.1], pred_class=[0],
            gt_xy=[[0.0, 100.0]], gt_class=[0], matched=[True],
        ),
    ]
    observed_thresholds = []

    def decode(*_args):
        observed_thresholds.append(head.score_threshold)
        from soma.benchmarks.detection_benchmark import stitch_tiles_to_rois

        return stitch_tiles_to_rois(tiles, manifest, head)

    monkeypatch.setattr(m, "_decode_split_points", decode)

    from soma.detection.matching import sweep_score_thresholds

    tile_threshold = sweep_score_thresholds(
        [np.asarray(s.pred_xy) for s in tiles],
        [np.asarray(s.pred_class) for s in tiles],
        [np.asarray(s.pred_score) for s in tiles],
        [np.asarray(s.gt_xy) for s in tiles],
        [np.asarray(s.gt_class) for s in tiles],
        num_classes=1,
        delta=30.0,
        method="hungarian",
    )

    threshold = m._sweep_thresholds_on_rois(object(), object(), head, object(), manifest)

    assert tile_threshold == [0.1]  # overlap copy wins on the old per-tile surrogate
    assert threshold == [0.9]
    assert observed_thresholds == [0.0]
    assert head.score_threshold == [0.25]


def test_score_cell_persists_roi_threshold_and_checkpoint_proxy(tmp_path: Path, monkeypatch):
    m = _load_driver()
    sample = lambda sid: SamplePrediction(  # noqa: E731
        sid, [[10.0, 10.0]], [0.9], [0], [[10.0, 10.0]], [0], [True]
    )
    tune = CellPredictions("uni2", "midog", 2, "f1", 0.25, [sample("tune_roi")])
    test = CellPredictions("uni2", "midog", 2, "f1", 0.25, [sample("test_roi")])
    monkeypatch.setattr(m, "_decode_cell_points", lambda *a, **k: (tune, test, [0.73]))

    out = tmp_path / "out"
    m.score_cell("uni2", "midog", 2, "seeds", tmp_path / "data", out)

    metrics = json.loads((m.cell_dir(out, "midog", "uni2", 2) / "metrics.json").read_text())
    assert metrics == {
        "test": {"f1": 1.0},
        "tune": {"f1": 1.0},
        "replicate_axis": "seeds",
        "test_source": "local_holdout",
        "metric_name": "f1",
        "score_threshold_per_class": [0.73],
        "score_threshold_selection_frame": "stitched_roi",
        "checkpoint_selection_frame": "tile_proxy",
    }


def test_score_cell_marks_ocelot_selection_as_native_sample_frame(tmp_path: Path, monkeypatch):
    m = _load_driver()
    sample = SamplePrediction(
        "image",
        pred_xy=[[10.0, 10.0], [20.0, 20.0]], pred_score=[0.9, 0.8], pred_class=[0, 1],
        gt_xy=[[10.0, 10.0], [20.0, 20.0]], gt_class=[0, 1], matched=[True, True],
    )
    predictions = CellPredictions("uni2", "ocelot", 0, "mean_f1", 0.2, [sample])
    monkeypatch.setattr(
        m, "_decode_cell_points", lambda *a, **k: (predictions, predictions, [0.6, 0.7])
    )

    out = tmp_path / "out"
    m.score_cell("uni2", "ocelot", 0, "seeds", tmp_path / "data", out)

    metrics = json.loads((m.cell_dir(out, "ocelot", "uni2", 0) / "metrics.json").read_text())
    assert metrics["score_threshold_selection_frame"] == "sample_native"
    assert metrics["checkpoint_selection_frame"] == "sample_native"


def test_rank_scores_out_of_process(tmp_path: Path, monkeypatch):
    """``run_rank`` must score in a child process, and be able to parse its own command back.

    Scoring loads the decoder + dense grids onto the GPU. Run in-process it strands those GiB
    in the driver for its whole life (PyTorch's allocator does not hand them back), and since
    the launcher pins the driver and every cell it trains to one GPU, the *next* ``train_cell``
    starts several GiB down and OOMs on a 12 GB card. Scoring must therefore shell out, exactly
    as training does. The round-trip half matters too: the parent builds the argv and the child
    parses it, so a drift between the two only shows up as a crash mid-sweep.
    """
    m = _load_driver()
    cmds: list[list[str]] = []
    data_root, out_root = tmp_path / "data", tmp_path / "out"
    monkeypatch.setattr(m, "plan_cells", lambda *a, **k: [
        {"encoder": "uni2", "dataset": "ocelot", "replicate": 1, "replicate_axis": "seeds"}
    ])
    monkeypatch.setattr(m, "metrics_exists", lambda *a: False)
    monkeypatch.setattr(m, "training_done", lambda *a: True)  # trained already -> score only
    monkeypatch.setattr(m, "aggregate_and_report", lambda *a, **k: {})
    monkeypatch.setattr(m, "train_cell", lambda *a: pytest.fail("must not retrain a done cell"))
    monkeypatch.setattr(m, "score_cell", lambda *a: pytest.fail("scoring must not run in-process"))
    monkeypatch.setattr(m, "_run", lambda cmd, **kw: cmds.append([str(c) for c in cmd]))

    m.run_rank(data_root, out_root, DEFAULT_ROSTER[:1], ["ocelot"], [1],
               dry_run=False, git_sha=None)

    assert len(cmds) == 1, "the one cell must be scored by exactly one subprocess"
    cmd = cmds[0]
    assert cmd[1].endswith("campaign.py") and cmd[2] == "score"

    # Now feed that argv back through the script's own parser: it must reconstitute the cell.
    scored: list[tuple] = []
    monkeypatch.setattr(m, "score_cell", lambda *a: scored.append(a))
    assert m.main(cmd[2:]) == 0
    assert scored == [("uni2", "ocelot", 1, "seeds", data_root, out_root)]


def _fabricate_scored_cell(m, out, dataset, encoder, replicate, axis, test_val):
    cd = m.cell_dir(out, dataset, encoder, replicate)
    cd.mkdir(parents=True)
    metric = dataset_spec(dataset).metric_name
    (cd / "metrics.json").write_text(json.dumps({
        "test": {metric: test_val}, "tune": {metric: test_val + 0.01},
        "replicate_axis": axis, "test_source": "local_holdout",
    }))
    samples = [SamplePrediction(f"s{i}", [[10 + i, 10]], [0.9], [0], [[10 + i, 10.1]], [0]) for i in range(3)]
    write_cell_predictions(cd / "predictions.json",
                           CellPredictions(encoder, dataset, replicate, metric, dataset_spec(dataset).spacing_um, samples))


def test_driver_aggregate_from_disk_singlefold(tmp_path: Path):
    m = _load_driver()
    out = tmp_path / "out"
    roster = (RosterEntry("virchow2"), RosterEntry("uni2"), RosterEntry("dinov2-vitb14", is_compact=True, is_control=True))
    vals = {"virchow2": 0.72, "uni2": 0.66, "dinov2-vitb14": 0.50}
    for enc, v in vals.items():
        for rid in (0, 1, 2):  # seeds axis
            _fabricate_scored_cell(m, out, "ocelot", enc, rid, "seeds", v + rid * 0.003)
    report = m.aggregate_and_report(out, roster=roster, datasets=["ocelot"], write=True)
    # One cell per encoder, aggregated over 3 seed replicates.
    assert {c["encoder"] for c in report["cells"]} == set(vals)
    assert all(c["n_replicates"] == 3 and c["replicate_axis"] == "seeds" for c in report["cells"])
    ranks = {r["encoder"]: r["rank"] for r in report["ranking"]["per_dataset"]["ocelot"]["test"]}
    assert ranks["virchow2"] == 1 and ranks["dinov2-vitb14"] == 3
    assert (out / "ranking_report.json").is_file()
    # The stability bootstrap ran purely off the persisted predictions (no training).
    assert "ocelot" in report["ranking"]["stability"]


def test_driver_aggregate_from_disk_multifold(tmp_path: Path):
    m = _load_driver()
    out = tmp_path / "out"
    roster = (RosterEntry("virchow2"), RosterEntry("uni2"))
    for enc, v in {"virchow2": 0.6, "uni2": 0.5}.items():
        for fold in (0, 1, 2):  # folds axis
            _fabricate_scored_cell(m, out, "monkey", enc, fold, "folds", v + fold * 0.01)
    report = m.aggregate_and_report(out, roster=roster, datasets=["monkey"], write=False)
    cell = next(c for c in report["cells"] if c["encoder"] == "virchow2")
    assert cell["replicate_axis"] == "folds" and cell["n_replicates"] == 3
    assert cell["metric_name"] == "mean_froc"


def test_driver_aggregate_skips_unscored_pairs(tmp_path: Path):
    m = _load_driver()
    out = tmp_path / "out"
    roster = (RosterEntry("virchow2"), RosterEntry("uni2"))
    _fabricate_scored_cell(m, out, "ocelot", "virchow2", 0, "seeds", 0.7)  # only one encoder scored
    report = m.aggregate_and_report(out, roster=roster, datasets=["ocelot"], write=False)
    assert {c["encoder"] for c in report["cells"]} == {"virchow2"}  # uni2 has no metrics → skipped
