"""Unit tests for the annotation-efficiency ladder (issue #237).

Cold on synthetic fixtures — no GPU, no real data. Covers the pure protocol math in
``examples/detection_benchmark/efficiency.py`` (study logic, kept out of ``soma/``) and the
driver's pure seams (split variants, rung
roots, full-rung references, aggregation from disk) in
``examples/detection_benchmark/campaign.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DRIVER = REPO_ROOT / "examples" / "detection_benchmark" / "campaign.py"
EFFICIENCY = REPO_ROOT / "examples" / "detection_benchmark" / "efficiency.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Study logic beside the driver (not a soma package): import it from its directory.
if str(EFFICIENCY.parent) not in sys.path:
    sys.path.insert(0, str(EFFICIENCY.parent))
from efficiency import (  # noqa: E402
    LADDERS, FullReference, RungResult, atom_column_for, build_dataset_efficiency,
    build_efficiency_report, cross_dataset_nstar, curve_points, ladder_for, n_star,
    nested_prefix, noise_band_crossing, plot_learning_curves, rank_crossing, rank_of,
    recipe_scale, scaled_recipe, train_atoms, write_split_variant,
)


def _load_driver():
    return _load("db_campaign_eff", DRIVER)


# --- fixtures ------------------------------------------------------------------------


def _write_tiled_dataset(root: Path, *, rois: int = 12, tiles_per_roi: int = 3, points_per_tile: int = 2):
    """A MIDOG-shaped curated manifest: ROIs -> tiles, one points CSV per tile."""
    curated = root / "midog" / "curated"
    (curated / "points").mkdir(parents=True)
    rows, splits = [], []
    for r in range(rois):
        roi = f"roi_{r:03d}"
        split = "train" if r < rois - 4 else ("tune" if r < rois - 2 else "test")
        for t in range(tiles_per_roi):
            sid = f"{roi}_t{t}"
            pts = curated / "points" / f"{sid}.csv"
            pd.DataFrame({"x": range(points_per_tile), "y": range(points_per_tile)}).to_csv(pts, index=False)
            rows.append({"sample_id": sid, "image_path": f"{sid}.png", "points_path": str(pts), "source_wsi": roi})
            splits.append({"sample_id": sid, "split": split, "fold": 0})
    pd.DataFrame(rows).to_csv(curated / "dataset.csv", index=False)
    pd.DataFrame(splits).to_csv(curated / "splits.csv", index=False)
    return curated / "dataset.csv", curated / "splits.csv"


def _write_flat_dataset(root: Path, *, n_train: int = 20):
    """An OCELOT-shaped manifest: the sample is the atom, no points files."""
    curated = root / "ocelot" / "curated"
    curated.mkdir(parents=True)
    ids = [f"s{i:03d}" for i in range(n_train + 6)]
    pd.DataFrame({"sample_id": ids, "image_path": [f"{i}.jpg" for i in ids]}).to_csv(curated / "dataset.csv", index=False)
    split = ["train"] * n_train + ["tune"] * 3 + ["test"] * 3
    pd.DataFrame({"sample_id": ids, "split": split, "fold": 0}).to_csv(curated / "splits.csv", index=False)
    return curated / "dataset.csv", curated / "splits.csv"


# --- protocol constants + nesting ----------------------------------------------------


def test_ladders_are_descending_and_atoms_known():
    for dataset, ladder in LADDERS.items():
        assert list(ladder) == sorted(ladder, reverse=True)
        assert atom_column_for(dataset)
    assert ladder_for("ocelot")[0] == 256 and ladder_for("midog")[-1] == 4
    with pytest.raises(KeyError):
        ladder_for("nope")


def test_nested_prefix_is_nested_deterministic_and_seed_dependent():
    atoms = [f"a{i}" for i in range(50)]
    small, big = nested_prefix(atoms, 3, 8), nested_prefix(atoms, 3, 32)
    assert set(small) <= set(big) and len(small) == 8 and len(big) == 32
    assert nested_prefix(atoms, 3, 8) == small  # deterministic
    assert nested_prefix(list(reversed(atoms)), 3, 8) == small  # order-independent (sorted)
    assert nested_prefix(atoms, 4, 8) != small


def test_train_atoms_roi_and_sample_atoms(tmp_path: Path):
    ds, sp = _write_tiled_dataset(tmp_path)
    assert len(train_atoms(ds, sp, "source_wsi")) == 8  # 12 rois - 4 held out
    ds2, sp2 = _write_flat_dataset(tmp_path)
    assert len(train_atoms(ds2, sp2, "sample_id")) == 20


# --- split variants ------------------------------------------------------------------


def test_split_variant_keeps_roi_tiles_together_and_tune_test_verbatim(tmp_path: Path):
    ds, sp = _write_tiled_dataset(tmp_path)
    out = tmp_path / "variants" / "seed0_n2.csv"
    v = write_split_variant(ds, sp, dataset="midog", seed=0, n=2, out_path=out)
    assert v.n_atoms == 2 and v.n_train_samples == 6 and v.n_train_objects == 12
    var = pd.read_csv(out)
    full = pd.read_csv(sp)
    # tune/test rows byte-identical (same ids, same order, fold column kept)
    pd.testing.assert_frame_equal(
        var[var.split != "train"].reset_index(drop=True),
        full[full.split != "train"].reset_index(drop=True),
    )
    # the paired manifest keeps exactly the variant's rows (pipeline needs every sample split)
    ds_var = pd.read_csv(v.dataset_path)
    assert v.dataset_path == out.with_suffix(".dataset.csv")
    assert set(ds_var.sample_id) == set(var.sample_id) and len(ds_var) == 6 + 12
    train_rois = {s.rsplit("_t", 1)[0] for s in var.loc[var.split == "train", "sample_id"]}
    assert train_rois == set(v.atoms) and len(train_rois) == 2
    # nested: the n=1 variant's ROI is inside the n=2 one
    v1 = write_split_variant(ds, sp, dataset="midog", seed=0, n=1, out_path=tmp_path / "v1.csv")
    assert set(v1.atoms) <= set(v.atoms)


def test_split_variant_rejects_rung_above_full(tmp_path: Path):
    ds, sp = _write_tiled_dataset(tmp_path)
    with pytest.raises(ValueError, match="exceeds"):
        write_split_variant(ds, sp, dataset="midog", seed=0, n=9, out_path=tmp_path / "x.csv")


def test_split_variant_flat_atoms_no_points(tmp_path: Path):
    ds, sp = _write_flat_dataset(tmp_path)
    v = write_split_variant(ds, sp, dataset="ocelot", seed=1, n=5, out_path=tmp_path / "v.csv")
    assert v.n_train_samples == 5 and v.n_train_objects is None
    assert (pd.read_csv(tmp_path / "v.csv").split == "train").sum() == 5


# --- recipe scaling ------------------------------------------------------------------


@pytest.mark.parametrize(
    "full_n,n,expected",
    [(400, 256, 1), (400, 64, 1), (400, 50, 1), (400, 32, 2), (400, 16, 4), (400, 8, 7),
     (212, 128, 1), (212, 32, 1), (212, 16, 2), (212, 8, 4), (212, 4, 7)],
)
def test_recipe_scale_floor_at_full_over_eight(full_n, n, expected):
    assert recipe_scale(full_n, n) == expected


def test_scaled_recipe_scales_epochs_and_patience_together():
    assert scaled_recipe(400, 8, epochs=50, patience=10) == {"epochs": 350, "patience": 70, "scale": 7}
    assert scaled_recipe(400, 256, epochs=50, patience=None) == {"epochs": 50, "scale": 1}
    with pytest.raises(ValueError):
        recipe_scale(0, 8)


# --- learning-curve math -------------------------------------------------------------


def _rung(enc, n, *vals):
    return RungResult(enc, n, tuple(vals))


def test_n_star_and_noise_band():
    full = FullReference("e", 400, mean=0.80, std=0.02, per_replicate=(0.78, 0.80, 0.82))
    rungs = [_rung("e", 8, 0.5), _rung("e", 32, 0.7), _rung("e", 128, 0.77), _rung("e", 256, 0.79)]
    pts = curve_points(rungs, full)
    assert pts[-1] == (400, 0.80)
    assert n_star(pts, 0.95 * 0.80) == 128  # 0.76 target -> first at 128
    assert n_star(pts, 0.99) is None
    assert noise_band_crossing(pts, full) == 256  # band floor 0.78


def test_rank_of_dense_descending_with_name_tiebreak():
    assert rank_of({"b": 0.5, "a": 0.5, "c": 0.9}) == {"c": 1, "a": 2, "b": 3}


def test_rank_crossing_reports_per_rung_agreement_and_first_full_order():
    full = {
        "top": FullReference("top", 100, 0.9, 0.01),
        "mid": FullReference("mid", 100, 0.8, 0.01),
        "low": FullReference("low", 100, 0.7, 0.01),
    }
    rungs = {
        "top": [_rung("top", 8, 0.3), _rung("top", 32, 0.7), _rung("top", 64, 0.85)],
        "mid": [_rung("mid", 8, 0.5), _rung("mid", 32, 0.6), _rung("mid", 64, 0.75)],
        "low": [_rung("low", 8, 0.4), _rung("low", 32, 0.5), _rung("low", 64, 0.6)],
    }
    rc = rank_crossing(rungs, full)
    assert rc["full_ranking"] == ["top", "mid", "low"]
    by_n = {e["n"]: e for e in rc["per_rung"]}
    assert by_n[8]["ranking"] == ["mid", "low", "top"] and not by_n[8]["full_order_holds"]
    assert by_n[32]["full_order_holds"] and by_n[64]["full_order_holds"]
    assert by_n[32]["spearman"] == 1.0
    assert rc["first_n_full_order"] == 32


def test_rank_crossing_skips_rungs_missing_an_encoder():
    full = {e: FullReference(e, 100, s, 0.01) for e, s in [("a", 0.9), ("b", 0.8), ("c", 0.7)]}
    rungs = {"a": [_rung("a", 8, 0.5), _rung("a", 32, 0.8)], "b": [_rung("b", 32, 0.7)], "c": [_rung("c", 32, 0.6)]}
    rc = rank_crossing(rungs, full)
    assert [e["n"] for e in rc["per_rung"]] == [32]


def test_build_dataset_efficiency_headline_and_global_columns():
    full = {"top": FullReference("top", 400, 0.8, 0.01), "low": FullReference("low", 400, 0.6, 0.01)}
    rungs = {
        "top": [_rung("top", 8, 0.5), _rung("top", 64, 0.77), _rung("top", 256, 0.79)],
        "low": [_rung("low", 8, 0.58), _rung("low", 64, 0.59), _rung("low", 256, 0.6)],
    }
    block = build_dataset_efficiency("ocelot", rungs, full)
    assert block["global_best"]["encoder"] == "top"
    assert block["encoders"]["top"]["n_star_own"] == 64
    assert block["encoders"]["low"]["n_star_own"] == 8       # 0.57 target, reached at 8
    assert block["encoders"]["low"]["n_star_global"] is None  # never reaches 0.76
    assert block["atom"] == "sample_id" and block["metric_name"] == "mean_f1"
    with pytest.raises(ValueError, match="no full-data references"):
        build_dataset_efficiency("ocelot", rungs, {})


def test_cross_dataset_nstar_agreement_and_report_plot(tmp_path: Path):
    def block(dataset, nstars, fulls):
        return {
            "dataset": dataset,
            "encoders": {
                e: {"n_star_own": n, "full": {"n": 100, "mean": f, "std": 0.01, "per_replicate": [], "source": "t"},
                    "rungs": [{"n": 8, "mean": f - 0.2, "std": 0.01, "n_replicates": 1, "per_replicate": [f - 0.2],
                               "n_train_objects": [None], "realized_epochs": [None]}]}
                for e, n, f in zip(["a", "b", "c"], nstars, fulls)
            },
            "metric_name": "f1", "atom": "x",
        }
    per = {"ocelot": block("ocelot", [8, 32, 128], [0.9, 0.8, 0.7]), "midog": block("midog", [16, 64, 128], [0.9, 0.8, 0.7])}
    cross = cross_dataset_nstar(per)
    assert cross["ranks"]["ocelot"] == {"a": 1, "b": 2, "c": 3} == cross["ranks"]["midog"]
    assert cross["pairs"]["ocelot|midog"]["spearman"] == 1.0
    report = build_efficiency_report(per, git_sha="abc", seeds=[0, 1, 2])
    assert report["cross_dataset"]["pairs"] and report["config"]["git_sha"] == "abc"
    assert list(report["datasets"]) == ["ocelot", "midog"]
    png = plot_learning_curves(report, tmp_path / "curves.png")
    assert png.is_file() and png.stat().st_size > 0


# --- driver seams --------------------------------------------------------------------


def test_driver_rung_root_and_variant_paths(tmp_path: Path):
    m = _load_driver()
    assert m.efficiency_rung_root(tmp_path, 32) == tmp_path / "n32"
    assert m.cell_dir(m.efficiency_rung_root(tmp_path, 32), "midog", "e", 1) == tmp_path / "n32" / "midog" / "e" / "replicate_1"
    assert m.efficiency_variant_path(tmp_path, "midog", 2, 8) == tmp_path / "splits" / "midog" / "seed2_n8.csv"


def test_driver_link_feature_cache_is_idempotent(tmp_path: Path):
    m = _load_driver()
    shared = tmp_path / "out" / "midog" / "feature_cache"
    shared.mkdir(parents=True)
    rung = m.efficiency_rung_root(tmp_path / "eff", 8)
    link = m.link_feature_cache(rung, "midog", shared)
    assert link.is_symlink() and link.resolve() == shared.resolve()
    assert m.link_feature_cache(rung, "midog", shared) == link


def test_driver_plan_variants_drops_rungs_at_or_above_full(tmp_path: Path):
    m = _load_driver()
    _write_tiled_dataset(tmp_path / "data")  # 8 train ROIs
    full_n, variants = m.plan_efficiency_variants(tmp_path / "data", tmp_path / "eff", "midog", [0, 1], ladder=[16, 8, 4, 2])
    assert full_n == 8
    assert sorted({v.n_atoms for v in variants}) == [2, 4]
    assert len(variants) == 4
    manifest = json.loads((tmp_path / "eff" / "splits" / "midog" / "variants.json").read_text())
    assert {(e["seed"], e["n_atoms"]) for e in manifest} == {(0, 2), (0, 4), (1, 2), (1, 4)}


def test_driver_base_recipe_reads_committed_config():
    m = _load_driver()
    assert m.base_recipe("midog") == {"epochs": 50, "patience": 10}


def _write_cell(root: Path, dataset: str, encoder: str, rep: int, value: float, epochs: int | None = None):
    d = root / dataset / encoder / f"replicate_{rep}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps({"test": {"f1": value}, "tune": {"f1": value}, "replicate_axis": "seeds"}))
    if epochs is not None:
        run = d / "experiments" / "exp" / "runs" / "r"
        run.mkdir(parents=True)
        (run / "training_history.json").write_text(json.dumps({"epochs": [{"epoch": i} for i in range(epochs)]}))


def test_driver_full_references_sweep_then_report_fallback(tmp_path: Path):
    m = _load_driver()
    roster = (m.RosterEntry("a"), m.RosterEntry("b"))
    full = tmp_path / "out"
    # report only
    full.mkdir()
    (full / "ranking_report.json").write_text(json.dumps({"cells": [
        {"encoder": "a", "dataset": "midog", "metric_name": "f1", "mean": 0.7, "std": 0.02, "per_replicate": [], "test_source": "recorded"},
        {"encoder": "a", "dataset": "ocelot", "metric_name": "mean_f1", "mean": 0.5, "std": 0.0, "per_replicate": []},
    ]}))
    refs = m.full_references(full, "midog", roster, full_n=212)
    assert set(refs) == {"a"} and refs["a"].source.startswith("report:") and refs["a"].n == 212
    # sweep cells take precedence under auto
    _write_cell(full, "midog", "b", 0, 0.8); _write_cell(full, "midog", "b", 1, 0.9)
    refs = m.full_references(full, "midog", roster, full_n=212)
    assert set(refs) == {"b"} and refs["b"].per_replicate == (0.8, 0.9) and refs["b"].source.startswith("sweep:")
    assert set(m.full_references(full, "midog", roster, full_n=212, source="report")) == {"a"}


def test_driver_collect_rungs_and_aggregate_report(tmp_path: Path):
    m = _load_driver()
    data = tmp_path / "data"; _write_tiled_dataset(data)  # 8 train ROIs
    eff, full = tmp_path / "eff", tmp_path / "out"
    roster = (m.RosterEntry("a"), m.RosterEntry("b"))
    for rep, v in enumerate([0.80, 0.82, 0.81]):
        _write_cell(full, "midog", "a", rep, v); _write_cell(full, "midog", "b", rep, v - 0.1)
    for n, va, vb in [(2, 0.5, 0.55), (4, 0.78, 0.6)]:
        for rep in (0, 1):
            _write_cell(m.efficiency_rung_root(eff, n), "midog", "a", rep, va, epochs=3)
            _write_cell(m.efficiency_rung_root(eff, n), "midog", "b", rep, vb, epochs=5)
    report = m.aggregate_efficiency(eff, data, full, roster, ["midog"], [0, 1], ladder=[4, 2], write=True)
    block = report["datasets"]["midog"]
    assert block["encoders"]["a"]["n_star_own"] == 4          # 0.95*0.81=0.77 -> reached at 4
    assert block["encoders"]["b"]["n_star_own"] == 8          # full=0.71 -> 0.6745 target; 0.55/0.6 miss, full rung reaches
    assert block["encoders"]["a"]["rungs"][0]["realized_epochs"] == [3, 3]
    assert block["encoders"]["a"]["rungs"][0]["n_train_objects"] == [12, 12]
    assert block["rank_crossing"]["per_rung"][0]["ranking"] == ["b", "a"]
    assert (eff / "efficiency_report.json").is_file() and (eff / "efficiency_curves.png").is_file()
