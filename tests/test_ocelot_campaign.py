"""Pure-logic tests for examples/ocelot/campaign.py (#152).

The GPU orchestration (train → greedy-rescore) needs the dataset + an encoder, so it is
not unit-tested; the testable seams are the seed aggregation, the model×magnification
interaction, the winner pick, the JSON extraction, and the markdown formatter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "examples" / "ocelot" / "campaign.py"


def _load_campaign():
    spec = importlib.util.spec_from_file_location("ocelot_campaign", CAMPAIGN)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module-level @dataclass can resolve its own __module__.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cells_cover_the_2x2_plus_anchor():
    m = _load_campaign()
    keys = {c.key for c in m.CELLS}
    assert keys == {"virchow2_0.20", "virchow2_0.25", "virchow2_0.50", "uni2_0.25", "uni2_0.50"}
    assert sum(c.is_anchor for c in m.CELLS) == 1
    assert m.ANCHOR.key == "virchow2_0.20"


def test_every_cell_uses_the_one_native_manifest(tmp_path: Path):
    m = _load_campaign()
    expected = [
        "--set",
        f"data.dataset_csv={tmp_path / 'curated' / 'dataset.csv'}",
        "--set",
        f"data.splits_csv={tmp_path / 'curated' / 'splits.csv'}",
    ]

    assert [m._data_overrides(cell, tmp_path) for cell in m.CELLS] == [
        expected,
        expected,
        expected,
        expected,
        expected,
    ]


def test_campaign_resolves_the_current_dense_image_cache_layout(monkeypatch, tmp_path: Path):
    import soma

    m = _load_campaign()
    curated = tmp_path / "curated"
    curated.mkdir()
    (curated / "dataset.csv").write_text(
        "sample_id,image_path,points_path,spacing_at_level_0\n"
        "train_001,/images/001.jpg,/points/001.csv,0.2\n"
    )
    (curated / "splits.csv").write_text(
        "sample_id,split,fold\ntrain_001,train,0\n"
    )
    feature_dir = (
        tmp_path
        / "feature_cache"
        / "dense_image"
        / "cache-key"
        / "dense_image_embeddings"
    )
    calls: list[dict] = []

    class _FakeFeatureExtractor:
        def __init__(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

        def extract(self):
            return SimpleNamespace(
                artifacts=SimpleNamespace(feature_dir=feature_dir)
            )

    monkeypatch.setattr(soma, "FeatureExtractor", _FakeFeatureExtractor)

    path = m.dense_embeddings_dir(m.ANCHOR, tmp_path)

    assert path is not None
    assert path.parts[-3] == "dense_image"
    assert path.name == "dense_image_embeddings"
    assert len(calls) == 1
    assert calls[0]["kwargs"]["output_root"].name == "rescore_extraction"


def test_score_run_reuses_the_trained_runs_saved_config(monkeypatch, tmp_path: Path):
    m = _load_campaign()
    run_subdir = tmp_path / "experiments" / "exp" / "runs" / "seed-0"
    run_subdir.mkdir(parents=True)
    saved_config = run_subdir / "config.yaml"
    saved_config.write_text("data:\n  dataset_csv: /chosen/curated/dataset.csv\n")
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return type(
            "Result", (), {"stdout": '{\n  "tune": {"mean_f1": 0.7}\n}\n'}
        )()

    monkeypatch.setattr(m, "_run", fake_run)

    assert m.score_run(m.ANCHOR, run_subdir, tune_only=True)["tune"] == {
        "mean_f1": 0.7
    }
    cmd = captured["cmd"]
    assert cmd[cmd.index("--config") + 1] == str(saved_config)


def test_extract_trailing_json():
    m = _load_campaign()
    stdout = "feature store: /x\n  [tune] mF1=0.71\n{\n  \"tune\": {\"mean_f1\": 0.71}\n}\n"
    assert m.extract_trailing_json(stdout)["tune"]["mean_f1"] == pytest.approx(0.71)


def test_extract_trailing_json_raises_without_json():
    m = _load_campaign()
    with pytest.raises(ValueError):
        m.extract_trailing_json("only text, no object\n")


def test_summarize_seed_metrics_mean_std_and_recall():
    m = _load_campaign()
    per_seed = [
        {"mean_f1": 0.70, "recall_class_0": 0.6, "recall_class_1": 0.8},
        {"mean_f1": 0.72, "recall_class_0": 0.7, "recall_class_1": 0.9},
        {"mean_f1": 0.74, "recall_class_0": 0.8, "recall_class_1": 1.0},
    ]
    s = m.summarize_seed_metrics(per_seed)
    assert s["n_seeds"] == 3
    assert s["mean_f1_mean"] == pytest.approx(0.72)
    assert s["mean_f1_std"] == pytest.approx(0.016329931618554516)  # population std
    assert s["recall_bc_mean"] == pytest.approx(0.7)
    assert s["recall_tc_mean"] == pytest.approx(0.9)
    assert s["mean_f1_per_seed"] == [0.70, 0.72, 0.74]


def test_summarize_seed_metrics_single_seed_zero_std():
    m = _load_campaign()
    s = m.summarize_seed_metrics([{"mean_f1": 0.6995, "recall_class_0": 0.5, "recall_class_1": 0.6}])
    assert s["mean_f1_mean"] == pytest.approx(0.6995)
    assert s["mean_f1_std"] == 0.0
    assert s["n_seeds"] == 1


def test_summarize_seed_metrics_empty_raises():
    m = _load_campaign()
    with pytest.raises(ValueError):
        m.summarize_seed_metrics([])


def test_magnification_interaction():
    m = _load_campaign()
    summaries = {
        "virchow2_0.25": {"mean_f1_mean": 0.70},
        "virchow2_0.50": {"mean_f1_mean": 0.64},  # finer helps V2 by +0.06
        "uni2_0.25": {"mean_f1_mean": 0.60},
        "uni2_0.50": {"mean_f1_mean": 0.62},       # finer hurts UNI2 by -0.02
    }
    inter = m.magnification_interaction(summaries)
    assert inter["virchow2"] == pytest.approx(0.06)
    assert inter["uni2"] == pytest.approx(-0.02)
    assert inter["interaction"] == pytest.approx(0.08)


def test_magnification_interaction_missing_cell_omits_encoder():
    m = _load_campaign()
    summaries = {"virchow2_0.25": {"mean_f1_mean": 0.70}}  # no 0.50 → no virchow2 delta
    assert m.magnification_interaction(summaries) == {}


def test_pick_winner_is_max_tune_mf1():
    m = _load_campaign()
    summaries = {
        "virchow2_0.20": {"mean_f1_mean": 0.71},
        "virchow2_0.25": {"mean_f1_mean": 0.73},
        "uni2_0.50": {"mean_f1_mean": 0.60},
    }
    assert m.pick_winner(summaries) == "virchow2_0.25"


def test_test_split_names_and_headline_metrics():
    m = _load_campaign()
    report = {
        "matching": "greedy",
        "score_threshold_per_class": [0.5, 0.49],
        "tune": {"mean_f1": 0.71},
        "test": {"headline": {"metrics": {"mean_f1": 0.6995, "recall_class_0": 0.6}}},
    }
    assert m.test_split_names(report) == ["test"]
    assert m.test_headline_metrics(report)["mean_f1"] == pytest.approx(0.6995)


def test_test_headline_metrics_requires_exactly_one_test_split():
    m = _load_campaign()
    two = {"matching": "greedy", "score_threshold_per_class": [], "tune": {},
           "test": {"headline": {"metrics": {}}}, "test_2": {"headline": {"metrics": {}}}}
    with pytest.raises(ValueError):
        m.test_headline_metrics(two)
    tune_only = {"matching": "greedy", "score_threshold_per_class": [], "tune": {"mean_f1": 0.7}}
    with pytest.raises(ValueError):
        m.test_headline_metrics(tune_only)


def test_confirmation_cells_dedups_anchor_winner():
    m = _load_campaign()
    # non-anchor winner → winner + anchor
    keys = [c.key for c in m.confirmation_cells("virchow2_0.25")]
    assert keys == ["virchow2_0.25", "virchow2_0.20"]
    # anchor winner → just the anchor, no duplicate
    assert [c.key for c in m.confirmation_cells("virchow2_0.20")] == ["virchow2_0.20"]


def test_format_confirmation_markdown_head_to_head():
    m = _load_campaign()
    results = {
        "virchow2_0.25": {
            "greedy": {"mean_f1_mean": 0.71, "mean_f1_std": 0.01, "n_seeds": 3,
                       "recall_bc_mean": 0.66, "recall_tc_mean": 0.73},
            "hungarian_mean_f1_mean": 0.711,
        },
        "virchow2_0.20": {
            "greedy": {"mean_f1_mean": 0.6995, "mean_f1_std": 0.005, "n_seeds": 3,
                       "recall_bc_mean": 0.67, "recall_tc_mean": 0.72},
            "hungarian_mean_f1_mean": 0.6996,
        },
    }
    md = m.format_confirmation_markdown(results, "virchow2_0.25")
    assert "confirmation (test)" in md.lower()
    assert "⭐winner" in md
    assert "(anchor)" in md
    assert "0.7100" in md and "0.6995" in md


def test_run_selection_isolates_a_failing_cell(monkeypatch, tmp_path):
    """A (cell, seed) whose scoring raises is recorded in ``failures`` and skipped; the rest of
    the grid still aggregates and yields a winner — so one deterministic OOM can't sink the
    unattended run. Monkeypatches the GPU/subprocess seams so no encoder or dataset is needed."""
    m = _load_campaign()
    monkeypatch.setattr(m, "output_root_for", lambda cell: tmp_path / cell.key)
    monkeypatch.setattr(m, "find_seed_runs", lambda root: {0: root / "run0"})  # seed 0 "trained"
    monkeypatch.setattr(m, "OUT_DIR", tmp_path / "out")

    def fake_score(cell, run_subdir, *, tune_only, matching="greedy"):
        assert tune_only  # selection always scores tune-only
        if cell.key == "uni2_0.25":
            raise ValueError("simulated OOM during scoring")
        return {"tune": {"mean_f1": 0.5, "recall_class_0": 0.4, "recall_class_1": 0.6}}

    monkeypatch.setattr(m, "score_run", fake_score)

    report = m.run_selection(tmp_path, [0], train=False, dry_run=False)

    # the failing cell is recorded and omitted; the survivors still produce a winner.
    assert any(f["cell"] == "uni2_0.25" and f["stage"] == "score" for f in report["failures"])
    assert "uni2_0.25" not in report["summaries"]
    assert set(report["summaries"]) == {"virchow2_0.20", "virchow2_0.25", "virchow2_0.50", "uni2_0.50"}
    assert report["winner"] in report["summaries"]
    assert (tmp_path / "out" / "selection_report.json").exists()


def test_format_selection_markdown_marks_winner_and_interaction():
    m = _load_campaign()
    summaries = {
        "virchow2_0.20": {"mean_f1_mean": 0.7146, "mean_f1_std": 0.0, "n_seeds": 1,
                          "recall_bc_mean": 0.6, "recall_tc_mean": 0.7},
        "virchow2_0.25": {"mean_f1_mean": 0.73, "mean_f1_std": 0.01, "n_seeds": 3,
                          "recall_bc_mean": 0.65, "recall_tc_mean": 0.72},
    }
    md = m.format_selection_markdown(summaries, "virchow2_0.25", {"virchow2": 0.05, "interaction": 0.05})
    assert "Winner (tune): virchow2_0.25" in md
    assert "⭐" in md
    assert "0.7300" in md
    assert "interaction" in md
