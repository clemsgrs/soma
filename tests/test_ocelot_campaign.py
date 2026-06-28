"""Pure-logic tests for examples/ocelot/campaign.py (#152).

The GPU orchestration (train → greedy-rescore) needs the dataset + an encoder, so it is
not unit-tested; the testable seams are the seed aggregation, the model×magnification
interaction, the winner pick, the JSON extraction, and the markdown formatter.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
