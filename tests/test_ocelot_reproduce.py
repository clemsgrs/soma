"""Pure-logic tests for examples/ocelot/reproduce.py + the recorded reference card.

The subprocess orchestration (curate → train → score) needs a GPU and the dataset, so it
is not unit-tested; the testable seams are the JSON extraction, the override builder, and
the tolerance check. We also assert ``expected_metrics.json`` keeps the shape the runner
relies on, so editing the reference can't silently break reproduction.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
REPRODUCE = REPO_ROOT / "examples" / "ocelot" / "reproduce.py"
REFERENCE = REPO_ROOT / "examples" / "ocelot" / "expected_metrics.json"


def _load_reproduce():
    spec = importlib.util.spec_from_file_location("ocelot_reproduce", REPRODUCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAMPLE_STDOUT = """feature store: /x/dense/abc
checkpoint:    /x/runs/ts/best_model.pt

matching = greedy
tune-frozen per-class thresholds (headline): [0.5, 0.49]
  [test] headline mF1=0.6995  |  oracle mF1 (diagnostic ceiling, NOT reported)=0.7062
{
  "matching": "greedy",
  "tune": {"mean_f1": 0.7146},
  "test": {
    "headline": {"metrics": {"mean_f1": 0.6995}},
    "oracle": {"metrics": {"mean_f1": 0.7062}}
  }
}
"""


def test_extract_greedy_report_parses_trailing_json():
    m = _load_reproduce()
    report = m.extract_greedy_report(SAMPLE_STDOUT)
    assert report["matching"] == "greedy"
    assert m.greedy_test_mean_f1(report) == pytest.approx(0.6995)


def test_extract_greedy_report_raises_without_json():
    m = _load_reproduce()
    with pytest.raises(ValueError):
        m.extract_greedy_report("no json here\njust text\n")


def test_build_path_overrides_without_output_root():
    m = _load_reproduce()
    pairs = m.build_path_overrides(Path("/data/curated"), None)
    assert pairs == [
        "data.dataset_csv=/data/curated/dataset.csv",
        "data.splits_csv=/data/curated/splits.csv",
    ]


def test_build_path_overrides_with_output_root():
    m = _load_reproduce()
    pairs = m.build_path_overrides(Path("/data/curated"), Path("/out"))
    assert pairs[-1] == "run.output_root=/out"


def test_check_within_tolerance_pass():
    m = _load_reproduce()
    reference = {"expected": {"test_greedy": {"mean_f1": 0.6995}}, "tolerance": {"mean_f1_abs": 0.02}}
    ok, msg = m.check_within_tolerance(0.69, reference)
    assert ok
    assert "PASS" in msg


def test_check_within_tolerance_fail():
    m = _load_reproduce()
    reference = {"expected": {"test_greedy": {"mean_f1": 0.6995}}, "tolerance": {"mean_f1_abs": 0.02}}
    ok, msg = m.check_within_tolerance(0.60, reference)
    assert not ok
    assert "FAIL" in msg


def test_reference_card_has_required_shape():
    reference = json.loads(REFERENCE.read_text())
    assert reference["canonical_seed"] == 0
    assert reference["config"].endswith("ocelot_virchow2_0.20.yaml")
    assert reference["expected"]["test_greedy"]["mean_f1"] == pytest.approx(0.6995, abs=1e-4)
    assert reference["tolerance"]["mean_f1_abs"] > 0
    # the runner reads these exact paths
    m = _load_reproduce()
    assert m.greedy_test_mean_f1(
        {"test": {"headline": {"metrics": {"mean_f1": reference["expected"]["test_greedy"]["mean_f1"]}}}}
    ) == pytest.approx(0.6995, abs=1e-4)
