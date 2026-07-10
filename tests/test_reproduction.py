"""reproduction_report — the A/B/C soundness analysis over results ⋈ reference.

Pure logic, so these tests craft a small ``(dataset, encoder)`` grid (monkeypatching the two
loaders) and assert the per-cell delta (A), the pooled resolvable-tagged pairwise concordance
+ per-task Spearman (B), and that an empty ledger yields an empty report. This is the headline
of soma's HEST reproduction: does the native pipeline re-derive the benchmark's *ranking*?
"""

from __future__ import annotations

import pytest

from soma.benchmarks import reproduction
from soma.benchmarks.registry import EXTERNAL, MeasuredRow, ReferenceRow
from soma.benchmarks.reproduction import RESOLVABLE_EPS, reproduction_report

METRIC = "test/mean_pearson_mean"


def _ref(dataset: str, encoder: str, value: float) -> ReferenceRow:
    return ReferenceRow(
        key={"dataset": dataset, "encoder": encoder},
        metric=METRIC,
        expected=value,
        tolerance=0.0,
        source="HEST leaderboard (test fixture)",
        kind=EXTERNAL,
        label=f"{encoder} (published)",
        url="http://example/hest",
    )


def _measured(dataset: str, encoder: str, value: float, commit: str = "abc1234") -> MeasuredRow:
    return MeasuredRow(
        key={"dataset": dataset, "encoder": encoder},
        metric=METRIC,
        measured=value,
        std=None,
        n_seeds=1,
        date="2026-07-10",
        soma_commit=commit,
        slide2vec_version="5.3.0",
    )


# A controlled grid: T1 orders A>B>C (all gaps resolvable); T2 orders C>B>A but B/A is a
# within-noise tie (gap 0.002 < eps). soma reproduces every resolvable ordering and flips only
# the within-noise T2 pair -> headline concordance 100 %, all-pairs concordance < 100 %.
REFERENCE = [
    _ref("T1", "A", 0.60), _ref("T1", "B", 0.55), _ref("T1", "C", 0.40),
    _ref("T2", "A", 0.300), _ref("T2", "B", 0.302), _ref("T2", "C", 0.50),
]
MEASURED = [
    _measured("T1", "A", 0.61), _measured("T1", "B", 0.54), _measured("T1", "C", 0.42),
    _measured("T2", "A", 0.305), _measured("T2", "B", 0.300), _measured("T2", "C", 0.49),
]


@pytest.fixture
def stub_tables(monkeypatch):
    monkeypatch.setattr(reproduction, "load_reference", lambda name: list(REFERENCE))
    monkeypatch.setattr(reproduction, "load_results", lambda name: list(MEASURED))


# --- A: per-cell delta ----------------------------------------------------------------


def test_cells_join_measured_to_reference_with_signed_delta(stub_tables):
    report = reproduction_report("hest", metric=METRIC)
    by_cell = {(c.dataset, c.encoder): c for c in report.cells}
    assert len(report.cells) == 6
    assert by_cell[("T1", "A")].measured == pytest.approx(0.61)
    assert by_cell[("T1", "A")].reference == pytest.approx(0.60)
    assert by_cell[("T1", "A")].delta == pytest.approx(0.01)
    assert by_cell[("T2", "C")].delta == pytest.approx(-0.01)


def test_measured_cell_without_reference_is_omitted(monkeypatch):
    monkeypatch.setattr(reproduction, "load_reference", lambda name: list(REFERENCE))
    monkeypatch.setattr(
        reproduction,
        "load_results",
        lambda name: [*MEASURED, _measured("T1", "UNPUBLISHED", 0.99)],
    )
    report = reproduction_report("hest", metric=METRIC)
    # The extra cell has no published reference to compare against -> not a scored cell.
    assert ("T1", "UNPUBLISHED") not in {(c.dataset, c.encoder) for c in report.cells}
    assert len(report.cells) == 6


def test_latest_measurement_wins_for_a_repeated_cell(monkeypatch):
    monkeypatch.setattr(reproduction, "load_reference", lambda name: [_ref("T1", "A", 0.60)])
    monkeypatch.setattr(
        reproduction,
        "load_results",
        lambda name: [_measured("T1", "A", 0.61, "old0000"), _measured("T1", "A", 0.63, "new1111")],
    )
    report = reproduction_report("hest", metric=METRIC)
    assert len(report.cells) == 1
    assert report.cells[0].measured == pytest.approx(0.63)  # append-only: last row wins
    assert report.cells[0].soma_commit == "new1111"


# --- B: pooled pairwise concordance + resolvable tagging ------------------------------


def test_resolvable_headline_excludes_within_noise_pair(stub_tables):
    report = reproduction_report("hest", metric=METRIC)
    # 6 unordered pairs total (3 per task). The T2 A/B pair (ref gap 0.002 < eps) is the only
    # non-resolvable one -> 5 resolvable.
    assert len(report.pairs) == 6
    assert report.n_resolvable == 5
    # soma reproduces every resolvable ordering -> headline concordance 100 %.
    assert report.n_resolvable_concordant == 5
    assert report.concordance_resolvable == pytest.approx(1.0)
    # Over ALL pairs, the within-noise T2 A/B pair is a flip -> 5/6.
    assert report.concordance_all == pytest.approx(5 / 6)


def test_within_noise_pair_is_tagged_not_resolvable(stub_tables):
    report = reproduction_report("hest", metric=METRIC)
    t2_ab = next(
        p for p in report.pairs
        if p.dataset == "T2" and {p.encoder_high, p.encoder_low} == {"A", "B"}
    )
    assert not t2_ab.resolvable  # ref gap 0.002 < RESOLVABLE_EPS
    assert abs(t2_ab.reference_gap) < RESOLVABLE_EPS
    assert not t2_ab.concordant  # soma reverses it (harmless: within noise, excluded from headline)


def test_per_task_spearman(stub_tables):
    report = reproduction_report("hest", metric=METRIC)
    # T1 ordering is reproduced exactly -> ρ = +1. T2 has one adjacent (within-noise) swap -> ρ = 0.5.
    assert report.spearman_by_dataset["T1"] == pytest.approx(1.0)
    assert report.spearman_by_dataset["T2"] == pytest.approx(0.5)


# --- empty / degenerate ---------------------------------------------------------------


def test_empty_ledger_yields_empty_report(monkeypatch):
    monkeypatch.setattr(reproduction, "load_reference", lambda name: list(REFERENCE))
    monkeypatch.setattr(reproduction, "load_results", lambda name: [])
    report = reproduction_report("hest", metric=METRIC)
    assert report.cells == [] and report.pairs == []
    assert report.concordance_resolvable is None
    assert report.concordance_all is None


def test_rankdata_and_spearman_helpers():
    # Average-tie ranks, matching scipy.stats.rankdata.
    assert reproduction._rankdata([0.4, 0.6, 0.5]) == [1.0, 3.0, 2.0]
    assert reproduction._rankdata([0.5, 0.5, 0.9]) == [1.5, 1.5, 3.0]
    assert reproduction._spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    assert reproduction._spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert reproduction._spearman([1.0], [1.0]) is None  # < 2 points
    assert reproduction._spearman([1.0, 1.0], [2.0, 3.0]) is None  # constant ranking
