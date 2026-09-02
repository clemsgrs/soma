"""Reproduced-results ledger: the counterpart to the reference table (issue-driven).

``soma/benchmarks/results/<name>.csv`` records soma's OWN produced numbers with the
provenance that makes a reproduced value meaningful. These tests cover the loader
(``load_results``/``reproduced_rows``), the append-only ``append_result`` ledger, and one
consistency guard: every committed ``results/eva.csv`` row must join a gate reference band
and fall inside its tolerance — so the committed evidence can never silently contradict the
target it claims to reproduce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soma.benchmarks import (
    MeasuredRow,
    append_result,
    expected_rows,
    load_results,
    reproduced_rows,
)
from soma.benchmarks import registry


# --- loader: schema + absent table ----------------------------------------------------


def test_load_results_absent_table_is_empty_not_error():
    # A benchmark may carry a reference band with no reproduced measurement yet.
    assert load_results("no_such_benchmark_table") == []


def test_load_results_parses_eva_seed_rows():
    rows = load_results("eva")
    assert rows, "results/eva.csv should ship seeded historical rows"
    # The ledger is append-only, so a re-recorded cell holds several rows; file order is
    # oldest-first and the latest reproduction of a cell is the last match.
    bach_uni2 = [r for r in rows if r.key == {"dataset": "bach", "encoder": "uni2"}]
    assert len(bach_uni2) == 2  # historical epoch-mapping row + fixed-step re-record
    historical, rerecorded = bach_uni2
    assert historical.metric == "test/balanced_accuracy"
    assert historical.measured == pytest.approx(0.914)
    assert historical.std == pytest.approx(0.007)
    assert historical.n_seeds == 5
    assert historical.soma_commit == "7ef2d7c"
    # slide2vec version was not recorded for the historical rows -> blank, not fabricated.
    assert historical.slide2vec_version == ""
    assert rerecorded.metric == "test/balanced_accuracy"
    assert rerecorded.measured == pytest.approx(0.914, abs=1e-3)
    assert rerecorded.n_seeds == 5
    assert rerecorded.slide2vec_version == "5.8.2"


def test_load_results_requires_metric_and_measured(tmp_path, monkeypatch):
    bad = tmp_path / "bad.csv"
    bad.write_text("dataset,encoder,measured\nbach,uni2,0.9\n")  # no `metric` column
    monkeypatch.setattr(registry, "_results_file", lambda name: bad)
    with pytest.raises(ValueError, match="missing required column"):
        load_results("bad")


# --- reproduced_rows filtering --------------------------------------------------------


def test_reproduced_rows_filters_by_axes_and_metric():
    rows = reproduced_rows("eva", dataset="crc", encoder="virchow2", metric="test/balanced_accuracy")
    assert len(rows) == 1
    assert rows[0].key == {"dataset": "crc", "encoder": "virchow2"}
    assert rows[0].measured == pytest.approx(0.966)
    # A non-existent cell selects nothing (rather than erroring).
    assert reproduced_rows("eva", dataset="crc", encoder="nope") == []


def test_ocelot_spacing_migration_rows_are_provenance_pinned():
    expected = {
        "0.25": (0.7148, "c5d36b5"),
        "0.5": (0.6085, "c5d36b5"),
    }

    for spacing, (measured, commit) in expected.items():
        rows = reproduced_rows(
            "ocelot-spacing-migration",
            encoder="virchow2",
            spacing=spacing,
            metric="mean_f1",
        )
        assert rows
        assert rows[-1].measured == pytest.approx(measured)
        assert rows[-1].n_seeds == 1
        assert rows[-1].soma_commit == commit
        assert rows[-1].slide2vec_version == "5.7.0"


# --- append_result: append-only ledger ------------------------------------------------


def test_append_result_creates_header_then_appends(tmp_path, monkeypatch):
    target = tmp_path / "toy.csv"
    monkeypatch.setattr(registry, "_results_file", lambda name: target)

    first = MeasuredRow(
        key={"dataset": "d", "encoder": "e"},
        metric="test/balanced_accuracy",
        measured=0.9137,
        std=0.0072,
        n_seeds=5,
        date="2026-07-09",
        soma_commit="abc1234",
        slide2vec_version="5.3.0",
        source="soma reproduce --record",
    )
    path = append_result("toy", first, key_order=["dataset", "encoder"])
    assert path == target
    header = target.read_text().splitlines()[0]
    assert header == (
        "dataset,encoder,metric,measured,std,n_seeds,date,soma_commit,"
        "slide2vec_version,croma_version,source"
    )

    # Re-running the same cell appends a second row (history), never overwrites.
    second = MeasuredRow(
        key={"dataset": "d", "encoder": "e"},
        metric="test/balanced_accuracy",
        measured=0.9200,
        std=None,
        n_seeds=None,
        date="2026-07-10",
        soma_commit="def5678",
    )
    append_result("toy", second)

    rows = load_results("toy")
    assert len(rows) == 2  # append-only ledger, same key twice
    # A single re-scored run has no seed spread -> blank std/n_seeds round-trip to None.
    assert rows[1].std is None and rows[1].n_seeds is None
    assert rows[1].measured == pytest.approx(0.92)
    # reproduced_rows keeps file order; latest is the last match.
    latest = reproduced_rows("toy", dataset="d", encoder="e")[-1]
    assert latest.soma_commit == "def5678"


# --- consistency guard: committed evidence must match the reference band ---------------


def test_eva_seed_rows_join_and_stay_within_reference_band():
    for row in load_results("eva"):
        gates = expected_rows(
            "eva",
            metric=row.metric,
            dataset=row.key["dataset"],
            encoder=row.key["encoder"],
        )
        gates = [g for g in gates if not g.is_external]
        assert len(gates) == 1, f"no unique gate band for {row.key} {row.metric}"
        assert gates[0].within_tolerance(row.measured), (
            f"recorded {row.key} {row.measured} outside band "
            f"{gates[0].expected}±{gates[0].tolerance}"
        )


def test_append_result_extends_an_existing_header_in_place(tmp_path, monkeypatch):
    """A row keyed on an axis the ledger has never seen must not lose that cell.

    New key columns are inserted left of ``metric`` (after the existing keys) and old
    rows get blanks; previously ``extrasaction="ignore"`` dropped the cell silently.
    """
    target = tmp_path / "toy.csv"
    monkeypatch.setattr(registry, "_results_file", lambda name: target)
    append_result(
        "toy",
        MeasuredRow(key={"encoder": "e"}, metric="m", measured=0.5, date="2026-07-09",
                    soma_commit="abc"),
        key_order=["encoder"],
    )
    append_result(
        "toy",
        MeasuredRow(key={"encoder": "e", "spacing": "0.2"}, metric="m", measured=0.6,
                    date="2026-07-10", soma_commit="def"),
    )
    header = target.read_text().splitlines()[0]
    assert header.startswith("encoder,spacing,metric,measured,")
    rows = load_results("toy")
    assert rows[0].key == {"encoder": "e"}  # old row: blank spacing stays out of the key
    assert rows[1].key == {"encoder": "e", "spacing": "0.2"}
    assert [r.measured for r in rows] == pytest.approx([0.5, 0.6])
