"""The Leaderboard: a pure faceted projection over self-describing run dirs (ADR 0003, #214).

Everything here is exercised against **fixture output_roots** built from hand-written run
dirs (real ``config.yaml`` + ``summary.json`` + ``run.yaml`` + ``experiment.json`` via the
production identity code) — no pipeline execution, no training, no GPU. That covers the
projection, facet filtering, seed-collapse, the never-pool rule, triple
discovery/disambiguation, reference-row injection (broad banner + keyed join via a fixture
benchmark), and a concurrency test that simultaneous completions lose no rows.
"""

from __future__ import annotations

import json
import threading
from hashlib import sha256
from pathlib import Path

import pytest

from soma.benchmarks import get_benchmark
from soma.benchmarks import registry as registry_mod
from soma.benchmarks.registry import Facet, ReferenceRow, register_benchmark
from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
    save_config,
)
from soma.leaderboard import (
    LeaderboardFacet,
    LeaderboardRow,
    LeaderboardTable,
    discover_triples,
    format_table,
    load_run_record,
    project_leaderboard,
    render_csv,
    render_html,
    render_json,
    run_primary_metric,
    write_leaderboard,
)
from soma.output_layout import (
    create_run_metadata,
    resolve_managed_output_paths,
    write_experiment_metadata,
    write_run_metadata,
)

# Real, registry-valid tile encoders (config validates names). Enough distinct names for
# the concurrency fan-out.
ENC = [
    "virchow2", "uni2", "phikon", "conch", "gigapath", "hibou-b",
    "hibou-l", "lunit", "midnight", "musk", "uni", "virchow",
]

_TASK_METRICS = {
    "binary_classification": ["accuracy"],
    "multiclass_classification": ["accuracy"],
    "regression": ["mae"],
    "detection": ["mean_f1"],
}


# --- fixture run-dir factory ---------------------------------------------------------


def _dataset_csv(tmp_path: Path, name: str = "dataset", body: str | None = None) -> Path:
    path = tmp_path / f"{name}.csv"
    path.write_text(body or "sample_id,image_path,label\ns0,/slides/s0.svs,tumor\n", encoding="utf-8")
    return path


def _splits_csv(tmp_path: Path, name: str = "splits") -> Path:
    path = tmp_path / f"{name}.csv"
    path.write_text("fold,sample_id,split\n0,s0,train\n", encoding="utf-8")
    return path


def _make_config(
    *,
    output_root: Path,
    dataset_csv: Path,
    splits_csv: Path,
    encoder: str = "uni2",
    aggregator: str = "abmil",
    task: str = "binary_classification",
    seed: int = 0,
    learning_rate: float = 1e-4,
    spacing: float = 0.5,
    hidden_dim: int = 128,
) -> PipelineConfig:
    return PipelineConfig(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=output_root,
        dataset_type="slide",
        preprocessing=PreprocessingConfig(requested_tile_size_px=224, requested_spacing_um=spacing),
        cache=CacheConfig(),
        encoder=EncoderConfig(name=encoder),
        aggregator=AggregatorConfig(name=aggregator, params={"hidden_dim": hidden_dim}),
        task=TaskConfig(name=task),
        evaluation=EvalConfig(metrics=_TASK_METRICS[task]),
        training=TrainingConfig(seed=seed, epochs=5, learning_rate=learning_rate),
    )


_RUN_COUNTER = {"n": 0}


def make_run_dir(
    config: PipelineConfig,
    summary: dict[str, float],
    *,
    run_id: str | None = None,
    status: str = "completed",
    finished_at: str = "2026-07-03T00:00:00+00:00",
) -> Path:
    """Materialise a completed run dir from a config + summary (production identity code)."""
    if run_id is None:
        _RUN_COUNTER["n"] += 1
        run_id = f"2026-07-03_00-00-{_RUN_COUNTER['n']:02d}__seed{config.training.seed}"
    layout = resolve_managed_output_paths(config, run_id=run_id)
    layout.experiment_dir.mkdir(parents=True, exist_ok=True)
    (layout.experiment_dir / "runs").mkdir(parents=True, exist_ok=True)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    write_experiment_metadata(layout.experiment_dir, layout.experiment)
    save_config(config, layout.run_dir / "config.yaml")
    metadata = create_run_metadata(
        config=config,
        experiment=layout.experiment,
        run_dir=layout.run_dir,
        run_id=run_id,
        status=status,
        summary_metrics=summary,
    ).with_updates(finished_at=finished_at)
    write_run_metadata(layout.run_dir, metadata)
    (layout.run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return layout.run_dir


def _cfg(root, ds, sp, **kw):
    return _make_config(output_root=root, dataset_csv=ds, splits_csv=sp, **kw)


# --- projection basics ---------------------------------------------------------------


def test_project_leaderboard_ranks_by_metric_descending(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    d_low = make_run_dir(_cfg(root, ds, sp, encoder="phikon"), {"test/accuracy": 0.70})
    d_high = make_run_dir(_cfg(root, ds, sp, encoder="virchow2"), {"test/accuracy": 0.90})
    d_mid = make_run_dir(_cfg(root, ds, sp, encoder="uni2"), {"test/accuracy": 0.80})

    table = project_leaderboard([d_low, d_high, d_mid], LeaderboardFacet(vary=("encoder",)))

    assert [r.rank for r in table.rows] == [1, 2, 3]
    assert [r.vary_values["encoder"] for r in table.rows] == ["virchow2", "uni2", "phikon"]
    assert table.higher_is_better is True
    assert table.metric == "accuracy"


def test_lower_is_better_metric_ranks_ascending(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    d_a = make_run_dir(_cfg(root, ds, sp, encoder="uni2", task="regression"), {"test/mae": 0.5})
    d_b = make_run_dir(_cfg(root, ds, sp, encoder="phikon", task="regression"), {"test/mae": 0.2})

    table = project_leaderboard([d_a, d_b], LeaderboardFacet(vary=("encoder",)), metric="mae")

    assert table.higher_is_better is False
    assert table.rows[0].vary_values["encoder"] == "phikon"  # lower MAE wins


def test_raw_primary_metric_resolves_from_run(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(
        _cfg(root, ds, sp, encoder="uni2"),
        {"test/auroc": 0.88, "test/accuracy": 0.80, "test/num_samples": 10, "test/coverage": 1.0},
    )
    # Bookkeeping keys (num_samples, coverage) are skipped; first real metric is primary.
    assert run_primary_metric(json.loads((run / "summary.json").read_text()), "test") == "auroc"
    table = project_leaderboard([run], LeaderboardFacet(vary=("encoder",)))
    assert table.metric == "auroc"


def test_metric_override_beats_run(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(_cfg(root, ds, sp, encoder="uni2"), {"test/auroc": 0.88, "test/accuracy": 0.80})
    table = project_leaderboard([run], LeaderboardFacet(vary=("encoder",)), metric="accuracy")
    assert table.metric == "accuracy"
    assert table.rows[0].mean == pytest.approx(0.80)


# --- seed collapse -------------------------------------------------------------------


def test_seed_runs_collapse_to_mean_std_n(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    # Same config, two seeds -> same experiment_id -> one collapsed row.
    d0 = make_run_dir(_cfg(root, ds, sp, seed=0), {"test/accuracy": 0.80})
    d1 = make_run_dir(_cfg(root, ds, sp, seed=1), {"test/accuracy": 0.90})

    table = project_leaderboard([d0, d1], LeaderboardFacet(vary=("encoder",)))

    assert len(table.rows) == 1
    row = table.rows[0]
    assert row.n == 2
    assert row.mean == pytest.approx(0.85)
    assert row.std is not None and row.std == pytest.approx(0.05)
    assert row.seeds == (0, 1)


def test_single_seed_leaves_std_blank(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    d0 = make_run_dir(_cfg(root, ds, sp, seed=0), {"test/accuracy": 0.80})
    table = project_leaderboard([d0], LeaderboardFacet(vary=("encoder",)))
    assert table.rows[0].n == 1
    assert table.rows[0].std is None


# --- the never-pool rule -------------------------------------------------------------


def test_unfixed_multivalued_axis_yields_one_row_per_combo_with_config_diff(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    # Same encoder, but a HIDDEN axis (learning_rate) differs -> must NOT be pooled.
    d_lo = make_run_dir(_cfg(root, ds, sp, encoder="uni2", learning_rate=1e-4), {"test/accuracy": 0.80})
    d_hi = make_run_dir(_cfg(root, ds, sp, encoder="uni2", learning_rate=1e-3), {"test/accuracy": 0.85})

    table = project_leaderboard([d_lo, d_hi], LeaderboardFacet(vary=("encoder",)))

    # Two distinct configs -> two rows (never averaged into one), both encoder=uni2.
    assert len(table.rows) == 2
    assert {r.vary_values["encoder"] for r in table.rows} == {"uni2"}
    # Each row is annotated with the diff that distinguishes it (the hidden lr axis).
    for row in table.rows:
        assert "training.learning_rate" in row.config_diff
    lrs = {row.config_diff["training.learning_rate"] for row in table.rows}
    assert lrs == {1e-4, 1e-3}


def test_vary_axis_is_stripped_from_config_diff(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    d_a = make_run_dir(_cfg(root, ds, sp, encoder="uni2"), {"test/accuracy": 0.80})
    d_b = make_run_dir(_cfg(root, ds, sp, encoder="phikon"), {"test/accuracy": 0.85})
    table = project_leaderboard([d_a, d_b], LeaderboardFacet(vary=("encoder",)))
    # The encoder name is the headline column, so it is not repeated in the diff.
    for row in table.rows:
        assert "encoder.name" not in row.config_diff


# --- facet filtering: cross-encoder vs encoder-specific vs flat -----------------------


def test_fix_filters_rows_out(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    keep = make_run_dir(_cfg(root, ds, sp, encoder="uni2", aggregator="abmil"), {"test/accuracy": 0.80})
    drop = make_run_dir(_cfg(root, ds, sp, encoder="uni2", aggregator="mean_pool"), {"test/accuracy": 0.90})

    table = project_leaderboard(
        [keep, drop], LeaderboardFacet(vary=("encoder",), fixed={"aggregator": "abmil"})
    )
    assert len(table.rows) == 1
    assert table.rows[0].mean == pytest.approx(0.80)


def test_encoder_specific_facet_fixes_encoder_varies_aggregator(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    a = make_run_dir(_cfg(root, ds, sp, encoder="uni2", aggregator="abmil"), {"test/accuracy": 0.80})
    b = make_run_dir(_cfg(root, ds, sp, encoder="uni2", aggregator="mean_pool"), {"test/accuracy": 0.85})
    other = make_run_dir(_cfg(root, ds, sp, encoder="virchow2", aggregator="abmil"), {"test/accuracy": 0.99})

    table = project_leaderboard(
        [a, b, other], LeaderboardFacet(vary=("aggregator",), fixed={"encoder": "uni2"})
    )
    assert len(table.rows) == 2
    assert [r.vary_values["aggregator"] for r in table.rows] == ["mean_pool", "abmil"]


def test_flat_facet_ranks_all_no_vary(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    a = make_run_dir(_cfg(root, ds, sp, encoder="uni2"), {"test/accuracy": 0.80})
    b = make_run_dir(_cfg(root, ds, sp, encoder="phikon"), {"test/accuracy": 0.90})
    table = project_leaderboard([a, b], LeaderboardFacet(vary=()))
    assert len(table.rows) == 2
    # No vary axis -> the distinguishing config surfaces in the diff instead.
    assert all("encoder.name" in r.config_diff for r in table.rows)


# --- triple discovery + disambiguation ----------------------------------------------


def test_discover_triples_groups_by_dataset_splits_task(tmp_path: Path):
    root = tmp_path / "out"
    ds_a = _dataset_csv(tmp_path, "ds_a", "sample_id,image_path,label\ns0,/a.svs,tumor\n")
    ds_b = _dataset_csv(tmp_path, "ds_b", "sample_id,image_path,label\ns0,/b.svs,tumor\n")
    sp = _splits_csv(tmp_path)
    make_run_dir(_cfg(root, ds_a, sp, encoder="uni2"), {"test/accuracy": 0.8})
    make_run_dir(_cfg(root, ds_b, sp, encoder="uni2"), {"test/accuracy": 0.7})

    triples = discover_triples(root)
    assert len(triples) == 2


def test_project_leaderboard_rejects_multiple_triples(tmp_path: Path):
    root = tmp_path / "out"
    ds_a = _dataset_csv(tmp_path, "ds_a", "sample_id,image_path,label\ns0,/a.svs,tumor\n")
    ds_b = _dataset_csv(tmp_path, "ds_b", "sample_id,image_path,label\ns0,/b.svs,tumor\n")
    sp = _splits_csv(tmp_path)
    d_a = make_run_dir(_cfg(root, ds_a, sp, encoder="uni2"), {"test/accuracy": 0.8})
    d_b = make_run_dir(_cfg(root, ds_b, sp, encoder="uni2"), {"test/accuracy": 0.7})
    with pytest.raises(ValueError, match="single"):
        project_leaderboard([d_a, d_b], LeaderboardFacet(vary=("encoder",)))


# --- reference rows: broad banner + keyed join --------------------------------------


def test_broad_reference_renders_as_banner(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(_cfg(root, ds, sp, encoder="virchow2", task="detection"), {"test/mean_f1": 0.70})
    table = project_leaderboard([run], LeaderboardFacet(vary=("encoder",)), benchmark=get_benchmark("ocelot"))
    # OCELOT ships a broad, config-agnostic band -> a threshold banner, not a per-row join.
    assert table.banner is not None
    assert table.banner.metric == "mean_f1"
    assert table.banner.expected == pytest.approx(0.6995, abs=1e-6)
    assert table.rows[0].reference_expected is None  # broad band never joins per-row


def _detection_table(tmp_path: Path):
    """A one-row detection leaderboard projected with the real OCELOT benchmark."""
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(
        _cfg(root, ds, sp, encoder="virchow2", task="detection"), {"test/mean_f1": 0.70}
    )
    return project_leaderboard(
        [run], LeaderboardFacet(vary=("encoder",)), benchmark=get_benchmark("ocelot")
    )


def test_external_anchors_collected_as_guidance_distinct_from_gate(tmp_path: Path):
    table = _detection_table(tmp_path)
    # The gate band stays the soma-reproduced anchor; external anchors land in `guidance`.
    assert table.banner is not None and table.banner.expected == pytest.approx(0.6995, abs=1e-6)
    assert len(table.guidance) >= 2  # official baseline + best-reported
    assert any("best reported" in a.label for a in table.guidance)
    for anchor in table.guidance:
        assert anchor.label and anchor.url.startswith("http")
    # Non-gating: an external anchor is never the gate value, and never joins per-row.
    assert all(a.expected != table.banner.expected for a in table.guidance)
    assert table.rows[0].reference_expected is None


def test_format_table_renders_guidance_section_separate_from_gate(tmp_path: Path):
    text = format_table(_detection_table(tmp_path))
    assert "reference band" in text  # the gate band
    assert "guidance" in text.lower()  # a distinct, labelled guidance section
    assert "best reported" in text  # an anchor label
    assert "https://wearewaiv.github.io/histoboard/" in text  # its linkable source


def test_render_html_and_json_expose_clickable_guidance(tmp_path: Path):
    table = _detection_table(tmp_path)
    html = render_html(table)
    assert 'href="https://wearewaiv.github.io/histoboard/"' in html  # clickable link
    assert "guidance" in html.lower()
    data = json.loads(render_json(table))
    assert data["guidance"], "guidance anchors are serialised to JSON"
    assert data["guidance"][0]["label"]
    assert data["guidance"][0]["url"].startswith("http")


# --- HEST: external-only benchmark renders Measured + external Reference together (#260) --


def _hest_run_dir(tmp_path: Path, *, encoder: str, mean_pearson: float) -> Path:
    """A completed spatial_expression run dir built from the real hest/IDC config."""
    ds = _dataset_csv(tmp_path, name=f"ds_{encoder}")
    sp = _splits_csv(tmp_path, name=f"sp_{encoder}")
    config = get_benchmark("hest/IDC").build_config(
        encoder=encoder,
        dataset_csv=ds,
        splits_csv=sp,
        output_root=tmp_path / "out",
    )
    return make_run_dir(config, {"test/mean_pearson_mean": mean_pearson})


def test_hest_leaderboard_shows_measured_rows_and_external_reference(tmp_path: Path):
    # An external-only benchmark: soma's Measured rows rank normally AND HEST's published
    # external Reference renders as a non-gating guidance anchor beside them (never a gate).
    bench = get_benchmark("hest/IDC")
    uni2 = _hest_run_dir(tmp_path, encoder="uni2", mean_pearson=0.51)
    table = project_leaderboard([uni2], LeaderboardFacet(vary=("encoder",)), benchmark=bench)

    # The Measured row resolves (HEST's primary_metric already carries the test/ prefix).
    assert table.metric == "test/mean_pearson_mean"
    assert len(table.rows) == 1
    assert table.rows[0].mean == pytest.approx(0.51)
    assert table.rows[0].reference_expected is None  # external rows never gate per-row
    assert table.banner is None  # no gate band exists

    # HEST's published external Reference is surfaced as guidance beside the Measured table.
    assert table.guidance, "the external HEST Reference must render as a guidance anchor"
    anchor = table.guidance[0]
    assert anchor.expected == pytest.approx(0.5898)  # uni2's HEST IDC Pearson
    assert anchor.url.startswith("http") and anchor.label
    text = format_table(table)
    assert "guidance" in text.lower()
    assert "0.5898" in text  # Reference rendered
    assert "0.5100" in text  # Measured rendered beside it
    assert "PASS" not in text and "FAIL" not in text  # nothing is gated


class _KeyedBenchmark:
    """A fixture benchmark whose reference is KEYED per encoder (join-on-varied-axis)."""

    name = "keyed_fixture"
    facet = Facet(fixed={"task": "binary_classification"}, varied=("encoder",))
    canonical_seeds = (0,)
    primary_metric = "accuracy"
    reference_environment: dict[str, str] = {}

    _rows = [
        ReferenceRow(key={}, metric="accuracy", expected=0.85, tolerance=0.05, source="banner"),
        ReferenceRow(key={"encoder": "uni2"}, metric="accuracy", expected=0.80, tolerance=0.03, source="k"),
        ReferenceRow(key={"encoder": "virchow2"}, metric="accuracy", expected=0.90, tolerance=0.03, source="k"),
    ]

    def curate(self, raw_root, out_dir):  # pragma: no cover - unused
        raise NotImplementedError

    def build_config(self, **axes):  # pragma: no cover - unused
        raise NotImplementedError

    def expected(self, **axes):
        return [r for r in self._rows if r.matches(axes)]

    def score(self, run_dir):  # pragma: no cover - unused
        raise NotImplementedError


class _MultiMetricBenchmark(_KeyedBenchmark):
    name = "multi_metric_fixture"
    primary_metric = "croma_median"
    reported_metrics = ("croma_median", "croma_f0", "croma_ltm10")
    ranking_metrics = ("croma_median", "croma_ltm10")
    _rows: list[ReferenceRow] = []


class _MultiReferenceBenchmark(_MultiMetricBenchmark):
    _rows = [
        ReferenceRow(
            key={"encoder": "uni2"},
            metric="croma_median",
            expected=0.18,
            tolerance=0.01,
            source="published",
            kind="external",
            label="PathoROB",
            url="https://example.test/pathorob",
        ),
        ReferenceRow(
            key={"encoder": "uni2"},
            metric="croma_f0",
            expected=0.15,
            tolerance=0.0,
            source="published",
            kind="external",
            label="PathoROB",
            url="https://example.test/pathorob",
        ),
        ReferenceRow(
            key={"encoder": "uni2"},
            metric="croma_ltm10",
            expected=0.10,
            tolerance=0.0,
            source="published",
            kind="external",
            label="PathoROB",
            url="https://example.test/pathorob",
        ),
    ]


class _ControlBenchmark(_MultiReferenceBenchmark):
    @staticmethod
    def is_ranking_eligible(**axes):
        return axes["encoder"] != "uni2"


class _MixedDirectionBenchmark(_MultiMetricBenchmark):
    primary_metric = "accuracy"
    reported_metrics = ("accuracy", "mae", "croma_f0")
    ranking_metrics = ("accuracy", "mae")


class _MultiGateBenchmark(_MultiMetricBenchmark):
    _rows = [
        ReferenceRow(
            key={"encoder": "uni2"},
            metric=metric,
            expected=expected,
            tolerance=0.05,
            source="gate fixture",
        )
        for metric, expected in (
            ("croma_median", 0.18),
            ("croma_f0", 0.15),
            ("croma_ltm10", 0.10),
        )
    ]


class _SingleMetricBenchmark(_KeyedBenchmark):
    name = "single_metric_fixture"
    reported_metrics = ("accuracy",)
    ranking_metrics = ("accuracy",)
    _rows: list[ReferenceRow] = []


class _LegacyMetricBenchmark(_KeyedBenchmark):
    name = "legacy_metric_fixture"
    _rows: list[ReferenceRow] = []


def test_multi_metric_benchmark_projects_one_wide_row_per_experiment(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    first = make_run_dir(
        _cfg(root, ds, sp, encoder="uni2"),
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.13,
            "test/croma_ltm10": 0.11,
        },
    )
    second = make_run_dir(
        _cfg(root, ds, sp, encoder="virchow2"),
        {
            "test/croma_median": 0.21,
            "test/croma_f0": 0.12,
            "test/croma_ltm10": 0.10,
        },
    )

    table = project_leaderboard(
        [first, second],
        LeaderboardFacet(vary=("encoder",)),
        benchmark=_MultiMetricBenchmark(),
    )

    assert [metric.metric for metric in table.reported_metrics] == [
        "croma_median",
        "croma_f0",
        "croma_ltm10",
    ]
    assert [metric.metric for metric in table.ranking_metrics] == [
        "croma_median",
        "croma_ltm10",
    ]
    assert len(table.rows) == 2
    assert all(row.ranking_eligible for row in table.rows)
    assert list(table.rows[0].metrics) == [
        "croma_median",
        "croma_f0",
        "croma_ltm10",
    ]
    assert table.rows[0].metrics["croma_f0"].rank is None


def test_multi_metric_statistics_are_aggregated_independently(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    seed_0 = make_run_dir(
        _cfg(root, ds, sp, seed=0),
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.10,
            "test/croma_ltm10": 0.30,
        },
    )
    seed_1 = make_run_dir(
        _cfg(root, ds, sp, seed=1),
        {
            "test/croma_median": 0.40,
            "test/croma_f0": 0.14,
            "test/croma_ltm10": 0.50,
        },
    )

    table = project_leaderboard(
        [seed_0, seed_1], LeaderboardFacet(), benchmark=_MultiMetricBenchmark()
    )

    metrics = table.rows[0].metrics
    assert (metrics["croma_median"].mean, metrics["croma_median"].std) == pytest.approx(
        (0.30, 0.10)
    )
    assert (metrics["croma_f0"].mean, metrics["croma_f0"].std) == pytest.approx(
        (0.12, 0.02)
    )
    assert (metrics["croma_ltm10"].mean, metrics["croma_ltm10"].std) == pytest.approx(
        (0.40, 0.10)
    )
    assert all(metric.n == 2 and metric.seeds == (0, 1) for metric in metrics.values())


def test_multi_metric_projection_reports_every_incomplete_experiment(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    missing_f0 = make_run_dir(
        _cfg(root, ds, sp, encoder="uni2"),
        {"test/croma_median": 0.20, "test/croma_ltm10": 0.11},
    )
    missing_two = make_run_dir(
        _cfg(root, ds, sp, encoder="virchow2"),
        {"test/croma_median": 0.21},
    )

    with pytest.raises(ValueError) as error:
        project_leaderboard(
            [missing_f0, missing_two],
            LeaderboardFacet(vary=("encoder",)),
            benchmark=_MultiMetricBenchmark(),
        )

    message = str(error.value)
    first_id = load_run_record(missing_f0).experiment_id
    second_id = load_run_record(missing_two).experiment_id
    assert f"{first_id}: croma_f0" in message
    assert f"{second_id}: croma_f0, croma_ltm10" in message


def test_completed_attempt_with_no_reported_metrics_is_named_in_error(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(_cfg(root, ds, sp), {}, run_id="empty-attempt")
    experiment = json.loads((run.parent.parent / "experiment.json").read_text())

    with pytest.raises(ValueError) as error:
        project_leaderboard(
            [run], LeaderboardFacet(), benchmark=_MultiMetricBenchmark()
        )

    message = str(error.value)
    assert experiment["experiment_id"] in message
    assert "croma_median, croma_f0, croma_ltm10" in message


def test_repeated_experiment_seed_uses_latest_completed_attempt(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    config = _cfg(root, ds, sp, seed=0)
    older = make_run_dir(
        config,
        {
            "test/croma_median": 0.10,
            "test/croma_f0": 0.20,
            "test/croma_ltm10": 0.30,
        },
        run_id="older",
        finished_at="2026-07-03T00:00:00+00:00",
    )
    newer = make_run_dir(
        config,
        {
            "test/croma_median": 0.40,
            "test/croma_f0": 0.50,
            "test/croma_ltm10": 0.60,
        },
        run_id="newer",
        finished_at="2026-07-04T00:00:00+00:00",
    )

    table = project_leaderboard(
        [newer, older], LeaderboardFacet(), benchmark=_MultiMetricBenchmark()
    )

    metrics = table.rows[0].metrics
    assert metrics["croma_median"].mean == pytest.approx(0.40)
    assert all(metric.n == 1 and metric.std is None for metric in metrics.values())
    assert all(metric.seeds == (0,) for metric in metrics.values())


def test_equal_completion_times_use_stable_run_id_tie_break(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    config = _cfg(root, ds, sp, seed=0)
    attempt_a = make_run_dir(
        config,
        {
            "test/croma_median": 0.10,
            "test/croma_f0": 0.20,
            "test/croma_ltm10": 0.30,
        },
        run_id="attempt-a",
    )
    attempt_z = make_run_dir(
        config,
        {
            "test/croma_median": 0.90,
            "test/croma_f0": 0.80,
            "test/croma_ltm10": 0.70,
        },
        run_id="attempt-z",
    )

    forward = project_leaderboard(
        [attempt_a, attempt_z], LeaderboardFacet(), benchmark=_MultiMetricBenchmark()
    )
    reverse = project_leaderboard(
        [attempt_z, attempt_a], LeaderboardFacet(), benchmark=_MultiMetricBenchmark()
    )

    assert forward.rows[0].metrics["croma_median"].mean == pytest.approx(0.90)
    assert reverse.rows[0].metrics["croma_median"].mean == pytest.approx(0.90)


def test_incomplete_newer_attempt_does_not_replace_completed_attempt(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    config = _cfg(root, ds, sp, seed=0)
    complete = make_run_dir(
        config,
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.13,
            "test/croma_ltm10": 0.11,
        },
        run_id="complete",
        finished_at="2026-07-03T00:00:00+00:00",
    )
    incomplete = make_run_dir(
        config,
        {"test/croma_median": 0.99, "test/croma_ltm10": 0.99},
        run_id="incomplete",
        finished_at="2026-07-04T00:00:00+00:00",
    )

    table = project_leaderboard(
        [incomplete, complete], LeaderboardFacet(), benchmark=_MultiMetricBenchmark()
    )

    assert table.rows[0].metrics["croma_median"].mean == pytest.approx(0.20)


def test_running_attempt_does_not_hide_completed_attempt_missing_metric(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    config = _cfg(root, ds, sp, seed=0)
    completed = make_run_dir(
        config,
        {"test/croma_median": 0.20, "test/croma_ltm10": 0.11},
        run_id="completed-partial",
        finished_at="2026-07-03T00:00:00+00:00",
    )
    running = make_run_dir(
        config,
        {
            "test/croma_median": 0.99,
            "test/croma_f0": 0.99,
            "test/croma_ltm10": 0.99,
        },
        run_id="running",
        status="running",
        finished_at="2026-07-04T00:00:00+00:00",
    )

    with pytest.raises(ValueError) as error:
        project_leaderboard(
            [running, completed], LeaderboardFacet(), benchmark=_MultiMetricBenchmark()
        )

    experiment_id = load_run_record(completed).experiment_id
    assert str(error.value).endswith(f"{experiment_id}: croma_f0")


def test_croma_f0_is_displayed_lower_is_better_but_not_ranked(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(
        _cfg(root, ds, sp),
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.13,
            "test/croma_ltm10": 0.11,
        },
    )

    table = project_leaderboard(
        [run], LeaderboardFacet(), benchmark=_MultiMetricBenchmark()
    )

    f0 = table.rows[0].metrics["croma_f0"]
    assert f0.higher_is_better is False
    assert f0.ranking is False
    assert f0.rank is None


def test_multi_metric_references_are_metric_local_and_external_never_gates(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(
        _cfg(root, ds, sp, encoder="uni2"),
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.13,
            "test/croma_ltm10": 0.11,
        },
    )

    table = project_leaderboard(
        [run],
        LeaderboardFacet(vary=("encoder",)),
        benchmark=_MultiReferenceBenchmark(),
    )

    metrics = table.rows[0].metrics
    assert metrics["croma_median"].reference.expected == pytest.approx(0.18)
    assert metrics["croma_median"].reference.delta == pytest.approx(0.02)
    assert metrics["croma_f0"].reference.expected == pytest.approx(0.15)
    assert metrics["croma_f0"].reference.delta == pytest.approx(-0.02)
    assert metrics["croma_ltm10"].reference.expected == pytest.approx(0.10)
    assert metrics["croma_ltm10"].reference.delta == pytest.approx(0.01)
    assert all(
        metric.reference.within_tolerance is None for metric in metrics.values()
    )


def test_only_primary_metric_reference_exposes_tolerance_verdict(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(
        _cfg(root, ds, sp, encoder="uni2"),
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.13,
            "test/croma_ltm10": 0.11,
        },
    )

    table = project_leaderboard(
        [run],
        LeaderboardFacet(vary=("encoder",)),
        benchmark=_MultiGateBenchmark(),
    )

    metrics = table.rows[0].metrics
    assert metrics["croma_median"].reference.within_tolerance is True
    assert metrics["croma_median"].reference.tolerance == pytest.approx(0.05)
    assert metrics["croma_f0"].reference.expected == pytest.approx(0.15)
    assert metrics["croma_ltm10"].reference.expected == pytest.approx(0.10)
    assert metrics["croma_f0"].reference.within_tolerance is None
    assert metrics["croma_ltm10"].reference.within_tolerance is None
    assert metrics["croma_f0"].reference.tolerance is None
    assert metrics["croma_ltm10"].reference.tolerance is None


def _multi_reference_table(tmp_path: Path) -> LeaderboardTable:
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    run = make_run_dir(
        _cfg(root, ds, sp, encoder="uni2"),
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.13,
            "test/croma_ltm10": 0.11,
        },
    )
    return project_leaderboard(
        [run],
        LeaderboardFacet(vary=("encoder",)),
        benchmark=_MultiReferenceBenchmark(),
    )


def test_multi_metric_json_emits_ordered_wide_schema(tmp_path: Path):
    data = json.loads(render_json(_multi_reference_table(tmp_path)))
    assert data["reported_metrics"] == [
        {"metric": "croma_median", "higher_is_better": True},
        {"metric": "croma_f0", "higher_is_better": False},
        {"metric": "croma_ltm10", "higher_is_better": True},
    ]
    assert data["ranking_metrics"] == [
        {"metric": "croma_median", "higher_is_better": True},
        {"metric": "croma_ltm10", "higher_is_better": True},
    ]
    json_row = data["rows"][0]
    assert "rank" not in json_row and "mean" not in json_row
    assert list(json_row["metrics"]) == ["croma_median", "croma_f0", "croma_ltm10"]
    assert json_row["metrics"]["croma_f0"]["reference"] == {
        "expected": 0.15,
        "delta": pytest.approx(-0.02),
        "tolerance": None,
        "pass": None,
        "source": "published",
        "label": "PathoROB",
        "url": "https://example.test/pathorob",
    }


def test_multi_metric_csv_emits_ordered_wide_schema(tmp_path: Path):
    lines = render_csv(_multi_reference_table(tmp_path)).splitlines()
    assert lines[0].split(",") == [
        "encoder",
        "croma_median_mean",
        "croma_median_std",
        "croma_median_n",
        "croma_median_seeds",
        "croma_median_reference_expected",
        "croma_median_reference_delta",
        "croma_median_reference_pass",
        "croma_f0_mean",
        "croma_f0_std",
        "croma_f0_n",
        "croma_f0_seeds",
        "croma_f0_reference_expected",
        "croma_f0_reference_delta",
        "croma_ltm10_mean",
        "croma_ltm10_std",
        "croma_ltm10_n",
        "croma_ltm10_seeds",
        "croma_ltm10_reference_expected",
        "croma_ltm10_reference_delta",
        "croma_median_rank",
        "croma_ltm10_rank",
        "ranking_eligible",
        "pareto",
        "config_diff",
    ]
    csv_row = lines[1].split(",")
    assert csv_row[5:8] == ["0.180000", "0.020000", ""]
    assert csv_row[12:14] == ["0.150000", "-0.020000"]


def test_multi_metric_plain_text_emits_ordered_wide_schema(tmp_path: Path):
    plain_header = format_table(_multi_reference_table(tmp_path)).splitlines()[1]
    assert plain_header.index("croma_median") < plain_header.index("croma_f0")
    assert plain_header.index("croma_f0") < plain_header.index("croma_ltm10")
    assert "eligible" in plain_header and "pareto" in plain_header


def test_multi_metric_html_emits_ordered_wide_schema(tmp_path: Path):
    html = render_html(_multi_reference_table(tmp_path))
    assert html.index("<th>croma_median</th>") < html.index("<th>croma_f0</th>")
    assert html.index("<th>croma_f0</th>") < html.index("<th>croma_ltm10</th>")
    assert "<th>eligible</th>" in html and "<th>pareto</th>" in html


def test_multi_metric_ties_use_standard_competition_ranks(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    runs = [
        make_run_dir(
            _cfg(root, ds, sp, encoder=encoder),
            {
                "test/croma_median": median,
                "test/croma_f0": 0.10,
                "test/croma_ltm10": ltm10,
            },
        )
        for encoder, median, ltm10 in (
            ("conch", 0.90, 0.70),
            ("phikon", 0.90, 0.80),
            ("uni2", 0.70, 0.90),
        )
    ]

    table = project_leaderboard(
        runs, LeaderboardFacet(vary=("encoder",)), benchmark=_MultiMetricBenchmark()
    )

    by_encoder = {row.vary_values["encoder"]: row for row in table.rows}
    assert [by_encoder[name].metrics["croma_median"].rank for name in ("conch", "phikon", "uni2")] == [1, 1, 3]
    assert [by_encoder[name].metrics["croma_ltm10"].rank for name in ("uni2", "phikon", "conch")] == [1, 2, 3]


def test_mixed_direction_pareto_uses_only_ranking_metrics(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    runs = [
        make_run_dir(
            _cfg(root, ds, sp, encoder=encoder),
            {"test/accuracy": accuracy, "test/mae": mae, "test/croma_f0": diagnostic},
        )
        for encoder, accuracy, mae, diagnostic in (
            ("conch", 0.90, 0.50, 0.20),
            ("phikon", 0.80, 0.40, 0.30),
            ("uni2", 0.70, 0.60, 0.01),
        )
    ]

    table = project_leaderboard(
        runs,
        LeaderboardFacet(vary=("encoder",)),
        benchmark=_MixedDirectionBenchmark(),
    )

    by_encoder = {row.vary_values["encoder"]: row for row in table.rows}
    assert by_encoder["conch"].metrics["accuracy"].rank == 1
    assert by_encoder["conch"].metrics["mae"].rank == 2
    assert by_encoder["phikon"].metrics["accuracy"].rank == 2
    assert by_encoder["phikon"].metrics["mae"].rank == 1
    assert by_encoder["conch"].pareto is True
    assert by_encoder["phikon"].pareto is True
    assert by_encoder["uni2"].pareto is False


def test_ranking_ineligible_control_is_visible_but_excluded_from_rank_and_pareto(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    runs = [
        make_run_dir(
            _cfg(root, ds, sp, encoder=encoder),
            {
                "test/croma_median": median,
                "test/croma_f0": f0,
                "test/croma_ltm10": ltm10,
            },
        )
        for encoder, median, f0, ltm10 in (
            ("uni2", 0.99, 0.01, 0.99),
            ("conch", 0.80, 0.20, 0.70),
            ("phikon", 0.70, 0.30, 0.60),
        )
    ]

    table = project_leaderboard(
        runs, LeaderboardFacet(vary=("encoder",)), benchmark=_ControlBenchmark()
    )

    by_encoder = {row.vary_values["encoder"]: row for row in table.rows}
    control = by_encoder["uni2"]
    assert control.ranking_eligible is False
    assert control.pareto is None
    assert all(metric.rank is None for metric in control.metrics.values())
    assert control.metrics["croma_median"].mean == pytest.approx(0.99)
    assert control.metrics["croma_median"].reference.expected == pytest.approx(0.18)
    assert by_encoder["conch"].metrics["croma_median"].rank == 1
    assert by_encoder["phikon"].metrics["croma_median"].rank == 2
    assert "control (ranking-ineligible)" in format_table(table)


def test_multi_metric_rows_have_neutral_deterministic_order(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    runs = [
        make_run_dir(
            _cfg(root, ds, sp, encoder=encoder),
            {
                "test/croma_median": median,
                "test/croma_f0": 0.10,
                "test/croma_ltm10": ltm10,
            },
        )
        for encoder, median, ltm10 in (
            ("uni2", 0.99, 0.99),
            ("conch", 0.50, 0.50),
            ("phikon", 0.80, 0.80),
        )
    ]

    table = project_leaderboard(
        list(reversed(runs)),
        LeaderboardFacet(vary=("encoder",)),
        benchmark=_MultiMetricBenchmark(),
    )

    assert [row.vary_values["encoder"] for row in table.rows] == [
        "conch",
        "phikon",
        "uni2",
    ]
    assert all(row.rank is None for row in table.rows)


def test_metric_override_uses_scalar_schema_without_ranking_diagnostic(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    runs = [
        make_run_dir(
            _cfg(root, ds, sp, encoder=encoder),
            {
                "test/croma_median": 0.50,
                "test/croma_f0": f0,
                "test/croma_ltm10": 0.50,
            },
        )
        for encoder, f0 in (("uni2", 0.01), ("conch", 0.20))
    ]

    table = project_leaderboard(
        runs,
        LeaderboardFacet(vary=("encoder",)),
        metric="croma_f0",
        benchmark=_MultiMetricBenchmark(),
    )

    assert table.reported_metrics == ()
    assert [row.vary_values["encoder"] for row in table.rows] == ["conch", "uni2"]
    assert [row.rank for row in table.rows] == [None, None]
    assert "reported_metrics" not in json.loads(render_json(table))
    assert "None" not in format_table(table)
    assert "None" not in render_html(table)
    assert render_csv(table).splitlines()[1].startswith(",conch,")


def test_metric_override_keeps_control_ineligible_for_scalar_rank(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    runs = [
        make_run_dir(
            _cfg(root, ds, sp, encoder=encoder),
            {
                "test/croma_median": median,
                "test/croma_f0": 0.10,
                "test/croma_ltm10": 0.50,
            },
        )
        for encoder, median in (("uni2", 0.99), ("conch", 0.80))
    ]

    table = project_leaderboard(
        runs,
        LeaderboardFacet(vary=("encoder",)),
        metric="croma_median",
        benchmark=_ControlBenchmark(),
    )

    by_encoder = {row.vary_values["encoder"]: row for row in table.rows}
    assert by_encoder["uni2"].ranking_eligible is False
    assert by_encoder["uni2"].rank is None
    assert by_encoder["conch"].rank == 1


@pytest.mark.parametrize("benchmark", [_SingleMetricBenchmark(), _LegacyMetricBenchmark()])
def test_single_or_undeclared_metric_benchmark_retains_scalar_model(
    tmp_path: Path, benchmark
):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    runs = [
        make_run_dir(
            _cfg(root, ds, sp, encoder=encoder), {"test/accuracy": accuracy}
        )
        for encoder, accuracy in (("uni2", 0.80), ("conch", 0.90))
    ]

    actual = project_leaderboard(
        runs,
        LeaderboardFacet(vary=("encoder",)),
        benchmark=benchmark,
    )
    uni2 = load_run_record(runs[0])
    conch = load_run_record(runs[1])
    expected = LeaderboardTable(
        triple=conch.triple,
        metric="accuracy",
        higher_is_better=True,
        vary=("encoder",),
        split="test",
        rows=[
            LeaderboardRow(
                rank=1,
                experiment_id=conch.experiment_id,
                vary_values={"encoder": "conch"},
                metric="accuracy",
                mean=0.90,
                std=None,
                n=1,
                seeds=(0,),
                config_diff={},
            ),
            LeaderboardRow(
                rank=2,
                experiment_id=uni2.experiment_id,
                vary_values={"encoder": "uni2"},
                metric="accuracy",
                mean=0.80,
                std=None,
                n=1,
                seeds=(0,),
                config_diff={},
            ),
        ],
    )

    assert actual == expected


@pytest.fixture()
def keyed_benchmark():
    bench = _KeyedBenchmark()
    register_benchmark(bench)
    try:
        yield bench
    finally:
        registry_mod._REGISTRY.pop(bench.name, None)


@pytest.fixture()
def multi_metric_benchmark():
    benchmark = _MultiMetricBenchmark()
    register_benchmark(benchmark)
    try:
        yield benchmark
    finally:
        registry_mod._REGISTRY.pop(benchmark.name, None)


def test_keyed_reference_joins_on_varied_axis_with_pass_fail(tmp_path: Path, keyed_benchmark):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    # uni2 measured 0.79 vs expected 0.80±0.03 -> PASS; virchow2 measured 0.70 vs 0.90±0.03 -> FAIL.
    uni = make_run_dir(_cfg(root, ds, sp, encoder="uni2"), {"test/accuracy": 0.79})
    vir = make_run_dir(_cfg(root, ds, sp, encoder="virchow2"), {"test/accuracy": 0.70})

    table = project_leaderboard([uni, vir], LeaderboardFacet(vary=("encoder",)), benchmark=keyed_benchmark)
    by_encoder = {r.vary_values["encoder"]: r for r in table.rows}

    assert by_encoder["uni2"].reference_expected == pytest.approx(0.80)
    assert by_encoder["uni2"].reference_pass is True
    assert by_encoder["virchow2"].reference_expected == pytest.approx(0.90)
    assert by_encoder["virchow2"].reference_pass is False
    # The keyed benchmark also carries a banner row.
    assert table.banner is not None and table.banner.source == "banner"


# --- render outputs ------------------------------------------------------------------


def _legacy_scalar_table() -> LeaderboardTable:
    row = LeaderboardRow(
        rank=1,
        experiment_id="exp-a",
        vary_values={"encoder": "uni2"},
        metric="accuracy",
        mean=0.8,
        std=None,
        n=1,
        seeds=(0,),
        config_diff={},
    )
    return LeaderboardTable(
        triple=("dataset", "splits", "binary_classification"),
        metric="accuracy",
        higher_is_better=True,
        vary=("encoder",),
        split="test",
        rows=[row],
    )


def test_legacy_scalar_plain_text_matches_exact_golden():
    assert format_table(_legacy_scalar_table()) == (
        "Leaderboard — task=binary_classification · metric=accuracy "
        "(higher is better) · split=test\n"
        "#  encoder  accuracy  n  config diff\n"
        "1  uni2     0.8000    1  —          "
    )


def test_legacy_scalar_csv_matches_exact_golden():
    assert render_csv(_legacy_scalar_table()) == (
        "rank,encoder,mean,std,n,seeds,config_diff\r\n"
        "1,uni2,0.800000,,1,0,\r\n"
    )


def test_legacy_scalar_json_matches_exact_golden():
    assert render_json(_legacy_scalar_table()) == """{
  "triple": {
    "dataset_checksum": "dataset",
    "splits_checksum": "splits",
    "task": "binary_classification"
  },
  "metric": "accuracy",
  "higher_is_better": true,
  "split": "test",
  "vary": [
    "encoder"
  ],
  "banner": null,
  "guidance": [],
  "rows": [
    {
      "rank": 1,
      "experiment_id": "exp-a",
      "vary": {
        "encoder": "uni2"
      },
      "metric": "accuracy",
      "mean": 0.8,
      "std": null,
      "n": 1,
      "seeds": [
        0
      ],
      "config_diff": {},
      "reference": null
    }
  ]
}"""


def test_legacy_scalar_html_matches_exact_golden():
    rendered = render_html(_legacy_scalar_table()).encode()
    assert len(rendered) == 14890
    assert sha256(rendered).hexdigest() == (
        "29163d82fce3b9d0c84ec5d7c11067a9fb944e94736ddbdbba310435422bfc5c"
    )


def test_write_leaderboard_emits_csv_json_html(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    a = make_run_dir(_cfg(root, ds, sp, encoder="uni2"), {"test/accuracy": 0.80})
    b = make_run_dir(_cfg(root, ds, sp, encoder="virchow2"), {"test/accuracy": 0.90})
    table = project_leaderboard([a, b], LeaderboardFacet(vary=("encoder",)))

    paths = write_leaderboard(table, root)
    for kind in ("csv", "json", "html"):
        assert paths[kind].exists()
        assert paths[kind].parent == root / "leaderboards"

    assert "virchow2" in render_csv(table).splitlines()[1]  # winner first
    assert json.loads(render_json(table))["rows"][0]["vary"]["encoder"] == "virchow2"
    assert "<table" in render_html(table)
    assert "virchow2" in format_table(table)


# --- concurrency: simultaneous completion loses no rows ------------------------------


def test_concurrent_completions_lose_no_rows(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    n_runs = len(ENC)
    configs = [_cfg(root, ds, sp, encoder=ENC[i]) for i in range(n_runs)]
    barrier = threading.Barrier(n_runs)

    def _finish(index: int) -> None:
        barrier.wait()  # release all writers at the same instant
        make_run_dir(configs[index], {"test/accuracy": 0.5 + index / 100.0}, run_id=f"sim_{index:02d}")

    threads = [threading.Thread(target=_finish, args=(i,)) for i in range(n_runs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    triples = discover_triples(root)
    assert len(triples) == 1  # all share one dataset+splits+task
    run_dirs = next(iter(triples.values()))
    assert len(run_dirs) == n_runs

    table = project_leaderboard(run_dirs, LeaderboardFacet(vary=("encoder",)))
    # No shared mutable index means every distinct config survives as its own row.
    assert len(table.rows) == n_runs
    assert sum(r.n for r in table.rows) == n_runs
    assert {r.vary_values["encoder"] for r in table.rows} == set(ENC)


# --- known-ranking reproduction over a fixture output_root ---------------------------


def test_fixture_output_root_reproduces_known_ranking(tmp_path: Path):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    known = {"virchow2": 0.91, "uni2": 0.87, "phikon": 0.83}
    for encoder, score in known.items():
        make_run_dir(_cfg(root, ds, sp, encoder=encoder), {"test/accuracy": score})

    run_dirs = next(iter(discover_triples(root).values()))
    table = project_leaderboard(run_dirs, LeaderboardFacet(vary=("encoder",)))
    assert [r.vary_values["encoder"] for r in table.rows] == ["virchow2", "uni2", "phikon"]
    assert [round(r.mean, 2) for r in table.rows] == [0.91, 0.87, 0.83]


# --- CLI verb ------------------------------------------------------------------------


def _run_cli(argv: list[str]) -> int:
    from soma.cli import main

    try:
        main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    return 0


def test_cli_leaderboard_renders_and_writes(tmp_path: Path, capsys):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    make_run_dir(_cfg(root, ds, sp, encoder="uni2"), {"test/accuracy": 0.80})
    make_run_dir(_cfg(root, ds, sp, encoder="virchow2"), {"test/accuracy": 0.90})

    code = _run_cli(["leaderboard", "--root", str(root), "--vary", "encoder"])
    out = capsys.readouterr().out

    assert code == 0
    assert "virchow2" in out  # winner shown
    assert "Wrote:" in out
    assert (root / "leaderboards").is_dir()
    assert list((root / "leaderboards").glob("*.html"))


def test_cli_multiple_triples_requires_disambiguation(tmp_path: Path, capsys):
    root = tmp_path / "out"
    ds_a = _dataset_csv(tmp_path, "ds_a", "sample_id,image_path,label\ns0,/a.svs,tumor\n")
    ds_b = _dataset_csv(tmp_path, "ds_b", "sample_id,image_path,label\ns0,/b.svs,tumor\n")
    sp = _splits_csv(tmp_path)
    make_run_dir(_cfg(root, ds_a, sp, encoder="uni2"), {"test/accuracy": 0.8})
    make_run_dir(_cfg(root, ds_b, sp, encoder="uni2"), {"test/accuracy": 0.7})

    code = _run_cli(["leaderboard", "--root", str(root), "--vary", "encoder"])
    err = capsys.readouterr().err
    assert code == 2
    assert "triples" in err


def test_cli_like_selects_one_triple(tmp_path: Path, capsys):
    root = tmp_path / "out"
    ds_a = _dataset_csv(tmp_path, "ds_a", "sample_id,image_path,label\ns0,/a.svs,tumor\n")
    ds_b = _dataset_csv(tmp_path, "ds_b", "sample_id,image_path,label\ns0,/b.svs,tumor\n")
    sp = _splits_csv(tmp_path)
    like = make_run_dir(_cfg(root, ds_a, sp, encoder="uni2"), {"test/accuracy": 0.81})
    make_run_dir(_cfg(root, ds_b, sp, encoder="uni2"), {"test/accuracy": 0.72})

    code = _run_cli(["leaderboard", "--root", str(root), "--vary", "encoder", "--like", str(like)])
    out = capsys.readouterr().out
    assert code == 0
    assert "0.81" in out  # only the --like triple rendered
    assert "0.72" not in out


def test_cli_benchmark_uses_canonical_facet(tmp_path: Path, capsys, keyed_benchmark):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    make_run_dir(_cfg(root, ds, sp, encoder="uni2"), {"test/accuracy": 0.79})
    make_run_dir(_cfg(root, ds, sp, encoder="virchow2"), {"test/accuracy": 0.70})

    code = _run_cli(["leaderboard", "keyed_fixture", "--root", str(root)])
    out = capsys.readouterr().out
    assert code == 0
    assert "reference band" in out  # banner from the benchmark
    assert "PASS" in out and "FAIL" in out  # keyed per-row tolerance verdicts


def test_cli_metric_requests_scalar_schema_without_promoting_diagnostic(
    tmp_path: Path, capsys, multi_metric_benchmark
):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    make_run_dir(
        _cfg(root, ds, sp, encoder="uni2"),
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.13,
            "test/croma_ltm10": 0.11,
        },
    )

    code = _run_cli(
        [
            "leaderboard",
            multi_metric_benchmark.name,
            "--root",
            str(root),
            "--metric",
            "croma_f0",
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(
        (root / "leaderboards" / f"{multi_metric_benchmark.name}.json").read_text()
    )

    assert code == 0
    assert "None" not in out
    assert "reported_metrics" not in payload
    assert payload["rows"][0]["rank"] is None


def test_cli_multi_metric_reports_logical_sample_with_no_completed_attempt(
    tmp_path: Path, capsys, multi_metric_benchmark
):
    root = tmp_path / "out"
    ds, sp = _dataset_csv(tmp_path), _splits_csv(tmp_path)
    make_run_dir(
        _cfg(root, ds, sp, encoder="uni2"),
        {
            "test/croma_median": 0.20,
            "test/croma_f0": 0.13,
            "test/croma_ltm10": 0.11,
        },
    )
    incomplete = make_run_dir(
        _cfg(root, ds, sp, encoder="phikon"),
        {"test/croma_median": 0.30},
        status="running",
    )
    experiment_id = load_run_record(incomplete).experiment_id

    code = _run_cli(
        ["leaderboard", multi_metric_benchmark.name, "--root", str(root)]
    )
    error = capsys.readouterr().err

    assert code == 2
    assert experiment_id in error
    assert "croma_median, croma_f0, croma_ltm10" in error


def test_cli_unknown_benchmark_exits_nonzero(tmp_path: Path, capsys):
    code = _run_cli(["leaderboard", "nope", "--root", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown benchmark" in err
