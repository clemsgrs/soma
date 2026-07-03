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


def make_run_dir(config: PipelineConfig, summary: dict[str, float], *, run_id: str | None = None) -> Path:
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
        status="completed",
        summary_metrics=summary,
    ).with_updates(finished_at="2026-07-03T00:00:00+00:00")
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


@pytest.fixture()
def keyed_benchmark():
    bench = _KeyedBenchmark()
    register_benchmark(bench)
    try:
        yield bench
    finally:
        registry_mod._REGISTRY.pop(bench.name, None)


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


def test_cli_unknown_benchmark_exits_nonzero(tmp_path: Path, capsys):
    code = _run_cli(["leaderboard", "nope", "--root", str(tmp_path)])
    err = capsys.readouterr().err
    assert code == 2
    assert "Unknown benchmark" in err
