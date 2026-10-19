"""HEST's closed-form probe as a benchmark-side predictor over the verbs.

``HestBenchmark.run`` extracts a rank-1 tile set source through soma's feature cache, fits
``RidgePCAProbe`` on each fold's train spots, predicts the test spots and scores them with
``soma.evaluate`` (``mean_pearson`` + one Pearson per gene). Synthetic spots carry gene
expression that is a linear function of their colour, so only a probe that receives each
spot's own features and expression can correlate.

The same runner serves both benchmark entry points: ``soma reproduce`` / ``run_benchmark``
call the registered benchmark's ``run``, and ``run_benchmark_spec`` calls the ``run`` of an
external ``BenchmarkSpec`` that wraps the recipe.
"""

from __future__ import annotations

import functools
import json

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from soma.benchmark import BenchmarkSpec, get_benchmark, run_benchmark, run_benchmark_spec
from soma.cli import main as soma_main
from slide2vec import ExecutionOptions
from soma.benchmark.leaderboard import axis_value, load_run_record
from tests.e2e.harness import CPU_EXECUTION

GENES = ["ERBB2", "ESR1", "MKI67"]


def _spots(root, n: int = 40, size: int = 224, seed: int = 0, folds: int = 1):
    """``n`` one-colour spots whose expression is a linear function of their colour.

    One fold holds the last 10 spots out for test. ``folds=2`` adds a second fold that
    holds the first 10 out, so the summary carries HEST's fold-averaged headline
    (``test/mean_pearson_mean``)."""
    rng = np.random.default_rng(seed)
    tiles = root / "tiles"
    tiles.mkdir(parents=True)
    colours = rng.uniform(40, 215, size=(n, 3))
    weight = np.array([[1.0, 0.2, -0.5], [-0.3, 1.0, 0.4], [0.6, -0.4, 1.0]])
    expression = np.log1p(np.abs(colours / 64.0 @ weight + 0.05 * rng.standard_normal((n, 3))))
    ids = [f"spot{i:02d}" for i in range(n)]
    for sample_id, colour in zip(ids, colours):
        image = np.broadcast_to(colour.astype(np.uint8), (size, size, 3)).copy()
        Image.fromarray(image).save(tiles / f"{sample_id}.png")
    pd.DataFrame(
        {
            "sample_id": ids,
            "image_path": [str(tiles / f"{s}.png") for s in ids],
            "target_index": list(range(n)),
            "spacing_at_level_0": 0.5,
        }
    ).to_csv(root / "dataset.csv", index=False)
    np.save(root / "targets.npy", expression)
    (root / "genes.json").write_text(json.dumps(GENES))
    split = ["train"] * 30 + ["test"] * (n - 30)
    splits = pd.DataFrame({"sample_id": ids, "split": split})
    if folds == 2:
        second = pd.DataFrame({"sample_id": ids, "split": ["test"] * 10 + ["train"] * (n - 10)})
        splits = pd.concat([splits.assign(fold=0), second.assign(fold=1)], ignore_index=True)
    splits.to_csv(root / "splits.csv", index=False)
    return root / "dataset.csv", root / "splits.csv"


def _cpu_sized(monkeypatch, benchmark) -> None:
    """Shrink the registered recipe to the synthetic spots: PCA dim 2 (the ``fake``
    encoder's features are 3-d) and CPU execution. The recipe is otherwise untouched."""
    monkeypatch.setattr(
        benchmark,
        "build_config",
        functools.partial(
            benchmark.build_config, pca_components=2, execution=ExecutionOptions(**CPU_EXECUTION)
        ),
    )


def test_reproduce_hest_runs_the_benchmark_runner_from_the_cli_and_run_benchmark(
    tmp_path, monkeypatch, encoder_name, new_artifact
):
    """``soma reproduce hest/IDC`` and ``run_benchmark`` run HEST through its own runner (no
    ``Pipeline`` trains the probe) and score the same fold-averaged Pearson."""
    artifact = new_artifact("reproduce_hest")
    curated = tmp_path / "curated"
    _spots(curated, folds=2)
    benchmark = get_benchmark("hest/IDC")
    _cpu_sized(monkeypatch, benchmark)
    code = 0
    try:
        soma_main(
            [
                "reproduce",
                "hest/IDC",
                "--curated-dir",
                str(curated),
                "--encoder",
                encoder_name,
                "--output-root",
                str(tmp_path / "cli"),
                "--cache-root",
                str(tmp_path / "cache"),
            ]
        )
    except SystemExit as exc:
        code = int(exc.code or 0)
    artifact.check_equal("cli/exit_code", code, 0)
    cli_score = benchmark.score(tmp_path / "cli" / "seed_0")[benchmark.primary_metric]
    artifact.check_at_least("cli/test/mean_pearson_mean", cli_score, 0.8)

    result = run_benchmark(
        "hest/IDC",
        encoder=encoder_name,
        curated_dir=curated,
        output_root=tmp_path / "api",
        cache_root=tmp_path / "cache",
        return_result=True,
    )

    artifact.check_equal("api/status", result.status, 0)
    artifact.check_equal("api/metrics", [m.metric for m in result.metrics], ["test/mean_pearson_mean"])
    artifact.check_equal("api/score_matches_cli", result.metrics[0].measured, cli_score)
    artifact.assert_passed()


def test_hest_run_composes_extract_fit_predict_evaluate(tmp_path, encoder_name, new_artifact):
    artifact = new_artifact("hest_probe_verbs")
    dataset_csv, splits_csv = _spots(tmp_path / "data")
    benchmark = get_benchmark("hest/IDC")
    config = benchmark.build_config(
        encoder=encoder_name,
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "run",
        overrides={"cache": {"enabled": True, "root_dir": str(tmp_path / "cache")}},
        pca_components=2,
        execution=ExecutionOptions(**CPU_EXECUTION),
    )

    summary = benchmark.run(config).summary

    (run,) = (tmp_path / "run").glob("experiments/*/runs/*")
    metrics = json.loads((run / "metrics.json").read_text())["test"]
    artifact.check_equal("summary/keys", sorted(summary)[:1], ["test/mean_pearson"])
    artifact.check_equal(
        "per_gene_keys", sorted(k for k in metrics if k.startswith("pearson_gene")),
        [f"pearson_gene{k:02d}_{gene}" for k, gene in enumerate(GENES)],
    )
    artifact.check_equal(
        "mean_is_mean_over_genes",
        round(metrics["mean_pearson"], 9),
        round(float(np.mean([metrics[f"pearson_gene{k:02d}_{g}"] for k, g in enumerate(GENES)])), 9),
    )
    artifact.check_at_least("test/mean_pearson", summary["test/mean_pearson"], 0.8)
    artifact.check_equal(
        "score/reads_summary",
        benchmark.score(tmp_path / "run")["test/mean_pearson"],
        summary["test/mean_pearson"],
    )
    artifact.check_equal("predictions/written", (run / "predictions_test.parquet").is_file(), True)
    # Managed run metadata, as a Pipeline run writes it: the leaderboard and
    # ``reproduce --from-run-dir`` read the run's identity from these.
    artifact.check_equal("metadata/experiment_json", (run.parent.parent / "experiment.json").is_file(), True)
    run_yaml = run / "run.yaml"
    status = yaml.safe_load(run_yaml.read_text())["status"] if run_yaml.is_file() else None
    artifact.check_equal("metadata/run_status", status, "completed")
    artifact.check_equal("metadata/run_record", load_run_record(run) is not None, True)
    axes = {axis: axis_value(load_run_record(run).canonical_spec, axis) for axis in benchmark.facet.varied}
    artifact.check_equal("metadata/encoder_axis", axes, {"encoder": encoder_name})
    artifact.assert_passed()


def test_run_benchmark_spec_runs_hest_through_the_specs_runner(
    tmp_path, monkeypatch, encoder_name, new_artifact
):
    """An external harness wraps the registered recipe in a ``BenchmarkSpec`` and passes its
    runner: ``run_benchmark_spec`` runs HEST with it, never builds a ``Pipeline``, and
    returns the manifest checksums the run recorded in ``experiment.json`` and ``run.yaml``."""
    import soma.pipeline

    artifact = new_artifact("benchmark_spec_hest")
    dataset_csv, splits_csv = _spots(tmp_path / "data", folds=2)
    recipe = get_benchmark("hest/IDC")
    _cpu_sized(monkeypatch, recipe)

    class NoPipeline:
        def __init__(self, config):
            raise AssertionError("run_benchmark_spec built a Pipeline for a spec with a runner")

    monkeypatch.setattr(soma.pipeline, "Pipeline", NoPipeline)
    spec = BenchmarkSpec(
        name="fmtf/hest-IDC",
        canonical_seeds=recipe.canonical_seeds,
        primary_metric=recipe.primary_metric,
        build_config=recipe.build_config,
        score=recipe.score,
        run=getattr(recipe, "run", None),
    )

    result = run_benchmark_spec(
        spec,
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        encoder=encoder_name,
        output_root=tmp_path / "evidence",
    )

    (run_dir,) = (tmp_path / "evidence" / "seed_0").glob("experiments/*/runs/*")
    experiment = json.loads((run_dir.parent.parent / "experiment.json").read_text())
    recorded = yaml.safe_load((run_dir / "run.yaml").read_text())
    artifact.check_equal("status", result.status, 0)
    artifact.check_equal("metrics", [m.metric for m in result.metrics], ["test/mean_pearson_mean"])
    artifact.check_at_least("test/mean_pearson_mean", result.metrics[0].measured, 0.8)
    artifact.check_equal(
        "checksums/non_empty",
        all((result.dataset_checksum, result.splits_checksum, result.test_checksum)),
        True,
    )
    artifact.check_equal(
        "checksums/dataset_as_recorded", result.dataset_checksum == experiment["dataset_checksum"], True
    )
    artifact.check_equal(
        "checksums/splits_as_recorded", result.splits_checksum == experiment["splits_checksum"], True
    )
    artifact.check_equal("checksums/test_as_recorded", result.test_checksum == recorded["test_checksum"], True)
    artifact.assert_passed()

def _twin_encoder(name: str) -> str:
    """Register a second name for the ``fake`` encoder: an encoder axis value of its own."""
    from slide2vec.encoders.registry import encoder_registry

    twin = f"{name}-twin"
    metadata = {k: v for k, v in encoder_registry.info(name).items() if k != "name"}
    encoder_registry.register(twin, encoder_registry.require(name), metadata=metadata)
    return twin


def test_hest_runs_of_two_encoders_under_one_root_keep_their_own_outputs(tmp_path, encoder_name):
    """``soma reproduce`` hands every encoder the same ``seed_<n>`` root: each run lands in
    its own managed run directory, so a second encoder never overwrites the first."""
    from slide2vec.encoders.registry import encoder_registry

    dataset_csv, splits_csv = _spots(tmp_path / "data")
    benchmark = get_benchmark("hest/IDC")
    twin = _twin_encoder(encoder_name)
    try:
        run_dirs = {}
        for encoder in (encoder_name, twin):
            config = benchmark.build_config(
                encoder=encoder,
                dataset_csv=dataset_csv,
                splits_csv=splits_csv,
                output_root=tmp_path / "seed_0",
                overrides={"cache": {"enabled": True, "root_dir": str(tmp_path / "cache")}},
                pca_components=2,
                execution=ExecutionOptions(**CPU_EXECUTION),
            )
            benchmark.run(config)
            (run_dir,) = {p.parent for p in (tmp_path / "seed_0").glob("experiments/*/runs/*/config.yaml")} - set(
                run_dirs.values()
            )
            run_dirs[encoder] = run_dir
    finally:
        encoder_registry._entries.pop(twin, None)

    assert run_dirs[encoder_name] != run_dirs[twin]
    for encoder, run_dir in run_dirs.items():
        assert (run_dir / "summary.json").is_file()
        assert (run_dir / "predictions_test.parquet").is_file()
        assert f"name: {encoder}\n" in (run_dir / "config.yaml").read_text()
