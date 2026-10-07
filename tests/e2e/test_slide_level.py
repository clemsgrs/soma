"""Slide-level MIL: synthetic WSIs -> tiling -> extraction -> cache -> train -> evaluate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soma.cli import main as soma_main

from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma, sha256
from soma.features import FeatureStore
from tests.e2e.synthetic import (
    SPACING_UM,
    SlideCohort,
    make_slide_cohort,
    write_coordinates_artifact,
)

TILE_PX = 32


@pytest.fixture(scope="module")
def cohort(tmp_path_factory) -> SlideCohort:
    return make_slide_cohort(tmp_path_factory.mktemp("slide-cohort"))


def _config(
    cohort: SlideCohort,
    workdir: Path,
    encoder_name: str,
    *,
    task: dict,
    metrics: list[str],
    cache_root: Path,
    aggregator: str = "abmil",
    splits_csv: Path | None = None,
    epochs: int = 25,
    learning_rate: float = 5e-3,
) -> dict:
    return {
        "run": {"output_root": str(workdir / "runs"), "seed": 0},
        "data": {
            "dataset_csv": str(cohort.manifest(task["name"])),
            "splits_csv": str(splits_csv or cohort.splits_csv),
            "dataset_type": "slide",
        },
        "preprocessing": {
            "backend": "openslide",
            "requested_tile_size_px": TILE_PX,
            "requested_spacing_um": SPACING_UM,
            "tissue_method": "otsu",
            "seg_downsample": 16,
            "a_t": 1,
            "min_coverage": {"tissue": 0.5},
        },
        "execution": CPU_EXECUTION,
        "cache": {"enabled": True, "root_dir": str(cache_root)},
        "encoder": {"name": encoder_name, "precision": "fp32", "batch_size": 256},
        "aggregation": {"name": aggregator},
        "task": task,
        "evaluation": {"metrics": metrics},
        "normalization": {"method": "zscore"},
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": 1,
            "patience": epochs,
            **CPU_LOADER,
        },
    }


def test_slide_classification_is_learned_cached_and_repeatable(
    cohort, tmp_path, encoder_name, encoded_images, new_artifact
):
    """The flagship journey, run twice from the same config.

    The first run extracts every slide; the second must be served entirely from the
    feature cache (zero images encoded) and reproduce the first run's predictions
    byte for byte.
    """
    artifact = new_artifact("slide_mil_binary_classification")
    config = _config(
        cohort,
        tmp_path,
        encoder_name,
        task={"name": "binary_classification"},
        metrics=["auroc", "balanced_accuracy"],
        cache_root=tmp_path / "cache",
    )

    cold = run_soma(config, tmp_path / "config.yaml")
    tiles_encoded_cold = sum(encoded_images)
    encoded_images.clear()
    warm = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("cold", cold)
    artifact.check_training_reduced_loss("cold", cold)
    artifact.facts["tiles_encoded_cold"] = tiles_encoded_cold
    artifact.facts["cache_keys"] = sorted(p.name for p in (tmp_path / "cache").glob("*/*"))
    artifact.check_at_least("cold/test/auroc", cold.summary["test/auroc"], 0.95)
    artifact.check_at_least(
        "cold/test/balanced_accuracy", cold.summary["test/balanced_accuracy"], 0.8
    )
    artifact.check_equal("cold/test/num_samples", cold.summary["test/num_samples"], 8.0)
    artifact.check_at_least("cold/tiles_encoded", tiles_encoded_cold, 1)
    artifact.check_equal("warm/tiles_encoded", sum(encoded_images), 0)
    artifact.check_equal(
        "warm/predictions_identical",
        artifact.runs["cold"]["predictions_sha256"],
        {k: sha256(p) for k, p in warm.predictions().items()},
    )
    artifact.check_equal("warm/metrics_identical", warm.summary, cold.summary)
    for name in ("config.yaml", "best_model.pt", "report.html", "predictions_test.csv"):
        artifact.check_equal(f"cold/bundle/{name}", (cold.run_dir / name).is_file(), True)
    artifact.assert_passed()


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("slide-cache")


@pytest.mark.parametrize(
    ("scenario", "task", "metric", "threshold", "extra"),
    [
        (
            "multiclass_classification",
            {"name": "multiclass_classification"},
            "balanced_accuracy",
            0.75,
            {"epochs": 100},
        ),
        (
            "ordinal_classification",
            {"name": "ordinal_classification"},
            "qwk",
            0.75,
            {"epochs": 150},
        ),
        (
            "regression",
            {"name": "regression"},
            "spearman",
            0.8,
            {"epochs": 100, "learning_rate": 2e-2},
        ),
        ("survival_nll", {"name": "survival", "params": {"num_bins": 4}}, "c_index", 0.75, {}),
        (
            "survival_cox",
            {"name": "survival", "params": {"loss": "cox", "cox_window": 6}},
            "c_index",
            0.75,
            {"learning_rate": 5e-3, "epochs": 20},
        ),
    ],
)
def test_slide_task_heads_learn_from_the_same_features(
    scenario,
    task,
    metric,
    threshold,
    extra,
    cohort,
    shared_cache,
    tmp_path,
    encoder_name,
    new_artifact,
):
    """Each task head trains on one cohort; labels differ, the cached features do not."""
    artifact = new_artifact(f"slide_mil_{scenario}")
    config = _config(
        cohort,
        tmp_path,
        encoder_name,
        task=task,
        metrics=[metric, "mae"] if scenario == "regression" else [metric],
        cache_root=shared_cache,
        **extra,
    )
    run = run_soma(config, tmp_path / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least(f"test/{metric}", run.summary[f"test/{metric}"], threshold)
    if scenario == "regression":
        # Targets span 0-10: ranking alone is not enough, the scale must be learned too.
        artifact.check_at_most("test/mae", run.summary["test/mae"], 1.5)
    artifact.assert_passed()


def test_cross_validated_aggregators_are_ranked_on_a_leaderboard(
    cohort, shared_cache, tmp_path, encoder_name, new_artifact
):
    """Two aggregators, 2-fold CV each, into one output root; the leaderboard ranks them."""
    artifact = new_artifact("slide_mil_cv_leaderboard")
    splits_csv = cohort.cv_splits(2)
    runs = {}
    for aggregator in ("abmil", "mean_pool"):
        config = _config(
            cohort,
            tmp_path,
            encoder_name,
            task={"name": "binary_classification"},
            metrics=["auroc"],
            cache_root=shared_cache,
            aggregator=aggregator,
            splits_csv=splits_csv,
        )
        runs[aggregator] = run_soma(config, tmp_path / f"{aggregator}.yaml")
        artifact.record_run(aggregator, runs[aggregator])
        artifact.check_training_reduced_loss(aggregator, runs[aggregator])
        for fold in (0, 1):
            fold_dir = runs[aggregator].run_dir / f"fold_{fold}"
            artifact.check_equal(
                f"{aggregator}/fold_{fold}/metrics.json", (fold_dir / "metrics.json").is_file(), True
            )
        artifact.check_at_least(
            f"{aggregator}/test/auroc_mean", runs[aggregator].summary["test/auroc_mean"], 0.9
        )

    with pytest.raises(SystemExit) as exit_info:
        soma_main(["leaderboard", "--root", str(tmp_path / "runs"), "--vary", "aggregator"])
    artifact.check_equal("leaderboard/exit_code", exit_info.value.code, 0)
    (board_json,) = (tmp_path / "runs" / "leaderboards").glob("*.json")
    board = json.loads(board_json.read_text())
    artifact.facts["leaderboard"] = board
    rows = {row["vary"]["aggregator"]: row for row in board["rows"]}
    artifact.check_equal("leaderboard/aggregators", sorted(rows), ["abmil", "mean_pool"])
    for aggregator, run in runs.items():
        artifact.check_equal(
            f"leaderboard/{aggregator}/mean",
            rows.get(aggregator, {}).get("mean"),
            run.summary["test/auroc_mean"],
        )
    artifact.assert_passed()


def test_flat_png_slides_render_their_previews(cohort, tmp_path, encoder_name, new_artifact):
    """PNG slides take their spacing from the manifest, previews included.

    Both previews are on by default and reopen each slide; one that drops the
    manifest's ``spacing_at_level_0`` cannot open a PNG and fails the slide.
    """
    artifact = new_artifact("slide_mil_flat_png")
    config = _config(
        cohort,
        tmp_path,
        encoder_name,
        task={"name": "binary_classification"},
        metrics=["auroc"],
        cache_root=tmp_path / "cache",
    )
    config["data"]["dataset_csv"] = str(cohort.flat_manifest("binary_classification"))
    config["preprocessing"]["backend"] = "auto"
    run = run_soma(config, tmp_path / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("test/auroc", run.summary["test/auroc"], 0.95)
    for kind in ("mask", "tiling"):
        previews = list(tmp_path.glob(f"tiling_cache/*/previews/{kind}/*.jpg"))
        artifact.check_equal(f"previews/{kind}", len(previews), len(cohort.sample_ids))
    artifact.assert_passed()


def _tissue_cells(size: int = 384, tile: int = TILE_PX) -> list[tuple[int, int]]:
    """Level-0 origins of the grid cells that lie wholly inside the cohort's tissue disc."""
    centre, radius = size / 2, size * 0.4
    cells = []
    for y in range(0, size - tile + 1, tile):
        for x in range(0, size - tile + 1, tile):
            corners = [(x, y), (x + tile, y), (x, y + tile), (x + tile, y + tile)]
            if all((cx - centre) ** 2 + (cy - centre) ** 2 < radius**2 for cx, cy in corners):
                cells.append((x, y))
    return cells


def _bring_own_coordinates(cohort: SlideCohort, root: Path) -> tuple[Path, dict[str, int]]:
    """Write one tiling artifact per slide (3 to 8 tissue tiles) and a manifest naming it."""
    rng = np.random.default_rng(0)
    cells = _tissue_cells()
    frame = pd.read_csv(cohort.manifest("binary_classification"))
    paths, counts = [], {}
    for index, row in frame.iterrows():
        count = 3 + index % 6
        chosen = rng.choice(len(cells), count, replace=False)
        paths.append(
            str(
                write_coordinates_artifact(
                    root,
                    sample_id=row["sample_id"],
                    image_path=Path(row["image_path"]),
                    x=np.array([cells[i][0] for i in chosen]),
                    y=np.array([cells[i][1] for i in chosen]),
                    tile_size_px=TILE_PX,
                    slide_size=384,
                )
            )
        )
        counts[row["sample_id"]] = count
    frame["coordinates_path"] = paths
    manifest = root / "dataset.csv"
    frame.to_csv(manifest, index=False)
    return manifest, counts


def test_user_supplied_coordinates_replace_tiling(
    cohort, tmp_path, encoder_name, encoded_images, new_artifact
):
    """A manifest's ``coordinates_path`` tiles each slide with exactly the supplied set.

    No hs2p tiling runs: every slide's bag holds as many tiles as its artifact lists. A
    second run reuses the feature cache; editing one slide's artifact re-encodes only
    that slide, because the feature cache identity folds in the artifact's content.
    """
    artifact = new_artifact("slide_mil_supplied_coordinates")
    manifest, counts = _bring_own_coordinates(cohort, tmp_path / "coordinates")
    config = _config(
        cohort,
        tmp_path,
        encoder_name,
        task={"name": "binary_classification"},
        metrics=["auroc"],
        cache_root=tmp_path / "cache",
    )
    config["data"]["dataset_csv"] = str(manifest)

    cold = run_soma(config, tmp_path / "config.yaml")
    artifact.record_run("cold", cold)
    artifact.check_training_reduced_loss("cold", cold)
    artifact.check_at_least("cold/test/auroc", cold.summary["test/auroc"], 0.95)
    store = FeatureStore(cold.run_dir / "features")
    bag_sizes = {sid: int(store.load(sid).shape[0]) for sid in counts}
    artifact.check_equal("cold/bag_sizes", bag_sizes, counts)
    artifact.check_equal("cold/tiles_encoded", sum(encoded_images), sum(counts.values()))
    artifact.check_equal(
        "cold/no_hs2p_tiling", sorted(p.name for p in tmp_path.glob("**/previews/*/*.jpg")), []
    )

    encoded_images.clear()
    run_soma(config, tmp_path / "config.yaml")
    artifact.check_equal("warm/tiles_encoded", sum(encoded_images), 0)

    # Drop one tile from one slide's artifact: only that slide is encoded again.
    edited = pd.read_csv(manifest).iloc[0]
    cells = _tissue_cells()[: counts[edited["sample_id"]] - 1]
    write_coordinates_artifact(
        tmp_path / "coordinates",
        sample_id=edited["sample_id"],
        image_path=Path(edited["image_path"]),
        x=np.array([c[0] for c in cells]),
        y=np.array([c[1] for c in cells]),
        tile_size_px=TILE_PX,
        slide_size=384,
    )
    encoded_images.clear()
    edited_run = run_soma(config, tmp_path / "config.yaml")
    artifact.check_equal("edited/tiles_encoded", sum(encoded_images), len(cells))
    artifact.check_equal(
        "edited/bag_size",
        int(FeatureStore(edited_run.run_dir / "features").load(edited["sample_id"]).shape[0]),
        len(cells),
    )
    artifact.assert_passed()
