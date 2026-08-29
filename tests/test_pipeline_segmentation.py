"""End-to-end (offline) routing smoke test for dataset_type='segmentation'.

Builds a pre-extracted dense feature store + mask manifest + splits on disk, then
runs ``train_one_segmentation_fold`` — exercising geometry-from-sidecar, decoder +
head construction, the SegmentationDataset loaders, ``Trainer.fit`` (with the
streaming ``_tune``), checkpoint reload, and ``_evaluate_segmentation``. No encoder
/ GPU needed: the grids are written directly.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from soma.config import (
    DecoderConfig,
    EvalConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import SegmentationManifest, Splits
from soma.dense import DenseFeatureStore
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.pipeline import train, train_one_segmentation_fold

NUM_CLASSES = 2
TARGET = 8
PATCH = 4
FEATURE_DIM = 4


def _build_dense_run(
    root: Path,
    sample_ids: list[str],
    *,
    grid_spacing_um: float | None = None,
    num_classes: int = NUM_CLASSES,
    train_count: int = 2,
    patient_ids: list[str] | None = None,
) -> tuple[SegmentationManifest, Splits, DenseFeatureStore]:
    dense_dir = root / "dense"
    masks_dir = root / "masks"
    dense_dir.mkdir()
    masks_dir.mkdir()

    geom = compute_dense_geometry(
        target_size=TARGET, patch_size=PATCH
    )  # grid 2x2, encoded 8
    meta = dense_grid_metadata(
        geom, feature_dim=FEATURE_DIM, pad_mode="reflect", spacing_um=grid_spacing_um
    )

    rows = []
    rng = np.random.default_rng(0)
    for sid in sample_ids:
        write_dense_grid(
            dense_dir, sid, torch.randn(FEATURE_DIM, *geom.grid_shape), meta
        )
        if num_classes == 4:
            mask = np.repeat(
                np.arange(4, dtype=np.uint8), TARGET * TARGET // 4
            ).reshape(TARGET, TARGET)
        else:
            mask = rng.integers(0, num_classes, size=(TARGET, TARGET), dtype=np.uint8)
        label_mask_path = masks_dir / f"{sid}.png"
        Image.fromarray(mask).save(label_mask_path)
        rows.append((sid, f"{sid}.jpg", str(label_mask_path)))

    manifest_csv = root / "manifest.csv"
    if patient_ids is None:
        manifest_csv.write_text(
            "sample_id,image_path,label_mask_path\n"
            + "\n".join(f"{sid},{img},{mask}" for sid, img, mask in rows)
            + "\n"
        )
    else:
        manifest_csv.write_text(
            "sample_id,image_path,label_mask_path,patient_id\n"
            + "\n".join(
                f"{sid},{img},{mask},{patient_id}"
                for (sid, img, mask), patient_id in zip(rows, patient_ids, strict=True)
            )
            + "\n"
        )
    splits_csv = root / "splits.csv"
    split_assign = {
        sid: (
            "train"
            if index < train_count
            else "tune" if index == train_count else "test"
        )
        for index, sid in enumerate(sample_ids)
    }
    splits_csv.write_text(
        "sample_id,split,fold\n"
        + "\n".join(f"{sid},{split},0" for sid, split in split_assign.items())
        + "\n"
    )

    manifest = SegmentationManifest(manifest_csv)
    splits = Splits(splits_csv, manifest)
    store = DenseFeatureStore(dense_dir)
    return manifest, splits, store


@pytest.mark.parametrize(
    ("strategy", "expected_requests", "expected_rois"),
    [
        (
            "uniform",
            [None, None, None, None, None, None, None, None],
            ["s3", "s1", "s7", "s5", "s4", "s2", "s0", "s6"],
        ),
        (
            "class_conditioned",
            [1, 2, 3, 0, 2, 1, 3, 0],
            ["s0", "s1", "s7", "s0", "s4", "s2", "s4", "s5"],
        ),
    ],
)
def test_cached_segmentation_fold_writes_exact_roi_batch_sampling_audit(
    tmp_path: Path,
    strategy: str,
    expected_requests: list[int | None],
    expected_rois: list[str],
):
    sample_ids = [f"s{i}" for i in range(10)]
    manifest, splits, store = _build_dense_run(
        tmp_path,
        sample_ids,
        num_classes=4,
        train_count=8,
    )
    fold_dir = tmp_path / "fold"

    train_one_segmentation_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": 4}),
        training=TrainingConfig(
            epochs=1,
            batch_size=4,
            roi_batch_sampling=strategy,
            roi_draws_per_epoch=8,
            seed=11,
        ),
        fold_dir=fold_dir,
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice"]),
    )

    audit = json.loads((fold_dir / "roi_batch_sampling.json").read_text())
    population = json.loads(
        (fold_dir / "segmentation_roi_population.json").read_text()
    )
    assert Path(population["cache_path"]).is_relative_to(
        fold_dir / "segmentation_roi_population"
    )
    assert {
        "roi_count": population["roi_count"],
        "num_classes": population["num_classes"],
        "class_pixel_totals": population["class_pixel_totals"],
    } == {
        "roi_count": 10,
        "num_classes": 4,
        "class_pixel_totals": [160, 160, 160, 160],
    }
    epoch = audit["epochs"][0]
    assert {
        "strategy": audit["strategy"],
        "draws_per_epoch": audit["draws_per_epoch"],
        "requested_classes": [row["requested_class"] for row in epoch["selections"]],
        "selected_rois": [row["selected_roi"] for row in epoch["selections"]],
        "actual_class_pixel_counts": epoch["actual_class_pixel_counts"],
    } == {
        "strategy": strategy,
        "draws_per_epoch": 8,
        "requested_classes": expected_requests,
        "selected_rois": expected_rois,
        "actual_class_pixel_counts": [128, 128, 128, 128],
    }


def test_cached_segmentation_default_roi_budget_reaches_sampler_audit(
    tmp_path: Path,
) -> None:
    sample_ids = [f"s{i}" for i in range(12)]
    manifest, splits, store = _build_dense_run(
        tmp_path,
        sample_ids,
        train_count=10,
    )
    fold_dir = tmp_path / "fold"

    train_one_segmentation_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            gradient_accumulation=4,
            roi_batch_sampling="uniform",
        ),
        fold_dir=fold_dir,
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice"]),
    )

    audit = json.loads((fold_dir / "roi_batch_sampling.json").read_text())

    assert {
        "physical_batch_size": audit["physical_batch_size"],
        "gradient_accumulation": audit["gradient_accumulation"],
        "effective_batch_size": audit["effective_batch_size"],
        "resolved_draws_per_epoch": audit["resolved_draws_per_epoch"],
        "draw_count": len(audit["epochs"][0]["selections"]),
    } == {
        "physical_batch_size": 2,
        "gradient_accumulation": 4,
        "effective_batch_size": 8,
        "resolved_draws_per_epoch": 8,
        "draw_count": 8,
    }


def test_cached_sampler_counts_arbitrary_classes_and_forwards_request_ratios(
    tmp_path: Path,
):
    sample_ids = [f"s{i}" for i in range(6)]
    manifest, splits, store = _build_dense_run(
        tmp_path,
        sample_ids,
        num_classes=3,
        train_count=4,
    )
    fold_dir = tmp_path / "fold"

    train_one_segmentation_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": 3}),
        training=TrainingConfig(
            epochs=1,
            batch_size=4,
            roi_batch_sampling="class_conditioned",
            class_request_ratios=[1, 2, 1],
            roi_draws_per_epoch=4,
            seed=11,
        ),
        fold_dir=fold_dir,
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice"]),
    )

    audit = json.loads((fold_dir / "roi_batch_sampling.json").read_text())
    assert audit["classes"] == [0, 1, 2]
    assert audit["class_request_ratios"] == [1.0, 2.0, 1.0]
    assert audit["epochs"][0]["target_request_counts"] == [1, 2, 1]


def test_train_one_segmentation_fold_end_to_end(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_dense_run(tmp_path, sample_ids)

    result = train_one_segmentation_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )

    # tune + test reports carry dense metrics (no scalar predictions in 1f-b).
    assert set(result.tune_report.metrics) >= {"mean_dice", "mean_iou"}
    assert "test" in result.test_reports
    assert set(result.test_reports["test"].metrics) >= {"mean_dice", "mean_iou"}
    assert result.tune_report.predictions == []
    assert 0.0 <= result.tune_report.metrics["mean_dice"] <= 1.0
    # metrics.json + checkpoint were written.
    assert (tmp_path / "fold" / "metrics.json").is_file()
    assert (tmp_path / "fold" / "best_model.pt").is_file()

    # 1g dense artifacts: per-tile rasters + predictions CSV land for each split.
    fold = tmp_path / "fold"
    assert (fold / "predictions_test.csv").is_file()
    test_raster = fold / "preds" / "test" / "s3.png"
    assert test_raster.is_file()
    raster = np.asarray(Image.open(test_raster))
    assert raster.shape == (TARGET, TARGET)  # head cropped logits to target_size
    assert raster.dtype == np.uint8
    assert raster.max() < NUM_CLASSES
    # Overlays are fail-soft: the cached-feature fixture has no real source tiles.
    assert not (fold / "pred_overlays" / "test" / "s3.png").exists()


def test_selected_checkpoint_exports_one_confusion_per_tune_sample(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_dense_run(
        tmp_path,
        sample_ids,
        num_classes=3,
    )
    fold_dir = tmp_path / "fold"

    train_one_segmentation_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": 3}),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=fold_dir,
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(
            metrics=["mean_dice"],
            save_segmentation_confusion_evidence=True,
        ),
        fold=0,
    )

    payload = __import__("json").loads(
        (fold_dir / "confusion_evidence_tune.json").read_text()
    )
    assert payload["schema_version"] == 1
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert record["sample_id"] == "s2"
    assert record["fold"] == 0
    assert record["class_vocabulary"] == ["class_0", "class_1", "class_2"]
    assert len(record["confusion_matrix"]) == 3
    assert all(len(row) == 3 for row in record["confusion_matrix"])
    assert sum(sum(row) for row in record["confusion_matrix"]) == TARGET * TARGET


def _run_confusion_evidence_cv(
    tmp_path: Path,
    *,
    save_evidence: bool = True,
    roi_batch_sampling: str | None = None,
    holdout_test: bool = False,
):
    sample_ids = ["s0", "s1", "s2", "s3", "external"]
    manifest, _, store = _build_dense_run(
        tmp_path,
        sample_ids,
        num_classes=3,
    )
    split_rows = [
        # Fold 0 holds out p0/p1 for tune. The external patient is never development tune.
        (0, "s0", "tune"),
        (0, "s1", "tune"),
        (0, "s2", "train"),
        (0, "s3", "test"),
        (0, "external", "test_external"),
        # Fold 1 holds out p2/p3, completing four unique development patients.
        (1, "s0", "train"),
        (1, "s1", "test"),
        (1, "s2", "tune"),
        (1, "s3", "tune"),
        (1, "external", "test_external"),
    ]
    splits_path = tmp_path / "cv_splits.csv"
    splits_path.write_text(
        "sample_id,split,fold\n"
        + "\n".join(f"{sample},{split},{fold}" for fold, sample, split in split_rows)
        + "\n"
    )
    splits = Splits(splits_path, manifest)
    run_dir = tmp_path / "run"
    evaluation = EvalConfig(
        metrics=["mean_dice"],
        save_segmentation_confusion_evidence=save_evidence,
        holdout_test=holdout_test,
    )
    train_kwargs = {
        "feature_store": store,
        "dataset": manifest,
        "splits": splits,
        "dataset_type": "segmentation",
        "task": TaskConfig(name="segmentation", params={"num_classes": 3}),
        "training": TrainingConfig(
            epochs=1,
            batch_size=2,
            roi_batch_sampling=roi_batch_sampling,
            roi_draws_per_epoch=2 if roi_batch_sampling is not None else None,
        ),
        "run_dir": run_dir,
        "decoder": DecoderConfig(name="lightweight_conv"),
        "evaluation": evaluation,
    }

    return train(**train_kwargs), run_dir, train_kwargs


def test_multifold_population_provenance_is_fold_local(tmp_path: Path) -> None:
    _, run_dir, _ = _run_confusion_evidence_cv(
        tmp_path,
        roi_batch_sampling="class_conditioned",
    )

    for fold in range(2):
        provenance = json.loads(
            (run_dir / f"fold_{fold}" / "segmentation_roi_population.json").read_text()
        )
        assert provenance["artifact_kind"] == "segmentation_roi_population"
        assert Path(provenance["cache_path"]).is_relative_to(
            run_dir / "segmentation_roi_population"
        )


def test_holdout_multifold_resolves_one_full_population_and_subsets_per_fold(
    tmp_path: Path,
) -> None:
    _, run_dir, _ = _run_confusion_evidence_cv(
        tmp_path,
        roi_batch_sampling="class_conditioned",
        holdout_test=True,
    )

    populations = [
        json.loads(
            (run_dir / f"fold_{fold}" / "segmentation_roi_population.json").read_text()
        )
        for fold in range(2)
    ]
    cache_entries = list((run_dir / "segmentation_roi_population").glob("*/population.json"))

    assert len(cache_entries) == 1
    assert json.loads(cache_entries[0].read_text())["sample_ids"] == [
        "s0",
        "s1",
        "s2",
        "s3",
        "external",
    ]
    assert [population["roi_count"] for population in populations] == [3, 3]
    assert [population["class_pixel_totals"] for population in populations] == [
        [66, 56, 70],
        [71, 58, 63],
    ]
    assert {population["cache_key"] for population in populations} == {
        populations[0]["cache_key"]
    }


def test_tune_is_test_multifold_reuses_one_unique_roi_population(tmp_path: Path) -> None:
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, _, store = _build_dense_run(tmp_path, sample_ids, num_classes=3)
    splits_path = tmp_path / "tune_is_test_cv.csv"
    splits_path.write_text(
        "sample_id,split,fold\n"
        "s0,train,0\n"
        "s1,train,0\n"
        "s2,test,0\n"
        "s3,test,0\n"
        "s0,test,1\n"
        "s1,test,1\n"
        "s2,train,1\n"
        "s3,train,1\n"
    )
    run_dir = tmp_path / "run"

    train(
        feature_store=store,
        dataset=manifest,
        splits=Splits(splits_path, manifest, tune_is_test=True),
        dataset_type="segmentation",
        task=TaskConfig(name="segmentation", params={"num_classes": 3}),
        training=TrainingConfig(
            epochs=1,
            batch_size=2,
            tune_is_test=True,
            roi_batch_sampling="class_conditioned",
            roi_draws_per_epoch=2,
        ),
        run_dir=run_dir,
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice"]),
        roi_population_cache_root=tmp_path / "population_cache",
    )

    populations = [
        json.loads(
            (run_dir / f"fold_{fold}" / "segmentation_roi_population.json").read_text()
        )
        for fold in range(2)
    ]
    assert {population["cache_key"] for population in populations} == {
        populations[0]["cache_key"]
    }
    assert [population["roi_count"] for population in populations] == [4, 4]


def test_train_validates_complete_fold_confusion_evidence(tmp_path: Path) -> None:
    _, run_dir, _ = _run_confusion_evidence_cv(tmp_path)

    fold_0 = json.loads(
        (run_dir / "fold_0" / "confusion_evidence_tune.json").read_text()
    )
    fold_1 = json.loads(
        (run_dir / "fold_1" / "confusion_evidence_tune.json").read_text()
    )
    assert [record["sample_id"] for record in fold_0["records"]] == ["s0", "s1"]
    assert [record["sample_id"] for record in fold_1["records"]] == ["s2", "s3"]
    assert {record["fold"] for record in fold_0["records"]} == {0}
    assert {record["fold"] for record in fold_1["records"]} == {1}


def test_resume_regenerates_missing_evidence_from_selected_checkpoints_only(
    tmp_path: Path,
) -> None:
    _, run_dir, train_kwargs = _run_confusion_evidence_cv(
        tmp_path, save_evidence=False
    )
    histories = {
        fold: (run_dir / f"fold_{fold}" / "training_history.json").read_bytes()
        for fold in range(2)
    }
    checkpoints = {
        fold: hashlib.sha256(
            (run_dir / f"fold_{fold}" / "best_model.pt").read_bytes()
        ).hexdigest()
        for fold in range(2)
    }
    train_kwargs["evaluation"] = replace(
        train_kwargs["evaluation"],
        save_segmentation_confusion_evidence=True,
    )
    train_kwargs["resume_completed_folds"] = True

    resumed = train(**train_kwargs)

    assert resumed.fold_results == []
    for fold in range(2):
        fold_dir = run_dir / f"fold_{fold}"
        assert (fold_dir / "confusion_evidence_tune.json").is_file()
        assert (fold_dir / "training_history.json").read_bytes() == histories[fold]
        assert hashlib.sha256((fold_dir / "best_model.pt").read_bytes()).hexdigest() == (
            checkpoints[fold]
        )


def test_tune_is_test_validates_evidence_against_effective_held_out_split(
    tmp_path: Path,
) -> None:
    sample_ids = ["train_a", "train_b", "held_a", "held_b"]
    manifest, _, store = _build_dense_run(tmp_path, sample_ids, num_classes=3)
    splits_path = tmp_path / "test_only_splits.csv"
    splits_path.write_text(
        "sample_id,split,fold\n"
        "train_a,train,0\n"
        "train_b,train,0\n"
        "held_a,test,0\n"
        "held_b,test,0\n"
    )
    splits = Splits(splits_path, manifest, tune_is_test=True)
    run_dir = tmp_path / "run"

    train(
        feature_store=store,
        dataset=manifest,
        splits=splits,
        dataset_type="segmentation",
        task=TaskConfig(name="segmentation", params={"num_classes": 3}),
        training=TrainingConfig(epochs=1, batch_size=2, tune_is_test=True),
        run_dir=run_dir,
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(
            metrics=["mean_dice"],
            save_segmentation_confusion_evidence=True,
        ),
    )

    payload = json.loads((run_dir / "confusion_evidence_tune.json").read_text())
    assert [record["sample_id"] for record in payload["records"]] == [
        "held_a",
        "held_b",
    ]


def test_segmentation_fold_requires_num_classes(tmp_path: Path):
    manifest, splits, store = _build_dense_run(tmp_path, ["s0", "s1", "s2", "s3"])
    with pytest.raises(ValueError, match="num_classes"):
        train_one_segmentation_fold(
            feature_store=store,
            dataset=manifest,
            fold_split=splits.folds[0],
            task=TaskConfig(name="segmentation"),  # no num_classes
            training=TrainingConfig(epochs=1, batch_size=2),
            fold_dir=tmp_path / "fold",
            decoder=DecoderConfig(name="lightweight_conv"),
        )


def test_segmentation_fold_rejects_mask_grid_spacing_mismatch(tmp_path: Path):
    # Grids extracted at 0.5 µm/px but masks would be read at 1.0 — the supervision
    # would misregister against the features. The fold must fail loud, not train.
    from soma.config import PreprocessingConfig

    manifest, splits, store = _build_dense_run(
        tmp_path, ["s0", "s1", "s2", "s3"], grid_spacing_um=0.5
    )
    with pytest.raises(ValueError, match="does not match"):
        train_one_segmentation_fold(
            feature_store=store,
            dataset=manifest,
            fold_split=splits.folds[0],
            task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
            training=TrainingConfig(epochs=1, batch_size=2),
            fold_dir=tmp_path / "fold",
            decoder=DecoderConfig(name="lightweight_conv"),
            preprocessing=PreprocessingConfig(requested_spacing_um=1.0),
        )


def _seg_pipeline_config(tmp_path: Path):
    from soma.config import PipelineConfig

    return PipelineConfig(
        dataset_csv=tmp_path / "manifest.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "out",
        dataset_type="segmentation",
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=2),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )


def _recoverable_seg_pipeline_config(tmp_path: Path, *, run_id: str | None = None):
    _build_dense_run(
        tmp_path,
        ["s0", "s1", "s2", "s3", "s4", "s5"],
        num_classes=4,
        train_count=4,
    )
    return replace(
        _seg_pipeline_config(tmp_path),
        mirror_root=tmp_path / "shared",
        run_id=run_id,
        task=TaskConfig(name="segmentation", params={"num_classes": 4}),
        training=TrainingConfig(
            epochs=1,
            batch_size=4,
            roi_batch_sampling="class_conditioned",
            roi_draws_per_epoch=4,
        ),
    )


def test_pipeline_uses_segmentation_manifest(tmp_path: Path):
    from soma.pipeline import Pipeline

    _build_dense_run(tmp_path, ["s0", "s1", "s2", "s3"])
    pipeline = Pipeline(_seg_pipeline_config(tmp_path), feature_dir=tmp_path / "dense")
    assert isinstance(pipeline.dataset, SegmentationManifest)
    assert (
        pipeline._get_feature_store(run_dir=tmp_path / "out").feature_dim == FEATURE_DIM
    )


def test_pipeline_run_end_to_end_segmentation(tmp_path: Path):
    """The real entry point: Pipeline.run() (run-summary panel, train, summary)."""
    from soma.pipeline import Pipeline

    _build_dense_run(tmp_path, ["s0", "s1", "s2", "s3"])
    pipeline = Pipeline(_seg_pipeline_config(tmp_path), feature_dir=tmp_path / "dense")
    result = pipeline.run()
    assert "test/mean_dice" in result.summary
    assert result.fold_results[0].test_reports["test"].metrics["mean_dice"] >= 0.0
    # Dense artifacts land via the real entry point too (run -> train -> eval).
    # run_dir is a structured layout path under output_root, so glob for them.
    out = tmp_path / "out"
    assert list(out.rglob("preds/test/s3.png")), "pred raster missing under run dir"
    assert list(
        out.rglob("predictions_test.csv")
    ), "predictions CSV missing under run dir"


def test_pipeline_mirrors_verified_checkpoint_and_completed_fold_bundles(
    tmp_path: Path,
):
    """The recovery surface is the real pipeline plus its local/shared files."""
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path)

    result = Pipeline(config, feature_dir=tmp_path / "dense").run()

    population = json.loads(
        (result.run_dir / "segmentation_roi_population.json").read_text()
    )
    assert Path(population["cache_path"]).is_relative_to(
        config.output_root / "feature_cache" / "segmentation_roi_population"
    )

    local_recovery = result.run_dir / "recovery"
    mirrored_run = config.mirror_root / result.run_dir.relative_to(config.output_root)
    checkpoint_spools = sorted(
        (local_recovery / "spool" / "checkpoints").rglob("epoch_*")
    )
    checkpoint_bundles = sorted(
        (mirrored_run / "recovery" / "checkpoints").rglob("epoch_*")
    )
    fold_spool = local_recovery / "spool" / "folds" / "fold_0"
    fold_bundle = mirrored_run / "recovery" / "folds" / "fold_0"

    assert len(checkpoint_spools) == len(checkpoint_bundles) == 1
    assert fold_spool.is_dir() and fold_bundle.is_dir()
    assert {
        "best_model.pt",
        "config.yaml",
        "training_history.json",
        "sampler_audit.json",
    } <= set(json.loads((checkpoint_bundles[0] / "manifest.json").read_text())["files"])
    assert "segmentation_roi_population.json" in json.loads(
        (fold_bundle / "manifest.json").read_text()
    )["files"]

    for local_bundle, mirrored_bundle in [
        (checkpoint_spools[0], checkpoint_bundles[0]),
        (fold_spool, fold_bundle),
    ]:
        manifest = json.loads((mirrored_bundle / "manifest.json").read_text())
        assert manifest["kind"] in {"checkpoint", "fold"}
        for relative, expected in manifest["files"].items():
            local_file = local_bundle / relative
            mirrored_file = mirrored_bundle / relative
            assert mirrored_file.read_bytes() == local_file.read_bytes()
            assert (
                hashlib.sha256(mirrored_file.read_bytes()).hexdigest()
                == expected["sha256"]
            )
            assert mirrored_file.stat().st_size == expected["size"]

    assert not list(mirrored_run.rglob("*.tmp.*"))


def test_transient_mirror_outage_is_recorded_retried_and_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import shutil

    from soma.pipeline import Pipeline

    mirror_root = tmp_path / "shared"
    config = _recoverable_seg_pipeline_config(tmp_path, run_id="pending-mirror-fold")
    real_copy2 = shutil.copy2
    outage_count = 0

    def fail_first_shared_copy(source, destination, *args, **kwargs):
        nonlocal outage_count
        if Path(destination).is_relative_to(mirror_root) and outage_count == 0:
            outage_count += 1
            raise OSError("simulated shared-storage outage")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr("soma.artifact_mirror.shutil.copy2", fail_first_shared_copy)

    result = Pipeline(config, feature_dir=tmp_path / "dense").run()

    state = json.loads((result.run_dir / "recovery" / "mirror_state.json").read_text())
    mirrored_run = mirror_root / result.run_dir.relative_to(config.output_root)
    assert "test/mean_dice" in result.summary
    assert outage_count == 1
    assert any(
        "simulated shared-storage outage" in failure["error"]
        for failure in state["failures"]
    )
    assert all(entry["status"] == "verified" for entry in state["entries"].values())
    checkpoint_entry = next(
        entry
        for bundle_id, entry in state["entries"].items()
        if bundle_id.startswith("checkpoints/")
    )
    assert checkpoint_entry["attempts"] >= 2
    assert list((mirrored_run / "recovery" / "checkpoints").rglob("manifest.json"))
    assert (mirrored_run / "recovery" / "folds" / "fold_0" / "manifest.json").is_file()


def test_verified_mirror_is_idempotent_on_pinned_run_resume(tmp_path: Path):
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="recoverable-fold")
    first = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / first.run_dir.relative_to(config.output_root)
    before = {
        path.relative_to(mirrored_run).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in mirrored_run.rglob("*")
        if path.is_file()
    }
    before_state = json.loads(
        (first.run_dir / "recovery" / "mirror_state.json").read_text()
    )

    second = Pipeline(config, feature_dir=tmp_path / "dense").run()

    after = {
        path.relative_to(mirrored_run).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in mirrored_run.rglob("*")
        if path.is_file()
    }
    after_state = json.loads(
        (second.run_dir / "recovery" / "mirror_state.json").read_text()
    )
    assert second.run_dir == first.run_dir
    assert after == before
    assert set(after_state["entries"]) == set(before_state["entries"])
    assert {
        key: entry["attempts"] for key, entry in after_state["entries"].items()
    } == {key: entry["attempts"] for key, entry in before_state["entries"].items()}
    assert len(list((mirrored_run / "recovery" / "folds").rglob("manifest.json"))) == 1


def test_foreign_valid_mirror_bundle_is_repaired_from_local_spool(tmp_path: Path):
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="foreign-mirror-fold")
    first = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / first.run_dir.relative_to(config.output_root)

    mirrored_checkpoint = next(
        (mirrored_run / "recovery" / "checkpoints").rglob("best_model.pt")
    )
    local_checkpoint = next(
        (first.run_dir / "recovery" / "spool" / "checkpoints").rglob(
            "best_model.pt"
        )
    )
    authoritative_local_bytes = local_checkpoint.read_bytes()
    mirrored_checkpoint.write_bytes(b"corrupt")
    foreign_manifest_path = mirrored_checkpoint.parent / "manifest.json"
    foreign_manifest = json.loads(foreign_manifest_path.read_text())
    foreign_manifest["files"]["best_model.pt"] = {
        "sha256": hashlib.sha256(b"corrupt").hexdigest(),
        "size": len(b"corrupt"),
    }
    foreign_manifest_path.write_text(
        json.dumps(foreign_manifest, indent=2, sort_keys=True)
    )
    third = Pipeline(config, feature_dir=tmp_path / "dense").run()
    local_checkpoint = next(
        (third.run_dir / "recovery" / "spool" / "checkpoints").rglob("best_model.pt")
    )
    assert local_checkpoint.read_bytes() == authoritative_local_bytes
    assert mirrored_checkpoint.read_bytes() == local_checkpoint.read_bytes()
    repaired_manifest = json.loads(
        (mirrored_checkpoint.parent / "manifest.json").read_text()
    )
    assert (
        hashlib.sha256(mirrored_checkpoint.read_bytes()).hexdigest()
        == repaired_manifest["files"]["best_model.pt"]["sha256"]
    )


def _make_valid_foreign_checkpoint(
    source: Path, destination: Path, *, epoch: int
) -> None:
    import shutil

    shutil.copytree(source, destination)
    (destination / "best_model.pt").write_bytes(b"foreign-checkpoint")
    saved_config = yaml.safe_load((destination / "config.yaml").read_text())
    saved_config["run"]["seed"] += 1
    (destination / "config.yaml").write_text(
        yaml.safe_dump(saved_config, sort_keys=False)
    )
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["epoch"] = epoch
    for relative in ("best_model.pt", "config.yaml"):
        payload = (destination / relative).read_bytes()
        manifest["files"][relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload),
        }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def test_restore_ignores_foreign_valid_higher_epoch_checkpoint(tmp_path: Path):
    import shutil

    from soma.artifact_mirror import restore_run_from_mirror
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="mixed-checkpoints")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / original.run_dir.relative_to(config.output_root)
    checkpoint_root = mirrored_run / "recovery" / "checkpoints" / "fold_0"
    compatible = next(
        path for path in checkpoint_root.iterdir() if path.name.startswith("epoch_")
    )
    compatible_bytes = (compatible / "best_model.pt").read_bytes()
    _make_valid_foreign_checkpoint(
        compatible, checkpoint_root / "epoch_9999", epoch=9999
    )
    shutil.rmtree(mirrored_run / "recovery" / "folds")
    shutil.rmtree(original.run_dir)

    restored = restore_run_from_mirror(replace(config, resume=True), num_folds=1)

    assert restored == "mixed-checkpoints"
    assert (original.run_dir / "best_model.pt").read_bytes() == compatible_bytes


def test_restore_ignores_valid_quarantined_checkpoint_bundle(tmp_path: Path):
    import shutil

    from soma.artifact_mirror import restore_run_from_mirror
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="quarantined-checkpoint")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / original.run_dir.relative_to(config.output_root)
    checkpoint_root = mirrored_run / "recovery" / "checkpoints" / "fold_0"
    compatible = next(
        path for path in checkpoint_root.iterdir() if path.name.startswith("epoch_")
    )
    compatible_bytes = (compatible / "best_model.pt").read_bytes()
    quarantined = checkpoint_root / ".epoch_9999.corrupt.test"
    shutil.copytree(compatible, quarantined)
    (quarantined / "best_model.pt").write_bytes(b"quarantined-checkpoint")
    manifest_path = quarantined / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["epoch"] = 9999
    payload = (quarantined / "best_model.pt").read_bytes()
    manifest["files"]["best_model.pt"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    shutil.rmtree(mirrored_run / "recovery" / "folds")
    shutil.rmtree(original.run_dir)

    restored = restore_run_from_mirror(replace(config, resume=True), num_folds=1)

    assert restored == "quarantined-checkpoint"
    assert (original.run_dir / "best_model.pt").read_bytes() == compatible_bytes


def test_interrupted_publish_never_exposes_a_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import os

    from soma.pipeline import Pipeline

    mirror_root = tmp_path / "shared"
    config = _recoverable_seg_pipeline_config(
        tmp_path, run_id="interrupted-mirror-fold"
    )
    real_replace = os.replace

    def interrupt_shared_publication(source, destination, *args, **kwargs):
        if Path(destination).is_relative_to(mirror_root):
            raise OSError("simulated interruption before atomic publish")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr("soma.artifact_mirror.os.replace", interrupt_shared_publication)

    result = Pipeline(config, feature_dir=tmp_path / "dense").run()

    state = json.loads((result.run_dir / "recovery" / "mirror_state.json").read_text())
    assert "test/mean_dice" in result.summary
    assert all(entry["status"] == "pending" for entry in state["entries"].values())
    assert not list(mirror_root.rglob("manifest.json"))
    assert not list(mirror_root.rglob("*.tmp.*"))
    assert (
        len(list((result.run_dir / "recovery" / "spool").rglob("manifest.json"))) == 2
    )

    monkeypatch.undo()
    resumed = Pipeline(config, feature_dir=tmp_path / "dense").run()
    resumed_state = json.loads(
        (resumed.run_dir / "recovery" / "mirror_state.json").read_text()
    )
    mirrored_run = mirror_root / resumed.run_dir.relative_to(config.output_root)
    assert all(
        entry["status"] == "verified" for entry in resumed_state["entries"].values()
    )
    assert any(
        "simulated interruption before atomic publish" in failure["error"]
        for failure in resumed_state["failures"]
    )
    assert len(list((mirrored_run / "recovery").rglob("manifest.json"))) == 2


def test_unpinned_resume_restores_completed_fold_after_local_node_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import shutil

    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="lost-node-run")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    original_summary = dict(original.summary)
    lost_run_dir = original.run_dir
    shutil.rmtree(lost_run_dir)

    def forbid_retraining(*args, **kwargs):
        raise AssertionError(
            "a verified completed fold must be restored, not retrained"
        )

    monkeypatch.setattr("soma.training.trainer.Trainer.fit", forbid_retraining)
    resumed_config = replace(config, run_id=None, resume=True)

    resumed = Pipeline(resumed_config, feature_dir=tmp_path / "dense").run()

    assert resumed.run_dir == lost_run_dir
    assert resumed.summary == original_summary
    assert resumed.fold_results == []
    assert (resumed.run_dir / "metrics.json").is_file()
    assert (resumed.run_dir / "best_model.pt").is_file()
    restored_state = json.loads(
        (resumed.run_dir / "recovery" / "mirror_state.json").read_text()
    )
    assert all(
        entry["status"] == "verified" for entry in restored_state["entries"].values()
    )


def test_checkpoint_only_mirror_restores_exact_files_then_training_continues(
    tmp_path: Path,
):
    import shutil

    from soma.artifact_mirror import restore_run_from_mirror
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="checkpoint-only-run")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / original.run_dir.relative_to(config.output_root)
    checkpoint = next(
        (mirrored_run / "recovery" / "checkpoints").rglob("manifest.json")
    ).parent
    expected = {
        relative: (checkpoint / relative).read_bytes()
        for relative in (
            "best_model.pt",
            "training_history.json",
            "sampler_audit.json",
            "config.yaml",
        )
    }
    shutil.rmtree(mirrored_run / "recovery" / "folds")
    shutil.rmtree(original.run_dir)

    restored = restore_run_from_mirror(replace(config, resume=True), num_folds=1)

    assert restored == "checkpoint-only-run"
    assert {
        relative: (original.run_dir / relative).read_bytes() for relative in expected
    } == expected
    assert not (original.run_dir / "metrics.json").exists()

    resumed = Pipeline(
        replace(config, resume=True), feature_dir=tmp_path / "dense"
    ).run()
    assert resumed.run_dir == original.run_dir
    assert "test/mean_dice" in resumed.summary


def test_pinned_resume_records_transient_restore_failure_and_continues_locally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="restore-outage")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()

    def fail_shared_traversal(*args, **kwargs):
        raise OSError("simulated restore traversal outage")

    monkeypatch.setattr(
        "soma.artifact_mirror._canonical_mirror_bundles", fail_shared_traversal
    )

    resumed = Pipeline(
        replace(config, resume=True), feature_dir=tmp_path / "dense"
    ).run()

    assert resumed.run_dir == original.run_dir
    assert resumed.summary == original.summary
    state_path = resumed.run_dir / "recovery" / "mirror_state.json"
    state = json.loads(state_path.read_text())
    assert any(
        error["status"] == "pending"
        and "simulated restore traversal outage" in error["error"]
        for error in state["restore_errors"]
    )

    monkeypatch.undo()
    retried = Pipeline(
        replace(config, resume=True), feature_dir=tmp_path / "dense"
    ).run()
    retried_state = json.loads(
        (retried.run_dir / "recovery" / "mirror_state.json").read_text()
    )
    assert all(
        error["status"] == "resolved" for error in retried_state["restore_errors"]
    )


def test_checkpoint_restore_failure_preserves_existing_local_artifact_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import os
    import shutil

    from soma.artifact_mirror import restore_run_from_mirror
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="restore-set-outage")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / original.run_dir.relative_to(config.output_root)
    shutil.rmtree(mirrored_run / "recovery" / "folds")
    (original.run_dir / "metrics.json").unlink()
    expected = {
        "best_model.pt": b"local-model",
        "training_history.json": b"local-history",
        "sampler_audit.json": b"local-audit",
    }
    for relative, payload in expected.items():
        (original.run_dir / relative).write_bytes(payload)
    real_replace = os.replace
    failed = False

    def fail_active_install(source, destination, *args, **kwargs):
        nonlocal failed
        if Path(destination) == original.run_dir and not failed:
            failed = True
            raise OSError("simulated checkpoint-set install outage")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr("soma.artifact_mirror.os.replace", fail_active_install)

    restored = restore_run_from_mirror(replace(config, resume=True), num_folds=1)

    assert restored is None
    assert {
        relative: (original.run_dir / relative).read_bytes() for relative in expected
    } == expected
    state = json.loads(
        (original.run_dir / "recovery" / "mirror_state.json").read_text()
    )
    assert any(
        error["status"] == "pending"
        and "simulated checkpoint-set install outage" in error["error"]
        for error in state["restore_errors"]
    )


def test_checkpoint_double_failure_recovers_quarantined_run_before_state_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import os
    import shutil

    from soma.artifact_mirror import restore_run_from_mirror
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="restore-rollback-outage")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / original.run_dir.relative_to(config.output_root)
    shutil.rmtree(mirrored_run / "recovery" / "folds")
    (original.run_dir / "metrics.json").unlink()
    expected = {
        "best_model.pt": b"local-model",
        "training_history.json": b"local-history",
        "sampler_audit.json": b"local-audit",
    }
    for relative, payload in expected.items():
        (original.run_dir / relative).write_bytes(payload)
    real_replace = os.replace
    failures_remaining = 2

    def fail_install_and_rollback(source, destination, *args, **kwargs):
        nonlocal failures_remaining
        if Path(destination) == original.run_dir and failures_remaining:
            failures_remaining -= 1
            raise OSError("simulated install and rollback outage")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr("soma.artifact_mirror.os.replace", fail_install_and_rollback)
    assert restore_run_from_mirror(replace(config, resume=True), num_folds=1) is None
    assert not original.run_dir.exists()
    transaction_marker = (
        original.run_dir.parent
        / ".restore_transactions"
        / original.run_dir.name
        / "fold_0.json"
    )
    assert transaction_marker.is_file()

    monkeypatch.undo()

    def fail_after_local_transaction_recovery(*args, **kwargs):
        raise OSError("keep recovered local run authoritative")

    monkeypatch.setattr(
        "soma.artifact_mirror._canonical_mirror_bundles",
        fail_after_local_transaction_recovery,
    )
    assert restore_run_from_mirror(replace(config, resume=True), num_folds=1) is None
    assert {
        relative: (original.run_dir / relative).read_bytes() for relative in expected
    } == expected
    state = json.loads(
        (original.run_dir / "recovery" / "mirror_state.json").read_text()
    )
    assert any(
        "simulated install and rollback outage" in error["error"]
        for error in state["restore_errors"]
    )


def test_checkpoint_crash_before_first_swap_keeps_authoritative_local_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import os
    import shutil

    from soma.artifact_mirror import restore_run_from_mirror
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="restore-preswap-crash")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / original.run_dir.relative_to(config.output_root)
    shutil.rmtree(mirrored_run / "recovery" / "folds")
    (original.run_dir / "metrics.json").unlink()
    expected = {
        "best_model.pt": b"local-model",
        "training_history.json": b"local-history",
        "sampler_audit.json": b"local-audit",
    }
    for relative, payload in expected.items():
        (original.run_dir / relative).write_bytes(payload)
    real_replace = os.replace

    def interrupt_before_first_swap(source, destination, *args, **kwargs):
        if Path(source) == original.run_dir:
            raise KeyboardInterrupt("simulated crash before first directory swap")
        return real_replace(source, destination, *args, **kwargs)

    monkeypatch.setattr("soma.artifact_mirror.os.replace", interrupt_before_first_swap)
    with pytest.raises(KeyboardInterrupt, match="before first directory swap"):
        restore_run_from_mirror(replace(config, resume=True), num_folds=1)

    monkeypatch.undo()

    def fail_after_transaction_recovery(*args, **kwargs):
        raise OSError("keep recovered pre-swap run authoritative")

    monkeypatch.setattr(
        "soma.artifact_mirror._canonical_mirror_bundles",
        fail_after_transaction_recovery,
    )
    assert restore_run_from_mirror(replace(config, resume=True), num_folds=1) is None
    assert {
        relative: (original.run_dir / relative).read_bytes() for relative in expected
    } == expected
    assert not (
        original.run_dir.parent
        / ".restore_transactions"
        / original.run_dir.name
        / "fold_0.json"
    ).exists()


def test_unpinned_resume_recovers_nested_multifold_transaction_marker(tmp_path: Path):
    import os

    from soma.artifact_mirror import restore_run_from_mirror
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="nested-marker-run")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    fold_target = original.run_dir / "fold_0"
    fold_target.mkdir()
    (fold_target / "best_model.pt").write_bytes(b"preserved-fold-checkpoint")
    other_fold = original.run_dir / "fold_1"
    other_fold.mkdir()
    (other_fold / "best_model.pt").write_bytes(b"other-fold-checkpoint")
    quarantine = original.run_dir / ".fold_0.checkpoint-previous.test"
    os.replace(fold_target, quarantine)
    marker = (
        original.run_dir.parent
        / ".restore_transactions"
        / original.run_dir.name
        / "fold_0.json"
    )
    marker.parent.mkdir(parents=True)
    marker.write_text(
        json.dumps(
            {
                "run_id": original.run_dir.name,
                "target": f"{original.run_dir.name}/fold_0",
                "staging": f"{original.run_dir.name}/.fold_0.checkpoint-stage.test",
                "quarantine": f"{original.run_dir.name}/{quarantine.name}",
                "files": {
                    "best_model.pt": {
                        "sha256": hashlib.sha256(b"new-fold-checkpoint").hexdigest(),
                        "size": len(b"new-fold-checkpoint"),
                    }
                },
            }
        )
    )

    restored = restore_run_from_mirror(
        replace(config, run_id=None, resume=True), num_folds=2
    )

    assert restored is None
    assert (fold_target / "best_model.pt").read_bytes() == b"preserved-fold-checkpoint"
    assert (other_fold / "best_model.pt").read_bytes() == b"other-fold-checkpoint"
    assert not marker.exists()
    assert not quarantine.exists()


@pytest.mark.parametrize("run_id", ["missing-everywhere", None])
def test_resume_without_local_or_mirrored_run_fails_clearly(
    tmp_path: Path, run_id: str | None
):
    from soma.pipeline import Pipeline

    config = replace(
        _recoverable_seg_pipeline_config(tmp_path, run_id="placeholder"),
        run_id=run_id,
        resume=True,
    )

    with pytest.raises(OSError, match="no local run is available"):
        Pipeline(config, feature_dir=tmp_path / "dense").run()


def test_unpinned_resume_refuses_multiple_compatible_mirrored_runs(tmp_path: Path):
    import shutil

    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="first-compatible-run")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / original.run_dir.relative_to(config.output_root)
    shutil.copytree(mirrored_run, mirrored_run.with_name("second-compatible-run"))
    shutil.rmtree(original.run_dir)

    with pytest.raises(ValueError, match="multiple compatible mirrored runs"):
        Pipeline(
            replace(config, run_id=None, resume=True),
            feature_dir=tmp_path / "dense",
        ).run()
