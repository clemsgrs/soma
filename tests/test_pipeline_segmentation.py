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


def test_pipeline_mirrors_only_a_verified_completed_fold(tmp_path: Path) -> None:
    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path)

    result = Pipeline(config, feature_dir=tmp_path / "dense").run()

    mirrored_run = config.mirror_root / result.run_dir.relative_to(config.output_root)
    fold_bundle = mirrored_run / "recovery" / "folds" / "fold_0"
    manifest = json.loads((fold_bundle / "manifest.json").read_text())
    assert manifest["kind"] == "fold"
    assert "metrics.json" in manifest["files"]
    assert not (mirrored_run / "recovery" / "checkpoints").exists()
    assert not (result.run_dir / "recovery").exists()
    assert not (result.run_dir / "sampler_audit.json").exists()


def test_unpinned_resume_restores_completed_fold_after_local_node_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="lost-node-run")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    original_summary = dict(original.summary)
    lost_run_dir = original.run_dir
    shutil.rmtree(lost_run_dir)

    def forbid_retraining(*args, **kwargs):
        raise AssertionError("a completed mirrored fold must be restored")

    monkeypatch.setattr("soma.training.trainer.Trainer.fit", forbid_retraining)
    resumed = Pipeline(
        replace(config, run_id=None, resume=True), feature_dir=tmp_path / "dense"
    ).run()

    assert resumed.run_dir == lost_run_dir
    assert resumed.summary == original_summary
    assert resumed.fold_results == []
    assert (resumed.run_dir / "metrics.json").is_file()
    assert (resumed.run_dir / "best_model.pt").is_file()


def test_restore_rejects_a_completed_fold_with_a_bad_checksum(tmp_path: Path) -> None:
    import shutil

    from soma.pipeline import Pipeline

    config = _recoverable_seg_pipeline_config(tmp_path, run_id="corrupt-fold")
    original = Pipeline(config, feature_dir=tmp_path / "dense").run()
    mirrored_run = config.mirror_root / original.run_dir.relative_to(config.output_root)
    (mirrored_run / "recovery" / "folds" / "fold_0" / "best_model.pt").write_bytes(
        b"corrupt"
    )
    shutil.rmtree(original.run_dir)

    with pytest.raises(OSError, match="no local run is available"):
        Pipeline(
            replace(config, run_id=None, resume=True),
            feature_dir=tmp_path / "dense",
        ).run()


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
