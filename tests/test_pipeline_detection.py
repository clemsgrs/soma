"""End-to-end (offline) routing test for dataset_type='detection'.

Builds a pre-extracted dense feature store + point-annotation manifest + splits on
disk, then runs ``train_one_detection_fold`` — exercising geometry-from-sidecar,
decoder + DetectionHead construction, the DetectionDataset loaders, ``Trainer.fit``
(streaming F1@δ tune metric), checkpoint reload, the tune-split threshold sweep, and
``_evaluate_detection`` (metrics + level-0 predictions CSV). No encoder / GPU needed:
the grids are written directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from soma.config import (
    DecoderConfig,
    EvalConfig,
    PreprocessingConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import DetectionManifest, Splits
from soma.dense import DenseFeatureStore, DenseSampleSpacing
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.pipeline import train_one_detection_fold

NUM_CLASSES = 2
TARGET = 16
PATCH = 4
FEATURE_DIM = 4
# Detection distances are µm-only; grids are read flat at this bookkeeping spacing, so
# source_spacing_um == effective_spacing_um == SPACING makes the transform identity, and
# match_distance/sigma in µm divide by SPACING to recover target-frame pixels.
SPACING = 0.2  # µm/px; 0.6 µm -> 3 px (δ), 0.3 µm -> 1.5 px (σ)


def test_detection_spacing_allows_heterogeneous_sources_with_one_effective_grid():
    from types import SimpleNamespace

    from soma.pipeline import _resolve_detection_sample_spacings

    by_id = {
        "a": DenseSampleSpacing(source_spacing_um=0.25, effective_spacing_um=0.5),
        "b": DenseSampleSpacing(source_spacing_um=0.4, effective_spacing_um=0.5),
    }
    source = SimpleNamespace(spacing=lambda sample_id: by_id[sample_id])

    resolved, effective = _resolve_detection_sample_spacings(
        source, [SimpleNamespace(sample_id="a"), SimpleNamespace(sample_id="b")]
    )

    assert resolved == by_id
    assert effective == 0.5


def test_detection_spacing_rejects_multiple_effective_grids_with_sample_ids():
    from types import SimpleNamespace

    import pytest

    from soma.pipeline import _resolve_detection_sample_spacings

    by_id = {
        "a": DenseSampleSpacing(source_spacing_um=0.25, effective_spacing_um=0.5),
        "b": DenseSampleSpacing(source_spacing_um=0.25, effective_spacing_um=1.0),
    }
    source = SimpleNamespace(spacing=lambda sample_id: by_id[sample_id])

    with pytest.raises(ValueError, match=r"effective_spacing_um.*0.5.*a.*1.0.*b"):
        _resolve_detection_sample_spacings(
            source, [SimpleNamespace(sample_id="a"), SimpleNamespace(sample_id="b")]
        )


def _build_detection_run(root: Path, sample_ids: list[str], make_images: bool = False):
    dense_dir = root / "dense"
    points_dir = root / "points"
    dense_dir.mkdir()
    points_dir.mkdir()

    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    meta = dense_grid_metadata(geom, feature_dim=FEATURE_DIM, pad_mode="reflect", spacing_um=SPACING)
    meta.update(source_spacing_um=SPACING, effective_spacing_um=SPACING)

    rng = np.random.default_rng(0)
    rows = []
    for sid in sample_ids:
        write_dense_grid(dense_dir, sid, torch.randn(FEATURE_DIM, *geom.grid_shape), meta)
        # A couple of points per tile, well inside the frame, classes 0/1.
        pts = points_dir / f"{sid}.csv"
        pts.write_text("x,y,class\n4,4,0\n11,11,1\n")
        if make_images:
            # A real source tile (absolute path) so the overlay writer can render — the
            # default builder leaves image_path dangling (overlays then fail-soft).
            from PIL import Image

            img_path = root / f"{sid}.jpg"
            Image.fromarray(np.full((TARGET, TARGET, 3), 127, dtype=np.uint8)).save(img_path)
            img_ref = str(img_path)
        else:
            img_ref = f"{sid}.jpg"
        rows.append((sid, img_ref, str(pts)))

    manifest_csv = root / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,points_path,spacing_at_level_0\n"
        + "\n".join(f"{sid},{img},{pp},{SPACING}" for sid, img, pp in rows)
        + "\n"
    )
    splits_csv = root / "splits.csv"
    assign = {sample_ids[0]: "train", sample_ids[1]: "train", sample_ids[2]: "tune", sample_ids[3]: "test"}
    splits_csv.write_text(
        "sample_id,split,fold\n" + "\n".join(f"{sid},{s},0" for sid, s in assign.items()) + "\n"
    )

    manifest = DetectionManifest(manifest_csv)
    splits = Splits(splits_csv, manifest)
    store = DenseFeatureStore(dense_dir)
    return manifest, splits, store


def test_train_one_detection_fold_end_to_end(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids)

    result = train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,  # µm -> 3 px at SPACING
                "sigma": 0.3,  # µm -> 1.5 px
            },
        ),
        training=TrainingConfig(epochs=2, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1", "f1_per_class"]),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )

    assert "mean_f1" in result.tune_report.metrics
    assert "test" in result.test_reports
    assert "mean_f1" in result.test_reports["test"].metrics

    # Frozen thresholds + per-split prediction CSVs are written.
    thr = json.loads((tmp_path / "fold" / "detection_thresholds.json").read_text())
    assert len(thr["score_threshold_per_class"]) == NUM_CLASSES
    assert (tmp_path / "fold" / "predictions_test.csv").exists()
    header = (tmp_path / "fold" / "predictions_test.csv").read_text().splitlines()[0]
    assert header == "sample_id,x,y,class,score"


def test_train_one_detection_fold_eval_only_from_checkpoint(tmp_path: Path):
    """``checkpoint_path`` loads a trained checkpoint and skips the train loop, so a
    finished fold's eval artifacts can be regenerated without retraining. The threshold
    sweep + scoring are deterministic, so the regenerated metrics reproduce the original;
    no epoch history exists (``train_result is None``, no ``training_history.json``) and no
    new checkpoint is written."""
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids)

    common = dict(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,
                "sigma": 0.3,
            },
        ),
        training=TrainingConfig(epochs=2, batch_size=2),
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1", "f1_per_class"]),
        preprocessing=PreprocessingConfig(
            requested_spacing_um=SPACING, requested_tile_size_px=TARGET
        ),
    )

    trained = train_one_detection_fold(fold_dir=tmp_path / "trained", **common)
    checkpoint = tmp_path / "trained" / "best_model.pt"
    assert checkpoint.exists()

    eval_only = train_one_detection_fold(
        fold_dir=tmp_path / "eval_only", checkpoint_path=checkpoint, **common
    )

    # Training was skipped: no epoch history, no history file, no fresh checkpoint.
    assert eval_only.train_result is None
    assert not (tmp_path / "eval_only" / "training_history.json").exists()
    assert not (tmp_path / "eval_only" / "best_model.pt").exists()
    # The same weights + deterministic decode/match reproduce the headline metric exactly.
    assert (
        eval_only.test_reports["test"].metrics["mean_f1"]
        == trained.test_reports["test"].metrics["mean_f1"]
    )


def test_train_one_detection_fold_on_attention_grids(tmp_path: Path):
    """Detection on cls_attention grids is a pure feature_kind flip — the head/decoder
    are input_dim-agnostic, so a K-channel attention grid trains the same fold."""
    dense_dir = tmp_path / "dense"
    points_dir = tmp_path / "points"
    dense_dir.mkdir()
    points_dir.mkdir()

    K = 6  # attention channels (blocks × heads), not a patch-feature dim
    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    meta = dense_grid_metadata(
        geom, feature_dim=K, pad_mode="reflect", spacing_um=SPACING,
        feature_kind="cls_attention", attention_blocks=(-1,),
    )
    meta.update(source_spacing_um=SPACING, effective_spacing_um=SPACING)
    sample_ids = ["s0", "s1", "s2", "s3"]
    rows = []
    for sid in sample_ids:
        write_dense_grid(dense_dir, sid, torch.rand(K, *geom.grid_shape), meta)
        pts = points_dir / f"{sid}.csv"
        pts.write_text("x,y,class\n4,4,0\n11,11,1\n")
        rows.append((sid, f"{sid}.jpg", str(pts)))
    (tmp_path / "manifest.csv").write_text(
        "sample_id,image_path,points_path,spacing_at_level_0\n"
        + "\n".join(f"{s},{i},{p},{SPACING}" for s, i, p in rows) + "\n"
    )
    assign = {sample_ids[0]: "train", sample_ids[1]: "train", sample_ids[2]: "tune", sample_ids[3]: "test"}
    (tmp_path / "splits.csv").write_text(
        "sample_id,split,fold\n" + "\n".join(f"{s},{v},0" for s, v in assign.items()) + "\n"
    )
    manifest = DetectionManifest(tmp_path / "manifest.csv")
    splits = Splits(tmp_path / "splits.csv", manifest)
    store = DenseFeatureStore(dense_dir)
    assert store.metadata("s0")["feature_kind"] == "cls_attention"
    assert store.feature_dim == K

    result = train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,
                "sigma": 0.3,
            },
        ),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1"]),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )
    assert "mean_f1" in result.test_reports["test"].metrics


def test_train_one_detection_fold_requires_num_classes(tmp_path: Path):
    import pytest

    manifest, splits, store = _build_detection_run(tmp_path, ["s0", "s1", "s2", "s3"])
    with pytest.raises(ValueError, match="num_classes"):
        train_one_detection_fold(
            feature_store=store,
            dataset=manifest,
            fold_split=splits.folds[0],
            task=TaskConfig(name="detection", params={}),
            training=TrainingConfig(epochs=1, batch_size=2),
            fold_dir=tmp_path / "fold",
            decoder=DecoderConfig(name="lightweight_conv"),
        )


def test_detection_fold_writes_qualitative_artifacts(tmp_path: Path):
    """A detection fold writes the per-image manifest + split-level metrics CSVs and,
    with real source tiles, the plain pred/GT point overlays (design §3) — all from the
    consolidated single decode+match, leaving the per-point CSV byte-for-byte unchanged."""
    import csv

    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids, make_images=True)

    train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,
                "sigma": 0.3,
            },
        ),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1", "f1_per_class"]),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )
    fold = tmp_path / "fold"

    # Per-point CSV header unchanged (the documented stitch-ready level-0 output).
    assert (fold / "predictions_test.csv").read_text().splitlines()[0] == "sample_id,x,y,class,score"

    for split in ("tune", "test"):
        # Per-image manifest CSV (always) with the specified columns.
        manifest_csv = fold / f"detection_per_image_{split}.csv"
        assert manifest_csv.exists()
        rows = list(csv.DictReader(manifest_csv.open()))
        assert rows  # one row per evaluated tile
        assert {
            "sample_id", "pred_overlay_path", "gt_overlay_path",
            *(f"match_overlay_class_{c}" for c in range(NUM_CLASSES)),
            "n_pred", "n_gt", "tp", "fp", "fn", "mean_f1",
        } <= set(rows[0])
        # Split-level per-class metrics CSV (always).
        metrics_rows = {r["metric"] for r in csv.DictReader((fold / f"metrics_{split}.csv").open())}
        assert "mean_f1" in metrics_rows
        assert {f"f1_class_{c}" for c in range(NUM_CLASSES)} <= metrics_rows

    # Real source tiles -> pred/GT overlays + one per-class match overlay per evaluated
    # tile (test split = s3).
    assert (fold / "pred_overlays" / "test" / "s3.png").is_file()
    assert (fold / "gt_overlays" / "test" / "s3.png").is_file()
    for c in range(NUM_CLASSES):
        assert (fold / "match_overlays" / f"class_{c}" / "test" / "s3.png").is_file()


def test_detection_fold_overlays_suppressible(tmp_path: Path):
    """save_detection_overlays=False suppresses the overlays; the per-point CSV, the
    per-image manifest, and the split-level metrics CSV are still written."""
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids, make_images=True)

    train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,
                "sigma": 0.3,
            },
        ),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1"], save_detection_overlays=False),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )
    fold = tmp_path / "fold"

    assert not (fold / "pred_overlays").exists()
    assert not (fold / "gt_overlays").exists()
    assert not (fold / "match_overlays").exists()
    assert (fold / "predictions_test.csv").exists()
    assert (fold / "detection_per_image_test.csv").exists()
    assert (fold / "metrics_test.csv").exists()


def test_detection_fold_heatmap_artifacts_opt_in(tmp_path: Path):
    """save_detection_heatmaps=True writes per-class viridis overlays + a float16 npz
    sidecar per evaluated tile, and threads the manifest's heatmap columns; off by default
    those artifacts are absent and the columns are empty."""
    import csv

    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids, make_images=True)

    train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,
                "sigma": 0.3,
            },
        ),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1"], save_detection_heatmaps=True),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )
    fold = tmp_path / "fold"

    # Test split = s3: per-class colormap overlay + raw float16 (C,H,W) npz sidecar.
    for c in range(NUM_CLASSES):
        assert (fold / "heatmap_overlays" / f"class_{c}" / "test" / "s3.png").is_file()
    npz = fold / "heatmaps" / "test" / "s3.npz"
    assert npz.is_file()
    arr = np.load(npz)["heatmap"]
    assert arr.dtype == np.float16
    assert arr.shape[0] == NUM_CLASSES

    row = next(r for r in csv.DictReader((fold / "detection_per_image_test.csv").open()) if r["sample_id"] == "s3")
    assert {f"heatmap_overlay_class_{c}" for c in range(NUM_CLASSES)} <= set(row)
    assert row["heatmap_npz_path"] == "heatmaps/test/s3.npz"
    assert row["heatmap_overlay_class_0"] == "heatmap_overlays/class_0/test/s3.png"


def test_train_one_detection_fold_holdout_test_skips_test(tmp_path: Path):
    """evaluation.holdout_test: no test inference, no test artifacts, tune-only metrics.

    The tune-split threshold sweep and tune evaluation still run, so the run is a
    valid model-selection candidate while never touching the declared test split.
    """
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits, store = _build_detection_run(tmp_path, sample_ids)

    result = train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={
                "num_classes": NUM_CLASSES,
                "match_distance": 0.6,
                "sigma": 0.3,
            },
        ),
        training=TrainingConfig(epochs=2, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_f1", "f1_per_class"], holdout_test=True),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )

    # Tune still evaluated; no test split touched.
    assert "mean_f1" in result.tune_report.metrics
    assert result.test_reports == {}

    # Threshold sweep (tune-only) still ran and was frozen.
    thr = json.loads((tmp_path / "fold" / "detection_thresholds.json").read_text())
    assert len(thr["score_threshold_per_class"]) == NUM_CLASSES

    # No test prediction CSV; metrics.json carries tune only.
    assert not (tmp_path / "fold" / "predictions_test.csv").exists()
    metrics = json.loads((tmp_path / "fold" / "metrics.json").read_text())
    assert list(metrics.keys()) == ["tune"]
