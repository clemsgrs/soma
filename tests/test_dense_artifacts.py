"""DenseArtifactWriter — streaming prediction rasters, overlays, per-tile CSV (1g).

Covers the writer in isolation (so the raster-shape/dtype and overlay-fail-soft
contracts are pinned without a full train loop) plus the integration assertion that
``train_one_segmentation_fold`` lands the artifacts on disk. The fail-soft overlay
path is the fragile one (cached-feature runs need not retain source tiles), so it is
asserted both ways: a real image → an overlay file; a missing image → no overlay,
raster still written.
"""

from __future__ import annotations

import csv
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from soma.evaluation.dense_artifacts import DenseArtifactWriter, class_palette
from soma.training.segmentation_dataset import SegmentationBatch

NUM_CLASSES = 3
H = W = 4


def _head(num_classes: int = NUM_CLASSES):
    return types.SimpleNamespace(num_classes=num_classes)


def _logits_from_pred(pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, H, W) class indices -> sharp one-hot logits (B, C, H, W)."""
    return F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float() * 10.0


def _stat_row(num_images: int) -> torch.Tensor:
    # Shape only matters for indexing; reduce_dice_iou tolerates these counts.
    return torch.zeros(num_images, NUM_CLASSES, 3)


def _dataset_with_images(tmp_path: Path, sample_ids: list[str], make_image: bool):
    samples = {}
    for sid in sample_ids:
        if make_image:
            img_path = tmp_path / f"{sid}.png"
            Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(img_path)
        else:
            img_path = tmp_path / f"{sid}-missing.png"  # never created
        samples[sid] = types.SimpleNamespace(sample_id=sid, image_path=img_path)
    return types.SimpleNamespace(samples=samples)


def test_class_palette_shape_and_background():
    palette = class_palette(NUM_CLASSES)
    assert palette.shape == (NUM_CLASSES, 3)
    assert palette.dtype == np.uint8
    assert tuple(palette[0]) == (0, 0, 0)  # class 0 = background, black


def test_writer_emits_raster_overlay_and_csv(tmp_path):
    sample_ids = ["s0", "s1"]
    dataset = _dataset_with_images(tmp_path, sample_ids, make_image=True)
    writer = DenseArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset
    )

    pred = torch.tensor(
        [[[0, 1, 1, 0], [0, 1, 1, 0], [2, 2, 0, 0], [2, 2, 0, 0]],
         [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]],
        dtype=torch.long,
    )
    logits = _logits_from_pred(pred, NUM_CLASSES)
    batch = SegmentationBatch(features=torch.zeros(2, 1, 2, 2), targets={"mask": pred}, sample_ids=tuple(sample_ids))

    writer(batch, logits, _stat_row(2))
    csv_path = writer.finalize()

    for i, sid in enumerate(sample_ids):
        raster_path = tmp_path / "preds" / "test" / f"{sid}.png"
        assert raster_path.is_file()
        raster = np.asarray(Image.open(raster_path))
        # Raster is target-res, uint8, raw class indices in [0, num_classes).
        assert raster.shape == (H, W)
        assert raster.dtype == np.uint8
        assert raster.max() < NUM_CLASSES
        np.testing.assert_array_equal(raster, pred[i].numpy().astype(np.uint8))
        # A real source image -> prediction + GT overlay files (RGB, target-res).
        overlay_path = tmp_path / "pred_overlays" / "test" / f"{sid}.png"
        assert overlay_path.is_file()
        assert np.asarray(Image.open(overlay_path)).shape == (H, W, 3)
        gt_overlay_path = tmp_path / "gt_overlays" / "test" / f"{sid}.png"
        assert gt_overlay_path.is_file()
        assert np.asarray(Image.open(gt_overlay_path)).shape == (H, W, 3)

    rows = list(csv.DictReader(csv_path.open()))
    assert [r["sample_id"] for r in rows] == sample_ids
    assert rows[0]["pred_path"] == "preds/test/s0.png"
    assert rows[0]["pred_overlay_path"] == "pred_overlays/test/s0.png"
    assert rows[0]["gt_overlay_path"] == "gt_overlays/test/s0.png"
    assert {"dice", "iou"} <= set(rows[0])
    # Probabilities are opt-in: off by default -> empty column, no probs dir.
    assert rows[0]["probs_path"] == ""
    assert not (tmp_path / "probs").exists()

    # Split-level metrics.csv: per-class Dice + means, always (independent of monitor).
    metrics_rows = {r["metric"]: r for r in csv.DictReader((tmp_path / "metrics_test.csv").open())}
    assert "mean_dice" in metrics_rows and "mean_iou" in metrics_rows
    assert {f"dice_class_{c}" for c in range(NUM_CLASSES)} <= set(metrics_rows)
    # The aggregation column disambiguates the mixed conventions in one file.
    assert metrics_rows["mean_dice"]["aggregation"] == "per_image_macro"
    assert metrics_rows["dice_class_0"]["aggregation"] == "dataset_global"


def test_overlay_fail_soft_on_missing_image(tmp_path):
    """A missing source image skips the overlay but still writes the raster + CSV row."""
    dataset = _dataset_with_images(tmp_path, ["s0"], make_image=False)
    writer = DenseArtifactWriter(head=_head(), split="test", output_dir=tmp_path, dataset=dataset)

    pred = torch.zeros(1, H, W, dtype=torch.long)
    batch = SegmentationBatch(features=torch.zeros(1, 1, 2, 2), targets={"mask": pred}, sample_ids=("s0",))
    writer(batch, _logits_from_pred(pred, NUM_CLASSES), _stat_row(1))
    csv_path = writer.finalize()

    assert (tmp_path / "preds" / "test" / "s0.png").is_file()
    assert not (tmp_path / "pred_overlays" / "test" / "s0.png").exists()
    assert not (tmp_path / "gt_overlays" / "test" / "s0.png").exists()
    row = next(csv.DictReader(csv_path.open()))
    assert row["pred_overlay_path"] == ""  # skipped, recorded as empty
    assert row["gt_overlay_path"] == ""


def test_overlay_reads_roi_window_for_slide_manifest(tmp_path, monkeypatch):
    """A record with ``region`` set (whole-slide image_path) reads just the ROI window
    at the run spacing — never opening the gigapixel slide with PIL."""
    import soma.dense.reader as reader

    calls = {}

    def fake_region(path, *, location, size, spacing_um, backend, tolerance):
        calls["args"] = dict(path=str(path), location=location, size=size, spacing_um=spacing_um)
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)  # (h, w, 3)

    monkeypatch.setattr(reader, "read_image_region_at_spacing", fake_region)

    record = types.SimpleNamespace(
        sample_id="roi0", image_path=tmp_path / "slide.tif", region=(64, 128)
    )
    dataset = types.SimpleNamespace(samples={"roi0": record})
    writer = DenseArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset,
        spacing_um=0.5, backend="auto", tolerance=0.1,
    )
    pred = torch.zeros(1, H, W, dtype=torch.long)
    batch = SegmentationBatch(features=torch.zeros(1, 1, 2, 2), targets={"mask": pred}, sample_ids=("roi0",))
    writer(batch, _logits_from_pred(pred, NUM_CLASSES), _stat_row(1))
    writer.finalize()

    # Read the ROI window (not the whole slide), at the run geometry.
    assert calls["args"] == {
        "path": str(tmp_path / "slide.tif"), "location": (64, 128), "size": (W, H), "spacing_um": 0.5,
    }
    assert (tmp_path / "pred_overlays" / "test" / "roi0.png").is_file()
    assert (tmp_path / "gt_overlays" / "test" / "roi0.png").is_file()


def test_overlay_fail_soft_on_decompression_bomb(tmp_path, monkeypatch):
    """A slide-manifest ROI whose window read raises (e.g. PIL bomb) skips the overlay
    but still writes the raster — the eval must not crash."""
    import soma.dense.reader as reader

    def boom(*args, **kwargs):
        raise Image.DecompressionBombError("too big")

    monkeypatch.setattr(reader, "read_image_region_at_spacing", boom)
    record = types.SimpleNamespace(
        sample_id="roi0", image_path=tmp_path / "slide.tif", region=(0, 0)
    )
    dataset = types.SimpleNamespace(samples={"roi0": record})
    writer = DenseArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset, spacing_um=0.5,
    )
    pred = torch.zeros(1, H, W, dtype=torch.long)
    batch = SegmentationBatch(features=torch.zeros(1, 1, 2, 2), targets={"mask": pred}, sample_ids=("roi0",))
    writer(batch, _logits_from_pred(pred, NUM_CLASSES), _stat_row(1))
    writer.finalize()

    assert (tmp_path / "preds" / "test" / "roi0.png").is_file()
    assert not (tmp_path / "pred_overlays" / "test" / "roi0.png").exists()


def test_save_segmentation_overlays_false_suppresses_overlays(tmp_path):
    """save_segmentation_overlays=False -> no overlay PNGs, but rasters + CSV still written
    (rasters/probabilities/CSV are unaffected by the overlay toggle)."""
    sample_ids = ["s0", "s1"]
    dataset = _dataset_with_images(tmp_path, sample_ids, make_image=True)
    writer = DenseArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset,
        save_segmentation_overlays=False,
    )
    pred = torch.zeros(2, H, W, dtype=torch.long)
    batch = SegmentationBatch(
        features=torch.zeros(2, 1, 2, 2), targets={"mask": pred}, sample_ids=tuple(sample_ids)
    )
    writer(batch, _logits_from_pred(pred, NUM_CLASSES), _stat_row(2))
    csv_path = writer.finalize()

    rows = {r["sample_id"]: r for r in csv.DictReader(csv_path.open())}
    for sid in sample_ids:
        # Raster is unaffected; overlay PNGs are suppressed (no overlay dirs).
        assert (tmp_path / "preds" / "test" / f"{sid}.png").is_file()
        assert not (tmp_path / "pred_overlays" / "test" / f"{sid}.png").exists()
        assert not (tmp_path / "gt_overlays" / "test" / f"{sid}.png").exists()
        assert rows[sid]["pred_overlay_path"] == ""
        assert rows[sid]["gt_overlay_path"] == ""
    assert not (tmp_path / "pred_overlays").exists()
    assert not (tmp_path / "gt_overlays").exists()


def test_save_segmentation_probabilities_writes_float16_sidecar(tmp_path):
    """save_segmentation_probabilities=True -> a float16 (C,H,W) softmax npz whose argmax matches the raster."""
    sample_ids = ["s0", "s1"]
    dataset = _dataset_with_images(tmp_path, sample_ids, make_image=True)
    writer = DenseArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset,
        save_segmentation_probabilities=True,
    )
    pred = torch.tensor(
        [[[0, 1, 1, 0], [0, 1, 1, 0], [2, 2, 0, 0], [2, 2, 0, 0]],
         [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]],
        dtype=torch.long,
    )
    logits = _logits_from_pred(pred, NUM_CLASSES)
    batch = SegmentationBatch(features=torch.zeros(2, 1, 2, 2), targets={"mask": pred}, sample_ids=tuple(sample_ids))
    writer(batch, logits, _stat_row(2))
    csv_path = writer.finalize()

    rows = {r["sample_id"]: r for r in csv.DictReader(csv_path.open())}
    for i, sid in enumerate(sample_ids):
        probs_path = tmp_path / "probs" / "test" / f"{sid}.npz"
        assert probs_path.is_file()
        assert rows[sid]["probs_path"] == f"probs/test/{sid}.npz"
        probs = np.load(probs_path)["probs"]
        assert probs.shape == (NUM_CLASSES, H, W)
        assert probs.dtype == np.float16
        # argmax over channels recovers the canonical raster.
        np.testing.assert_array_equal(probs.argmax(axis=0).astype(np.uint8), pred[i].numpy().astype(np.uint8))


def test_gt_overlay_tolerates_ignore_index(tmp_path):
    """ignore_index pixels in the GT mask are treated as background, not palette-indexed."""
    dataset = _dataset_with_images(tmp_path, ["s0"], make_image=True)
    head = types.SimpleNamespace(num_classes=NUM_CLASSES, ignore_index=255)
    writer = DenseArtifactWriter(head=head, split="test", output_dir=tmp_path, dataset=dataset)

    pred = torch.zeros(1, H, W, dtype=torch.long)
    mask = torch.full((1, H, W), 255, dtype=torch.long)  # all ignore_index
    mask[0, 0, 0] = 1  # one in-range foreground pixel
    batch = SegmentationBatch(features=torch.zeros(1, 1, 2, 2), targets={"mask": mask}, sample_ids=("s0",))
    writer(batch, _logits_from_pred(pred, NUM_CLASSES), _stat_row(1))
    writer.finalize()

    gt_overlay = tmp_path / "gt_overlays" / "test" / "s0.png"
    assert gt_overlay.is_file()
    assert np.asarray(Image.open(gt_overlay)).shape == (H, W, 3)


def test_writer_without_dataset_skips_overlays(tmp_path):
    writer = DenseArtifactWriter(head=_head(), split="tune", output_dir=tmp_path, dataset=None)
    pred = torch.zeros(1, H, W, dtype=torch.long)
    batch = SegmentationBatch(features=torch.zeros(1, 1, 2, 2), targets={"mask": pred}, sample_ids=("s0",))
    writer(batch, _logits_from_pred(pred, NUM_CLASSES), _stat_row(1))
    writer.finalize()
    assert (tmp_path / "preds" / "tune" / "s0.png").is_file()
    assert not (tmp_path / "pred_overlays").exists()  # overlay dir never created
    assert not (tmp_path / "gt_overlays").exists()
