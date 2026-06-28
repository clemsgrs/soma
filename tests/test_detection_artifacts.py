"""DetectionArtifactWriter — plain pred/GT point overlays, per-image manifest, metrics.

Mirrors ``tests/test_dense_artifacts.py`` (the segmentation writer prior art) but for
the detection path: each per-image payload is decoded+matched once upstream and handed
in as ``(pred points, assignment, gt points)``. The writer renders the plain
predicted-point and ground-truth overlays (color = class via ``class_palette``),
accumulates a per-image manifest row whose ``mean_f1``/counts come from the *same*
``reduce_f1`` reduction the split metric uses, and flushes a per-image manifest CSV +
an unconditional split-level per-class metrics CSV. The fail-soft overlay path (cached
runs need not retain the source tile) is asserted both ways.
"""

from __future__ import annotations

import csv
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from soma.detection.matching import match_assignment, reduce_f1
from soma.evaluation.detection_artifacts import DetectionArtifactWriter

NUM_CLASSES = 2
H = W = 16
DELTA = 3.0


def _head(num_classes: int = NUM_CLASSES):
    # The writer needs num_classes (palette), crop_box (target-frame size) and delta_px
    # (the match-overlay ring radius = the matching tolerance).
    return types.SimpleNamespace(num_classes=num_classes, delta_px=DELTA, _crop_box=(0, 0, H, W))


def _dataset_with_images(tmp_path: Path, sample_ids: list[str], make_image: bool, size=(8, 8)):
    samples = {}
    for sid in sample_ids:
        if make_image:
            img_path = tmp_path / f"{sid}.jpg"
            # A mid-gray tile, deliberately smaller than the target frame so the writer
            # exercises the resize-to-target path.
            Image.fromarray(np.full((size[1], size[0], 3), 127, dtype=np.uint8)).save(img_path)
        else:
            img_path = tmp_path / f"{sid}-missing.jpg"  # never created
        samples[sid] = types.SimpleNamespace(sample_id=sid, image_path=img_path)
    return types.SimpleNamespace(samples=samples)


def _payload(pred_xy, pred_class, pred_score, gt_xy, gt_class):
    """Build the once-per-image decode+match payload the eval loop hands the writer."""
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(-1, 2)
    pred_class = np.asarray(pred_class, dtype=np.int64).reshape(-1)
    pred_score = np.asarray(pred_score, dtype=np.float64).reshape(-1)
    gt_xy = np.asarray(gt_xy, dtype=np.float64).reshape(-1, 2)
    gt_class = np.asarray(gt_class, dtype=np.int64).reshape(-1)
    assignment = match_assignment(
        pred_xy, pred_class, pred_score, gt_xy, gt_class,
        num_classes=NUM_CLASSES, delta=DELTA, method="hungarian",
    )
    return dict(
        heatmap=torch.zeros(NUM_CLASSES, H, W),
        pred_xy=pred_xy, pred_class=pred_class, pred_score=pred_score,
        assignment=assignment, gt_xy=gt_xy, gt_class=gt_class,
    )


def test_writer_emits_overlays_manifest_and_metrics(tmp_path):
    sample_ids = ["s0", "s1"]
    dataset = _dataset_with_images(tmp_path, sample_ids, make_image=True)
    writer = DetectionArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset
    )

    # One matched TP (class 1) + one stray FP (class 0); 1 GT (class 1).
    for sid in sample_ids:
        writer.add_image(
            sample_id=sid,
            **_payload(
                pred_xy=[[4.0, 4.0], [10.0, 10.0]],
                pred_class=[1, 0],
                pred_score=[0.9, 0.8],
                gt_xy=[[4.0, 4.0]],
                gt_class=[1],
            ),
        )
    manifest_path = writer.finalize()
    assert manifest_path == tmp_path / "detection_per_image_test.csv"

    for sid in sample_ids:
        pred_overlay = tmp_path / "pred_overlays" / "test" / f"{sid}.png"
        gt_overlay = tmp_path / "gt_overlays" / "test" / f"{sid}.png"
        assert pred_overlay.is_file()
        assert gt_overlay.is_file()
        # Overlays are RGB and resized to the target frame (not the 8x8 source).
        assert np.asarray(Image.open(pred_overlay)).shape == (H, W, 3)
        assert np.asarray(Image.open(gt_overlay)).shape == (H, W, 3)

    rows = {r["sample_id"]: r for r in csv.DictReader(manifest_path.open())}
    assert set(rows) == set(sample_ids)
    r0 = rows["s0"]
    assert r0["pred_overlay_path"] == "pred_overlays/test/s0.png"
    assert r0["gt_overlay_path"] == "gt_overlays/test/s0.png"
    assert int(r0["n_pred"]) == 2
    assert int(r0["n_gt"]) == 1
    assert int(r0["tp"]) == 1
    assert int(r0["fp"]) == 1
    assert int(r0["fn"]) == 0

    # Split-level metrics CSV: per-class F1/precision/recall, dataset-global, always.
    metrics_rows = {
        r["metric"]: r for r in csv.DictReader((tmp_path / "metrics_test.csv").open())
    }
    assert "mean_f1" in metrics_rows
    assert {f"f1_class_{c}" for c in range(NUM_CLASSES)} <= set(metrics_rows)
    assert {f"precision_class_{c}" for c in range(NUM_CLASSES)} <= set(metrics_rows)
    assert {f"recall_class_{c}" for c in range(NUM_CLASSES)} <= set(metrics_rows)
    assert metrics_rows["f1_class_0"]["aggregation"] == "dataset_global"


def test_writer_emits_per_class_match_overlays(tmp_path):
    """One match overlay per class per tile + a manifest column per class (slice #175).

    The payload yields, for class 1, a TP (matched pred+GT at (4,4)) and an FN (lone GT
    at (12,12)); for class 0, a lone FP pred at (10,10). The overlay grammar is therefore
    exercised end to end: green (TP) + blue (FN) in the class-1 image, red (FP) in the
    class-0 image, GT as δ-radius rings, predictions as filled dots.
    """
    dataset = _dataset_with_images(tmp_path, ["s0"], make_image=True)
    writer = DetectionArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset
    )
    writer.add_image(
        sample_id="s0",
        **_payload(
            pred_xy=[[4.0, 4.0], [10.0, 10.0]],
            pred_class=[1, 0],
            pred_score=[0.9, 0.8],
            gt_xy=[[4.0, 4.0], [12.0, 12.0]],
            gt_class=[1, 1],
        ),
    )
    manifest_path = writer.finalize()

    # One overlay per class per tile, under match_overlays/class_<c>/<split>/, RGB, target frame.
    for c in range(NUM_CLASSES):
        overlay = tmp_path / "match_overlays" / f"class_{c}" / "test" / "s0.png"
        assert overlay.is_file()
        assert np.asarray(Image.open(overlay)).shape == (H, W, 3)

    # The per-image manifest lists each class's match overlay path.
    row = next(csv.DictReader(manifest_path.open()))
    for c in range(NUM_CLASSES):
        assert row[f"match_overlay_class_{c}"] == f"match_overlays/class_{c}/test/s0.png"

    # Hue grammar (TP green / FP red / FN blue), asserted as channel-dominant pixels —
    # external behavior, not pixel-exact rendering.
    def _has_hue(arr: np.ndarray, channel: int) -> bool:
        others = [i for i in range(3) if i != channel]
        return bool(
            np.any(
                (arr[..., channel] > 150)
                & (arr[..., others[0]] < 90)
                & (arr[..., others[1]] < 90)
            )
        )

    c0 = np.asarray(Image.open(tmp_path / "match_overlays" / "class_0" / "test" / "s0.png"))
    c1 = np.asarray(Image.open(tmp_path / "match_overlays" / "class_1" / "test" / "s0.png"))
    assert _has_hue(c0, 0)  # class-0: a lone red FP dot
    assert _has_hue(c1, 1)  # class-1: a green TP
    assert _has_hue(c1, 2)  # class-1: a blue FN ring


def test_manifest_mean_f1_matches_reduce_f1(tmp_path):
    """Each row's mean_f1 is reduce_f1 on that image's own (1, C, 3) counts — so the
    per-image number cannot drift from the headline reduction."""
    dataset = _dataset_with_images(tmp_path, ["s0"], make_image=True)
    writer = DetectionArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset
    )
    payload = _payload(
        pred_xy=[[4.0, 4.0], [10.0, 10.0]],
        pred_class=[1, 0],
        pred_score=[0.9, 0.8],
        gt_xy=[[4.0, 4.0]],
        gt_class=[1],
    )
    writer.add_image(sample_id="s0", **payload)
    manifest_path = writer.finalize()

    counts = np.zeros((NUM_CLASSES, 3), dtype=np.int64)
    for c, m in enumerate(payload["assignment"]):
        counts[c] = m.counts
    expected = reduce_f1(
        torch.from_numpy(counts).to(torch.long).unsqueeze(0),
        num_classes=NUM_CLASSES,
        aggregation="dataset_global",
    )["mean_f1"]

    row = next(csv.DictReader(manifest_path.open()))
    assert float(row["mean_f1"]) == pytest.approx(expected)


def test_save_detection_overlays_false_suppresses_overlays(tmp_path):
    """Off -> no overlay PNGs, but the manifest + metrics CSVs are still written."""
    dataset = _dataset_with_images(tmp_path, ["s0"], make_image=True)
    writer = DetectionArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset,
        save_detection_overlays=False,
    )
    writer.add_image(
        sample_id="s0",
        **_payload([[4.0, 4.0]], [1], [0.9], [[4.0, 4.0]], [1]),
    )
    manifest_path = writer.finalize()

    assert not (tmp_path / "pred_overlays").exists()
    assert not (tmp_path / "gt_overlays").exists()
    assert not (tmp_path / "match_overlays").exists()
    assert (tmp_path / "metrics_test.csv").is_file()
    row = next(csv.DictReader(manifest_path.open()))
    assert row["pred_overlay_path"] == ""
    assert row["gt_overlay_path"] == ""
    for c in range(NUM_CLASSES):
        assert row[f"match_overlay_class_{c}"] == ""
    assert int(row["tp"]) == 1


def test_overlay_fail_soft_on_missing_image(tmp_path):
    """A missing source tile skips the overlay but still records the manifest row."""
    dataset = _dataset_with_images(tmp_path, ["s0"], make_image=False)
    writer = DetectionArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=dataset
    )
    writer.add_image(
        sample_id="s0",
        **_payload([[4.0, 4.0]], [1], [0.9], [[4.0, 4.0]], [1]),
    )
    manifest_path = writer.finalize()

    assert not (tmp_path / "pred_overlays" / "test" / "s0.png").exists()
    assert not (tmp_path / "gt_overlays" / "test" / "s0.png").exists()
    assert not (tmp_path / "match_overlays").exists()
    row = next(csv.DictReader(manifest_path.open()))
    assert row["pred_overlay_path"] == ""  # skipped, recorded empty
    assert row["gt_overlay_path"] == ""
    for c in range(NUM_CLASSES):
        assert row[f"match_overlay_class_{c}"] == ""  # skipped, recorded empty
    assert int(row["tp"]) == 1  # counts still recorded


def test_writer_without_dataset_skips_overlays(tmp_path):
    writer = DetectionArtifactWriter(
        head=_head(), split="tune", output_dir=tmp_path, dataset=None
    )
    writer.add_image(
        sample_id="s0",
        **_payload([[4.0, 4.0]], [1], [0.9], [[4.0, 4.0]], [1]),
    )
    manifest_path = writer.finalize()
    assert manifest_path.is_file()
    assert not (tmp_path / "pred_overlays").exists()
    assert not (tmp_path / "gt_overlays").exists()
    assert not (tmp_path / "match_overlays").exists()
    assert (tmp_path / "metrics_tune.csv").is_file()


def test_metrics_csv_written_with_no_images(tmp_path):
    """The split-level metrics CSV is emitted unconditionally (even an empty split)."""
    writer = DetectionArtifactWriter(
        head=_head(), split="test", output_dir=tmp_path, dataset=None
    )
    manifest_path = writer.finalize()
    assert manifest_path.is_file()
    metrics_rows = {
        r["metric"]: r for r in csv.DictReader((tmp_path / "metrics_test.csv").open())
    }
    assert "mean_f1" in metrics_rows
    assert {f"f1_class_{c}" for c in range(NUM_CLASSES)} <= set(metrics_rows)
