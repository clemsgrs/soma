"""Unit tests for the detection-v1 dense-feature machinery (design §4-§7).

Pure pieces (coordinate transform, peak-heatmap encoder, peak extraction, F1@δ
matcher) tested against hand-computed values on tiny explicit inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.detection.encode import render_peak_heatmap, transform_points_to_target
from soma.detection.matching import (
    detection_counts,
    match_points,
    reduce_f1,
    sweep_score_thresholds,
)
from soma.detection.peaks import extract_peaks


# --------------------------------------------------------------------------- #
# Coordinate transform (level-0 -> target frame)
# --------------------------------------------------------------------------- #


def test_transform_identity_when_spacing_and_crop_match():
    pts = np.array([[10.0, 20.0], [0.0, 0.0]])
    out = transform_points_to_target(pts, level0_spacing=0.2, run_spacing=0.2)
    np.testing.assert_allclose(out, pts)


def test_transform_scales_by_spacing_ratio():
    pts = np.array([[10.0, 20.0]])
    # level-0 finer than run: a level-0 px maps to half a run px.
    out = transform_points_to_target(pts, level0_spacing=0.25, run_spacing=0.5)
    np.testing.assert_allclose(out, [[5.0, 10.0]])


def test_transform_applies_crop_offset():
    pts = np.array([[10.0, 20.0]])
    out = transform_points_to_target(
        pts, level0_spacing=0.5, run_spacing=0.5, crop_top=3, crop_left=4
    )
    np.testing.assert_allclose(out, [[6.0, 17.0]])


def test_transform_empty_passthrough():
    out = transform_points_to_target(np.zeros((0, 2)), level0_spacing=0.5, run_spacing=0.5)
    assert out.shape == (0, 2)


def test_transform_rejects_nonpositive_spacing():
    with pytest.raises(ValueError, match="positive"):
        transform_points_to_target(np.array([[1.0, 1.0]]), level0_spacing=0.0, run_spacing=0.5)


# --------------------------------------------------------------------------- #
# Peak-heatmap encoder
# --------------------------------------------------------------------------- #


def test_heatmap_peak_is_one_at_center():
    hm = render_peak_heatmap(
        np.array([[5.0, 7.0]]), np.array([0]),
        target_size=(16, 16), num_classes=1, sigma=2.0,
    )
    assert hm.shape == (1, 16, 16)
    assert hm[0, 7, 5] == pytest.approx(1.0)
    assert hm.max() == pytest.approx(1.0)


def test_heatmap_multiclass_channels_independent():
    hm = render_peak_heatmap(
        np.array([[2.0, 2.0], [8.0, 8.0]]), np.array([0, 1]),
        target_size=(12, 12), num_classes=2, sigma=1.0,
    )
    assert hm[0, 2, 2] == pytest.approx(1.0)
    assert hm[1, 8, 8] == pytest.approx(1.0)
    assert hm[1, 2, 2] < 1e-6  # the class-0 point does not bleed into channel 1
    assert hm[0, 8, 8] < 1e-6


def test_heatmap_overlap_is_max_not_sum():
    # Two coincident points: max-merge keeps the peak at 1 (sum would give 2).
    hm = render_peak_heatmap(
        np.array([[5.0, 5.0], [5.0, 5.0]]), np.array([0, 0]),
        target_size=(11, 11), num_classes=1, sigma=2.0,
    )
    assert hm[0, 5, 5] == pytest.approx(1.0)


def test_heatmap_drops_out_of_frame_points():
    hm = render_peak_heatmap(
        np.array([[-5.0, 5.0], [100.0, 5.0]]), np.array([0, 0]),
        target_size=(16, 16), num_classes=1, sigma=2.0,
    )
    assert hm.max() == pytest.approx(0.0)


def test_heatmap_empty_is_zero():
    hm = render_peak_heatmap(
        np.zeros((0, 2)), np.zeros((0,)), target_size=(8, 8), num_classes=3, sigma=1.0
    )
    assert hm.shape == (3, 8, 8)
    assert hm.sum() == pytest.approx(0.0)


def test_heatmap_rejects_class_out_of_range():
    with pytest.raises(ValueError, match="class id"):
        render_peak_heatmap(
            np.array([[1.0, 1.0]]), np.array([5]),
            target_size=(8, 8), num_classes=2, sigma=1.0,
        )


# --------------------------------------------------------------------------- #
# Peak extraction (round-trip with the encoder)
# --------------------------------------------------------------------------- #


def test_extract_peaks_roundtrips_rendered_points():
    pts = np.array([[5.0, 7.0], [20.0, 22.0]])
    cls = np.array([0, 0])
    hm = render_peak_heatmap(pts, cls, target_size=(32, 32), num_classes=1, sigma=2.0)
    xy, classes, scores = extract_peaks(hm, min_distance=3, score_threshold=0.5)
    assert xy.shape == (2, 2)
    # Recovered centres match the rendered ones (order by score; both peak ~1).
    found = {tuple(p) for p in xy.astype(int)}
    assert (5, 7) in found and (20, 22) in found
    assert np.all(scores > 0.99)


def test_extract_peaks_threshold_filters():
    hm = torch.zeros(1, 10, 10)
    hm[0, 5, 5] = 0.9
    hm[0, 1, 1] = 0.3
    xy, _, _ = extract_peaks(hm, min_distance=2, score_threshold=0.5)
    assert xy.shape == (1, 2)
    np.testing.assert_array_equal(xy.astype(int)[0], [5, 5])


def test_extract_peaks_nms_suppresses_close_lower_peak():
    hm = torch.zeros(1, 10, 10)
    hm[0, 5, 5] = 0.9
    hm[0, 5, 6] = 0.8  # within min_distance of the stronger peak
    xy, _, scores = extract_peaks(hm, min_distance=3, score_threshold=0.5)
    assert xy.shape == (1, 2)
    assert scores[0] == pytest.approx(0.9)


def test_extract_peaks_per_class_threshold():
    hm = torch.zeros(2, 10, 10)
    hm[0, 5, 5] = 0.6
    hm[1, 2, 2] = 0.6
    # class 0 threshold 0.5 (keep), class 1 threshold 0.7 (drop)
    xy, classes, _ = extract_peaks(hm, min_distance=2, score_threshold=[0.5, 0.7])
    assert xy.shape == (1, 2)
    assert classes[0] == 0


def test_extract_peaks_empty():
    xy, classes, scores = extract_peaks(torch.zeros(1, 8, 8), min_distance=2, score_threshold=0.5)
    assert xy.shape == (0, 2) and classes.shape == (0,) and scores.shape == (0,)


def test_extract_peaks_flat_heatmap_yields_no_peaks():
    # A uniform heatmap (e.g. sigmoid(0)=0.5 on an untrained/all-background tile) has no
    # peaks: the plateau-safe criterion must return nothing rather than flag every pixel
    # and flood NMS, even when the threshold is 0 (as the tune sweep can set).
    flat = torch.full((2, 64, 64), 0.5)
    xy, classes, scores = extract_peaks(flat, min_distance=3, score_threshold=0.0)
    assert xy.shape == (0, 2) and classes.shape == (0,) and scores.shape == (0,)


# --------------------------------------------------------------------------- #
# F1@δ matching
# --------------------------------------------------------------------------- #


def test_match_perfect_one_class():
    pred = np.array([[1.0, 1.0], [5.0, 5.0]])
    gt = np.array([[1.0, 1.0], [5.0, 5.0]])
    counts = match_points(
        pred, np.array([0, 0]), np.array([0.9, 0.8]), gt, np.array([0, 0]),
        num_classes=1, delta=2.0,
    )
    np.testing.assert_array_equal(counts, [[2, 0, 0]])


def test_match_distance_threshold_misses_far_pred():
    pred = np.array([[1.0, 1.0]])
    gt = np.array([[10.0, 10.0]])
    counts = match_points(
        pred, np.array([0]), np.array([0.9]), gt, np.array([0]), num_classes=1, delta=3.0
    )
    np.testing.assert_array_equal(counts, [[0, 1, 1]])  # 1 FP, 1 FN


def test_match_is_class_aware():
    # Same location, wrong class: not a match.
    pred = np.array([[1.0, 1.0]])
    gt = np.array([[1.0, 1.0]])
    counts = match_points(
        pred, np.array([1]), np.array([0.9]), gt, np.array([0]), num_classes=2, delta=2.0
    )
    np.testing.assert_array_equal(counts, [[0, 0, 1], [0, 1, 0]])


def test_match_hungarian_one_to_one():
    # Two preds near one GT: only one can match (one-to-one), other is FP.
    pred = np.array([[1.0, 1.0], [1.5, 1.0]])
    gt = np.array([[1.0, 1.0]])
    counts = match_points(
        pred, np.array([0, 0]), np.array([0.9, 0.8]), gt, np.array([0]),
        num_classes=1, delta=2.0, method="hungarian",
    )
    np.testing.assert_array_equal(counts, [[1, 1, 0]])


def test_match_greedy_matches_by_confidence():
    pred = np.array([[1.0, 1.0], [1.2, 1.0]])
    gt = np.array([[1.0, 1.0]])
    counts = match_points(
        pred, np.array([0, 0]), np.array([0.4, 0.9]), gt, np.array([0]),
        num_classes=1, delta=2.0, method="greedy",
    )
    np.testing.assert_array_equal(counts, [[1, 1, 0]])


def test_reduce_f1_dataset_global():
    # class 0: tp=2 fp=0 fn=0 -> f1=1 ; class 1: tp=1 fp=1 fn=1 -> p=r=0.5 f1=0.5
    counts = torch.tensor([[[2, 0, 0], [1, 1, 1]]])
    out = reduce_f1(counts, num_classes=2, aggregation="dataset_global")
    assert out["f1_class_0"] == pytest.approx(1.0)
    assert out["f1_class_1"] == pytest.approx(0.5)
    assert out["mean_f1"] == pytest.approx(0.75)


def test_reduce_f1_missed_class_scores_zero():
    # class 0 perfect; class 1 entirely missed with GT present (tp=0 fp=0 fn=2) is a
    # real failure -> f1=0, NOT dropped from the macro mean. Likewise an all-FP class.
    counts = torch.tensor([[[2, 0, 0], [0, 0, 2]]])
    out = reduce_f1(counts, num_classes=2, aggregation="dataset_global")
    assert out["f1_class_1"] == pytest.approx(0.0)
    assert out["mean_f1"] == pytest.approx(0.5)
    # class 1 with predictions but no matches and no GT (tp=0 fp=3 fn=0) -> f1=0.
    counts = torch.tensor([[[2, 0, 0], [0, 3, 0]]])
    out = reduce_f1(counts, num_classes=2, aggregation="dataset_global")
    assert out["f1_class_1"] == pytest.approx(0.0)
    assert out["mean_f1"] == pytest.approx(0.5)
    # class 1 absent from both pred and GT (tp=fp=fn=0) stays undefined -> excluded.
    counts = torch.tensor([[[2, 0, 0], [0, 0, 0]]])
    out = reduce_f1(counts, num_classes=2, aggregation="dataset_global")
    assert out["mean_f1"] == pytest.approx(1.0)


def test_reduce_f1_per_image_macro_excludes_undefined():
    # image 0: class 0 perfect (f1=1), class 1 undefined (no pred/gt) -> excluded
    # image 1: class 0 f1=0.5
    counts = torch.tensor([[[1, 0, 0], [0, 0, 0]], [[1, 1, 1], [0, 0, 0]]])
    out = reduce_f1(counts, num_classes=2, aggregation="per_image_macro")
    # image 0 mean over defined = 1.0 ; image 1 mean over defined = 0.5 ; mean = 0.75
    assert out["mean_f1"] == pytest.approx(0.75)


def test_detection_counts_shape_and_axis():
    row = detection_counts(
        np.array([[1.0, 1.0]]), np.array([0]), np.array([0.9]),
        np.array([[1.0, 1.0]]), np.array([0]), num_classes=2, delta=2.0,
    )
    assert row.shape == (1, 2, 3)
    assert row.dtype == torch.long


def test_sweep_thresholds_picks_separating_value():
    # One image: a true peak at high score, a false peak at low score, far from GT.
    pred_xy = [np.array([[5.0, 5.0], [0.0, 0.0]])]
    pred_cls = [np.array([0, 0])]
    pred_score = [np.array([0.9, 0.2])]
    gt_xy = [np.array([[5.0, 5.0]])]
    gt_cls = [np.array([0])]
    thr = sweep_score_thresholds(
        pred_xy, pred_cls, pred_score, gt_xy, gt_cls, num_classes=1, delta=2.0
    )
    # Optimal threshold drops the 0.2 false peak but keeps the 0.9 true peak.
    assert 0.2 < thr[0] <= 0.9


def test_sweep_thresholds_suppresses_all_false_positive_class():
    # A class with GT but whose only predictions are far-away false positives: the best
    # operating point emits nothing, so the chosen threshold must drop every prediction.
    pred_xy = [np.array([[0.0, 0.0], [1.0, 1.0]])]
    pred_cls = [np.array([0, 0])]
    pred_score = [np.array([0.9, 0.8])]
    gt_xy = [np.array([[50.0, 50.0]])]
    gt_cls = [np.array([0])]
    thr = sweep_score_thresholds(
        pred_xy, pred_cls, pred_score, gt_xy, gt_cls, num_classes=1, delta=2.0
    )
    assert thr[0] > 0.9  # above the max score -> keeps nothing (score >= thr is False)


# --------------------------------------------------------------------------- #
# DetectionHead (target encoding, loss, eval round-trip)
# --------------------------------------------------------------------------- #


def _make_head(**kwargs):
    from soma.dense.geometry import compute_dense_geometry
    from soma.tasks.detection import DetectionHead

    geom = compute_dense_geometry(target_size=32, patch_size=4)
    params = dict(num_classes=2, geometry=geom, delta_px=3.0, sigma_px=1.5, score_threshold=0.5)
    params.update(kwargs)
    return DetectionHead(**params)


def test_head_registered():
    from soma.tasks.registry import task_registry

    assert "detection" in task_registry.list()


def test_head_forward_applies_sigmoid_and_crops():
    head = _make_head()
    # decoder output at grid resolution (B, C, h', w'); head -> (B, C, 32, 32) in [0,1]
    out = head.forward(torch.randn(2, 2, 8, 8))
    assert out.shape == (2, 2, 32, 32)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_head_extract_targets_renders_points(tmp_path):
    from soma.dataset import SampleRecord

    pts = tmp_path / "p.csv"
    pts.write_text("x,y,class\n10,12,0\n20,8,1\n")
    record = SampleRecord(sample_id="s", image_path=tmp_path / "i.jpg", label=None, points_path=pts)
    head = _make_head()
    targets = head.extract_targets(record)
    assert targets["heatmap"].shape == (2, 32, 32)
    assert targets["heatmap"][0, 12, 10] == pytest.approx(1.0)
    assert targets["heatmap"][1, 8, 20] == pytest.approx(1.0)
    # gt_points carries both points as (x, y, class)
    assert targets["gt_points"].shape == (2, 3)


def test_head_level0_spacing_defaults_to_run_spacing(tmp_path):
    # Grids extracted at run_spacing=0.2 with no level0_spacing override: points must be
    # treated as already in the grid frame (identity), NOT scaled by 1.0/0.2 = 5 out of
    # the tile. The point (10, 12) should land exactly, giving two in-frame targets.
    from soma.dataset import SampleRecord

    pts = tmp_path / "p.csv"
    pts.write_text("x,y,class\n10,12,0\n20,8,1\n")
    record = SampleRecord(sample_id="s", image_path=tmp_path / "i.jpg", label=None, points_path=pts)
    head = _make_head(run_spacing=0.2)
    assert head.resolve_spacings(record) == (0.2, 0.2)
    targets = head.extract_targets(record)
    assert targets["gt_points"].shape == (2, 3)
    assert targets["heatmap"][0, 12, 10] == pytest.approx(1.0)
    # An explicit per-sample level0_spacing override still wins and rescales.
    record_override = SampleRecord(
        sample_id="s", image_path=tmp_path / "i.jpg", label=None,
        points_path=pts, metadata={"level0_spacing": 0.4},
    )
    assert head.resolve_spacings(record_override) == (0.4, 0.2)


def test_head_eval_roundtrip_perfect_on_gt_heatmap():
    """Feeding the GT heatmap as the prediction yields perfect F1."""
    head = _make_head(score_threshold=0.5, metrics=["mean_f1"])
    xy = np.array([[10.0, 12.0], [20.0, 8.0]])
    cls = np.array([0, 1])
    from soma.detection.encode import render_peak_heatmap

    hm = render_peak_heatmap(xy, cls, target_size=(32, 32), num_classes=2, sigma=1.5)
    raw = hm.unsqueeze(0)  # (1, C, H, W)
    gt = torch.tensor([[[10.0, 12.0, 0.0], [20.0, 8.0, 1.0]]])  # (1, K, 3)
    metrics = head.compute_metrics(raw, {"gt_points": gt})
    assert metrics["mean_f1"] == pytest.approx(1.0)


def test_head_loss_zero_when_prediction_matches_target():
    head = _make_head()
    target = torch.rand(2, 2, 32, 32)
    loss = head.compute_loss(target.clone(), {"heatmap": target})
    assert float(loss) == pytest.approx(0.0)


def test_head_loss_upweights_foreground():
    head = _make_head(foreground_weight=9.0)
    target = torch.zeros(1, 2, 4, 4)
    target[0, 0, 0, 0] = 1.0  # one foreground (peak) pixel
    # Start from an exact match (loss 0), then inject one unit error — on the
    # foreground pixel vs. on a background pixel. The foreground error must cost more.
    pred_fg = target.clone(); pred_fg[0, 0, 0, 0] = 0.0  # unit error at the peak
    pred_bg = target.clone(); pred_bg[0, 1, 3, 3] = 1.0  # unit error at background
    loss_fg = head.compute_loss(pred_fg, {"heatmap": target})
    loss_bg = head.compute_loss(pred_bg, {"heatmap": target})
    assert float(loss_fg) > float(loss_bg)


# --------------------------------------------------------------------------- #
# DetectionManifest
# --------------------------------------------------------------------------- #


def test_detection_manifest_loads_points_path(tmp_path):
    from soma.dataset import DetectionManifest

    (tmp_path / "a.csv").write_text("x,y,class\n1,1,0\n")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "sample_id,image_path,points_path,level0_spacing\n"
        f"s0,img0.jpg,{tmp_path / 'a.csv'},0.25\n"
    )
    manifest = DetectionManifest(csv)
    rec = manifest.samples["s0"]
    assert rec.points_path == tmp_path / "a.csv"
    assert rec.metadata["level0_spacing"] == pytest.approx(0.25)


def test_detection_manifest_retains_tile_origin(tmp_path):
    # source_wsi/tile_x/tile_y are recognized columns; they must survive into metadata
    # for deferred WSI stitching rather than being silently dropped.
    from soma.dataset import DetectionManifest

    (tmp_path / "a.csv").write_text("x,y,class\n1,1,0\n")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "sample_id,image_path,points_path,source_wsi,tile_x,tile_y\n"
        f"s0,img0.jpg,{tmp_path / 'a.csv'},wsi0.tif,512,1024\n"
    )
    rec = DetectionManifest(csv).samples["s0"]
    assert rec.metadata["source_wsi"] == "wsi0.tif"
    assert rec.metadata["tile_x"] == 512
    assert rec.metadata["tile_y"] == 1024


def test_detection_manifest_requires_points_path(tmp_path):
    csv = tmp_path / "manifest.csv"
    csv.write_text("sample_id,image_path\ns0,img0.jpg\n")
    with pytest.raises(ValueError, match="points_path"):
        __import__("soma.dataset", fromlist=["DetectionManifest"]).DetectionManifest(csv)
