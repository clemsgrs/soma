"""Unit tests for the detection-v1 dense-feature machinery (design §4-§7).

Pure pieces (coordinate transform, peak-heatmap encoder, peak extraction, F1@δ
matcher) tested against hand-computed values on tiny explicit inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.detection.encode import (
    render_peak_heatmap,
    transform_points_to_level0,
    transform_points_to_target,
)
from soma.detection.matching import (
    ClassMatch,
    detection_counts,
    match_assignment,
    match_points,
    reduce_f1,
    sweep_score_thresholds,
)
from soma.detection.peaks import extract_peaks


# --------------------------------------------------------------------------- #
# Coordinate transform (level-0 -> target frame)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("effective_spacing_um", "expected_target"),
    [
        (0.2, [[100.0, 250.0]]),
        (0.25, [[80.0, 200.0]]),
        (0.5, [[40.0, 100.0]]),
    ],
)
def test_ocelot_point_transform_round_trips_every_protocol_exactly(
    effective_spacing_um: float, expected_target: list[list[float]]
):
    level0 = np.array([[100.0, 250.0]])
    target = transform_points_to_target(
        level0,
        source_spacing_um=0.2,
        effective_spacing_um=effective_spacing_um,
    )
    np.testing.assert_array_equal(target, np.array(expected_target))
    np.testing.assert_array_equal(
        transform_points_to_level0(
            target,
            source_spacing_um=0.2,
            effective_spacing_um=effective_spacing_um,
        ),
        level0,
    )


def test_transform_applies_crop_offset():
    pts = np.array([[10.0, 20.0]])
    out = transform_points_to_target(
        pts,
        source_spacing_um=0.5,
        effective_spacing_um=0.5,
        crop_top=3,
        crop_left=4,
    )
    np.testing.assert_allclose(out, [[6.0, 17.0]])


def test_transform_empty_passthrough():
    out = transform_points_to_target(
        np.zeros((0, 2)), source_spacing_um=0.5, effective_spacing_um=0.5
    )
    assert out.shape == (0, 2)


@pytest.mark.parametrize("invalid", [0.0, -0.1, float("nan"), float("inf"), True])
def test_transform_rejects_invalid_spacing(invalid):
    with pytest.raises(ValueError, match="positive and finite"):
        transform_points_to_target(
            np.array([[1.0, 1.0]]),
            source_spacing_um=invalid,
            effective_spacing_um=0.5,
        )


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


# --------------------------------------------------------------------------- #
# Matching assignment primitive (pairs -> derived counts)
# --------------------------------------------------------------------------- #


def test_classmatch_derives_counts_from_pairs():
    # tp = #pairs; fp = unmatched preds; fn = unmatched GT — all derived from one place.
    m = ClassMatch(pairs=np.array([[0, 1], [2, 0]]), n_pred=4, n_gt=3)
    assert m.tp == 2
    assert m.fp == 2  # 4 preds, 2 matched
    assert m.fn == 1  # 3 GT, 2 matched
    assert m.counts == (2, 2, 1)


def test_match_assignment_labels_tp_fp_fn_indices():
    # pred 0 lands on gt 0 (TP); pred 1 is far away (FP); gt 1 is unmatched (FN).
    pred = np.array([[1.0, 1.0], [50.0, 50.0]])
    gt = np.array([[1.0, 1.0], [9.0, 9.0]])
    matches = match_assignment(
        pred, np.array([0, 0]), np.array([0.9, 0.8]), gt, np.array([0, 0]),
        num_classes=1, delta=2.0,
    )
    (m,) = matches
    np.testing.assert_array_equal(m.pairs, [[0, 0]])
    assert m.n_pred == 2 and m.n_gt == 2
    assert m.counts == (1, 1, 1)
    # The assignment labels predicted index 0 as TP, index 1 as FP; GT index 1 as FN.
    matched_pred = set(m.pairs[:, 0].tolist())
    matched_gt = set(m.pairs[:, 1].tolist())
    assert matched_pred == {0}  # pred 1 (unmatched) is the FP
    assert matched_gt == {0}    # gt 1 (unmatched) is the FN


def test_match_assignment_class_aware_two_channels():
    # One location fires in BOTH class channels; GT has it only in class 0. The class-0
    # prediction is a TP, the class-1 prediction an FP. Global pred indices disambiguate
    # which channel fired, so the pair points back to the right original prediction.
    pred = np.array([[3.0, 3.0], [3.0, 3.0]])
    pred_cls = np.array([0, 1])
    pred_score = np.array([0.9, 0.8])
    gt = np.array([[3.0, 3.0]])
    gt_cls = np.array([0])
    matches = match_assignment(
        pred, pred_cls, pred_score, gt, gt_cls, num_classes=2, delta=2.0,
    )
    # class 0: global pred index 0 matches global gt index 0.
    np.testing.assert_array_equal(matches[0].pairs, [[0, 0]])
    assert matches[0].counts == (1, 0, 0)
    # class 1: global pred index 1 is an FP with no GT in the channel.
    assert matches[1].pairs.shape == (0, 2)
    assert matches[1].n_pred == 1 and matches[1].n_gt == 0
    assert matches[1].counts == (0, 1, 0)


@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_match_points_counts_equal_assignment_derivation(method):
    # match_points' (C, 3) must be exactly what the assignment derives — no second path.
    pred = np.array([[1.0, 1.0], [1.4, 1.0], [9.0, 9.0]])
    pred_cls = np.array([0, 0, 1])
    pred_score = np.array([0.9, 0.8, 0.7])
    gt = np.array([[1.0, 1.0], [9.0, 9.0]])
    gt_cls = np.array([0, 1])
    counts = match_points(
        pred, pred_cls, pred_score, gt, gt_cls,
        num_classes=2, delta=2.0, method=method,
    )
    matches = match_assignment(
        pred, pred_cls, pred_score, gt, gt_cls,
        num_classes=2, delta=2.0, method=method,
    )
    derived = np.array([list(m.counts) for m in matches], dtype=np.int64)
    np.testing.assert_array_equal(counts, derived)
    # class 0: two preds near one GT -> 1 TP, 1 FP, 0 FN; class 1: clean hit.
    np.testing.assert_array_equal(counts, [[1, 1, 0], [1, 0, 0]])


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


def test_sweep_thresholds_survives_junk_dominated_candidate_pool():
    # Threshold-0 decoding floods the pool with near-zero junk peaks (MIDOG regression):
    # with quantile-only candidates every quantile lands in the junk mass, the mid-range
    # where F1 peaks is never evaluated, and the sweep freezes a near-zero threshold.
    # 40 images, one GT + one well-scored true peak each, plus 500 junk peaks per image
    # far from GT with scores <= 0.01. True-peak scores spread over [0.5, 0.9]; the
    # junk-range optimum (keep everything) and the max-score bracket (keep one peak)
    # are both far worse than any mid-range cut.
    rng_scores = np.linspace(0.5, 0.9, 40)
    pred_xy, pred_cls, pred_score, gt_xy, gt_cls = [], [], [], [], []
    for k, true_score in enumerate(rng_scores):
        junk_xy = np.stack(
            [1000.0 + 5.0 * np.arange(500), np.full(500, 1000.0 + k)], axis=1
        )
        junk_scores = np.linspace(1e-4, 0.01, 500)
        pred_xy.append(np.concatenate([[[5.0, 5.0]], junk_xy]))
        pred_cls.append(np.zeros(501, dtype=np.int64))
        pred_score.append(np.concatenate([[true_score], junk_scores]))
        gt_xy.append(np.array([[5.0, 5.0]]))
        gt_cls.append(np.array([0]))
    thr = sweep_score_thresholds(
        pred_xy, pred_cls, pred_score, gt_xy, gt_cls, num_classes=1, delta=2.0
    )
    # Any threshold in (0.01, 0.5] keeps all 40 true peaks and no junk -> F1 = 1.0.
    assert 0.01 < thr[0] <= 0.5


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
# DetectionHead (raw level-0 targets encoded at loss time, loss, eval, predictions)
# --------------------------------------------------------------------------- #


def _make_head(*, scale: float = 1.0, **kwargs):
    from soma.data import GridGeometry
    from soma.tasks.detection import DetectionHead

    geometry = GridGeometry.from_sizes(target_size=32, patch_size=4, level0_px_per_token_px=scale)
    params = dict(num_classes=2, geometry=geometry, spacing_um=1.0, match_distance=3.0, sigma=1.5)
    params.update(kwargs)
    return DetectionHead(**params)


def _points(rows) -> torch.Tensor:
    """One image's level-0 ``(x, y, class)`` rows as a ``(1, K, 3)`` batch."""
    return torch.tensor([rows], dtype=torch.float32).reshape(1, -1, 3)


def test_head_registered():
    from soma.tasks.registry import task_registry

    assert "detection" in task_registry.list()


def test_head_forward_applies_sigmoid_and_crops():
    head = _make_head()
    out = head.forward(torch.randn(2, 2, 8, 8))
    assert out.shape == (2, 2, 32, 32)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_head_resolves_its_um_distances_at_the_frame_spacing():
    head = _make_head(spacing_um=0.5, match_distance=3.0, sigma=None, nms_distance=None, scale=2.0)
    assert head.delta_px == 6.0 and head.sigma_px == 2.0 and head.nms_distance_px == 6.0
    # The predictions' tolerance is in level-0 pixels: two per frame pixel here.
    assert head.match_distance_px == 12.0


def test_head_renders_the_reference_heatmap_on_device():
    """The loss-time renderer equals the CPU reference renderer (peak 1, max-merged)."""
    head = _make_head()
    rows = [[10.0, 12.0, 0.0], [20.0, 8.0, 1.0], [11.0, 13.0, 0.0]]
    heatmap, valid, frames = head.encode_targets({"points": _points(rows)})
    expected = render_peak_heatmap(
        np.array([r[:2] for r in rows]), np.array([int(r[2]) for r in rows]),
        target_size=(32, 32), num_classes=2, sigma=1.5,
    )
    assert heatmap.shape == (1, 2, 32, 32)
    torch.testing.assert_close(heatmap[0], expected)
    assert bool(valid.all())
    assert frames[0][0].tolist() == [[10.0, 12.0], [20.0, 8.0], [11.0, 13.0]]


def test_head_maps_level0_points_into_its_frame():
    # Two level-0 pixels per frame pixel: level-0 (20, 10) is frame (10, 5).
    head = _make_head(scale=2.0)
    heatmap, _, frames = head.encode_targets({"points": _points([[20.0, 10.0, 0.0]])})
    assert frames[0][0].tolist() == [[10.0, 5.0]]
    assert heatmap[0, 0, 5, 10] == pytest.approx(1.0)


def test_head_drops_nan_padding_and_points_outside_the_frame():
    head = _make_head()
    points = torch.tensor(
        [[[10.0, 12.0, 0.0], [40.0, 3.0, 1.0], [float("nan")] * 3]], dtype=torch.float32
    )
    _, _, frames = head.encode_targets({"points": points})
    assert frames[0][0].tolist() == [[10.0, 12.0]]


def test_head_refuses_a_class_outside_its_classes():
    with pytest.raises(ValueError, match=r"class id\(s\) \[5\]"):
        _make_head().encode_targets({"points": _points([[10.0, 12.0, 5.0]])})


def test_head_eval_roundtrip_perfect_on_gt_heatmap():
    head = _make_head(validation_threshold=0.5, metrics=["mean_f1"])
    rows = [[10.0, 12.0, 0.0], [20.0, 8.0, 1.0]]
    heatmap, _, _ = head.encode_targets({"points": _points(rows)})
    metrics = head.compute_metrics(heatmap, {"points": _points(rows)})
    assert metrics["mean_f1"] == pytest.approx(1.0)


def test_head_loss_zero_when_prediction_matches_target():
    head = _make_head()
    targets = {"points": _points([[10.0, 12.0, 0.0], [20.0, 8.0, 1.0]])}
    heatmap, _, _ = head.encode_targets(targets)
    assert float(head.compute_loss(heatmap.clone(), targets)) == pytest.approx(0.0)


def test_head_loss_upweights_foreground():
    head = _make_head(foreground_weight=9.0)
    targets = {"points": _points([[10.0, 12.0, 0.0]])}
    heatmap, _, _ = head.encode_targets(targets)
    pred_fg = heatmap.clone()
    pred_fg[0, 0, 12, 10] = 0.0  # unit error at the peak
    pred_bg = heatmap.clone()
    pred_bg[0, 1, 30, 30] = 1.0  # unit error on background
    assert float(head.compute_loss(pred_fg, targets)) > float(head.compute_loss(pred_bg, targets))


def test_head_postprocess_emits_level0_peaks_above_min_score():
    head = _make_head(scale=2.0, min_score=0.3)
    heatmap = render_peak_heatmap(
        np.array([[10.0, 5.0], [20.0, 20.0]]), np.array([0, 1]),
        target_size=(32, 32), num_classes=2, sigma=1.5,
    ).unsqueeze(0)
    heatmap[0, 1] *= 0.2  # the class-1 peak scores 0.2, under the floor
    (rows,) = head.postprocess(heatmap)["points"]
    assert rows.tolist() == [[20.0, 10.0, 0.0, 1.0]]  # level-0 = 2 x frame


# --------------------------------------------------------------------------- #
# DetectionManifest
# --------------------------------------------------------------------------- #


def test_detection_manifest_loads_points_path(tmp_path):
    from soma.data._legacy import legacy_samples_from_csv

    (tmp_path / "a.csv").write_text("x,y,class\n1,1,0\n")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "sample_id,image_path,points_path,spacing_at_level_0\n"
        f"s0,img0.jpg,{tmp_path / 'a.csv'},0.25\n"
    )
    manifest = legacy_samples_from_csv(csv)
    rec = manifest.samples["s0"]
    assert rec.points_path == tmp_path / "a.csv"
    assert rec.spacing_at_level_0 == pytest.approx(0.25)


def test_detection_manifest_retains_tile_origin(tmp_path):
    # source_wsi/tile_x/tile_y are recognized columns; they must survive into metadata
    # for deferred WSI stitching rather than being silently dropped.
    from soma.data._legacy import legacy_samples_from_csv

    (tmp_path / "a.csv").write_text("x,y,class\n1,1,0\n")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "sample_id,image_path,points_path,source_wsi,tile_x,tile_y\n"
        f"s0,img0.jpg,{tmp_path / 'a.csv'},wsi0.tif,512,1024\n"
    )
    rec = legacy_samples_from_csv(csv).samples["s0"]
    assert rec.metadata["source_wsi"] == "wsi0.tif"
    assert rec.metadata["tile_x"] == 512
    assert rec.metadata["tile_y"] == 1024


def test_detection_manifest_reads_ignore_mask_path_as_typed_column(tmp_path):
    from soma.data._legacy import legacy_samples_from_csv

    (tmp_path / "a.csv").write_text("x,y,class\n1,1,0\n")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "sample_id,image_path,points_path,ignore_mask_path\n"
        f"s0,img0.jpg,{tmp_path / 'a.csv'},{tmp_path / 'm0.png'}\n"
        f"s1,img1.jpg,{tmp_path / 'a.csv'},\n"
    )
    samples = legacy_samples_from_csv(csv).samples
    assert samples["s0"].ignore_mask_path == tmp_path / "m0.png"
    assert samples["s1"].ignore_mask_path is None
    assert "ignore_mask_path" not in samples["s0"].metadata


# --------------------------------------------------------------------------- #
# PointSource class scheme and ignore masks (255 = not annotated, 0 = supervised)
# --------------------------------------------------------------------------- #


def _point_source(tmp_path, rows: str, *, mask=None, **kwargs):
    from soma.data import PointEntry, PointSource

    pts = tmp_path / "p.csv"
    pts.write_text("x,y,class\n" + rows)
    return PointSource([PointEntry(sample_id="s", points_path=pts, ignore_mask_path=mask)], **kwargs)


def _ignore_mask(tmp_path, ignored_cols: slice, *, size=(32, 32), value=255, name="m.png"):
    from PIL import Image

    mask = np.zeros(size, dtype=np.uint8)
    mask[:, ignored_cols] = value
    path = tmp_path / name
    Image.fromarray(mask).save(path)
    return path


def test_point_source_class_remap_merges_ids_into_one_class(tmp_path):
    # classes: {mnl: [1, 2], other: [7]} -> ids 1 and 2 share class 0, id 7 is class 1.
    source = _point_source(tmp_path, "10,12,1\n20,8,2\n5,25,7\n", class_remap={1: 0, 2: 0, 7: 1})
    points = source.load("s")
    assert points.classes.tolist() == [0, 0, 1]
    assert points.xy.tolist() == [[10.0, 12.0], [20.0, 8.0], [5.0, 25.0]]


def test_point_source_dropped_point_becomes_negative_supervision(tmp_path):
    source = _point_source(tmp_path, "10,12,0\n20,8,2\n", class_remap={0: 0}, drop=[2])
    points = source.load("s")
    assert points.xy.tolist() == [[10.0, 12.0]]
    # A prediction on the kept point and one on the dropped point: 1 TP, 1 FP, 0 FN.
    head = _make_head(num_classes=1)
    batch = {"points": torch.cat([points.xy, points.classes[:, None].float()], dim=1)[None].float()}
    predicted = render_peak_heatmap(
        np.array([[10.0, 12.0], [20.0, 8.0]]), np.array([0, 0]),
        target_size=(32, 32), num_classes=1, sigma=1.5,
    )
    assert head.dense_stats(predicted.unsqueeze(0), batch).tolist() == [[[1, 1, 0]]]


def test_point_source_rejects_a_point_id_declared_nowhere(tmp_path):
    source = _point_source(tmp_path, "10,12,0\n20,8,3\n", class_remap={0: 0}, drop=[2])
    with pytest.raises(ValueError, match=r"points for 's' have class id\(s\) \[3\] declared in neither"):
        source.load("s")


def test_point_source_rejects_ignore_mask_value_naming_file_and_sample(tmp_path):
    source = _point_source(
        tmp_path, "10,12,0\n", mask=_ignore_mask(tmp_path, slice(16, None), value=1, name="bad_mask.png")
    )
    with pytest.raises(ValueError, match=r"(?s)sample 's'.*bad_mask\.png"):
        source.load("s")


def test_ignore_mask_follows_the_target_frame():
    from soma.data import GridGeometry
    from soma.learn.grid import ignore_to_frame

    # Level-0 is twice the frame resolution: level-0 columns 32.. map to frame 16..
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, 32:] = 255
    geometry = GridGeometry.from_sizes(target_size=32, patch_size=4, level0_px_per_token_px=2.0)
    ignored = ignore_to_frame(mask, geometry)
    assert ignored.shape == (32, 32)
    assert not bool(ignored[:, :16].any()) and bool(ignored[:, 16:].all())


def test_head_drops_points_on_ignored_pixels_from_heatmap_and_ground_truth():
    head = _make_head()
    ignore = torch.zeros(1, 32, 32, dtype=torch.bool)
    ignore[:, :, 16:] = True
    heatmap, valid, frames = head.encode_targets(
        {"points": _points([[10.0, 12.0, 0.0], [20.0, 8.0, 1.0]]), "ignore": ignore}
    )
    assert bool(valid[0, :, :16].all()) and not bool(valid[0, :, 16:].any())
    assert frames[0][0].tolist() == [[10.0, 12.0]]
    assert float(heatmap[0, 1].max()) == 0.0 and heatmap[0, 0, 12, 10] == pytest.approx(1.0)


def test_head_loss_ignores_predictions_on_ignored_pixels():
    head = _make_head()
    targets = {"points": _points([[10.0, 12.0, 0.0]]), "ignore": torch.zeros(1, 32, 32, dtype=torch.bool)}
    targets["ignore"][:, :, 16:] = True
    pred = torch.rand(1, 2, 32, 32)
    changed = pred.clone()
    changed[..., 16:] = torch.rand(1, 2, 32, 16)
    loss = head.compute_loss(pred, targets)
    assert float(loss) > 0.0
    assert float(head.compute_loss(changed, targets)) == float(loss)


def test_head_loss_averages_over_valid_pixels_of_every_channel():
    head = _make_head(foreground_weight=0.0)
    ignore = torch.ones(1, 32, 32, dtype=torch.bool)
    ignore[0, 0, 0] = False  # one supervised pixel, two channels
    pred = torch.zeros(1, 2, 32, 32)
    pred[0, 0, 0, 0] = 1.0  # squared error 1 on one of the two supervised entries
    targets = {"points": torch.full((1, 0, 3), float("nan")), "ignore": ignore}
    assert float(head.compute_loss(pred, targets)) == 0.5


def test_head_dense_stats_drops_predictions_on_ignored_pixels():
    head = _make_head(num_classes=1)
    ignore = torch.zeros(1, 32, 32, dtype=torch.bool)
    ignore[:, :, 16:] = True
    # One prediction on the supervised point, one on the ignored point, one on an ignored
    # pixel with no point at all: only the first is scored (1 TP, 0 FP, 0 FN).
    predicted = render_peak_heatmap(
        np.array([[10.0, 12.0], [20.0, 8.0], [28.0, 28.0]]), np.array([0, 0, 0]),
        target_size=(32, 32), num_classes=1, sigma=1.5,
    )
    targets = {"points": _points([[10.0, 12.0, 0.0], [20.0, 8.0, 0.0]]), "ignore": ignore}
    assert head.dense_stats(predicted.unsqueeze(0), targets).tolist() == [[[1, 0, 0]]]
