"""Unit tests for the detection-v1 dense-feature machinery (design §4-§7).

Pure pieces (coordinate transform, peak-heatmap encoder, peak extraction, F1@δ
matcher) tested against hand-computed values on tiny explicit inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from soma.models.detection_targets import (
    render_peak_heatmap,
    transform_points_to_level0,
    transform_points_to_target,
)
from soma.evaluation.detection.matching import (
    ClassMatch,
    detection_counts,
    match_assignment,
    match_points,
    reduce_f1,
    sweep_score_thresholds,
)
from soma.evaluation.detection.peaks import extract_peaks


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


def test_extract_peaks_nms_on_dense_plateau_keeps_a_maximal_spread_set():
    # One dip in a flat 0.5 map makes every plateau pixel whose window holds the dip a
    # tied local maximum: 61*61 - 1 candidates, almost all within min_distance of each
    # other. Greedy NMS must keep a set that is spread out (no two kept peaks closer than
    # min_distance) and maximal (every candidate lies closer than that to a kept peak).
    hm = torch.full((1, 96, 96), 0.5)
    hm[0, 48, 48] = 0.49
    xy, _, scores = extract_peaks(hm, min_distance=30, score_threshold=0.0)
    assert xy.shape[0] >= 1 and np.all(scores == 0.5)
    pair_d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    assert np.all(pair_d2[~np.eye(xy.shape[0], dtype=bool)] >= 30**2)
    gy, gx = np.mgrid[18:79, 18:79]
    cand = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    cand = cand[~((cand[:, 0] == 48) & (cand[:, 1] == 48))]
    cand_d2 = ((cand[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    assert np.all(cand_d2.min(axis=1) < 30**2)


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


@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_sweep_thresholds_picks_separating_value(method):
    # One image: a true peak at high score, a false peak at low score, far from GT.
    pred_xy = [np.array([[5.0, 5.0], [0.0, 0.0]])]
    pred_cls = [np.array([0, 0])]
    pred_score = [np.array([0.9, 0.2])]
    gt_xy = [np.array([[5.0, 5.0]])]
    gt_cls = [np.array([0])]
    thr = sweep_score_thresholds(
        pred_xy, pred_cls, pred_score, gt_xy, gt_cls, num_classes=1, delta=2.0, method=method
    )
    # Optimal threshold drops the 0.2 false peak but keeps the 0.9 true peak.
    assert 0.2 < thr[0] <= 0.9


@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_sweep_thresholds_survives_junk_dominated_candidate_pool(method):
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
        pred_xy, pred_cls, pred_score, gt_xy, gt_cls, num_classes=1, delta=2.0, method=method
    )
    # Any threshold in (0.01, 0.5] keeps all 40 true peaks and no junk -> F1 = 1.0.
    assert 0.01 < thr[0] <= 0.5


@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_sweep_thresholds_suppresses_all_false_positive_class(method):
    # A class with GT but whose only predictions are far-away false positives: the best
    # operating point emits nothing, so the chosen threshold must drop every prediction.
    pred_xy = [np.array([[0.0, 0.0], [1.0, 1.0]])]
    pred_cls = [np.array([0, 0])]
    pred_score = [np.array([0.9, 0.8])]
    gt_xy = [np.array([[50.0, 50.0]])]
    gt_cls = [np.array([0])]
    thr = sweep_score_thresholds(
        pred_xy, pred_cls, pred_score, gt_xy, gt_cls, num_classes=1, delta=2.0, method=method
    )
    assert thr[0] > 0.9  # above the max score -> keeps nothing (score >= thr is False)


@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_sweep_thresholds_keep_every_prediction_of_a_class_never_predicted(method):
    # Class 1 has annotated points but no prediction on tune: nothing to tune, threshold 0.
    thr = sweep_score_thresholds(
        [np.array([[5.0, 5.0]])], [np.array([0])], [np.array([0.9])],
        [np.array([[5.0, 5.0], [20.0, 20.0]])], [np.array([0, 1])],
        num_classes=2, delta=2.0, method=method,
    )
    assert thr[1] == 0.0


# --------------------------------------------------------------------------- #
# DetectionHead (raw level-0 targets encoded at loss time, loss, eval, predictions)
# --------------------------------------------------------------------------- #


def _make_head(*, scale: float = 1.0, **kwargs):
    from soma.data import GridGeometry
    from soma.models.heads.detection import DetectionHead

    geometry = GridGeometry.from_sizes(target_size=32, patch_size=4, level0_px_per_token_px=scale)
    params = dict(num_classes=2, geometry=geometry, spacing_um=1.0, match_distance=3.0, sigma=1.5)
    params.update(kwargs)
    return DetectionHead(**params)


def _points(rows) -> torch.Tensor:
    """One image's level-0 ``(x, y, class)`` rows as a ``(1, K, 3)`` batch."""
    return torch.tensor([rows], dtype=torch.float32).reshape(1, -1, 3)


def test_head_registered():
    from soma.models.heads.registry import task_registry

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
    head = _make_head(fixed_threshold=0.5, metrics=["mean_f1"])
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
    from soma.data import AnnotationManifest, ImageManifest

    (tmp_path / "a.csv").write_text("x,y,class\n1,1,0\n")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "sample_id,image_path,points_path,spacing_at_level_0\n"
        f"s0,img0.jpg,{tmp_path / 'a.csv'},0.25\n"
    )
    assert AnnotationManifest.from_csv(csv)["s0"].points_path == tmp_path / "a.csv"
    assert ImageManifest.from_csv(csv)["s0"].spacing_at_level_0 == pytest.approx(0.25)


def test_detection_manifest_retains_tile_origin(tmp_path):
    # source_wsi/tile_x/tile_y are recognized columns; they must survive into metadata
    # for deferred WSI stitching rather than being silently dropped.
    import pandas as pd

    from soma.data import Cohort

    (tmp_path / "a.csv").write_text("x,y,class\n1,1,0\n")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "sample_id,image_path,points_path,source_wsi,tile_x,tile_y\n"
        f"s0,img0.jpg,{tmp_path / 'a.csv'},wsi0.tif,512,1024\n"
    )
    rec = Cohort.from_frames(pd.read_csv(csv).assign(split="test")).record("s0")
    assert rec.metadata["source_wsi"] == "wsi0.tif"
    assert rec.metadata["tile_x"] == 512
    assert rec.metadata["tile_y"] == 1024


def test_detection_manifest_reads_ignore_mask_path_as_typed_column(tmp_path):
    import pandas as pd

    from soma.data import AnnotationManifest, Cohort

    (tmp_path / "a.csv").write_text("x,y,class\n1,1,0\n")
    csv = tmp_path / "manifest.csv"
    csv.write_text(
        "sample_id,image_path,points_path,ignore_mask_path\n"
        f"s0,img0.jpg,{tmp_path / 'a.csv'},{tmp_path / 'm0.png'}\n"
        f"s1,img1.jpg,{tmp_path / 'a.csv'},\n"
    )
    annotations = AnnotationManifest.from_csv(csv)
    assert annotations["s0"].ignore_mask_path == tmp_path / "m0.png"
    assert annotations["s1"].ignore_mask_path is None
    cohort = Cohort.from_frames(pd.read_csv(csv).assign(split="test"))
    assert "ignore_mask_path" not in cohort.record("s0").metadata


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
    # A prediction on the kept point and one on the dropped point, tied in score: the best
    # operating point keeps both, 1 TP and 1 FP.
    head = _make_head(num_classes=1, metrics=["precision", "recall"])
    batch = {"points": torch.cat([points.xy, points.classes[:, None].float()], dim=1)[None].float()}
    predicted = render_peak_heatmap(
        np.array([[10.0, 12.0], [20.0, 8.0]]), np.array([0, 0]),
        target_size=(32, 32), num_classes=1, sigma=1.5,
    )
    metrics = head.compute_metrics(predicted.unsqueeze(0), batch)
    assert (metrics["precision_class_0"], metrics["recall_class_0"]) == (0.5, 1.0)


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


def test_grid_dataset_drops_points_on_ignored_level0_pixels_before_framing():
    from soma.data import GridGeometry, Points
    from soma.learn.grid import GridDataset

    # Two level-0 px per frame px: frame pixel (0, 0) reads level-0 (0, 0), which is
    # supervised, so the framed ignore map alone would keep a point on the ignored
    # level-0 pixel (1, 1). ``evaluate`` drops it against the level-0 raster; so must training.
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[1, 1] = 255
    geometry = GridGeometry.from_sizes(target_size=32, patch_size=4, level0_px_per_token_px=2.0)

    class _Grids:
        def load(self, sample_id):
            return torch.zeros(1, 8, 8)

    class _Targets:
        def load(self, sample_id):
            return Points(
                xy=torch.tensor([[1.0, 1.0], [20.0, 20.0]]),
                classes=torch.tensor([0, 1]),
                ignore_mask=mask,
            )

    dataset = GridDataset(
        ["s"], _Grids(), _Targets(), inputs="features", geometry=geometry, target_kind="points"
    )
    _, _, encoded, _ = dataset[0]
    assert encoded["points"].tolist() == [[20.0, 20.0, 1.0]]
    assert not bool(encoded["ignore"][0, 0])
    heatmap, _, _ = _make_head(scale=2.0).encode_targets(
        {"points": encoded["points"][None], "ignore": encoded["ignore"][None]}
    )
    assert float(heatmap[0, 0].max()) == 0.0


def test_grid_dataset_augments_points_in_the_pixel_frame_and_returns_them_at_level0():
    from soma.data import GridGeometry, Points
    from soma.learn import SegmentationAugment
    from soma.learn.grid import GridDataset

    # The pixels start at level-0 (100, 40), two level-0 px per pixel. Level-0 (130, 50)
    # is pixel (15, 5); a horizontal flip of the 32-px frame puts it at pixel (16, 5),
    # level-0 (132, 50). Level-0 columns 100..109 are ignored: pixels 0..4, flipped to
    # 27..31; the point at level-0 (102, 60) sits on them and is dropped.
    geometry = GridGeometry.from_sizes(
        target_size=32, patch_size=1, origin_level0=(100.0, 40.0), level0_px_per_token_px=2.0
    )
    mask = np.zeros((128, 200), dtype=np.uint8)
    mask[:, 100:110] = 255

    class _Images:
        def load(self, sample_id):
            return torch.zeros(3, 32, 32, dtype=torch.uint8)

    class _Targets:
        def load(self, sample_id):
            return Points(
                xy=torch.tensor([[130.0, 50.0], [102.0, 60.0]]),
                classes=torch.tensor([1, 0]),
                ignore_mask=mask,
            )

    dataset = GridDataset(
        ["s"],
        _Images(),
        _Targets(),
        inputs="pixels",
        geometry=geometry,
        target_kind="points",
        augment=SegmentationAugment(horizontal_flip=1.0),
    )
    _, pixels, encoded, _ = dataset[0]
    assert tuple(pixels.shape) == (3, 32, 32)
    assert encoded["points"].tolist() == [[132.0, 50.0, 1.0]]
    assert encoded["ignore"].shape == (32, 32)
    assert bool(encoded["ignore"][:, 27:].all()) and not bool(encoded["ignore"][:, :27].any())


def test_grid_dataset_frames_the_ignore_raster_onto_an_image_larger_than_the_target():
    from soma.data import GridGeometry, Points
    from soma.learn import SegmentationAugment
    from soma.learn.grid import GridDataset

    # A 64x64 image under a 32-px head: the crop resizes it to the target size, so the
    # ignore raster must start on the 64x64 pixels, not on the 32x32 target frame.
    geometry = GridGeometry.from_sizes(target_size=32, patch_size=1, level0_px_per_token_px=1.0)
    mask = np.zeros((64, 64), dtype=np.uint8)

    class _Images:
        def load(self, sample_id):
            return torch.zeros(3, 64, 64, dtype=torch.uint8)

    class _Targets:
        def load(self, sample_id):
            return Points(xy=torch.tensor([[40.0, 40.0]]), classes=torch.tensor([0]), ignore_mask=mask)

    def dataset(augment):
        return GridDataset(
            ["s"], _Images(), _Targets(), inputs="pixels", geometry=geometry,
            target_kind="points", augment=augment,
        )

    _, pixels, encoded, _ = dataset(SegmentationAugment(crop_scale=(0.5, 1.0), target_size=32))[0]
    assert tuple(pixels.shape) == (3, 32, 32)
    assert encoded["ignore"].shape == (32, 32)
    _, _, plain, _ = dataset(None)[0]
    assert plain["ignore"].shape == (32, 32)


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


def test_head_validation_drops_predictions_on_ignored_pixels():
    head = _make_head(num_classes=1, metrics=["precision", "recall"])
    ignore = torch.zeros(1, 32, 32, dtype=torch.bool)
    ignore[:, :, 16:] = True
    # One prediction on the supervised point, one on the ignored point, one on an ignored
    # pixel with no point at all, all three tied in score: only the first is scored (1 TP,
    # 0 FP, 0 FN).
    predicted = render_peak_heatmap(
        np.array([[10.0, 12.0], [20.0, 8.0], [28.0, 28.0]]), np.array([0, 0, 0]),
        target_size=(32, 32), num_classes=1, sigma=1.5,
    )
    targets = {"points": _points([[10.0, 12.0, 0.0], [20.0, 8.0, 0.0]]), "ignore": ignore}
    metrics = head.compute_metrics(predicted.unsqueeze(0), targets)
    assert (metrics["precision_class_0"], metrics["recall_class_0"]) == (1.0, 1.0)


# --------------------------------------------------------------------------- #
# Training-time validation: the one-pass greedy sweep
# --------------------------------------------------------------------------- #


def _bumps(head, peaks) -> torch.Tensor:
    """A ``(C, H, W)`` heatmap with a Gaussian bump of height ``h`` at each ``(x, y, c, h)``."""
    height, width = head.geometry.target_size
    heat = torch.zeros(head.num_classes, height, width)
    rows = torch.tensor(peaks, dtype=torch.float32)
    for h in torch.unique(rows[:, 3]):
        level = rows[rows[:, 3] == h]
        heat = torch.maximum(heat, head.render_heatmap(level[:, :2], level[:, 2].long()) * h)
    return heat


def _sweep_scene(head, *, num_images: int, seed: int):
    """Synthetic multi-class images: jittered predictions around most annotated points,
    false bumps, scores drawn from a few values (so many tie), order-sensitive pairs (two
    tied predictions between two annotated points: A-then-B matches both, B-then-A one),
    and ignore regions."""
    rng = np.random.default_rng(seed)
    size = head.geometry.target_size[0]
    levels = [0.2, 0.4, 0.4, 0.6, 0.6, 0.7, 1.0]  # 0.7 is just below the 0.7 edge in float32
    cells = [(12 * i + 6.0, 12 * j + 6.0) for i in range(size // 12) for j in range(size // 12)]
    heatmaps, points, ignores = [], [], []
    for n in range(num_images):
        peaks, gt = [], []
        for c in range(head.num_classes):
            order = rng.permutation(len(cells))
            pairs, jittered, false = order[:12], order[12:72], order[72:96]
            for k in pairs:
                ax, ay = cells[k]
                gt += [(ax, ay, c), (ax + 4.5, ay, c)]
                tie = float(rng.choice(levels[1:3]))
                peaks += [(ax - 2.0, ay, c, tie), (ax + 2.0, ay, c, tie)]
            for k in jittered:
                x, y = np.asarray(cells[k]) + rng.uniform(-1.0, 1.0, 2)
                gt.append((x, y, c))
                if rng.random() < 0.75:  # a prediction, sometimes beyond delta
                    # On a pixel centre, so its peak scores exactly its level.
                    px, py = np.rint([x, y] + rng.uniform(-2.5, 2.5, 2))
                    peaks.append((px, py, c, float(rng.choice(levels))))
            for k in false:  # false bumps far from every annotated point of c
                x, y = cells[k]
                peaks.append((x, y, c, float(rng.choice(levels))))
        heatmaps.append(_bumps(head, peaks))
        points.append(torch.tensor(gt, dtype=torch.float32))
        ignore = torch.zeros(size, size, dtype=torch.bool)
        if n % 2 == 0:
            ignore[:, size - 20 :] = True  # a band holding points and predictions
        ignores.append(ignore)
    padded = torch.full((num_images, max(p.shape[0] for p in points), 3), float("nan"))
    for n, p in enumerate(points):
        padded[n, : p.shape[0]] = p
    return torch.stack(heatmaps), {"points": padded, "ignore": torch.stack(ignores)}


def _brute_force_greedy(head, predictions, targets, threshold: float) -> np.ndarray:
    """Pooled ``(C, 3)`` TP/FP/FN of greedy matching on the predictions scoring at least
    ``threshold``, with predictions and annotated points on ignored pixels left out."""
    total = np.zeros((head.num_classes, 3), dtype=np.int64)
    for n, rows in enumerate(predictions):
        ignore = targets["ignore"][n].numpy()
        rows = rows[rows[:, 3] >= threshold]
        on_ignored = ignore[np.rint(rows[:, 1]).astype(int), np.rint(rows[:, 0]).astype(int)]
        rows = rows[~on_ignored]
        gt = targets["points"][n].numpy()
        gt = gt[~np.isnan(gt).any(axis=1)]
        gt = gt[~ignore[np.rint(gt[:, 1]).astype(int), np.rint(gt[:, 0]).astype(int)]]
        total += match_points(
            rows[:, :2], rows[:, 2], rows[:, 3], gt[:, :2], gt[:, 2],
            num_classes=head.num_classes, delta=head.delta_px, method="greedy",
        )
    return total


def test_one_pass_sweep_matches_brute_force_greedy_at_every_threshold():
    from soma.data import GridGeometry
    from soma.evaluation.detection.matching import counts_at_thresholds, histogram_thresholds

    head = _make_head(
        num_classes=3, sigma=1.0, geometry=GridGeometry.from_sizes(target_size=144, patch_size=4)
    )
    heatmaps, targets = _sweep_scene(head, num_images=2, seed=0)
    swept = counts_at_thresholds(head.dense_stats(heatmaps, targets).sum(dim=0).numpy())

    thresholds = histogram_thresholds()
    predictions = head.postprocess(heatmaps)["points"]
    scores = np.concatenate([rows[:, 3] for rows in predictions])
    assert len(scores) - len(np.unique(scores)) > 100  # many ties in score
    assert swept.shape == (3, len(thresholds), 3)
    brute: dict[int, np.ndarray] = {}  # keyed by the kept set, which only its size identifies
    for k, threshold in enumerate(thresholds):
        kept = int((scores >= threshold).sum())
        if kept not in brute:
            brute[kept] = _brute_force_greedy(head, predictions, targets, threshold)
        np.testing.assert_array_equal(swept[:, k], brute[kept], err_msg=f"t={threshold}")
    assert len(brute) == len(np.unique(scores)) + 1  # every distinct operating point checked


def test_validation_metrics_are_the_best_f1_per_class_over_every_threshold():
    from soma.data import GridGeometry

    head = _make_head(
        num_classes=3,
        sigma=1.0,
        geometry=GridGeometry.from_sizes(target_size=144, patch_size=4),
        metrics=["mean_f1", "f1_per_class", "precision", "recall"],
    )
    heatmaps, targets = _sweep_scene(head, num_images=2, seed=1)
    metrics = head.compute_metrics(heatmaps, targets)

    predictions = head.postprocess(heatmaps)["points"]
    scores = np.concatenate([rows[:, 3] for rows in predictions])
    candidates = [*np.unique(scores), np.inf]  # every distinct operating point, and none kept

    def f1(counts):
        tp, fp, fn = counts
        return 2 * tp / (2 * tp + fp + fn)

    best = [max(f1(_brute_force_greedy(head, predictions, targets, t)[c]) for t in candidates) for c in range(3)]
    assert metrics["mean_f1"] == pytest.approx(np.mean(best))
    for c in range(3):
        tp, fp, fn = _brute_force_greedy(head, predictions, targets, metrics[f"threshold_class_{c}"])[c]
        assert metrics[f"f1_class_{c}"] == pytest.approx(best[c])
        assert metrics[f"f1_class_{c}"] == pytest.approx(f1((tp, fp, fn)))
        assert metrics[f"precision_class_{c}"] == pytest.approx(tp / (tp + fp))
        assert metrics[f"recall_class_{c}"] == pytest.approx(tp / (tp + fn))
