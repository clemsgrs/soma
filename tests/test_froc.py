"""Unit tests for the FROC scorer (Monkey's native detection metric).

FROC (Free-response ROC) plots sensitivity against the mean number of false
positives per mm² as a detection-score threshold is swept. The pieces are pure numpy
so they test against hand-built point sets with hand-computed sensitivity / FP-per-mm²
at known operating points.
"""

from __future__ import annotations

import numpy as np
import pytest

from soma.detection.froc import (
    DEFAULT_FP_THRESHOLDS,
    MNL_CLASS_NAME,
    MONKEY_CLASS_NAMES,
    MONKEY_MATCH_UM,
    FrocResult,
    compute_froc,
    froc_curve,
    froc_score_at_thresholds,
    label_detections,
    score_monkey_froc,
)


# --------------------------------------------------------------------------- #
# label_detections — greedy-by-confidence TP/FP labelling (single class)
# --------------------------------------------------------------------------- #


def test_label_detections_perfect_match_all_true_positive():
    pred = np.array([[1.0, 1.0], [5.0, 5.0]])
    gt = np.array([[1.0, 1.0], [5.0, 5.0]])
    tp, fp = label_detections(pred, np.array([0.9, 0.8]), gt, delta=2.0)
    assert sorted(tp.tolist()) == [0.8, 0.9]
    assert fp.tolist() == []


def test_label_detections_far_prediction_is_false_positive():
    tp, fp = label_detections(
        np.array([[1.0, 1.0]]), np.array([0.7]), np.array([[50.0, 50.0]]), delta=3.0
    )
    assert tp.tolist() == []
    assert fp.tolist() == [0.7]


def test_label_detections_greedy_one_to_one_leaves_extra_as_fp():
    # Two predictions near one GT: only the higher-confidence one matches (one-to-one),
    # the other is a false positive — the score labelling FROC sweeps over.
    tp, fp = label_detections(
        np.array([[1.0, 1.0], [1.2, 1.0]]),
        np.array([0.4, 0.9]),
        np.array([[1.0, 1.0]]),
        delta=2.0,
    )
    assert tp.tolist() == [0.9]  # higher score claims the GT
    assert fp.tolist() == [0.4]


def test_label_detections_no_gt_all_false_positive():
    tp, fp = label_detections(
        np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([0.5, 0.6]), np.zeros((0, 2)), delta=2.0
    )
    assert tp.tolist() == []
    assert sorted(fp.tolist()) == [0.5, 0.6]


# --------------------------------------------------------------------------- #
# froc_curve — sensitivity vs FP/mm² sweep (hand-computed operating points)
# --------------------------------------------------------------------------- #


def test_froc_curve_hand_computed_operating_points():
    # tp scores 0.9, 0.8 ; fp scores 0.7, 0.6 ; 2 GT ; 1 mm² of area.
    # Swept thresholds are every sorted unique score: 0.6, 0.7, 0.8, 0.9
    #   thr=0.6 -> fp>=0.6: {0.7,0.6}=2 ; tp>=0.6: {0.9,0.8}=2
    #   thr=0.7 -> fp>=0.7: {0.7}=1     ; tp>=0.7: {0.9,0.8}=2
    #   thr=0.8 -> fp:0                 ; tp:{0.9,0.8}=2
    #   thr=0.9 -> fp:0                 ; tp:{0.9}=1
    # plus the trailing (0,0) most-exclusive anchor.
    fp_per_mm2, sensitivity = froc_curve(
        np.array([0.9, 0.8]), np.array([0.7, 0.6]), num_targets=2, area_mm2=1.0
    )
    np.testing.assert_allclose(fp_per_mm2, [2.0, 1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(sensitivity, [1.0, 1.0, 1.0, 0.5, 0.0])


def test_froc_curve_fp_axis_scales_inversely_with_area():
    # The x-axis is *per mm²*: doubling the evaluated area halves every FP/mm² value,
    # while sensitivity (a GT fraction) is unchanged.
    args = (np.array([0.9, 0.8]), np.array([0.7, 0.6]))
    fp1, sens1 = froc_curve(*args, num_targets=2, area_mm2=1.0)
    fp2, sens2 = froc_curve(*args, num_targets=2, area_mm2=2.0)
    np.testing.assert_allclose(fp2, fp1 / 2.0)
    np.testing.assert_allclose(sens2, sens1)


def test_froc_curve_no_predictions_is_empty_flat():
    fp_per_mm2, sensitivity = froc_curve(
        np.zeros((0,)), np.zeros((0,)), num_targets=3, area_mm2=1.0
    )
    # No scores to sweep -> only the trailing (0 FP, 0 sensitivity) anchor.
    np.testing.assert_allclose(fp_per_mm2, [0.0])
    np.testing.assert_allclose(sensitivity, [0.0])


# --------------------------------------------------------------------------- #
# froc_score_at_thresholds — mean interpolated sensitivity at operating points
# --------------------------------------------------------------------------- #


def test_froc_score_interpolates_and_averages_operating_points():
    # Curve in descending-threshold order (as froc_curve emits); reversed internally to
    # ascending FP/mm² [0,20,40,60] with sensitivity [0,0.3,0.6,0.9].
    #   FP/mm²=10  -> between (0,0),(20,0.3)      -> 0.15
    #   FP/mm²=20  -> exactly 0.3
    #   FP/mm²=50  -> between (40,0.6),(60,0.9)   -> 0.75
    #   FP/mm²>=100 clamp to the max sensitivity  -> 0.90
    # mean(0.15, 0.30, 0.75, 0.90, 0.90, 0.90) = 0.65
    fp_per_mm2 = np.array([60.0, 40.0, 20.0, 0.0])
    sensitivity = np.array([0.9, 0.6, 0.3, 0.0])
    score = froc_score_at_thresholds(fp_per_mm2, sensitivity)
    assert score == pytest.approx(0.65)


def test_froc_score_default_thresholds_are_monkey_operating_points():
    assert DEFAULT_FP_THRESHOLDS == (10.0, 20.0, 50.0, 100.0, 200.0, 300.0)


def test_froc_score_custom_thresholds():
    # Only one operating point at FP/mm²=20 -> sensitivity 0.3.
    fp_per_mm2 = np.array([60.0, 40.0, 20.0, 0.0])
    sensitivity = np.array([0.9, 0.6, 0.3, 0.0])
    assert froc_score_at_thresholds(fp_per_mm2, sensitivity, fp_thresholds=(20.0,)) == pytest.approx(0.3)


def test_froc_score_empty_curve_is_zero():
    assert froc_score_at_thresholds(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# compute_froc — pool TP/FP + targets + area across images, then score (1 class)
# --------------------------------------------------------------------------- #


def test_compute_froc_pools_images_and_scores():
    # Image 0: one TP (0.9) + one far FP (0.2), 1 GT, 0.05 mm².
    # Image 1: one TP (0.8), 2 GT (one missed), 0.05 mm².
    # Pooled: tp=[0.9,0.8], fp=[0.2], 3 GT, 0.10 mm² -> the single FP is 1/0.10 = 10/mm².
    result = compute_froc(
        per_image_pred_xy=[np.array([[5.0, 5.0], [0.0, 0.0]]), np.array([[10.0, 10.0]])],
        per_image_pred_score=[np.array([0.9, 0.2]), np.array([0.8])],
        per_image_gt_xy=[np.array([[5.0, 5.0]]), np.array([[10.0, 10.0], [20.0, 20.0]])],
        delta=2.0,
        per_image_area_mm2=[0.05, 0.05],
    )
    assert isinstance(result, FrocResult)
    assert result.num_targets == 3  # 1 + 2 ground-truth points
    assert result.area_mm2 == pytest.approx(0.10)  # 0.05 + 0.05
    # The one false positive contributes 1 / 0.10 mm² = 10 FP/mm² at the low end; every
    # operating point (>= 10) clamps to the max sensitivity = 2 detected / 3 GT.
    assert result.fp_per_mm2.max() == pytest.approx(10.0)
    assert result.sensitivity.max() == pytest.approx(2 / 3)
    assert result.score == pytest.approx(2 / 3)


def test_compute_froc_equals_manual_label_curve_score_pipeline():
    # compute_froc must be exactly label_detections -> pool -> froc_curve -> score.
    pred_xy = [np.array([[5.0, 5.0], [50.0, 50.0]]), np.array([[10.0, 10.0], [11.0, 11.0]])]
    pred_score = [np.array([0.6, 0.9]), np.array([0.4, 0.7])]
    gt_xy = [np.array([[5.0, 5.0]]), np.array([[10.0, 10.0]])]
    areas = [0.03, 0.02]

    tp_all, fp_all = [], []
    for pxy, ps, gxy in zip(pred_xy, pred_score, gt_xy):
        tp, fp = label_detections(pxy, ps, gxy, delta=2.0)
        tp_all.append(tp)
        fp_all.append(fp)
    fpm, sens = froc_curve(
        np.concatenate(tp_all), np.concatenate(fp_all),
        num_targets=2, area_mm2=sum(areas),
    )
    expected = froc_score_at_thresholds(fpm, sens)

    result = compute_froc(
        per_image_pred_xy=pred_xy, per_image_pred_score=pred_score,
        per_image_gt_xy=gt_xy, delta=2.0, per_image_area_mm2=areas,
    )
    assert result.score == pytest.approx(expected)
    np.testing.assert_allclose(result.fp_per_mm2, fpm)
    np.testing.assert_allclose(result.sensitivity, sens)


def test_compute_froc_no_ground_truth_scores_zero():
    result = compute_froc(
        per_image_pred_xy=[np.array([[1.0, 1.0]])],
        per_image_pred_score=[np.array([0.9])],
        per_image_gt_xy=[np.zeros((0, 2))],
        delta=2.0,
        per_image_area_mm2=[0.1],
    )
    assert result.num_targets == 0
    assert result.score == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# score_monkey_froc — 2-class (lymphocytes/monocytes) + MNL (merged) paths
# --------------------------------------------------------------------------- #


def _monkey_single_image():
    """One synthetic image (points in level0 pixels), spacing 0.5 µm/px.

    Classes: 0 = lymphocytes, 1 = monocytes. GT: two lymphocytes + one monocyte.
    Predictions: a lymphocyte TP, a far lymphocyte FP, and a monocyte TP.
    """
    pred_xy = [np.array([[12.0, 10.0], [200.0, 200.0], [50.0, 50.0]])]
    pred_class = [np.array([0, 0, 1])]
    pred_score = [np.array([0.9, 0.5, 0.8])]
    gt_xy = [np.array([[10.0, 10.0], [100.0, 100.0], [50.0, 50.0]])]
    gt_class = [np.array([0, 0, 1])]
    area = [0.02]
    return pred_xy, pred_class, pred_score, gt_xy, gt_class, area


def test_score_monkey_two_class_paths_match_per_class_compute_froc():
    pred_xy, pred_class, pred_score, gt_xy, gt_class, area = _monkey_single_image()
    spacing_um = 0.5

    out = score_monkey_froc(
        pred_xy, pred_class, pred_score, gt_xy, gt_class, area, spacing_um=spacing_um
    )

    # Per-class result equals compute_froc on that class's filtered points, with the
    # class's µm margin converted to pixels through the spacing (4µm/0.5 = 8px lymph).
    for cls_idx, name in enumerate(MONKEY_CLASS_NAMES):
        delta_px = MONKEY_MATCH_UM[cls_idx] / spacing_um
        expected = compute_froc(
            per_image_pred_xy=[pred_xy[0][pred_class[0] == cls_idx]],
            per_image_pred_score=[pred_score[0][pred_class[0] == cls_idx]],
            per_image_gt_xy=[gt_xy[0][gt_class[0] == cls_idx]],
            delta=delta_px, per_image_area_mm2=area,
        )
        assert isinstance(out[name], FrocResult)
        assert out[name].score == pytest.approx(expected.score)
        assert out[name].num_targets == expected.num_targets


def test_score_monkey_mean_froc_is_mean_of_two_classes():
    pred_xy, pred_class, pred_score, gt_xy, gt_class, area = _monkey_single_image()
    out = score_monkey_froc(pred_xy, pred_class, pred_score, gt_xy, gt_class, area, spacing_um=0.5)
    expected = np.mean([out[n].score for n in MONKEY_CLASS_NAMES])
    assert out["mean_froc"] == pytest.approx(expected)


def test_score_monkey_mnl_merges_all_classes():
    pred_xy, pred_class, pred_score, gt_xy, gt_class, area = _monkey_single_image()
    spacing_um = 0.5
    out = score_monkey_froc(pred_xy, pred_class, pred_score, gt_xy, gt_class, area, spacing_um=spacing_um)

    # The MNL / inflammatory-cells path ignores class: every point is one merged class,
    # matched at the 5µm inflammation margin (5µm/0.5 = 10px).
    expected = compute_froc(
        per_image_pred_xy=pred_xy, per_image_pred_score=pred_score,
        per_image_gt_xy=gt_xy, delta=5.0 / spacing_um, per_image_area_mm2=area,
    )
    mnl = out[MNL_CLASS_NAME]
    assert mnl.num_targets == 3  # all GT, regardless of class
    assert mnl.score == pytest.approx(expected.score)


def test_score_monkey_spacing_scales_matching_distance():
    # Finer spacing (more px per µm) widens the pixel margin, so a prediction that was
    # too far at a coarse spacing can become a true positive at a finer one.
    pred_xy = [np.array([[19.0, 10.0]])]     # 9 px from GT
    pred_class = [np.array([0])]
    pred_score = [np.array([0.9])]
    gt_xy = [np.array([[10.0, 10.0]])]
    gt_class = [np.array([0])]
    area = [0.02]

    coarse = score_monkey_froc(pred_xy, pred_class, pred_score, gt_xy, gt_class, area, spacing_um=1.0)
    fine = score_monkey_froc(pred_xy, pred_class, pred_score, gt_xy, gt_class, area, spacing_um=0.25)
    # lymph margin 4µm -> 4px @1.0 (miss, dist 9) but 16px @0.25 (hit).
    assert coarse["lymphocytes"].score == pytest.approx(0.0)
    assert fine["lymphocytes"].score > 0.0
