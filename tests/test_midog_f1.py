"""Unit tests for the MIDOG-native F1 scorer, on hand-constructed point sets.

MIDOG 2022 scores mitosis detection with a single dataset-global F1 (one-to-one match
within a 7.5 µm tolerance). The scorer is a thin adapter over ``soma.detection.matching``,
so these tests fix its TP/FP/FN pooling + reduction against explicit tiny inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from soma.detection import MIDOG_MATCH_DISTANCE_UM, MidogF1Score, midog_f1


def test_perfect_single_image():
    score = midog_f1(
        per_image_pred_xy=[np.array([[1.0, 1.0], [5.0, 5.0]])],
        per_image_pred_score=[np.array([0.9, 0.8])],
        per_image_gt_xy=[np.array([[1.0, 1.0], [5.0, 5.0]])],
        delta=2.0,
    )
    assert isinstance(score, MidogF1Score)
    assert (score.tp, score.fp, score.fn) == (2, 0, 0)
    assert score.f1 == pytest.approx(1.0)
    assert score.precision == pytest.approx(1.0)
    assert score.recall == pytest.approx(1.0)


def test_false_positive_and_false_negative():
    # A prediction far from the only GT: 0 TP, 1 FP, 1 FN -> F1 = 0.
    score = midog_f1(
        per_image_pred_xy=[np.array([[1.0, 1.0]])],
        per_image_pred_score=[np.array([0.9])],
        per_image_gt_xy=[np.array([[50.0, 50.0]])],
        delta=3.0,
    )
    assert (score.tp, score.fp, score.fn) == (0, 1, 1)
    assert score.f1 == pytest.approx(0.0)


def test_distance_threshold_gates_the_match():
    pred = [np.array([[0.0, 0.0]])]
    gt = [np.array([[2.0, 0.0]])]
    scores = [np.array([0.9])]
    near = midog_f1(per_image_pred_xy=pred, per_image_pred_score=scores, per_image_gt_xy=gt, delta=3.0)
    far = midog_f1(per_image_pred_xy=pred, per_image_pred_score=scores, per_image_gt_xy=gt, delta=1.0)
    assert near.tp == 1 and near.fp == 0 and near.fn == 0
    assert far.tp == 0 and far.fp == 1 and far.fn == 1


def test_counts_pool_dataset_global_across_images():
    # image A: a clean hit (1 TP). image B: one FP + one FN. Pooled -> tp=1, fp=1, fn=1.
    score = midog_f1(
        per_image_pred_xy=[np.array([[1.0, 1.0]]), np.array([[0.0, 0.0]])],
        per_image_pred_score=[np.array([0.9]), np.array([0.5])],
        per_image_gt_xy=[np.array([[1.0, 1.0]]), np.array([[40.0, 40.0]])],
        delta=2.0,
    )
    assert (score.tp, score.fp, score.fn) == (1, 1, 1)
    assert score.precision == pytest.approx(0.5)
    assert score.recall == pytest.approx(0.5)
    assert score.f1 == pytest.approx(0.5)


def test_one_to_one_matching_leaves_extra_prediction_as_fp():
    # Two predictions near one GT: only one can match (one-to-one), the other is an FP.
    score = midog_f1(
        per_image_pred_xy=[np.array([[1.0, 1.0], [1.4, 1.0]])],
        per_image_pred_score=[np.array([0.9, 0.8])],
        per_image_gt_xy=[np.array([[1.0, 1.0]])],
        delta=2.0,
    )
    assert (score.tp, score.fp, score.fn) == (1, 1, 0)
    # F1 = 2*1 / (2*1 + 1 + 0) = 2/3.
    assert score.f1 == pytest.approx(2.0 / 3.0)


def test_empty_inputs_score_zero():
    score = midog_f1(
        per_image_pred_xy=[np.zeros((0, 2))],
        per_image_pred_score=[np.zeros((0,))],
        per_image_gt_xy=[np.zeros((0, 2))],
        delta=2.0,
    )
    assert (score.tp, score.fp, score.fn) == (0, 0, 0)
    assert score.f1 == pytest.approx(0.0)


def test_no_images_score_zero():
    score = midog_f1(
        per_image_pred_xy=[], per_image_pred_score=[], per_image_gt_xy=[], delta=2.0
    )
    assert (score.tp, score.fp, score.fn) == (0, 0, 0)
    assert score.f1 == pytest.approx(0.0)


def test_greedy_method_selectable():
    # A greedy alternative is available (OCELOT-style); on a clean set it agrees.
    score = midog_f1(
        per_image_pred_xy=[np.array([[1.0, 1.0]])],
        per_image_pred_score=[np.array([0.9])],
        per_image_gt_xy=[np.array([[1.0, 1.0]])],
        delta=2.0,
        method="greedy",
    )
    assert score.f1 == pytest.approx(1.0)


def test_rejects_mismatched_list_lengths():
    with pytest.raises(ValueError, match="same number of images"):
        midog_f1(
            per_image_pred_xy=[np.zeros((0, 2))],
            per_image_pred_score=[np.zeros((0,))],
            per_image_gt_xy=[],
            delta=2.0,
        )


def test_native_distance_constant_is_the_challenge_tolerance():
    # The MIDOG 2022 hit tolerance is ~7.5 µm (about one nucleus).
    assert MIDOG_MATCH_DISTANCE_UM == pytest.approx(7.5)
