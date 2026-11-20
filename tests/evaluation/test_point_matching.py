"""The class-aware point matcher against a dense oracle, and its scale bound.

The matcher only looks at prediction and annotation pairs within the matching distance.
The oracle below is the dense matcher it replaced: it builds the full distance matrix,
so it is only usable on small point sets. On every point set of the property test, the
two give the same true-positive count per class, for Hungarian and greedy matching.
Greedy matching also gives the same pairs: the one-pass threshold sweep reads them.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from soma.evaluation.detection import match_points
from soma.evaluation.detection.matching import match_single_class
from tests.whole_slides import MEMORY_PRELUDE, peak_rss_of


def dense_pairs(pred_xy, pred_score, gt_xy, *, delta, method):
    """The dense matcher: every prediction against every annotation."""
    if pred_xy.shape[0] == 0 or gt_xy.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    diff = pred_xy[:, None, :] - gt_xy[None, :, :]
    dist = np.sqrt((diff**2).sum(axis=2))
    within = dist <= float(delta)
    if method == "hungarian":
        cost = np.where(within, dist, float(delta) * 1e6 + 1.0)
        rows, cols = linear_sum_assignment(cost)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if within[r, c]]
    else:
        order = np.argsort(-pred_score, kind="stable")
        order = order[within[order].any(axis=1)]
        claimed = np.zeros(gt_xy.shape[0], dtype=bool)
        pairs = []
        for r in order:
            in_range = within[r] & ~claimed
            if in_range.any():
                g = int(np.argmin(np.where(in_range, dist[r], np.inf)))
                claimed[g] = True
                pairs.append((int(r), g))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def uniform(rng, n_pred, n_gt, side):
    """Points spread over a ``side`` px square: the density sets how many pairs are in reach."""
    return rng.uniform(0, side, (n_pred, 2)), rng.uniform(0, side, (n_gt, 2))


def clustered(rng, n_pred, n_gt, side):
    """Annotations in a few tight clusters and predictions jittered around them: large
    connected groups of pairs in reach."""
    centres = rng.uniform(0, side, (4, 2))
    gt = centres[rng.integers(0, 4, n_gt)] + rng.normal(0, 3, (n_gt, 2))
    pred = gt[rng.integers(0, n_gt, n_pred)] + rng.normal(0, 2, (n_pred, 2))
    return pred, gt


def tied(rng, n_pred, n_gt, side):
    """Points on an integer grid, with repeats: distances of exactly the matching distance,
    equal distances to several annotations, and coincident points."""
    grid = max(2, int(side) // 8)
    return (
        rng.integers(0, grid, (n_pred, 2)).astype(np.float64),
        rng.integers(0, grid, (n_gt, 2)).astype(np.float64),
    )


LAYOUTS = {"uniform": uniform, "clustered": clustered, "tied": tied}
#: (predictions, annotations, side px): the densities of the property test.
DENSITIES = [(40, 30, 400.0), (120, 90, 120.0), (200, 160, 40.0)]


def point_sets(layout):
    for seed in range(12):
        for n_pred, n_gt, side in DENSITIES:
            rng = np.random.default_rng(seed)
            pred, gt = LAYOUTS[layout](rng, n_pred, n_gt, side)
            # Few score levels, so equal scores are common too.
            score = rng.integers(0, 5, n_pred) / 4.0
            yield pred, score, gt


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("delta", [1.0, 3.0])
def test_hungarian_matching_finds_as_many_pairs_as_the_dense_matcher(layout, delta):
    for pred, score, gt in point_sets(layout):
        match = match_single_class(pred, score, gt, delta=delta, method="hungarian")
        expected = dense_pairs(pred, score, gt, delta=delta, method="hungarian")
        assert match.tp == expected.shape[0]
        # A one-to-one matching within the matching distance.
        assert len(set(match.pairs[:, 0])) == len(set(match.pairs[:, 1])) == match.tp
        gaps = np.linalg.norm(pred[match.pairs[:, 0]] - gt[match.pairs[:, 1]], axis=1)
        assert (gaps <= delta).all()


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize("delta", [1.0, 3.0])
def test_greedy_matching_makes_the_dense_matchers_pairs(layout, delta):
    for pred, score, gt in point_sets(layout):
        match = match_single_class(pred, score, gt, delta=delta, method="greedy")
        expected = dense_pairs(pred, score, gt, delta=delta, method="greedy")
        np.testing.assert_array_equal(match.pairs, expected)


@pytest.mark.parametrize("layout", ["uniform", "tied"])
def test_a_crowd_of_points_all_in_reach_of_each_other_matches_like_the_dense_matcher(layout):
    """One connected group of 1300 predictions and 1100 annotations, more than a dense cost
    matrix is built for."""
    rng = np.random.default_rng(3)
    pred, gt = LAYOUTS[layout](rng, 1300, 1100, 30.0 if layout == "uniform" else 96.0)
    score = rng.integers(0, 5, 1300) / 4.0
    for method in ("hungarian", "greedy"):
        match = match_single_class(pred, score, gt, delta=3.0, method=method)
        expected = dense_pairs(pred, score, gt, delta=3.0, method=method)
        assert match.tp == expected.shape[0]
        gaps = np.linalg.norm(pred[match.pairs[:, 0]] - gt[match.pairs[:, 1]], axis=1)
        assert (gaps <= 3.0).all() and len(set(match.pairs[:, 1])) == match.tp
    np.testing.assert_array_equal(match.pairs, expected)


@pytest.mark.parametrize("method", ["hungarian", "greedy"])
def test_class_aware_counts_match_the_dense_matcher_per_class(method):
    rng = np.random.default_rng(7)
    for _ in range(10):
        pred, gt = clustered(rng, 150, 120, 60.0)
        pred_class, gt_class = rng.integers(0, 3, 150), rng.integers(0, 3, 120)
        score = rng.integers(0, 5, 150) / 4.0
        counts = match_points(pred, pred_class, score, gt, gt_class, num_classes=3, delta=3.0, method=method)
        for c in range(3):
            p, g = pred_class == c, gt_class == c
            tp = dense_pairs(pred[p], score[p], gt[g], delta=3.0, method=method).shape[0]
            assert counts[c].tolist() == [tp, int(p.sum()) - tp, int(g.sum()) - tp]


#: Matching 100k predictions against 20k annotations of one class: a 4 mm x 4 mm slide
#: region at 0.5 um/px (8000 px a side), predictions around the annotations and spread
#: over the region, at a 3 um (6 px) matching distance.
SCALE_CHILD = MEMORY_PRELUDE + r"""
import time
import numpy as np
from soma.evaluation.detection import match_points

rng = np.random.default_rng(0)
side, delta = 8000.0, 6.0
gt = rng.uniform(0, side, (20000, 2))
near = gt[rng.integers(0, 20000, 60000)] + rng.normal(0, 3, (60000, 2))
pred = np.concatenate([near, rng.uniform(0, side, (40000, 2))])
score = rng.uniform(0, 1, 100000)
zeros_p, zeros_g = np.zeros(100000, dtype=np.int64), np.zeros(20000, dtype=np.int64)
result = {}
for method in ("hungarian", "greedy"):
    reset_peak()
    before = status("VmRSS")
    start = time.perf_counter()
    counts = match_points(pred, zeros_p, score, gt, zeros_g, num_classes=1, delta=delta, method=method)
    result[method] = {
        "seconds": time.perf_counter() - start,
        "peak_bytes": status("VmHWM") - before,
        "tp": int(counts[0, 0]),
    }
print(json.dumps(result))
"""


def test_matching_100k_predictions_against_20k_annotations_stays_within_bounds():
    """Measured on the development node: Hungarian 0.5 s and 13 MB over the inputs, greedy
    0.1 s and 5 MB. The bound of 10 s and 200 MB leaves room for a slower CI runner; the
    dense matcher needed 16 GB for the distance matrix alone.

    About 3 predictions fall around each annotation, each within 6 px with probability
    1 - exp(-2) (a normal of 3 px), so about 7.5% of the annotations have none in reach.
    """
    result = peak_rss_of(SCALE_CHILD)
    for method in ("hungarian", "greedy"):
        assert result[method]["seconds"] < 10.0, result
        assert result[method]["peak_bytes"] < 200 * 2**20, result
        assert 18000 < result[method]["tp"] < 19000, result
    assert result["hungarian"]["tp"] >= result["greedy"]["tp"]
