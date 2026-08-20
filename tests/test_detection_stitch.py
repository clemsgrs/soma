"""Tests for the per-ROI tile stitch (soma.benchmarks.detection_benchmark)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pytest

from soma.benchmarks.detection_benchmark import (
    SamplePrediction,
    _greedy_point_nms,
    score_dataset_points,
    stitch_tiles_to_rois,
)


def _manifest(meta_by_sid):
    return SimpleNamespace(
        samples={sid: SimpleNamespace(sample_id=sid, metadata=m) for sid, m in meta_by_sid.items()}
    )


def _head():
    from soma.dense import DenseSampleSpacing

    spacing = DenseSampleSpacing(source_spacing_um=0.25, effective_spacing_um=0.25)
    return SimpleNamespace(
        num_classes=1, delta_px=30.0, nms_distance_px=30.0, matching="hungarian",
        spacing_for_sample=lambda sid: spacing,
    )


def test_greedy_point_nms_per_class():
    xy = np.array([[100.0, 100.0], [102.0, 101.0], [300.0, 300.0]])  # first two ~within radius
    scores = np.array([0.5, 0.9, 0.7])
    classes = np.array([0, 0, 0])
    keep = _greedy_point_nms(xy, scores, classes, min_distance=30.0)
    assert keep.tolist() == [1, 2]  # the higher-scoring near-duplicate survives; distant kept
    # same coordinates in a different class are NOT suppressed.
    keep2 = _greedy_point_nms(xy[:2], scores[:2], np.array([0, 1]), min_distance=30.0)
    assert keep2.tolist() == [0, 1]


def test_stitch_dedups_gt_and_nms_predictions_per_roi():
    # ROI "roi_A": tile t0 origin (0,0), tile t1 origin (400,0); they overlap in x [400,512).
    meta = {
        "roi_A_t0": {"source_wsi": "roi_A", "tile_x": 0, "tile_y": 0, "roi_width": 912, "roi_height": 512},
        "roi_A_t1": {"source_wsi": "roi_A", "tile_x": 400, "tile_y": 0, "roi_width": 912, "roi_height": 512},
    }
    # M1 at ROI (450,100) sits in the overlap -> labelled in BOTH tiles; M2 at (100,100) only t0.
    t0 = SamplePrediction(
        sample_id="roi_A_t0",
        pred_xy=[[452.0, 101.0], [98.0, 99.0]], pred_score=[0.9, 0.8], pred_class=[0, 0],
        gt_xy=[[450.0, 100.0], [100.0, 100.0]], gt_class=[0, 0], matched=[True, True],
    )
    t1 = SamplePrediction(
        sample_id="roi_A_t1",
        pred_xy=[[49.0, 100.0]], pred_score=[0.85], pred_class=[0],  # ROI (449,100), dup of M1
        gt_xy=[[50.0, 100.0]], gt_class=[0], matched=[True],          # ROI (450,100), dup of M1
    )
    out = stitch_tiles_to_rois([t0, t1], _manifest(meta), _head())

    assert len(out) == 1 and out[0].sample_id == "roi_A"
    roi = out[0]
    # GT overlap copy of M1 deduped -> exactly the 2 distinct mitoses, in ROI coordinates.
    assert sorted(roi.gt_xy) == [[100.0, 100.0], [450.0, 100.0]]
    # The two overlapping predictions of M1 collapse under NMS -> 2 predictions total.
    assert len(roi.pred_xy) == 2
    # Both surviving predictions match their GT within delta (2 TP, 0 FP, 0 FN).
    assert all(roi.matched) and len(roi.matched) == 2
    assert roi.area_mm2 is not None and roi.area_mm2 > 0


def test_stitch_passthrough_when_no_tile_origins():
    # No source_wsi in metadata (e.g. OCELOT) -> samples returned unchanged.
    meta = {"img_1": {"domain": "x"}, "img_2": {"domain": "y"}}
    samples = [
        SamplePrediction(sample_id="img_1", pred_xy=[[1.0, 1.0]], pred_score=[0.5], pred_class=[0],
                         gt_xy=[[1.0, 1.0]], gt_class=[0], matched=[True]),
        SamplePrediction(sample_id="img_2", pred_xy=[], pred_score=[], pred_class=[],
                         gt_xy=[], gt_class=[], matched=[]),
    ]
    out = stitch_tiles_to_rois(samples, _manifest(meta), _head())
    assert out == samples


def test_stitch_score_equals_direct_per_roi_f1():
    # The central claim: stitching per-tile detections then scoring MIDOG-native == scoring
    # the equivalent whole-ROI prediction directly. Build a 2-tile ROI (t0@0, t1@400,
    # tile 512 -> overlap x[400,512)) with an overlap mitosis M1 (labelled + predicted in
    # both tiles), a t0-only mitosis M2 (TP), a t1-only mitosis M3 (FN, no prediction), and a
    # t0-only false positive. Expected: TP=2, FP=1, FN=1 -> F1 = 2/3.
    meta = {
        "roi_t0": {"source_wsi": "roi", "tile_x": 0, "tile_y": 0, "roi_width": 912, "roi_height": 512},
        "roi_t1": {"source_wsi": "roi", "tile_x": 400, "tile_y": 0, "roi_width": 912, "roi_height": 512},
    }
    t0 = SamplePrediction(
        sample_id="roi_t0",
        pred_xy=[[452.0, 101.0], [98.0, 99.0], [250.0, 250.0]],  # P1(M1), P2(M2), P3(FP)
        pred_score=[0.9, 0.8, 0.7], pred_class=[0, 0, 0],
        gt_xy=[[450.0, 100.0], [100.0, 100.0]], gt_class=[0, 0], matched=[True, True, False],
    )
    t1 = SamplePrediction(
        sample_id="roi_t1",
        pred_xy=[[52.0, 101.0]], pred_score=[0.9], pred_class=[0],  # ROI (452,101): dup of P1
        gt_xy=[[50.0, 100.0], [300.0, 100.0]], gt_class=[0, 0],      # ROI (450,100) dup M1; (700,100) M3
        matched=[True],
    )
    stitched = stitch_tiles_to_rois([t0, t1], _manifest(meta), _head())

    direct = SamplePrediction(
        sample_id="roi",
        pred_xy=[[452.0, 101.0], [98.0, 99.0], [250.0, 250.0]],
        pred_score=[0.9, 0.8, 0.7], pred_class=[0, 0, 0],
        gt_xy=[[450.0, 100.0], [100.0, 100.0], [700.0, 100.0]], gt_class=[0, 0, 0],
        matched=[True, True, False],
    )
    f1_stitched = score_dataset_points("midog", stitched)["f1"]
    f1_direct = score_dataset_points("midog", [direct])["f1"]
    assert f1_stitched == pytest.approx(f1_direct)
    assert f1_stitched == pytest.approx(2.0 / 3.0)


def test_stitch_four_tile_corner_overlap():
    # One mitosis at ROI (500,500) sits in the corner overlap of 4 tiles (origins (0,0),
    # (400,0), (0,400), (400,400); tile 512). All 4 label + predict it -> 1 GT, 1 prediction,
    # F1 = 1.0 after dedup/NMS.
    origins = {"t00": (0, 0), "t10": (400, 0), "t01": (0, 400), "t11": (400, 400)}
    meta = {
        f"roi_{k}": {"source_wsi": "roi", "tile_x": ox, "tile_y": oy, "roi_width": 912, "roi_height": 912}
        for k, (ox, oy) in origins.items()
    }
    samples = []
    for k, (ox, oy) in origins.items():
        samples.append(SamplePrediction(
            sample_id=f"roi_{k}",
            pred_xy=[[500.0 - ox + 1.0, 500.0 - oy + 1.0]], pred_score=[0.9], pred_class=[0],
            gt_xy=[[500.0 - ox, 500.0 - oy]], gt_class=[0], matched=[True],
        ))
    out = stitch_tiles_to_rois(samples, _manifest(meta), _head())
    assert len(out) == 1
    assert len(out[0].gt_xy) == 1 and len(out[0].pred_xy) == 1
    assert score_dataset_points("midog", out)["f1"] == pytest.approx(1.0)


def test_stitch_mixed_tiled_untiled_manifest_raises():
    meta = {"roi_t0": {"source_wsi": "roi", "tile_x": 0, "tile_y": 0}, "plain": {"domain": "x"}}
    samples = [
        SamplePrediction(sample_id="roi_t0", pred_xy=[], pred_score=[], pred_class=[],
                         gt_xy=[], gt_class=[], matched=[]),
        SamplePrediction(sample_id="plain", pred_xy=[], pred_score=[], pred_class=[],
                         gt_xy=[], gt_class=[], matched=[]),
    ]
    with pytest.raises(ValueError, match="mixed manifest"):
        stitch_tiles_to_rois(samples, _manifest(meta), _head())


def test_greedy_point_nms_matches_bruteforce():
    # The spatial-hash NMS must be identical to the O(n^2) reference on a dense point cloud.
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 200, size=(400, 2))
    scores = rng.uniform(0, 1, size=400)
    classes = rng.integers(0, 2, size=400)
    md = 15.0

    def brute(xy, scores, classes, md):
        md_sq = md * md
        kept: list[int] = []
        for i in np.argsort(-scores, kind="stable"):
            c = classes[i]
            if all(
                (xy[i, 0] - xy[j, 0]) ** 2 + (xy[i, 1] - xy[j, 1]) ** 2 >= md_sq
                for j in kept if classes[j] == c
            ):
                kept.append(int(i))
        return sorted(kept)

    assert _greedy_point_nms(xy, scores, classes, md).tolist() == brute(xy, scores, classes, md)


def test_stitch_handles_empty_prediction_roi():
    meta = {"roi_B_t0": {"source_wsi": "roi_B", "tile_x": 0, "tile_y": 0}}
    s = SamplePrediction(sample_id="roi_B_t0", pred_xy=[], pred_score=[], pred_class=[],
                         gt_xy=[[10.0, 10.0]], gt_class=[0], matched=[])
    out = stitch_tiles_to_rois([s], _manifest(meta), _head())
    assert len(out) == 1 and out[0].sample_id == "roi_B"
    assert out[0].pred_xy == [] and out[0].gt_xy == [[10.0, 10.0]]
    assert out[0].area_mm2 is None  # no roi_width/height in metadata


def test_stitch_drops_predictions_in_padded_region_before_matching():
    meta = {
        "roi_t0": {
            "source_wsi": "roi",
            "tile_x": 0,
            "tile_y": 0,
            "roi_width": 300,
            "roi_height": 250,
        }
    }
    sample = SamplePrediction(
        sample_id="roi_t0",
        # The invalid padding peak is higher-scored and within the NMS radius of the valid
        # edge peak, so clipping after NMS would incorrectly remove both.
        pred_xy=[[299.0, 100.0], [305.0, 100.0]],
        pred_score=[0.8, 0.9],
        pred_class=[0, 0],
        gt_xy=[[299.0, 100.0]],
        gt_class=[0],
        matched=[True, False],
    )

    (roi,) = stitch_tiles_to_rois([sample], _manifest(meta), _head())

    assert roi.pred_xy == [[299.0, 100.0]]
    assert roi.pred_score == [0.8]
    assert roi.matched == [True]


def test_stitch_rejects_inconsistent_roi_dimensions():
    meta = {
        "roi_t0": {
            "source_wsi": "roi", "tile_x": 0, "tile_y": 0,
            "roi_width": 300, "roi_height": 250,
        },
        "roi_t1": {
            "source_wsi": "roi", "tile_x": 100, "tile_y": 0,
            "roi_width": 301, "roi_height": 250,
        },
    }
    samples = [
        SamplePrediction(sid, [], [], [], [], [], []) for sid in ("roi_t0", "roi_t1")
    ]

    with pytest.raises(ValueError, match="inconsistent ROI dimensions"):
        stitch_tiles_to_rois(samples, _manifest(meta), _head())
