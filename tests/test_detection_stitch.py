"""Tests for the per-ROI tile stitch (soma.benchmarks.detection_benchmark)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from soma.benchmarks.detection_benchmark import (
    SamplePrediction,
    _greedy_point_nms,
    stitch_tiles_to_rois,
)


def _manifest(meta_by_sid):
    return SimpleNamespace(
        samples={sid: SimpleNamespace(sample_id=sid, metadata=m) for sid, m in meta_by_sid.items()}
    )


def _head():
    return SimpleNamespace(
        num_classes=1, delta_px=30.0, nms_distance_px=30.0, matching="hungarian", level0_spacing=0.25
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


def test_stitch_handles_empty_prediction_roi():
    meta = {"roi_B_t0": {"source_wsi": "roi_B", "tile_x": 0, "tile_y": 0}}
    s = SamplePrediction(sample_id="roi_B_t0", pred_xy=[], pred_score=[], pred_class=[],
                         gt_xy=[[10.0, 10.0]], gt_class=[0], matched=[])
    out = stitch_tiles_to_rois([s], _manifest(meta), _head())
    assert len(out) == 1 and out[0].sample_id == "roi_B"
    assert out[0].pred_xy == [] and out[0].gt_xy == [[10.0, 10.0]]
    assert out[0].area_mm2 is None  # no roi_width/height in metadata
