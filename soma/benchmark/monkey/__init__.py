"""MONKEY inflammatory-cell detection, a dataset of the detection harness.

The curator (:mod:`soma.benchmark.monkey.curate`) writes one sample per ROI polygon,
which is tiled into fixed supervision tiles; the native metric is the challenge's FROC
(:mod:`soma.benchmark.monkey.froc`). Importing this package registers the dataset with the
harness (:func:`~soma.benchmark.detection.register_dataset`) and its ``detection/monkey``
benchmark view.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from soma.benchmark.curation import CuratedManifest
from soma.benchmark.detection import (
    DatasetSpec,
    DetectionDatasetBenchmark,
    SamplePrediction,
    per_image_arrays,
    register_dataset,
    tile_roi_manifest,
)
from soma.benchmark.monkey.curate import curate_monkey_detection
from soma.benchmark.monkey.froc import MNL_CLASS_NAME, score_monkey_froc
from soma.benchmark.registry import register_benchmark

_CONFIG = Path(__file__).resolve().parent / "configs" / "detection.yaml"


def score_detection_points(samples: Sequence[SamplePrediction], spec: DatasetSpec) -> dict[str, float]:
    """MONKEY-native FROCs over persisted points, one per final-leaderboard column.

    ``froc_inflammatory`` is the merged lymphocyte + monocyte FROC (5 µm margin) the
    challenge ranks by (Track 1); ``froc_lymphocytes`` / ``froc_monocytes`` are the per-class
    columns (4 / 5 µm). ``mean_froc`` (their average, the Track-2 analogue) is kept for the
    class-averaged reference row. All use the per-mm² FP sweep of :func:`score_monkey_froc`.
    """
    pred_xy, pred_score, pred_class, gt_xy, gt_class, area = per_image_arrays(samples)
    out = score_monkey_froc(
        pred_xy, pred_class, pred_score, gt_xy, gt_class, area,
        spacing_um=float(spec.spacing_um), match_um=spec.match_um,
    )
    return {
        "froc_inflammatory": float(out[MNL_CLASS_NAME].score),
        "froc_lymphocytes": float(out["lymphocytes"].score),
        "froc_monocytes": float(out["monocytes"].score),
        "mean_froc": float(out["mean_froc"]),
    }


def curate_detection(raw_root: str | Path, out_dir: str | Path, spec: DatasetSpec) -> CuratedManifest:
    """Curate one sample per ROI under ``out_dir/roi`` and tile it into the Manifest at ``out_dir``."""
    roi_manifest = curate_monkey_detection(raw_root, Path(out_dir) / "roi", spacing_at_level_0=spec.spacing_um)
    return tile_roi_manifest(roi_manifest, out_dir, spec)


DETECTION_DATASET = DatasetSpec(
    name="monkey",
    spacing_um=0.24199951445730394,
    metric_name="froc_inflammatory",
    num_classes=2,
    match_um=(4.0, 5.0),
    match_method="greedy",
    tolerance=None,
    test_source="local_holdout",
    config=_CONFIG,
    score=score_detection_points,
    curate=curate_detection,
    stitch_to_roi=True,
    # The same fixed geometry as MIDOG: the curator crops one sample per ROI polygon and
    # these tile it.
    tile_px=1024,
    tile_overlap_px=128,
)

register_dataset(DETECTION_DATASET)
register_benchmark(DetectionDatasetBenchmark("monkey"))
