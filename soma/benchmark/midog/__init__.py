"""MIDOG 2022 mitotic-figure detection, a dataset of the detection harness.

The curator (:mod:`soma.benchmark.midog.curate`) writes one sample per ROI, which is tiled
into fixed supervision tiles; the native metric is the single-class F1 of
:mod:`soma.benchmark.midog.f1`. Importing this package registers the dataset with the
harness (:func:`~soma.benchmark.detection.register_dataset`) and its ``detection/midog``
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
from soma.benchmark.midog.curate import curate_midog_detection
from soma.benchmark.midog.f1 import midog_f1
from soma.benchmark.registry import register_benchmark

_CONFIG = Path(__file__).resolve().parent / "configs" / "detection.yaml"


def score_detection_points(samples: Sequence[SamplePrediction], spec: DatasetSpec) -> dict[str, float]:
    """MIDOG-native single-class F1 over persisted points (Hungarian one-to-one within 7.5 µm)."""
    pred_xy, pred_score, _pred_class, gt_xy, _gt_class, _ = per_image_arrays(samples)
    delta = float(spec.match_um[0]) / float(spec.spacing_um)
    score = midog_f1(pred_xy, pred_score, gt_xy, delta=delta, method=spec.match_method)
    return {"f1": float(score.f1)}


def curate_detection(raw_root: str | Path, out_dir: str | Path, spec: DatasetSpec) -> CuratedManifest:
    """Curate one sample per ROI under ``out_dir/roi`` and tile it into the Manifest at ``out_dir``."""
    roi_manifest = curate_midog_detection(
        raw_root,
        Path(out_dir) / "roi",
        annotations_json=Path(raw_root) / "MIDOG2022_training_enriched.json",
        force_spacing_at_level_0=spec.spacing_um,
    )
    return tile_roi_manifest(roi_manifest, out_dir, spec)


DETECTION_DATASET = DatasetSpec(
    name="midog",
    spacing_um=0.25,
    metric_name="f1",
    num_classes=1,
    match_um=(7.5,),
    match_method="hungarian",
    tolerance=0.10,
    test_source="local_holdout",
    config=_CONFIG,
    score=score_detection_points,
    curate=curate_detection,
    stitch_to_roi=True,
    # OCELOT's proven fixed geometry (1024² supervision tiles); overlap >= one mitotic-
    # figure diameter so every object is fully centred in at least one tile.
    tile_px=1024,
    tile_overlap_px=128,
)

register_dataset(DETECTION_DATASET)
register_benchmark(DetectionDatasetBenchmark("midog"))
