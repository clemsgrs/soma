"""Populate and validate the shared BEETLE dense cache without training decoders."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Sequence

from soma import extract
from soma.config import ExperimentConfig
from soma.data import AnnotationManifest, ImageManifest, require_coverage
from soma.pipeline.build import extraction_unit
from soma.plan import check, learner_shape
from soma.preprocessing.resolution import resolve_pipeline_dense, resolve_pipeline_preprocessing


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_verified_cache(
    *, config_path: str | Path, work_dir: str | Path, output_path: str | Path
) -> dict:
    """Run the pipeline's exact cache preparation seam and stop before training."""
    config_path = Path(config_path).resolve()
    work_dir = Path(work_dir).resolve()
    output_path = Path(output_path).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig.from_yaml(config_path)
    check(config)
    if learner_shape(config) != "grid":
        raise ValueError("BEETLE extraction is a grid run (a decoder over ROI token grids).")
    images = ImageManifest.from_csv(config.cohort.records)
    features = config.features
    if features.encoder is None:
        raise ValueError("BEETLE extraction requires a single encoder configuration.")
    cache = features.cache
    if cache.root_dir is None:
        cache = replace(cache, root_dir=Path(config.run.output_dir) / "feature_cache")
    store = extract(
        images,
        annotations=AnnotationManifest.from_csv(config.cohort.records),
        encoder=features.encoder,
        extraction=resolve_pipeline_preprocessing(config),
        unit=extraction_unit(config),
        dense=resolve_pipeline_dense(config),
        spacing_policy=features.spacing_policy,
        execution=features.execution,
        cache=cache,
        output_dir=work_dir,
    )
    roi_ids = list(store.sample_ids)
    require_coverage(store, roi_ids)
    slides_with_rois = sorted(set(store.rois.children.values()))
    parent_ids = list(images.sample_ids)
    zero_roi_slides = [slide for slide in parent_ids if slide not in set(slides_with_rois)]
    feature_dir = Path(store.feature_dir).resolve()
    tensors = sorted(feature_dir.rglob("*.pt"))
    sidecars = sorted(feature_dir.rglob("*.meta.json"))
    if len(tensors) != len(roi_ids) or len(sidecars) != len(roi_ids):
        raise ValueError(
            "BEETLE cache payload coverage mismatch: "
            f"ROIs={len(roi_ids)}, tensors={len(tensors)}, sidecars={len(sidecars)}"
        )

    payload = {
        "schema_version": 1,
        "status": "completed",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "work_dir": str(work_dir),
        "feature_dir": str(feature_dir),
        "parent_slides": len(parent_ids),
        "roi_grids": len(roi_ids),
        "slides_with_rois": len(slides_with_rois),
        "zero_roi_slides": zero_roi_slides,
        "tensor_files": len(tensors),
        "sidecar_files": len(sidecars),
        "payload_bytes": sum(path.stat().st_size for path in tensors),
        "sidecar_bytes": sum(path.stat().st_size for path in sidecars),
        "feature_dim": int(store.feature_dim),
        "grid_shape": list(store.grid_shape),
        "payload_hashes_pending": True,
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    extract_verified_cache(
        config_path=args.config,
        work_dir=args.work_dir,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["extract_verified_cache"]
