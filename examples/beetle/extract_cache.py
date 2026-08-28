"""Populate and validate the shared BEETLE dense cache without training decoders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from soma.config import load_config
from soma.pipeline import Pipeline


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

    pipeline = Pipeline(load_config(config_path))
    context = pipeline._build_slide_manifest_dense_context(run_dir=work_dir)
    store = context.feature_store
    roi_ids = list(context.dataset.sample_ids)
    store.validate_coverage(roi_ids)
    slides_with_rois = sorted(
        {str(record.slide_id) for record in context.dataset.samples.values()}
    )
    parent_ids = list(pipeline.dataset.sample_ids)
    zero_roi_slides = sorted(set(parent_ids) - set(slides_with_rois))
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
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
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
