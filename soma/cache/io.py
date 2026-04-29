"""Cache I/O utilities: path resolution, artifact writing, manifest/metadata helpers."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import slide2vec.progress as slide2vec_progress
from slide2vec.artifacts import TileEmbeddingArtifact

from soma.cache._types import CACHE_METADATA_NAME, MANIFEST_NAME
from soma.config import CacheConfig


def resolve_cache_root(
    cache_config: CacheConfig,
    *,
    feature_dir: Path | str,
    output_root: Path | str | None = None,
) -> Path:
    if cache_config.root_dir is not None:
        return Path(cache_config.root_dir)
    if output_root is not None:
        return Path(output_root) / "feature_cache"
    return Path(feature_dir).parent / "feature_cache"


def resolve_tiling_cache_root(
    cache_config: CacheConfig,
    *,
    tiling_dir: Path | str,
    output_root: Path | str | None = None,
) -> Path:
    feature_root = resolve_cache_root(
        cache_config,
        feature_dir=Path(tiling_dir).parent / "features",
        output_root=output_root,
    )
    return feature_root.parent / "tiling_cache"


def resolve_feature_payload_dir(path: Path | str) -> Path:
    """Resolve the directory containing feature .pt files.

    Handles soma cache dirs (cache_metadata.json + features/),
    slide2vec artifact dirs (slide_embeddings/, hierarchical_embeddings/, tile_embeddings/),
    and plain directories.
    """
    root = Path(path)
    for subdir in ("patient_embeddings", "slide_embeddings", "hierarchical_embeddings", "tile_embeddings"):
        candidate = root / subdir
        if candidate.is_dir():
            return candidate
    return root


def _feature_dim_from_tensor(tensor: torch.Tensor) -> int:
    return int(tensor.shape[0] if tensor.ndim == 1 else tensor.shape[-1])


def _materialize_pt_artifact(*, artifact_path: Path, output_path: Path) -> torch.Tensor:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.resolve() != output_path.resolve():
        if output_path.exists():
            output_path.unlink()
        try:
            os.link(artifact_path, output_path)
        except OSError:
            shutil.copyfile(artifact_path, output_path)
            with contextlib.suppress(OSError):
                artifact_path.unlink()
    return torch.load(output_path, weights_only=True, map_location="cpu")


def write_feature_payload(
    *,
    feature_dir: Path,
    sample_id: str,
    tensor: torch.Tensor,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a sample-level feature tensor directly into a cache feature directory."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    output_path = feature_dir / f"{sample_id}.pt"
    with tempfile.NamedTemporaryFile(prefix=f".{sample_id}.", suffix=".pt", dir=feature_dir, delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
    try:
        torch.save(tensor.detach().cpu(), tmp_path)
        os.replace(tmp_path, output_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
    if metadata is not None:
        metadata_path = feature_dir / f"{sample_id}.meta.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def write_cache_payload(
    artifacts: Sequence[object],
    *,
    feature_dir: Path,
    id_attr: str = "sample_id",
) -> int | None:
    """Write slide2vec artifacts to a soma cache directory as .pt files.

    Args:
        id_attr: Attribute on each artifact used as the filename stem. Defaults to
            ``"sample_id"``; pass ``"patient_id"`` for PatientEmbeddingArtifact.
    """
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_dim: int | None = None
    for artifact in artifacts:
        artifact_path = Path(artifact.path)
        artifact_id = getattr(artifact, id_attr)
        output_path = feature_dir / f"{artifact_id}.pt"
        if artifact_path.suffix != ".pt" or not artifact_path.is_file():
            raise ValueError(f"Expected a .pt artifact for cache materialization, got: {artifact_path}")
        tensor = _materialize_pt_artifact(
            artifact_path=artifact_path,
            output_path=output_path,
        )
        feature_dim = _feature_dim_from_tensor(tensor)
    return feature_dim


def build_tile_artifacts_from_cache_payload(
    *,
    features_dir: Path,
    loaded_tilings: Sequence[object],
    work_dir: Path,
    feature_path_by_sample_id: dict[str, Path] | None = None,
) -> list[TileEmbeddingArtifact]:
    """Reconstruct TileEmbeddingArtifact objects from cached .pt files."""
    work_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[TileEmbeddingArtifact] = []
    for loaded in loaded_tilings:
        sample_id = str(loaded.slide.sample_id)
        if feature_path_by_sample_id is not None:
            feature_path = feature_path_by_sample_id[sample_id]
        else:
            feature_path = features_dir / f"{sample_id}.pt"
        tensor = torch.load(feature_path, weights_only=True, map_location="cpu")
        metadata_path = work_dir / f"{sample_id}.meta.json"
        metadata = {
            "sample_id": sample_id,
            "artifact_type": "tile_embeddings",
            "format": "pt",
            "feature_dim": _feature_dim_from_tensor(tensor),
            "num_tiles": int(tensor.shape[0]),
            "image_path": str(loaded.slide.image_path),
            "mask_path": str(loaded.slide.mask_path) if loaded.slide.mask_path is not None else "",
            "coordinates_npz_path": str(getattr(loaded.tiling_result, "coordinates_npz_path", "")),
            "coordinates_meta_path": str(getattr(loaded.tiling_result, "coordinates_meta_path", "")),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        artifacts.append(
            TileEmbeddingArtifact(
                sample_id=sample_id,
                path=feature_path,
                metadata_path=metadata_path,
                format="pt",
                feature_dim=_feature_dim_from_tensor(tensor),
                num_tiles=int(tensor.shape[0]),
            )
        )
    return artifacts


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "image_path", "mask_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized_manifest_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        mask_path = row.get("mask_path")
        mask_text = ""
        if mask_path is not None:
            mask_text = str(mask_path)
            if mask_text.lower() == "nan":
                mask_text = ""
        normalized.append(
            {
                "sample_id": str(row["sample_id"]),
                "image_path": str(row["image_path"]),
                "mask_path": mask_text,
            }
        )
    return sorted(normalized, key=lambda row: row["sample_id"])


def _format_cache_metadata_mismatch(
    *,
    cache_label: str,
    cache_dir: Path,
    existing: dict[str, Any],
    expected: dict[str, Any],
    ignore_keys: set[str] | frozenset[str] = frozenset(),
) -> str:
    base_ignore_keys = {
        "backend",
        "backend_by_sample_id",
        "resolved_by_encoder_name",
        "requested_preprocessing",
    }
    effective_ignore_keys = base_ignore_keys | set(ignore_keys)
    comparable_existing = {
        key: value for key, value in existing.items() if key not in effective_ignore_keys
    }
    comparable_expected = {
        key: value for key, value in expected.items() if key not in effective_ignore_keys
    }
    if comparable_existing == comparable_expected:
        return ""

    all_keys = sorted(set(comparable_existing) | set(comparable_expected))
    missing_keys: list[str] = []
    extra_keys: list[str] = []
    changed_keys: list[str] = []
    for key in all_keys:
        if key not in comparable_existing:
            missing_keys.append(f"{key}={comparable_expected[key]!r}")
        elif key not in comparable_expected:
            extra_keys.append(f"{key}={comparable_existing[key]!r}")
        elif comparable_existing[key] != comparable_expected[key]:
            changed_keys.append(
                f"{key}: existing={comparable_existing[key]!r}, expected={comparable_expected[key]!r}"
            )

    sections: list[str] = []
    if missing_keys:
        sections.append(f"missing=[{', '.join(missing_keys)}]")
    if extra_keys:
        sections.append(f"extra=[{', '.join(extra_keys)}]")
    if changed_keys:
        sections.append(f"changed=[{'; '.join(changed_keys)}]")
    details = "; ".join(sections)
    return f"{cache_label} metadata mismatch for {cache_dir}: {details}"


def _emit_cache_state_log(
    *,
    cache_label: str,
    cache_dir: Path,
    complete: bool,
    partial: bool = False,
    complete_state: str = "hit",
    reason: str | None = None,
) -> None:
    cache_path = str(cache_dir.resolve())
    reporter = slide2vec_progress.get_progress_reporter()
    rich_viz = hasattr(reporter, "console") and hasattr(reporter, "progress")
    if complete:
        status = complete_state
        if rich_viz:
            status = f"\x1b[1;32m{complete_state}\x1b[0m"
        message = f"✓ {cache_label} cache {status}: {cache_path}"
    else:
        if partial:
            status = "partial"
            if rich_viz:
                status = "\x1b[1;33mpartial\x1b[0m"
            message = f"~ {cache_label} cache {status}: {cache_path}"
        else:
            status = "miss"
            if rich_viz:
                status = "\x1b[1;31mmiss\x1b[0m"
            message = f"✗ {cache_label} cache {status}: {cache_path}"
        if reason is not None:
            message = f"{message} ({reason})"
    slide2vec_progress.emit_progress_log(message)


def _emit_cache_resolve_log(
    *,
    cache_label: str,
    cache_dir: Path,
    key: str,
    scope_name: str,
    scope_count: int,
) -> None:
    slide2vec_progress.emit_progress_log(
        f"… resolving {cache_label} cache ({scope_name}={int(scope_count)}, key={str(key)[:16]}): {cache_dir.resolve()}"
    )


def _emit_cache_validation_log(
    *,
    cache_label: str,
    checked: int,
    total: int,
    stage: str = "progress",
) -> None:
    if total <= 0:
        return
    if stage == "start":
        message = f"… validating {cache_label} cache entries: 0/{total}"
    elif stage == "done":
        message = f"✓ validated {cache_label} cache entries: {checked}/{total}"
    else:
        message = f"… validating {cache_label} cache entries: {checked}/{total}"
    slide2vec_progress.emit_progress_log(message)
