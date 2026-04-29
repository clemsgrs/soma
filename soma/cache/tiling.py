"""Tiling cache: resolution, payload writing, and stub creation."""

from __future__ import annotations

import csv
import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

from slide2vec.utils.tiling_io import load_tiling_process_df

from soma.cache._types import (
    CACHE_METADATA_NAME,
    MANIFEST_NAME,
    PROCESS_LIST_NAME,
    SCHEMA_VERSION,
    CacheValidationResult,
    TilingCacheResolution,
)
from soma.cache.io import (
    _emit_cache_validation_log,
    _emit_cache_resolve_log,
    _emit_cache_state_log,
    _format_cache_metadata_mismatch,
    _load_metadata,
    _write_manifest,
    _write_metadata,
)
from soma.cache.keys import (
    _sample_stems_for_tiling,
    build_tiling_cache_key,
    dataset_manifest_rows,
    preprocessing_signature,
)
from soma.config import PreprocessingConfig
from soma.dataset import Dataset


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    value_str = str(value)
    if not value_str or value_str.lower() == "nan":
        return None
    return Path(value_str)


def _build_tiling_cache_metadata(
    *,
    preprocessing: PreprocessingConfig,
    backend_provenance: dict[str, Any],
    encoder_name: str | None = None,
    requested_preprocessing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "cache_kind": "tiling",
        "cache_key": build_tiling_cache_key(
            preprocessing=preprocessing,
        ),
        "preprocessing": preprocessing_signature(preprocessing),
        "requested_backend": str(backend_provenance["requested_backend"]),
    }
    if encoder_name is not None:
        metadata["resolved_by_encoder_name"] = str(encoder_name)
    if requested_preprocessing is not None:
        metadata["requested_preprocessing"] = requested_preprocessing
    return metadata


def _tiling_cache_dir(cache_root: Path, key: str) -> Path:
    return cache_root / key


def _canonical_artifact_destination(
    *,
    artifact_stem: str,
    column_name: str,
    source_path: Path,
    artifacts_dir: Path,
    previews_dir: Path,
) -> Path:
    if column_name == "mask_preview_path":
        return previews_dir / "mask" / f"{artifact_stem}.jpg"
    if column_name == "tiling_preview_path":
        return previews_dir / "tiling" / f"{artifact_stem}.jpg"
    if column_name == "coordinates_npz_path":
        return artifacts_dir / f"{artifact_stem}.coordinates.npz"
    if column_name == "coordinates_meta_path":
        return artifacts_dir / f"{artifact_stem}.coordinates.meta.json"
    suffix = "".join(source_path.suffixes) if source_path.suffixes else source_path.suffix
    stem = f"{artifact_stem}.{column_name.removesuffix('_path')}"
    return artifacts_dir / f"{stem}{suffix}"


def _copy_file_to_cache(*, source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _clear_directory_for_stub(tiling_dir: Path) -> None:
    tiling_dir.mkdir(parents=True, exist_ok=True)
    for path in list(tiling_dir.iterdir()):
        if path.name in {PROCESS_LIST_NAME, "README.txt"}:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _write_tiling_stub_marker(*, tiling_dir: Path, cache_dir: Path) -> None:
    (tiling_dir / "README.txt").write_text(
        (
            "This directory is a cache-backed tiling location placeholder.\n"
            f"Actual tiling payloads are stored under: {cache_dir.resolve()}\n"
            "Configure CacheConfig.root_dir to control the shared cache location.\n"
        ),
        encoding="utf-8",
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_tiling_cache_contents(
    *,
    dataset: Dataset,
    process_list_path: Path,
    artifacts_dir: Path,
    previews_dir: Path,
    cache_ids: Sequence[str],
    cache_stem_by_id: dict[str, str],
    preprocessing: PreprocessingConfig,
    expected_backend_provenance: dict[str, Any] | None,
) -> CacheValidationResult:
    total = len(cache_ids)
    _emit_cache_validation_log(cache_label="tiling", checked=0, total=total, stage="start")
    progress_interval = 100
    checked = 0
    if not process_list_path.is_file():
        return CacheValidationResult(complete=False, reason="missing process_list.csv")
    try:
        process_df = load_tiling_process_df(process_list_path)
    except Exception:
        logger.debug("Could not load process_list.csv at %s", process_list_path, exc_info=True)
        return CacheValidationResult(complete=False, reason="process_list.csv could not be loaded")
    rows = process_df.to_dict("records")
    rows_by_stem: dict[str, dict[str, Any]] = {}
    for row in rows:
        stem = row.get("sample_cache_stem")
        if stem is None or str(stem).strip() == "" or str(stem).lower() == "nan":
            stem = row.get("sample_id")
        if stem is None:
            continue
        rows_by_stem[str(stem)] = row

    for sample_id in cache_ids:
        checked += 1
        if checked % progress_interval == 0 or checked == total:
            _emit_cache_validation_log(cache_label="tiling", checked=checked, total=total)
        sample_id = str(sample_id)
        sample = dataset.samples[sample_id]
        row = rows_by_stem.get(str(cache_stem_by_id[sample_id]))
        if row is None:
            row = rows_by_stem.get(sample_id)
        if row is None or row.get("tiling_status") != "success":
            return CacheValidationResult(complete=False, reason=f"invalid tiling row for {sample_id}")
        for column_name in (
            "coordinates_npz_path",
            "coordinates_meta_path",
            "tiles_tar_path",
            "mask_preview_path",
            "tiling_preview_path",
        ):
            candidate = _optional_path(row.get(column_name))
            if candidate is None:
                continue
            if not candidate.is_file():
                return CacheValidationResult(complete=False, reason=f"missing artifact for {sample_id}")
            resolved_candidate = candidate.resolve()
            expected_root = previews_dir if column_name in {"mask_preview_path", "tiling_preview_path"} else artifacts_dir
            if not _is_relative_to(resolved_candidate, expected_root.resolve()):
                return CacheValidationResult(
                    complete=False,
                    reason=f"artifact path escapes cache entry for {sample_id}",
                )
        row_image_path = row.get("image_path")
        if row_image_path is not None and str(row_image_path) != str(sample.image_path):
            return CacheValidationResult(complete=False, reason=f"image path mismatch for {sample_id}")
        row_mask_path = row.get("mask_path")
        expected_mask_path = "" if sample.mask_path is None else str(sample.mask_path)
        if row_mask_path is not None:
            row_mask_str = str(row_mask_path)
            if row_mask_str.lower() == "nan":
                row_mask_str = ""
            if row_mask_str != expected_mask_path:
                return CacheValidationResult(complete=False, reason=f"mask path mismatch for {sample_id}")
        expected_requested_tile_size_px = (
            preprocessing.requested_region_size_px
            if preprocessing.requested_region_size_px is not None
            else preprocessing.requested_tile_size_px
        )
        row_tile_size = row.get("requested_tile_size_px")
        if row_tile_size is not None and str(row_tile_size).strip() not in {"", "nan", "NaN"}:
            if int(float(row_tile_size)) != int(expected_requested_tile_size_px):
                return CacheValidationResult(complete=False, reason=f"tile size mismatch for {sample_id}")
        row_spacing = row.get("requested_spacing_um")
        if row_spacing is not None and str(row_spacing).strip() not in {"", "nan", "NaN"}:
            if float(row_spacing) != float(preprocessing.requested_spacing_um):
                return CacheValidationResult(complete=False, reason=f"spacing mismatch for {sample_id}")
        expected_backend = None
        if expected_backend_provenance is not None:
            expected_backend = expected_backend_provenance.get("backend_by_sample_id", {}).get(str(sample_id))
        actual_backend = row.get("backend")
        if expected_backend is not None and str(expected_backend) != str(actual_backend):
            return CacheValidationResult(complete=False, reason=f"backend mismatch for {sample_id}")
    _emit_cache_validation_log(cache_label="tiling", checked=checked, total=total, stage="done")
    return CacheValidationResult(complete=True)


def resolve_tiling_cache(
    *,
    cache_root: Path,
    dataset: Dataset,
    preprocessing: PreprocessingConfig,
    backend_provenance: dict[str, Any],
    encoder_name: str | None = None,
    requested_preprocessing: dict[str, Any] | None = None,
    complete_state: str = "hit",
) -> TilingCacheResolution:
    metadata = _build_tiling_cache_metadata(
        preprocessing=preprocessing,
        backend_provenance=backend_provenance,
        encoder_name=encoder_name,
        requested_preprocessing=requested_preprocessing,
    )
    cache_ids = tuple(sorted(dataset.sample_ids))
    cache_stem_by_id = _sample_stems_for_tiling(
        dataset=dataset,
        cache_key=str(metadata["cache_key"]),
    )
    cache_dir = _tiling_cache_dir(cache_root, str(metadata["cache_key"]))
    metadata_path = cache_dir / CACHE_METADATA_NAME
    manifest_path = cache_dir / MANIFEST_NAME
    process_list_path = cache_dir / PROCESS_LIST_NAME
    artifacts_dir = cache_dir / "artifacts"
    previews_dir = cache_dir / "previews"
    cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)
    _emit_cache_resolve_log(
        cache_label="tiling",
        cache_dir=cache_dir,
        key=str(metadata["cache_key"]),
        scope_name="samples",
        scope_count=len(cache_ids),
    )

    if metadata_path.is_file():
        existing = _load_metadata(metadata_path)
        mismatch_message = _format_cache_metadata_mismatch(
            cache_label="Tiling cache",
            cache_dir=cache_dir,
            existing=existing,
            expected=metadata,
        )
        if mismatch_message:
            raise ValueError(mismatch_message)
        validation = _validate_tiling_cache_contents(
            dataset=dataset,
            process_list_path=process_list_path,
            artifacts_dir=artifacts_dir,
            previews_dir=previews_dir,
            cache_ids=cache_ids,
            cache_stem_by_id=cache_stem_by_id,
            preprocessing=preprocessing,
            expected_backend_provenance=backend_provenance,
        )
        _emit_cache_state_log(
            cache_label="tiling",
            cache_dir=cache_dir,
            complete=validation.complete,
            complete_state=complete_state,
            reason=validation.reason,
        )
        return TilingCacheResolution(
            key=str(existing["cache_key"]),
            cache_dir=cache_dir,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
            reused=validation.complete,
            complete=validation.complete,
            metadata=existing,
            process_list_path=process_list_path,
            artifacts_dir=artifacts_dir,
            cache_ids=cache_ids,
            cache_stem_by_id=cache_stem_by_id,
        )

    _write_manifest(manifest_path, dataset_manifest_rows(dataset))
    _write_metadata(metadata_path, metadata)
    _emit_cache_state_log(
        cache_label="tiling",
        cache_dir=cache_dir,
        complete=False,
        complete_state=complete_state,
        reason="initializing",
    )
    return TilingCacheResolution(
        key=str(metadata["cache_key"]),
        cache_dir=cache_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        reused=False,
        complete=False,
        metadata=metadata,
        process_list_path=process_list_path,
        artifacts_dir=artifacts_dir,
        cache_ids=cache_ids,
        cache_stem_by_id=cache_stem_by_id,
    )


def write_tiling_cache_payload(
    *,
    live_dir: Path,
    cache_resolution: TilingCacheResolution,
) -> None:
    process_df = load_tiling_process_df(live_dir / PROCESS_LIST_NAME)
    fieldnames = list(process_df.columns)
    if "sample_cache_stem" not in fieldnames:
        try:
            sample_idx = fieldnames.index("sample_id")
            fieldnames.insert(sample_idx + 1, "sample_cache_stem")
        except ValueError:
            fieldnames.insert(0, "sample_cache_stem")

    rows_by_stem: dict[str, dict[str, Any]] = {}
    if cache_resolution.process_list_path.is_file():
        with cache_resolution.process_list_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                stem = row.get("sample_cache_stem")
                if stem is None or str(stem).strip() == "" or str(stem).lower() == "nan":
                    stem = row.get("sample_id")
                if stem is None:
                    continue
                rows_by_stem[str(stem)] = dict(row)

    for row in process_df.to_dict("records"):
        rewritten = dict(row)
        sample_id = str(row["sample_id"])
        sample_cache_stem = cache_resolution.cache_stem_by_id.get(sample_id)
        if sample_cache_stem is None:
            continue
        rewritten["sample_cache_stem"] = sample_cache_stem
        for column_name in (
            "coordinates_npz_path",
            "coordinates_meta_path",
            "tiles_tar_path",
            "mask_preview_path",
            "tiling_preview_path",
        ):
            source_path = _optional_path(row.get(column_name))
            if source_path is None:
                rewritten[column_name] = None
                continue
            destination = _canonical_artifact_destination(
                artifact_stem=sample_cache_stem,
                column_name=column_name,
                source_path=source_path,
                artifacts_dir=cache_resolution.artifacts_dir,
                previews_dir=cache_resolution.previews_dir,
            )
            _copy_file_to_cache(source=source_path, destination=destination)
            rewritten[column_name] = str(destination.resolve())
        rows_by_stem[sample_cache_stem] = rewritten

    extra_columns: list[str] = []
    for row in rows_by_stem.values():
        for key in row:
            if key not in fieldnames and key not in extra_columns:
                extra_columns.append(key)
    resolved_fieldnames = [*fieldnames, *extra_columns]
    rows = [rows_by_stem[key] for key in sorted(rows_by_stem)]
    with cache_resolution.process_list_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tiling_cache_stub(
    tiling_dir: Path | str,
    *,
    cache_resolution: TilingCacheResolution,
) -> None:
    tiling_dir = Path(tiling_dir).resolve()
    _clear_directory_for_stub(tiling_dir)
    shutil.copyfile(cache_resolution.process_list_path, tiling_dir / PROCESS_LIST_NAME)
    _write_tiling_stub_marker(tiling_dir=tiling_dir, cache_dir=cache_resolution.cache_dir)
