"""ROI sampling cache: cross-run reuse of slide-manifest annotation-sampling coords.

The ``roi_sampling`` kind (sibling of ``tiling``) caches, per slide, the list of
level-0 integer ``(x, y)`` ROI origins that hs2p annotation sampling produced — the
entire contract the downstream slide-manifest dense path consumes. Layout is
``<cache_root>/roi_sampling/<key>/`` with ``cache_metadata.json``, ``manifest.csv``
(dataset rows, as in the other kinds) and one human-readable ``coords/<stem>.csv``
per slide. A slide hits iff its coords artifact loads successfully — loading *is*
the validation, so there is no ``validate_payloads`` plumbing for this kind; a
missing or unparseable file is a miss for that slide only, and a header-only CSV is
a legitimate cached zero-ROI answer.
"""

from __future__ import annotations

import csv
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any, Mapping, Sequence

from soma.cache._types import (
    CACHE_METADATA_NAME,
    MANIFEST_NAME,
    RoiSamplingCacheResolution,
)
from soma.cache.io import (
    _emit_cache_resolve_log,
    _emit_cache_state_log,
    _format_cache_metadata_mismatch,
    _load_metadata,
    _write_manifest,
    _write_metadata,
)
from soma.cache.keys import (
    _sample_stems_for_roi_sampling,
    build_roi_sampling_cache_key,
    dataset_manifest_rows,
    preprocessing_signature,
)
from soma.config import PreprocessingConfig

COORDS_DIR_NAME = "coords"
_COORDS_FIELDNAMES = ("x", "y")


def _build_roi_sampling_cache_metadata(
    *,
    preprocessing: PreprocessingConfig,
) -> dict[str, Any]:
    # No schema_version, matching the key payload (see build_roi_sampling_cache_key).
    return {
        "artifact_kind": "roi_sampling",
        "cache_key": build_roi_sampling_cache_key(preprocessing=preprocessing),
        "preprocessing": preprocessing_signature(preprocessing),
    }


def _load_coords_artifact(path: Path) -> list[tuple[int, int]] | None:
    """Load one per-slide coords CSV; ``None`` (a miss) on any failure.

    A header-only file loads as ``[]`` — "sampled, found nothing" is a cached answer.
    A file with no header row at all, a wrong header, or any non-integer row is
    unparseable, so the slide misses and gets re-sampled.
    """
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
            if header is None or tuple(header) != _COORDS_FIELDNAMES:
                return None
            coords: list[tuple[int, int]] = []
            for row in reader:
                if len(row) != 2:
                    return None
                coords.append((int(row[0]), int(row[1])))
    except (OSError, UnicodeDecodeError, ValueError, csv.Error):
        return None
    return coords


def resolve_roi_sampling_cache(
    *,
    cache_root: Path,
    dataset: Any,
    preprocessing: PreprocessingConfig,
) -> RoiSamplingCacheResolution:
    """Resolve the ``roi_sampling`` cache entry for ``dataset`` under ``cache_root``.

    Initializes a fresh directory (metadata + manifest) when none exists, hard-errors
    on a metadata mismatch (the existing cache-metadata contract), and otherwise loads
    every slide's coords artifact — hits land in ``coords_by_id``, the rest are misses
    to re-sample. ``dataset`` is duck-typed over ``samples``/``sample_ids`` so the
    segmentation slide manifest fits.
    """
    metadata = _build_roi_sampling_cache_metadata(preprocessing=preprocessing)
    key = str(metadata["cache_key"])
    cache_ids = tuple(sorted(str(sample_id) for sample_id in dataset.sample_ids))
    cache_stem_by_id = _sample_stems_for_roi_sampling(dataset)
    cache_dir = cache_root / "roi_sampling" / key
    metadata_path = cache_dir / CACHE_METADATA_NAME
    manifest_path = cache_dir / MANIFEST_NAME
    coords_dir = cache_dir / COORDS_DIR_NAME
    cache_dir.mkdir(parents=True, exist_ok=True)
    coords_dir.mkdir(parents=True, exist_ok=True)
    _emit_cache_resolve_log(
        cache_label="roi_sampling",
        cache_dir=cache_dir,
        key=key,
        scope_name="slides",
        scope_count=len(cache_ids),
    )

    if metadata_path.is_file():
        existing = _load_metadata(metadata_path)
        mismatch_message = _format_cache_metadata_mismatch(
            cache_label="ROI sampling cache",
            cache_dir=cache_dir,
            existing=existing,
            expected=metadata,
        )
        if mismatch_message:
            raise ValueError(mismatch_message)
        metadata = existing
    else:
        _write_manifest(manifest_path, dataset_manifest_rows(dataset))
        _write_metadata(metadata_path, metadata)

    coords_by_id: dict[str, list[tuple[int, int]]] = {}
    for cache_id in cache_ids:
        coords = _load_coords_artifact(coords_dir / f"{cache_stem_by_id[cache_id]}.csv")
        if coords is not None:
            coords_by_id[cache_id] = coords
    missing = len(cache_ids) - len(coords_by_id)
    complete = missing == 0
    _emit_cache_state_log(
        cache_label="roi_sampling",
        cache_dir=cache_dir,
        complete=complete,
        partial=bool(coords_by_id),
        reason=None if complete else f"{missing} slide(s) to sample",
    )
    return RoiSamplingCacheResolution(
        key=key,
        cache_dir=cache_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        reused=complete,
        complete=complete,
        metadata=metadata,
        coords_dir=coords_dir,
        cache_ids=cache_ids,
        cache_stem_by_id=cache_stem_by_id,
        coords_by_id=coords_by_id,
    )


def write_roi_sampling_coords(
    *,
    cache_resolution: RoiSamplingCacheResolution,
    coords_by_sample_id: Mapping[str, Sequence[tuple[int, int]]],
) -> None:
    """Publish fresh per-slide coords artifacts into a resolved cache entry.

    An empty coords list writes a real header-only artifact, so a zero-ROI slide is
    a hit on the next resolve. Each file lands atomically (temp file + ``os.replace``)
    so a crashed writer leaves a miss, never a half-written artifact that parses.
    """
    coords_dir = cache_resolution.coords_dir
    coords_dir.mkdir(parents=True, exist_ok=True)
    unknown = sorted(set(map(str, coords_by_sample_id)) - set(cache_resolution.cache_ids))
    if unknown:
        raise ValueError(
            f"Cannot write ROI sampling coords for sample_id(s) unknown to this cache "
            f"resolution: {unknown}"
        )
    for sample_id, coords in coords_by_sample_id.items():
        stem = cache_resolution.cache_stem_by_id[str(sample_id)]
        output_path = coords_dir / f"{stem}.csv"
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            prefix=f".{stem}.",
            suffix=".csv",
            dir=coords_dir,
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            writer = csv.writer(tmp_file)
            writer.writerow(_COORDS_FIELDNAMES)
            writer.writerows((int(x), int(y)) for x, y in coords)
        try:
            os.replace(tmp_path, output_path)
        finally:
            with suppress(FileNotFoundError):
                tmp_path.unlink()
