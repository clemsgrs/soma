"""Reusable class-pixel counts for cached segmentation ROI sampling."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from hs2p.wsi.reader import resolve_backend

from soma.dataset import SampleRecord

logger = logging.getLogger(__name__)

_FLAT_MASK_SUFFIXES = {".png", ".jpg", ".jpeg"}
_MASK_READER_SCHEMA_VERSION = 1


@contextmanager
def _exclusive_lock(path: Path):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class SegmentationRoiPopulation:
    """An immutable ROI population aligned by sample ID."""

    sample_ids: tuple[str, ...]
    class_pixel_counts: tuple[tuple[int, ...], ...]
    num_classes: int
    cache_key: str
    cache_path: Path
    payload_sha256: str

    def subset(self, sample_ids: Sequence[str]) -> "SegmentationRoiPopulation":
        """Return counts aligned to the caller's requested sample order."""
        row_by_id = dict(zip(self.sample_ids, self.class_pixel_counts, strict=True))
        requested = tuple(str(sample_id) for sample_id in sample_ids)
        missing = [sample_id for sample_id in requested if sample_id not in row_by_id]
        if missing:
            raise ValueError(f"ROI population is missing sample IDs: {missing}")
        return SegmentationRoiPopulation(
            sample_ids=requested,
            class_pixel_counts=tuple(row_by_id[sample_id] for sample_id in requested),
            num_classes=self.num_classes,
            cache_key=self.cache_key,
            cache_path=self.cache_path,
            payload_sha256=self.payload_sha256,
        )

    def provenance(self) -> dict[str, object]:
        totals = [
            sum(row[class_index] for row in self.class_pixel_counts)
            for class_index in range(self.num_classes)
        ]
        return {
            "artifact_kind": "segmentation_roi_population",
            "cache_key": self.cache_key,
            "cache_path": str(self.cache_path),
            "payload_sha256": self.payload_sha256,
            "roi_count": len(self.sample_ids),
            "num_classes": self.num_classes,
            "class_pixel_totals": totals,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mask_source_fingerprint(path: str) -> dict[str, object]:
    source = Path(path)
    try:
        resolved = source.resolve(strict=True)
        stat = resolved.stat()
    except OSError:
        return {"path": path, "missing": True}
    return {
        "path": path,
        "resolved_path": str(resolved),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "sha256": _sha256_file(resolved),
    }


def _resolved_mask_backend(
    path: str,
    *,
    requested_backend: str,
    spacing_aware: bool,
) -> str:
    if not spacing_aware or Path(path).suffix.lower() in _FLAT_MASK_SUFFIXES:
        return "pil"
    if requested_backend != "auto":
        return requested_backend
    return str(resolve_backend("auto", wsi_path=Path(path)).backend)


def _load_population(
    path: Path, *, sample_ids: tuple[str, ...], num_classes: int, cache_key: str
):
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored_ids = tuple(str(sample_id) for sample_id in payload["sample_ids"])
        if len(stored_ids) != len(sample_ids) or set(stored_ids) != set(sample_ids):
            return None
        counts = tuple(tuple(int(value) for value in row) for row in payload["class_pixel_counts"])
        if len(counts) != len(stored_ids) or any(len(row) != num_classes for row in counts):
            return None
        if any(value < 0 for row in counts for value in row):
            return None
        return SegmentationRoiPopulation(
            sample_ids=stored_ids,
            class_pixel_counts=counts,
            num_classes=num_classes,
            cache_key=cache_key,
            cache_path=path,
            payload_sha256=_sha256_file(path),
        ).subset(sample_ids)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return None


def resolve_segmentation_roi_population(
    cache_root: str | Path,
    records: Sequence[SampleRecord],
    target_fn: Callable[[SampleRecord], dict[str, torch.Tensor]],
    *,
    num_classes: int,
    target_identity: Mapping[str, object],
    workers: int = 1,
    source_fingerprint_cache: MutableMapping[str, dict[str, object]] | None = None,
    resolved_backend_cache: MutableMapping[tuple[str, str, bool], str] | None = None,
) -> SegmentationRoiPopulation:
    """Load exact ROI class counts, or compute and publish them atomically once."""
    mask_paths = sorted(
        {
            str(record.label_mask_path)
            for record in records
            if record.label_mask_path is not None
        }
    )
    fingerprints = source_fingerprint_cache if source_fingerprint_cache is not None else {}
    for path in mask_paths:
        if path not in fingerprints:
            fingerprints[path] = _mask_source_fingerprint(path)
    requested_backend = str(target_identity.get("backend", "auto"))
    spacing_aware = target_identity.get("spacing_um") is not None
    backends = resolved_backend_cache if resolved_backend_cache is not None else {}
    for path in mask_paths:
        backend_key = (path, requested_backend, spacing_aware)
        if backend_key not in backends:
            backends[backend_key] = _resolved_mask_backend(
                path,
                requested_backend=requested_backend,
                spacing_aware=spacing_aware,
            )
    identity_payload = {
        "artifact_kind": "segmentation_roi_population",
        "mask_sources": [fingerprints[path] for path in mask_paths],
        "mask_reader": {
            "schema_version": _MASK_READER_SCHEMA_VERSION,
            "requested_backend": requested_backend,
            "backend_by_source": {
                path: backends[(path, requested_backend, spacing_aware)]
                for path in mask_paths
            },
        },
        "records": sorted(
            [
                {
                    "sample_id": record.sample_id,
                    "slide_id": record.slide_id,
                    "label_mask_path": (
                        None if record.label_mask_path is None else str(record.label_mask_path)
                    ),
                    "region": None if record.region is None else list(record.region),
                    "spacing_at_level_0": record.spacing_at_level_0,
                }
                for record in records
            ],
            key=lambda row: str(row["sample_id"]),
        ),
        "num_classes": int(num_classes),
        "target": dict(target_identity),
    }
    canonical_identity = json.dumps(
        identity_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    cache_key = hashlib.sha256(canonical_identity.encode("utf-8")).hexdigest()[:16]
    path = Path(cache_root) / cache_key / "population.json"
    sample_ids = tuple(record.sample_id for record in records)
    cached = _load_population(
        path, sample_ids=sample_ids, num_classes=num_classes, cache_key=cache_key
    )
    if cached is not None:
        logger.info("ROI population cache hit: %s (%d ROIs)", path, len(cached.sample_ids))
        return cached

    logger.info("Resolving ROI population cache: %s (%d ROIs)", path, len(records))
    with _exclusive_lock(path.with_suffix(f"{path.suffix}.lock")):
        cached = _load_population(
            path, sample_ids=sample_ids, num_classes=num_classes, cache_key=cache_key
        )
        if cached is not None:
            logger.info(
                "ROI population cache hit after wait: %s (%d ROIs)",
                path,
                len(cached.sample_ids),
            )
            return cached

        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")

        def count_record(record: SampleRecord) -> tuple[int, ...]:
            mask = target_fn(record)["mask"].reshape(-1)
            annotated = mask[(mask >= 0) & (mask < num_classes)].to(torch.int64)
            return tuple(
                int(value)
                for value in torch.bincount(annotated, minlength=num_classes)
            )

        def iter_rows():
            if workers == 1:
                yield from map(count_record, records)
                return
            # Python 3.11's Executor.map submits its entire iterable eagerly. Bound
            # submissions so a 125k-ROI population does not allocate 125k Futures.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                for start in range(0, len(records), 1000):
                    yield from executor.map(count_record, records[start : start + 1000])

        rows = []
        progress_every = max(1, min(1000, len(records)))
        for completed, row in enumerate(iter_rows(), start=1):
            rows.append(row)
            if completed % progress_every == 0 or completed == len(records):
                logger.info(
                    "Building ROI population: %d/%d masks counted",
                    completed,
                    len(records),
                )
        population = SegmentationRoiPopulation(
            sample_ids=sample_ids,
            class_pixel_counts=tuple(rows),
            num_classes=num_classes,
            cache_key=cache_key,
            cache_path=path,
            payload_sha256="",
        )

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(
                {
                    "sample_ids": list(population.sample_ids),
                    "class_pixel_counts": [list(row) for row in population.class_pixel_counts],
                },
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
        try:
            os.replace(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        population = replace(population, payload_sha256=_sha256_file(path))
        logger.info("Published ROI population cache: %s", path)
        return population


__all__ = ["SegmentationRoiPopulation", "resolve_segmentation_roi_population"]
