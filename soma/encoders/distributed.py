"""Distributed multi-GPU feature extraction with slide-level sharding."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.multiprocessing as mp

import soma.encoders.models  # noqa: F401
from soma.encoders.extraction import (
    extract_slide_features,
    extract_tile_features,
    save_features,
)
from soma.encoders.progress import (
    JsonlProgressReporter,
    NullProgressReporter,
    ProgressEvent,
)
from soma.encoders.registry import encoder_registry
from soma.preprocessing.tiling import TilingResult
from soma.wsi.reader import open_slide


@dataclass(frozen=True)
class SlideTask:
    """Picklable unit of work for one slide."""

    slide_path: str
    tiling_result: TilingResult
    slide_id: str


@dataclass(frozen=True)
class ExtractionSummary:
    """Summary of a distributed extraction run."""

    completed: list[str]
    skipped: list[str]
    failed: list[str]
    duration_s: float


@dataclass(frozen=True)
class _WorkerConfig:
    encoder_name: str
    output_dir: str
    batch_size: int
    num_workers: int
    precision: str
    use_supertiles: bool
    backend: str
    progress: bool
    error_handling: str
    save_tile_features: bool


def extract_dataset(
    encoder_name: str,
    slides: list[SlideTask],
    output_dir: Path,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    precision: str = "fp16",
    use_supertiles: bool = True,
    backend: str = "auto",
    skip_existing: bool = True,
    progress: bool = True,
    num_gpus: int | None = None,
    error_handling: str = "skip",
    save_tile_features: bool = False,
) -> ExtractionSummary:
    """Extract features for a dataset of slides, optionally across multiple GPUs."""
    t0 = time.monotonic()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    remaining: list[SlideTask] = []
    if skip_existing:
        for task in slides:
            if (output_dir / f"{task.slide_id}.pt").exists():
                skipped.append(task.slide_id)
            else:
                remaining.append(task)
    else:
        remaining = list(slides)

    if not remaining:
        return ExtractionSummary(
            completed=[],
            skipped=skipped,
            failed=[],
            duration_s=time.monotonic() - t0,
        )

    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    num_gpus = max(1, num_gpus)

    config = _WorkerConfig(
        encoder_name=encoder_name,
        output_dir=str(output_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
        use_supertiles=use_supertiles,
        backend=backend,
        progress=progress,
        error_handling=error_handling,
        save_tile_features=save_tile_features,
    )

    if num_gpus == 1:
        _worker_fn(0, [remaining], config)
    else:
        assignments = _assign_slides_to_ranks(remaining, num_gpus)
        mp.spawn(_worker_fn, args=(assignments, config), nprocs=num_gpus, join=True)

    completed: list[str] = []
    failed: list[str] = []
    progress_dir = Path(output_dir) / ".progress"
    for rank in range(num_gpus):
        summary_path = progress_dir / f"rank_{rank}_summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text())
            completed.extend(data.get("completed", []))
            failed.extend(data.get("failed", []))

    return ExtractionSummary(
        completed=completed,
        skipped=skipped,
        failed=failed,
        duration_s=time.monotonic() - t0,
    )


def _assign_slides_to_ranks(
    slides: list[SlideTask], num_ranks: int
) -> list[list[SlideTask]]:
    """Load-balanced bin-packing by tile count."""
    assignments: list[list[SlideTask]] = [[] for _ in range(num_ranks)]
    rank_loads = [0] * num_ranks
    sorted_slides = sorted(
        slides, key=lambda s: (-len(s.tiling_result.coordinates), s.slide_id)
    )
    for task in sorted_slides:
        target_rank = min(range(num_ranks), key=lambda r: (rank_loads[r], r))
        assignments[target_rank].append(task)
        rank_loads[target_rank] += len(task.tiling_result.coordinates)
    return assignments


def _worker_fn(
    rank: int,
    assignments: list[list[SlideTask]],
    config: _WorkerConfig,
) -> None:
    if torch.cuda.is_available() and torch.cuda.device_count() > rank:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    output_dir = Path(config.output_dir)
    reporter = (
        JsonlProgressReporter(output_dir, rank)
        if config.progress
        else NullProgressReporter()
    )
    my_slides = assignments[rank] if rank < len(assignments) else []

    metadata = encoder_registry.info(config.encoder_name)
    level = metadata.get("level", "tile")
    encoder_cls = encoder_registry.get(config.encoder_name)
    encoder = encoder_cls().to(device)
    tile_encoder = None
    if level == "slide":
        tile_encoder_name = metadata["tile_encoder"]
        tile_encoder_cls = encoder_registry.get(tile_encoder_name)
        tile_encoder = tile_encoder_cls().to(device)

    reporter.emit(
        ProgressEvent(
            kind="extraction.started",
            timestamp=time.time(),
            rank=rank,
            payload={
                "num_slides": len(my_slides),
                "num_ranks": len(assignments),
                "encoder_name": config.encoder_name,
                "level": level,
            },
        )
    )

    t0 = time.monotonic()
    completed: list[str] = []
    failed: list[str] = []

    for task in my_slides:
        num_tiles = len(task.tiling_result.coordinates)
        reporter.emit(
            ProgressEvent(
                kind="extraction.slide.started",
                timestamp=time.time(),
                rank=rank,
                payload={"slide_id": task.slide_id, "num_tiles": num_tiles},
            )
        )
        slide_t0 = time.monotonic()
        try:
            reader = open_slide(task.slide_path, backend=config.backend)
            try:
                if level == "slide":
                    assert tile_encoder is not None
                    result = extract_slide_features(
                        encoder,
                        tile_encoder,
                        reader,
                        task.tiling_result,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        precision=config.precision,
                        use_supertiles=config.use_supertiles,
                        return_tile_features=config.save_tile_features,
                    )
                    save_features(result.slide_features, output_dir, task.slide_id)
                    if config.save_tile_features and result.tile_features is not None:
                        save_features(
                            result.tile_features,
                            output_dir / "tile_features",
                            task.slide_id,
                            tile_index=task.tiling_result.tile_index,
                        )
                else:
                    features = extract_tile_features(
                        encoder,
                        reader,
                        task.tiling_result,
                        batch_size=config.batch_size,
                        num_workers=config.num_workers,
                        precision=config.precision,
                        use_supertiles=config.use_supertiles,
                    )
                    save_features(
                        features,
                        output_dir,
                        task.slide_id,
                        tile_index=task.tiling_result.tile_index,
                    )
            finally:
                reader.close()

            duration = time.monotonic() - slide_t0
            completed.append(task.slide_id)
            reporter.emit(
                ProgressEvent(
                    kind="extraction.slide.completed",
                    timestamp=time.time(),
                    rank=rank,
                    payload={
                        "slide_id": task.slide_id,
                        "num_tiles": num_tiles,
                        "duration_s": round(duration, 2),
                    },
                )
            )
        except Exception as exc:
            if config.error_handling == "raise":
                raise
            failed.append(task.slide_id)
            reporter.emit(
                ProgressEvent(
                    kind="extraction.slide.failed",
                    timestamp=time.time(),
                    rank=rank,
                    payload={"slide_id": task.slide_id, "error": str(exc)},
                )
            )

    total_duration = time.monotonic() - t0
    reporter.emit(
        ProgressEvent(
            kind="extraction.completed",
            timestamp=time.time(),
            rank=rank,
            payload={
                "num_completed": len(completed),
                "num_failed": len(failed),
                "duration_s": round(total_duration, 2),
            },
        )
    )

    progress_dir = output_dir / ".progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    (progress_dir / f"rank_{rank}_summary.json").write_text(
        json.dumps({"completed": completed, "failed": failed})
    )

    if hasattr(reporter, "close"):
        reporter.close()
