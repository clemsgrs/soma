"""Multi-GPU slide-aggregation helpers using torch.multiprocessing.spawn."""

from __future__ import annotations

import json
import queue as py_queue
from pathlib import Path
from typing import Callable

import torch

from soma.extraction.orchestration import _aggregate_tiles


def _aggregate_slide_shard_worker(
    rank: int,
    shared: dict[str, object],
) -> tuple[list[str], int | None]:
    from slide2vec import ExecutionOptions
    from slide2vec.artifacts import TileEmbeddingArtifact

    shard_payloads = shared["shard_payloads_by_rank"][rank]
    if not shard_payloads:
        return [], None

    if torch.cuda.is_available():
        torch.cuda.set_device(int(rank))

    output_dir = Path(str(shared["output_dir"]))
    shard_dir = output_dir / f"tile_metadata_rank{int(rank)}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    tile_artifacts = []
    for payload in shard_payloads:
        sample_id = str(payload["sample_id"])
        feature_path = Path(payload["feature_path"])
        tensor = torch.load(feature_path, weights_only=True, map_location="cpu")
        feature_dim = int(tensor.shape[1])
        num_tiles = int(tensor.shape[0])
        metadata_path = shard_dir / f"{sample_id}.meta.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "artifact_type": "tile_embeddings",
                    "format": "pt",
                    "feature_dim": feature_dim,
                    "num_tiles": num_tiles,
                    "image_path": str(payload["image_path"]),
                    "mask_path": str(payload["mask_path"]),
                    "coordinates_npz_path": str(payload["coordinates_npz_path"]),
                    "coordinates_meta_path": str(payload["coordinates_meta_path"]),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        tile_artifacts.append(
            TileEmbeddingArtifact(
                sample_id=sample_id,
                path=feature_path,
                metadata_path=metadata_path,
                format="pt",
                feature_dim=feature_dim,
                num_tiles=num_tiles,
            )
        )

    slide_execution = ExecutionOptions(
        output_dir=output_dir,
        output_format="pt",
        batch_size=int(shared["execution_batch_size"]),
        num_workers_per_gpu=int(shared["execution_num_workers_per_gpu"]),
        num_preprocessing_workers=None,
        num_gpus=1,
        precision=shared["execution_precision"],
        prefetch_factor=int(shared["execution_prefetch_factor"]),
        save_tile_embeddings=False,
        save_slide_embeddings=False,
        save_latents=False,
    )
    slide_artifacts = _aggregate_tiles(
        model_name=str(shared["model_name"]),
        output_variant=shared["output_variant"],
        allow_non_recommended_settings=bool(shared["allow_non_recommended_settings"]),
        tile_artifacts=tile_artifacts,
        preprocessing=None,
        execution=slide_execution,
    )
    written_ids: list[str] = []
    feature_dim: int | None = None
    for artifact in slide_artifacts:
        written_ids.append(str(artifact.sample_id))
        if feature_dim is None:
            dim = getattr(artifact, "feature_dim", None)
            if dim is not None:
                feature_dim = int(dim)
    return written_ids, feature_dim


def _spawn_entrypoint(rank: int, packed: dict[str, object], result_queue) -> None:
    shard_written_ids, shard_feature_dim = _aggregate_slide_shard_worker(rank, packed)
    for _ in shard_written_ids:
        result_queue.put({"kind": "progress", "count": 1})
    result_queue.put(
        {
            "kind": "result",
            "written_ids": shard_written_ids,
            "feature_dim": shard_feature_dim,
        }
    )


def spawn_slide_aggregation_workers(
    *,
    num_workers: int,
    model_name: str,
    output_variant: str | None,
    allow_non_recommended_settings: bool,
    execution_precision: str | None,
    execution_batch_size: int,
    execution_num_workers_per_gpu: int,
    execution_prefetch_factor: int,
    output_dir: Path,
    shard_payloads_by_rank: list[list[dict[str, str]]],
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[set[str], int | None]:
    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    shared = {
        "model_name": model_name,
        "output_variant": output_variant,
        "allow_non_recommended_settings": allow_non_recommended_settings,
        "execution_precision": execution_precision,
        "execution_batch_size": execution_batch_size,
        "execution_num_workers_per_gpu": execution_num_workers_per_gpu,
        "execution_prefetch_factor": execution_prefetch_factor,
        "output_dir": str(output_dir),
        "shard_payloads_by_rank": shard_payloads_by_rank,
    }

    process_ctx = torch.multiprocessing.spawn(
        _spawn_entrypoint,
        args=(shared, result_queue),
        nprocs=num_workers,
        join=False,
    )

    total_slides = sum(len(shard_payload) for shard_payload in shard_payloads_by_rank)
    processed_slides = 0
    completed_workers = 0
    written_ids: set[str] = set()
    feature_dim: int | None = None
    while completed_workers < num_workers:
        try:
            message = result_queue.get(timeout=0.2)
        except py_queue.Empty:
            continue
        if not isinstance(message, dict):
            continue
        kind = str(message.get("kind", ""))
        if kind == "progress":
            processed_slides += int(message.get("count", 0))
            if on_progress is not None:
                on_progress(processed_slides, total_slides)
            continue
        if kind == "result":
            completed_workers += 1
            shard_written_ids = message.get("written_ids", [])
            shard_feature_dim = message.get("feature_dim")
            written_ids.update(str(sample_id) for sample_id in shard_written_ids)
            if feature_dim is None and shard_feature_dim is not None:
                feature_dim = int(shard_feature_dim)

    process_ctx.join()
    return written_ids, feature_dim
