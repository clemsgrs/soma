"""Multi-GPU helpers for tile-image feature extraction."""

from __future__ import annotations

import contextlib
import logging
import os
import queue as py_queue
import traceback
from pathlib import Path
from typing import Callable

import torch
from slide2vec.inference import load_model
from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx
from torch.utils.data import DataLoader

from soma.config import EncoderConfig
from soma.dataset import SampleRecord


_HF_TOKEN_NOTE_PREFIX = (
    "Note: Environment variable`HF_TOKEN` is set and is the current active token "
    "independently from the token you've just configured."
)


class _SuppressHfTokenNote(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(_HF_TOKEN_NOTE_PREFIX)


@contextlib.contextmanager
def _suppress_hf_login_token_note():
    logger = logging.getLogger("huggingface_hub._login")
    filter_ = _SuppressHfTokenNote()
    logger.addFilter(filter_)
    try:
        yield
    finally:
        logger.removeFilter(filter_)


def _atomic_save_tensor(tensor: torch.Tensor, path: Path, *, rank: int) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}.{rank}")
    torch.save(tensor, tmp_path)
    tmp_path.replace(path)


def _tile_extraction_shard_worker(rank: int, shared: dict[str, object], result_queue) -> None:
    from soma.tile_extraction import _TileImageDataset

    try:
        encoder = shared["encoder"]
        if not isinstance(encoder, EncoderConfig):
            raise TypeError("shared['encoder'] must be an EncoderConfig")

        records_by_rank = shared["records_by_rank"]
        shard_records = records_by_rank[rank]
        if not shard_records:
            load_events = shared.get("load_events")
            if isinstance(load_events, list) and rank + 1 < len(load_events):
                load_events[rank + 1].set()
            result_queue.put({"kind": "result", "written_ids": [], "feature_dim": None})
            return

        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        load_events = shared.get("load_events")
        if isinstance(load_events, list):
            load_events[rank].wait()
        with _suppress_hf_login_token_note():
            loaded = load_model(
                name=encoder.name,
                device=f"cuda:{rank}" if torch.cuda.is_available() else "cpu",
                output_variant=encoder.output_variant,
                allow_non_recommended_settings=encoder.allow_non_recommended_settings,
            )
        result_queue.put({"kind": "model_ready", "rank": rank, "device": str(loaded.device)})
        if isinstance(load_events, list) and rank + 1 < len(load_events):
            load_events[rank + 1].set()
        image_dataset = _TileImageDataset(shard_records, loaded.transforms)
        num_workers = int(shared["num_workers_per_gpu"])
        loader_kwargs = {
            "batch_size": int(shared["batch_size"]),
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = int(shared["prefetch_factor"])
        loader = DataLoader(image_dataset, **loader_kwargs)

        out_dir = Path(str(shared["out_dir"]))
        out_dir.mkdir(parents=True, exist_ok=True)
        precision = shared.get("precision")
        precision = str(precision) if precision is not None else None
        feature_torch_dtype = (
            torch.float16 if str(shared.get("feature_dtype", "fp32")) == "fp16" else torch.float32
        )
        written_ids: list[str] = []
        feature_dim: int | None = None
        with torch.inference_mode(), slide_encode_autocast_ctx(loaded.device, precision):
            for batch_images, batch_ids in loader:
                batch_images = batch_images.to(loaded.device, non_blocking=True)
                features = loaded.model.encode_tiles(batch_images).detach().to(feature_torch_dtype).cpu()
                if feature_dim is None:
                    feature_dim = int(features.shape[1])
                for feat, sample_id in zip(features, batch_ids):
                    sample_id = str(sample_id)
                    _atomic_save_tensor(feat, out_dir / f"{sample_id}.pt", rank=rank)
                    written_ids.append(sample_id)
                result_queue.put({"kind": "progress", "rank": rank, "count": len(batch_ids)})

        result_queue.put(
            {
                "kind": "result",
                "written_ids": written_ids,
                "feature_dim": feature_dim,
            }
        )
    except BaseException:
        load_events = shared.get("load_events")
        if isinstance(load_events, list):
            for event in load_events:
                event.set()
        result_queue.put(
            {
                "kind": "error",
                "rank": rank,
                "traceback": traceback.format_exc(),
            }
        )
        raise


def spawn_tile_feature_workers(
    *,
    num_workers: int,
    encoder: EncoderConfig,
    output_dir: Path,
    records_by_rank: list[list[SampleRecord]],
    batch_size: int,
    num_workers_per_gpu: int,
    prefetch_factor: int,
    precision: str | None,
    feature_dtype: str = "fp32",
    on_model_ready: Callable[[int, str], None] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[list[str], int | None]:
    def _terminate_workers() -> None:
        for process in getattr(process_ctx, "processes", []):
            if process.is_alive():
                process.terminate()

    def _handle_message(message: object) -> bool:
        nonlocal completed_workers, feature_dim
        if not isinstance(message, dict):
            return False
        kind = str(message.get("kind", ""))
        if kind == "model_ready":
            if on_model_ready is not None:
                on_model_ready(int(message.get("rank", -1)), str(message.get("device", "")))
            return False
        if kind == "error":
            rank = message.get("rank", "?")
            worker_traceback = str(message.get("traceback", "")).rstrip()
            _terminate_workers()
            raise RuntimeError(
                f"Tile feature worker {rank} failed before completing extraction:\n"
                f"{worker_traceback}"
            )
        if kind == "progress":
            if on_progress is not None:
                on_progress(int(message.get("rank", -1)), int(message.get("count", 0)))
            return False
        if kind == "result":
            completed_workers += 1
            written_ids.extend(str(sample_id) for sample_id in message.get("written_ids", []))
            shard_feature_dim = message.get("feature_dim")
            if feature_dim is None and shard_feature_dim is not None:
                feature_dim = int(shard_feature_dim)
            return True
        return False

    ctx = torch.multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    load_events = [ctx.Event() for _ in range(num_workers)]
    load_events[0].set()
    shared = {
        "encoder": encoder,
        "out_dir": str(output_dir),
        "records_by_rank": records_by_rank,
        "batch_size": int(batch_size),
        "num_workers_per_gpu": int(num_workers_per_gpu),
        "prefetch_factor": int(prefetch_factor),
        "precision": precision,
        "feature_dtype": feature_dtype,
        "load_events": load_events,
    }
    process_ctx = torch.multiprocessing.spawn(
        _tile_extraction_shard_worker,
        args=(shared, result_queue),
        nprocs=num_workers,
        join=False,
    )

    completed_workers = 0
    written_ids: list[str] = []
    feature_dim: int | None = None
    while completed_workers < num_workers:
        try:
            message = result_queue.get(timeout=0.2)
        except py_queue.Empty:
            try:
                joined = process_ctx.join(timeout=0.0)
            except Exception as exc:
                raise RuntimeError(
                    "A tile feature worker exited unexpectedly before reporting completion. "
                    "This is often caused by an OS or scheduler SIGKILL from memory pressure."
                ) from exc
            if not joined:
                continue
            while completed_workers < num_workers:
                try:
                    message = result_queue.get_nowait()
                except py_queue.Empty:
                    break
                _handle_message(message)
            if completed_workers < num_workers:
                break
            continue
        _handle_message(message)

    try:
        process_ctx.join()
    except Exception as exc:
        raise RuntimeError(
            "A tile feature worker exited unexpectedly during shutdown. "
            "This is often caused by an OS or scheduler SIGKILL from memory pressure."
        ) from exc
    if completed_workers < num_workers:
        raise RuntimeError("Tile feature worker exited before reporting completion")
    return written_ids, feature_dim
