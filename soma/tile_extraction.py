"""TileFeatureExtractor — encodes individual tile images into 1D feature vectors."""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

import slide2vec.progress as slide2vec_progress
import torch
from PIL import Image
from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from slide2vec.inference import load_model

from soma.cache import (
    FeatureCacheResolution,
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_tile_cache,
)
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig
from soma.dataset import Dataset, SampleRecord
from soma.features import FeatureStore
from soma.slide2vec_adapter import build_execution_options
from soma.tile_extraction_spawn import spawn_tile_feature_workers


logger = logging.getLogger(__name__)


class _TileImageDataset(TorchDataset):
    """Internal dataset that loads tile images from disk and applies a transform."""

    def __init__(self, records: list[SampleRecord], transform) -> None:
        self._records = records
        self._transform = transform

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[object, str]:
        record = self._records[idx]
        with Image.open(record.image_path) as image:
            return self._transform(image.convert("RGB")), record.sample_id


@contextlib.contextmanager
def _make_tile_extraction_reporter_ctx(feature_dir: Path):
    active = slide2vec_progress.get_progress_reporter()
    if not isinstance(active, slide2vec_progress.NullProgressReporter):
        yield
        return

    reporter = slide2vec_progress.create_api_progress_reporter(output_dir=feature_dir)
    if isinstance(reporter, slide2vec_progress.NullProgressReporter):
        yield
        return

    with slide2vec_progress.activate_progress_reporter(reporter):
        yield


def _install_tile_embedding_summary_patch() -> None:
    """Ensure slide2vec embedding summaries can render tile-oriented labels."""
    if getattr(slide2vec_progress, "_soma_tile_embedding_summary_patch_installed", False):
        return

    base_summary_rows = getattr(slide2vec_progress, "_embedding_summary_rows", None)
    if not callable(base_summary_rows):
        return

    def _patched_embedding_summary_rows(payload: dict[str, object]) -> list[tuple[str, str]]:
        summary_subject = payload.get("summary_subject")
        if summary_subject is not None:
            subject = str(summary_subject).strip() or "Samples"
            if subject.lower() == "tiles":
                total = int(payload.get("tile_count", payload.get("slide_count", 0)))
                completed = int(payload.get("tiles_completed", payload.get("slides_completed", 0)))
            else:
                total = int(payload.get("slide_count", 0))
                completed = int(payload.get("slides_completed", 0))
            failed = max(0, total - completed)
            return [
                (subject, str(total)),
                ("Completed", str(completed)),
                ("Failed", str(failed)),
            ]

        return base_summary_rows(payload)

    setattr(slide2vec_progress, "_embedding_summary_rows", _patched_embedding_summary_rows)
    setattr(slide2vec_progress, "_soma_tile_embedding_summary_patch_installed", True)


class TileFeatureExtractor:
    """Encode individual tile images into 1D feature vectors using a tile encoder.

    Loads each sample's tile image, applies the encoder's transform, runs the
    tile encoder, and saves a 1D ``.pt`` feature vector per sample. This is the
    entry point for ``dataset_type="tile"`` pipelines.

    Args:
        dataset: Dataset whose ``image_path`` fields point to tile images.
        encoder: Encoder configuration (name, precision, batch_size, etc.).
        cache: Optional cache configuration. When enabled, features are stored
            in a content-addressed cache directory and reused across runs.
    """

    def __init__(
        self,
        dataset: Dataset,
        encoder: EncoderConfig,
        *,
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig | None = None,
    ) -> None:
        self._dataset = dataset
        self._encoder = encoder
        self._execution = execution
        self._cache = cache or CacheConfig(enabled=False)

    def run(self, feature_dir: str | Path) -> FeatureStore:
        """Encode all tile images and return a FeatureStore over the results.

        Args:
            feature_dir: Directory to write ``.pt`` feature files into.
                Ignored when a complete cache hit is found.

        Returns:
            FeatureStore pointing at the directory containing 1D ``.pt`` files.
        """
        feature_dir = Path(feature_dir).resolve()

        cache_resolution: FeatureCacheResolution | None = None
        if self._cache.enabled:
            cache_root = resolve_cache_root(
                self._cache,
                feature_dir=feature_dir,
            )
            cache_resolution = resolve_tile_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                preprocessing=None,
                execution=self._encoder,
                output_variant=self._encoder.output_variant,
                feature_type="tile",
                fingerprint_files=self._cache.fingerprint_files,
                validate_payloads=self._cache.validate_payloads,
            )
            if cache_resolution.complete:
                logger.info(
                    "Reusing cached tile features from %s",
                    cache_resolution.features_dir,
                )
                return FeatureStore(cache_resolution.features_dir)

        out_dir = cache_resolution.features_dir if cache_resolution is not None else feature_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        records = list(self._dataset.samples.values())
        total_samples = len(records)

        feature_dim: int | None = None

        _install_tile_embedding_summary_patch()
        with _make_tile_extraction_reporter_ctx(feature_dir):
            execution = build_execution_options(
                self._encoder,
                execution=self._execution,
                encoder_name=self._encoder.name,
                output_dir=out_dir,
                num_gpus=self._execution.num_gpus,
                save_tile_embeddings=True,
            )
            resolved_num_workers = execution.resolved_num_workers_per_gpu()
            tile_num_workers = resolved_num_workers
            if execution.num_gpus > 1 and self._execution.num_workers_per_gpu is None:
                tile_num_workers = 0
            worker_source = (
                "explicit ExecutionConfig.num_workers_per_gpu"
                if self._execution.num_workers_per_gpu is not None
                else (
                    "tile multi-gpu conservative default"
                    if execution.num_gpus > 1
                    else "slide2vec cpu_worker_limit()"
                )
            )
            slide2vec_progress.emit_progress_log(
                f"Tile DataLoader workers: {tile_num_workers} ({worker_source})"
            )
            embedding_label = "Embedding tiles"

            if execution.num_gpus > 1 and len(records) > 1:
                logger.info(
                    "Encoding %d tile images with '%s' across %d GPUs (precision=%s, batch_size=%d)...",
                    len(records),
                    self._encoder.name,
                    execution.num_gpus,
                    execution.precision,
                    self._encoder.batch_size,
                )
                num_workers = min(execution.num_gpus, len(records))
                for _ in range(num_workers):
                    slide2vec_progress.emit_progress("model.loading", model_name=self._encoder.name)
                records_by_rank: list[list[SampleRecord]] = [[] for _ in range(num_workers)]
                for idx, record in enumerate(records):
                    records_by_rank[idx % num_workers].append(record)

                processed_samples = 0
                processed_by_rank = [0 for _ in range(num_workers)]
                ready_ranks: set[int] = set()

                def _on_model_ready(rank: int, _device: str) -> None:
                    if rank < 0 or rank >= len(records_by_rank):
                        return
                    if rank in ready_ranks:
                        return
                    ready_ranks.add(rank)
                    slide2vec_progress.emit_progress(
                        "embedding.slide.started",
                        sample_id=embedding_label,
                        progress_label=f"GPU {rank}",
                        total_tiles=len(records_by_rank[rank]),
                    )
                    slide2vec_progress.emit_progress(
                        "model.ready",
                        model_name=self._encoder.name,
                        device=f"GPU {rank}",
                    )

                def _on_progress(rank: int, count: int) -> None:
                    nonlocal processed_samples
                    if rank < 0 or rank >= len(processed_by_rank):
                        return
                    processed_samples += int(count)
                    processed_by_rank[rank] += int(count)
                    slide2vec_progress.emit_progress(
                        "embedding.tile.progress",
                        sample_id=embedding_label,
                        progress_label=f"GPU {rank}",
                        processed=processed_by_rank[rank],
                        total=len(records_by_rank[rank]),
                        unit="tile",
                    )

                written_ids, feature_dim = spawn_tile_feature_workers(
                    num_workers=num_workers,
                    encoder=self._encoder,
                    output_dir=out_dir,
                    records_by_rank=records_by_rank,
                    batch_size=self._encoder.batch_size,
                    num_workers_per_gpu=tile_num_workers,
                    prefetch_factor=execution.prefetch_factor,
                    precision=execution.precision,
                    on_model_ready=_on_model_ready,
                    on_progress=_on_progress,
                )
                if set(written_ids) != {record.sample_id for record in records}:
                    missing = sorted({record.sample_id for record in records} - set(written_ids))
                    raise RuntimeError(
                        "Multi-GPU tile extraction did not produce all expected feature files: "
                        f"missing={missing}"
                    )
                for rank, shard_records in enumerate(records_by_rank):
                    slide2vec_progress.emit_progress(
                        "embedding.slide.finished",
                        sample_id=embedding_label,
                        progress_label=f"GPU {rank}",
                        num_tiles=len(shard_records),
                    )
                slide2vec_progress.emit_progress(
                    "embedding.finished",
                    slide_count=1,
                    slides_completed=1,
                    summary_subject="Tiles",
                    tile_count=processed_samples,
                    tiles_completed=processed_samples,
                    tile_artifacts=processed_samples,
                    slide_artifacts=0,
                )
                logger.info("Saved tile features to %s (dim=%s)", out_dir, feature_dim)
                if cache_resolution is not None and feature_dim is not None:
                    record_feature_dim(cache_resolution, feature_dim)
                    record_sample_identity_signatures(
                        cache_resolution,
                        [record.sample_id for record in records],
                    )
                return FeatureStore(out_dir)

            logger.info("Loading tile encoder '%s'...", self._encoder.name)
            slide2vec_progress.emit_progress("model.loading", model_name=self._encoder.name)
            loaded = load_model(
                name=self._encoder.name,
                output_variant=self._encoder.output_variant,
                allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            )
            encoder = loaded.model
            transform = loaded.transforms
            device = loaded.device
            slide2vec_progress.emit_progress(
                "model.ready",
                model_name=self._encoder.name,
                device=str(device),
            )

            image_dataset = _TileImageDataset(records, transform)
            loader_kwargs = {
                "batch_size": self._encoder.batch_size,
                "shuffle": False,
                "num_workers": resolved_num_workers,
                "pin_memory": torch.cuda.is_available(),
            }
            if resolved_num_workers > 0:
                loader_kwargs["prefetch_factor"] = execution.prefetch_factor
            loader = DataLoader(image_dataset, **loader_kwargs)

            logger.info(
                "Encoding %d tile images with '%s' (precision=%s, batch_size=%d)...",
                len(records),
                self._encoder.name,
                execution.precision,
                self._encoder.batch_size,
            )
            slide2vec_progress.emit_progress(
                "embedding.slide.started",
                sample_id=embedding_label,
                total_tiles=total_samples,
            )

            processed_samples = 0
            with torch.inference_mode(), slide_encode_autocast_ctx(device, execution.precision):
                for batch_images, batch_ids in loader:
                    batch_images = batch_images.to(device, non_blocking=True)
                    features = encoder.encode_tiles(batch_images).detach().float().cpu()
                    if feature_dim is None:
                        feature_dim = features.shape[1]
                    for feat, sample_id in zip(features, batch_ids):
                        torch.save(feat, out_dir / f"{sample_id}.pt")
                    processed_samples += len(batch_ids)

                    slide2vec_progress.emit_progress(
                        "embedding.tile.progress",
                        sample_id=embedding_label,
                        processed=processed_samples,
                        total=total_samples,
                        unit="tile",
                    )

            slide2vec_progress.emit_progress(
                "embedding.slide.finished",
                sample_id=embedding_label,
                num_tiles=total_samples,
            )
            slide2vec_progress.emit_progress(
                "embedding.finished",
                slide_count=1,
                slides_completed=1,
                summary_subject="Tiles",
                tile_count=processed_samples,
                tiles_completed=processed_samples,
                tile_artifacts=processed_samples,
                slide_artifacts=0,
            )

        logger.info("Saved tile features to %s (dim=%s)", out_dir, feature_dim)

        if cache_resolution is not None and feature_dim is not None:
            record_feature_dim(cache_resolution, feature_dim)
            record_sample_identity_signatures(
                cache_resolution,
                [record.sample_id for record in records],
            )

        return FeatureStore(out_dir)
