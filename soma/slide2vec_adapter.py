"""Helpers for driving slide2vec from soma."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch
from hs2p import SlideSpec
from hs2p.preprocessing import validate_tiling_result_provenance

from slide2vec import (
    ExecutionOptions,
    PreprocessingConfig as Slide2VecPreprocessingConfig,
)
import slide2vec.api as slide2vec_api
from slide2vec.utils.tiling_io import load_tiling_process_df, load_tiling_result_from_row

from soma.config import EncoderConfig, ExecutionConfig, PreprocessingConfig, PreviewConfig
from soma.dataset import Dataset, SampleRecord
from soma.encoders.validation import resolve_encoder_precision


@dataclass(frozen=True)
class LoadedTiling:
    slide: SlideSpec
    tiling_result: object
    requested_backend: str | None = None
    backend: str | None = None


def to_slide_spec(record: SampleRecord) -> SlideSpec:
    return SlideSpec(
        sample_id=record.sample_id,
        image_path=record.image_path,
        mask_path=record.mask_path,
        spacing_at_level_0=record.metadata.get("spacing_at_level_0"),
    )


def build_slide_specs(dataset: Dataset) -> list[SlideSpec]:
    return [to_slide_spec(record) for record in dataset.samples.values()]


def tiling_num_tiles(tiling_result: object) -> int:
    """Return the number of tiles recorded in a tiling result."""
    num_tiles = getattr(tiling_result, "num_tiles", None)
    if num_tiles is not None:
        return int(num_tiles)
    coordinates = getattr(tiling_result, "x", None)
    if coordinates is not None:
        return int(len(coordinates))
    return 0


def ensure_supported_mask_value(
    dataset: Dataset,
    preprocessing: PreprocessingConfig,
) -> None:
    if int(preprocessing.tissue_mask_tissue_value) == 1:
        return
    if not any(record.mask_path is not None for record in dataset.samples.values()):
        return
    raise ValueError(
        "slide2vec-backed tiling currently supports only tissue_mask_tissue_value=1 "
        "when mask_path is provided."
    )


def build_preprocessing_config(
    preprocessing: PreprocessingConfig,
) -> Slide2VecPreprocessingConfig:
    if preprocessing.requested_tile_size_px is None:
        raise ValueError("requested_tile_size_px must be resolved before extraction")
    if preprocessing.requested_spacing_um is None:
        raise ValueError("requested_spacing_um must be resolved before extraction")
    if preprocessing.tissue_method is None:
        raise ValueError(
            "tissue_method is required unless the dataset provides precomputed masks"
        )
    payload: dict[str, object] = {
        "backend": preprocessing.backend,
        "requested_spacing_um": float(preprocessing.requested_spacing_um),
        "requested_tile_size_px": int(preprocessing.requested_tile_size_px),
        "tolerance": float(preprocessing.tolerance),
        "overlap": float(preprocessing.overlap),
        "tissue_threshold": float(preprocessing.tissue_threshold),
        "on_the_fly": True,
        "adaptive_batching": False,
        "use_supertiles": True,
        "segmentation": {
            "method": preprocessing.tissue_method,
            "downsample": int(preprocessing.seg_downsample),
        },
        "filtering": {
            "ref_tile_size": int(
                preprocessing.ref_tile_size_px
                if preprocessing.ref_tile_size_px is not None
                else preprocessing.requested_tile_size_px
            ),
            "a_t": int(preprocessing.a_t),
        },
        "preview": asdict(preprocessing.preview),
    }
    if preprocessing.sam2_num_workers is not None:
        payload["segmentation"]["sam2_num_workers"] = int(preprocessing.sam2_num_workers)
    if preprocessing.sam2_device is not None:
        payload["segmentation"]["sam2_device"] = str(preprocessing.sam2_device)
    if preprocessing.requested_region_size_px is not None:
        payload["requested_region_size_px"] = int(preprocessing.requested_region_size_px)
    if preprocessing.region_tile_multiple is not None:
        payload["region_tile_multiple"] = int(preprocessing.region_tile_multiple)
    if preprocessing.read_tile_size_px is not None:
        payload["read_tile_size_px"] = int(preprocessing.read_tile_size_px)
    if preprocessing.read_region_size_px is not None:
        payload["read_region_size_px"] = int(preprocessing.read_region_size_px)
    allowed_fields = set(getattr(Slide2VecPreprocessingConfig, "__dataclass_fields__", {}))
    filtered = {key: value for key, value in payload.items() if key in allowed_fields}
    return Slide2VecPreprocessingConfig(**filtered)


def build_preview_config(preview: dict[str, object] | None = None) -> PreviewConfig:
    return PreviewConfig(**({} if preview is None else dict(preview)))


def build_execution_options(
    encoder: EncoderConfig,
    *,
    execution: ExecutionConfig | None = None,
    encoder_name: str | None = None,
    output_dir: Path,
    num_gpus: int | None,
    save_tile_embeddings: bool,
) -> ExecutionOptions:
    execution = execution or ExecutionConfig()
    num_gpus_value = num_gpus if num_gpus is not None else execution.num_gpus
    resolved_num_gpus = (
        int(num_gpus_value)
        if num_gpus_value is not None
        else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    )
    precision = execution.precision
    if precision is None:
        precision = resolve_encoder_precision(encoder, encoder_name=encoder_name)
    prefetch_factor = 4 if execution.prefetch_factor is None else int(execution.prefetch_factor)
    if execution.num_workers_per_gpu is None:
        num_workers_per_gpu = min(16, max(1, slide2vec_api.cpu_worker_limit() // max(1, resolved_num_gpus)))
    else:
        num_workers_per_gpu = int(execution.num_workers_per_gpu)
    return ExecutionOptions(
        output_dir=output_dir,
        output_format="pt",
        batch_size=int(encoder.batch_size),
        num_workers_per_gpu=num_workers_per_gpu,
        num_preprocessing_workers=execution.num_preprocessing_workers,
        num_gpus=num_gpus_value,
        precision=precision,
        prefetch_factor=prefetch_factor,
        save_tile_embeddings=save_tile_embeddings,
        save_slide_embeddings=False,
        save_latents=False,
    )


def load_tilings(
    *,
    dataset: Dataset,
    tiling_dir: Path,
    tissue_mask_tissue_value: int,
) -> list[LoadedTiling]:
    process_list_path = tiling_dir / "process_list.csv"
    if not process_list_path.is_file():
        raise ValueError(
            f"Tiling directory '{tiling_dir}' is missing process_list.csv"
        )
    process_df = load_tiling_process_df(process_list_path)
    rows_by_sample_id = {
        str(row["sample_id"]): row
        for row in process_df.to_dict("records")
    }
    loaded: list[LoadedTiling] = []
    for record in dataset.samples.values():
        row = rows_by_sample_id.get(record.sample_id)
        if row is None:
            raise ValueError(f"No tiling result found for sample_id={record.sample_id}")
        if row["tiling_status"] != "success":
            raise RuntimeError(
                f"Tiling failed for {record.sample_id}: {row.get('error', '')}"
            )
        tiling_result = load_tiling_result_from_row(row)
        requested_backend = str(getattr(tiling_result, "requested_backend", "auto"))
        actual_backend = str(getattr(tiling_result, "backend", requested_backend))
        validate_tiling_result_provenance(
            tiling_result,
            sample_id=record.sample_id,
            image_path=record.image_path,
            mask_path=record.mask_path,
            tissue_mask_tissue_value=(
                int(tissue_mask_tissue_value) if record.mask_path is not None else None
            ),
        )
        loaded.append(
            LoadedTiling(
                slide=to_slide_spec(record),
                tiling_result=tiling_result,
                requested_backend=requested_backend,
                backend=actual_backend,
            )
        )
    return loaded
