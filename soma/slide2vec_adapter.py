"""Helpers for driving slide2vec from soma."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hs2p import SlideSpec
from hs2p.preprocessing import validate_tiling_result_provenance

from slide2vec import (
    ExecutionOptions,
    PreprocessingConfig as Slide2VecPreprocessingConfig,
)
from slide2vec.utils.tiling_io import load_process_df, load_tiling_result_from_row

from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset, SampleRecord


@dataclass(frozen=True)
class LoadedTiling:
    slide: SlideSpec
    tiling_result: object


def to_slide_spec(record: SampleRecord) -> SlideSpec:
    return SlideSpec(
        sample_id=record.sample_id,
        image_path=record.image_path,
        mask_path=record.mask_path,
        spacing_at_level_0=record.metadata.get("spacing_at_level_0"),
    )


def build_slide_specs(dataset: Dataset) -> list[SlideSpec]:
    return [to_slide_spec(record) for record in dataset.samples.values()]


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
    *,
    backend: str,
) -> Slide2VecPreprocessingConfig:
    if preprocessing.target_tile_size_px is None:
        raise ValueError("target_tile_size_px must be resolved before extraction")
    if preprocessing.target_spacing_um is None:
        raise ValueError("target_spacing_um must be resolved before extraction")
    payload: dict[str, object] = {
        "backend": backend,
        "target_spacing_um": float(preprocessing.target_spacing_um),
        "target_tile_size_px": int(preprocessing.target_tile_size_px),
        "tolerance": float(preprocessing.tolerance),
        "overlap": float(preprocessing.overlap),
        "tissue_threshold": float(preprocessing.tissue_threshold),
        "on_the_fly": True,
        "adaptive_batching": False,
        "use_supertiles": True,
        "segmentation": {
            "downsample": int(preprocessing.seg_downsample),
            "use_hsv": preprocessing.tissue_method == "hsv",
        },
        "filtering": {
            "ref_tile_size": int(
                preprocessing.ref_tile_size_px
                if preprocessing.ref_tile_size_px is not None
                else preprocessing.target_tile_size_px
            ),
            "a_t": int(preprocessing.a_t),
        },
    }
    if preprocessing.target_region_size_px is not None:
        payload["target_region_size_px"] = int(preprocessing.target_region_size_px)
    if preprocessing.region_tile_multiple is not None:
        payload["region_tile_multiple"] = int(preprocessing.region_tile_multiple)
    if preprocessing.effective_tile_size_px is not None:
        payload["effective_tile_size_px"] = int(preprocessing.effective_tile_size_px)
    if preprocessing.effective_region_size_px is not None:
        payload["effective_region_size_px"] = int(preprocessing.effective_region_size_px)
    allowed_fields = set(getattr(Slide2VecPreprocessingConfig, "__dataclass_fields__", {}))
    filtered = {key: value for key, value in payload.items() if key in allowed_fields}
    return Slide2VecPreprocessingConfig(**filtered)


def build_execution_options(
    encoder: EncoderConfig,
    *,
    output_dir: Path,
    num_gpus: int | None,
    save_tile_embeddings: bool,
) -> ExecutionOptions:
    return ExecutionOptions(
        output_dir=output_dir,
        output_format="pt",
        batch_size=int(encoder.batch_size),
        num_workers=int(encoder.num_workers),
        num_preprocessing_workers=8,
        num_gpus=1 if num_gpus is None else int(num_gpus),
        precision=encoder.precision,
        save_tile_embeddings=save_tile_embeddings,
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
    process_df = load_process_df(process_list_path)
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
        validate_tiling_result_provenance(
            tiling_result,
            sample_id=record.sample_id,
            image_path=record.image_path,
            tissue_mask_path=record.mask_path,
            tissue_mask_tissue_value=(
                int(tissue_mask_tissue_value) if record.mask_path is not None else None
            ),
        )
        loaded.append(LoadedTiling(slide=to_slide_spec(record), tiling_result=tiling_result))
    return loaded
