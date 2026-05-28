"""slide2vec orchestration: model loading, tile embedding, aggregation, CUDA cleanup."""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import Sequence

import torch
import slide2vec.progress as slide2vec_progress
from hs2p.utils.stderr import run_with_filtered_stderr
from slide2vec import (
    ExecutionOptions,
    Model,
    Pipeline,
    PreprocessingConfig as Slide2VecPreprocessingConfig,
)
from slide2vec.runtime.embedding_persist import persist_embedded_slide as _persist_embedded_slide
from slide2vec.runtime.embedding_pipeline import compute_embedded_slides as _compute_embedded_slides

from soma.extraction.process_list import (
    _normalize_process_list_for_embedding,
    _restore_process_list_after_embedding,
)


def _load_model(
    model_name: str,
    *,
    output_variant: str | None,
    allow_non_recommended_settings: bool,
) -> Model:
    return Model.from_preset(
        model_name,
        output_variant=output_variant,
        allow_non_recommended_settings=allow_non_recommended_settings,
    )


def _release_parent_cuda_state() -> None:
    """Flush stale CUDA allocations in the parent before spawning distributed workers."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def _embed_tiles(
    *,
    model_name: str,
    output_variant: str,
    allow_non_recommended_settings: bool = False,
    slides: Sequence[object],
    tiling_results: Sequence[object],
    preprocessing: Slide2VecPreprocessingConfig,
    execution: ExecutionOptions,
) -> list:
    slide_records = list(slides)
    resolved_tilings = list(tiling_results)
    artifacts: list[object] = []

    def _run_embedding() -> None:
        model = _load_model(
            model_name,
            output_variant=output_variant,
            allow_non_recommended_settings=allow_non_recommended_settings,
        )

        def _on_embedded_slide(slide, tiling_result, embedded_slide) -> None:
            tile_or_hier_artifact, _slide_artifact = _persist_embedded_slide(
                model,
                embedded_slide,
                tiling_result,
                preprocessing=preprocessing,
                execution=execution,
            )
            if tile_or_hier_artifact is not None:
                artifacts.append(tile_or_hier_artifact)

        slide2vec_progress.emit_progress("embedding.started", slide_count=len(slide_records))
        _compute_embedded_slides(
            model,
            slide_records,
            resolved_tilings,
            preprocessing=preprocessing,
            execution=execution,
            on_embedded_slide=_on_embedded_slide,
            collect_results=False,
        )
        slide2vec_progress.emit_progress(
            "embedding.finished",
            slide_count=len(slide_records),
            slides_completed=len(slide_records),
            tile_artifacts=len(artifacts),
            slide_artifacts=0,
        )

    run_with_filtered_stderr(_run_embedding)
    return artifacts


def _run_with_coordinates(
    *,
    model_name: str,
    output_variant: str,
    allow_non_recommended_settings: bool = False,
    preprocessing: Slide2VecPreprocessingConfig,
    execution: ExecutionOptions,
    tiling_dir: Path,
    slides: Sequence[object],
):
    staged_process_list = Path(execution.output_dir) / "process_list.csv"
    source_process_list = tiling_dir / "process_list.csv"
    if source_process_list.is_file():
        _normalize_process_list_for_embedding(source_process_list)
        if not staged_process_list.exists():
            staged_process_list.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_process_list, staged_process_list)
    try:
        def _run_pipeline():
            _release_parent_cuda_state()
            return Pipeline(
                _load_model(
                    model_name,
                    output_variant=output_variant,
                    allow_non_recommended_settings=allow_non_recommended_settings,
                ),
                preprocessing,
                execution=execution,
            ).run_with_coordinates(
                tiling_dir,
                slides=list(slides),
            )

        return run_with_filtered_stderr(_run_pipeline)
    finally:
        if source_process_list.is_file():
            _restore_process_list_after_embedding(source_process_list)
            source_resolved = source_process_list.resolve()
            staged_resolved = staged_process_list.resolve()
            if source_resolved != staged_resolved:
                shutil.copyfile(source_process_list, staged_process_list)


def _require_artifact_list(result, attr_name: str, *, artifact_label: str) -> list:
    artifacts = getattr(result, attr_name, None)
    if artifacts is None:
        raise ValueError(f"slide2vec did not return {attr_name} for {artifact_label} extraction")
    return list(artifacts)


def _embed_tile_artifacts_with_coordinates(
    *,
    model_name: str,
    output_variant: str,
    allow_non_recommended_settings: bool = False,
    preprocessing: Slide2VecPreprocessingConfig,
    execution: ExecutionOptions,
    tiling_dir: Path,
    slides: Sequence[object],
) -> list:
    result = _run_with_coordinates(
        model_name=model_name,
        output_variant=output_variant,
        allow_non_recommended_settings=allow_non_recommended_settings,
        preprocessing=preprocessing,
        execution=execution,
        tiling_dir=tiling_dir,
        slides=slides,
    )
    return _require_artifact_list(result, "tile_artifacts", artifact_label="tile")


def _embed_hierarchical_artifacts_with_coordinates(
    *,
    model_name: str,
    output_variant: str,
    allow_non_recommended_settings: bool = False,
    preprocessing: Slide2VecPreprocessingConfig,
    execution: ExecutionOptions,
    tiling_dir: Path,
    slides: Sequence[object],
) -> list:
    result = _run_with_coordinates(
        model_name=model_name,
        output_variant=output_variant,
        allow_non_recommended_settings=allow_non_recommended_settings,
        preprocessing=preprocessing,
        execution=execution,
        tiling_dir=tiling_dir,
        slides=slides,
    )
    return _require_artifact_list(result, "hierarchical_artifacts", artifact_label="hierarchical")


def _aggregate_tiles(
    *,
    model_name: str,
    output_variant: str,
    allow_non_recommended_settings: bool = False,
    tile_artifacts,
    preprocessing: Slide2VecPreprocessingConfig | None,
    execution: ExecutionOptions,
):
    model = _load_model(
        model_name,
        output_variant=output_variant,
        allow_non_recommended_settings=allow_non_recommended_settings,
    )
    return model.aggregate_tiles(
        tile_artifacts,
        preprocessing=preprocessing,
        execution=execution,
    )


def _aggregate_patients(
    *,
    model_name: str,
    output_variant: str,
    allow_non_recommended_settings: bool = False,
    tile_artifacts,
    patient_id_map: dict[str, str],
    preprocessing: Slide2VecPreprocessingConfig | None,
    slide_execution: ExecutionOptions,
    patient_execution: ExecutionOptions,
):
    """Aggregate per-slide tile artifacts into patient-level embeddings.

    Two-phase process:
    1. Run the model's slide encoder on each set of tile artifacts
       (using aggregate_tiles, which calls encode_slide).
    2. Group slide embeddings by patient_id and call encode_patient
       for each patient.

    Args:
        tile_artifacts: List of TileEmbeddingArtifact objects per slide.
        patient_id_map: Mapping from sample_id to patient_id.
        slide_execution: ExecutionOptions with a temporary output_dir for
            intermediate slide embeddings.
        patient_execution: ExecutionOptions with output_dir for patient
            embedding artifacts.

    Returns:
        List of PatientEmbeddingArtifact objects, one per unique patient.
    """
    from slide2vec.artifacts import load_array, write_patient_embeddings

    model = _load_model(
        model_name,
        output_variant=output_variant,
        allow_non_recommended_settings=allow_non_recommended_settings,
    )

    # Step 1: Compute per-slide embeddings from tile artifacts.
    slide_artifacts = model.aggregate_tiles(
        tile_artifacts,
        preprocessing=preprocessing,
        execution=slide_execution,
    )

    # Step 2: Group slide embeddings by patient_id.
    patient_slide_embs: dict[str, list[torch.Tensor]] = {}
    for art in slide_artifacts:
        try:
            pid = patient_id_map[art.sample_id]
        except KeyError as exc:
            raise ValueError(
                f"Missing patient_id for sample '{art.sample_id}' during patient-level aggregation."
            ) from exc
        emb = load_array(art.path)
        if not torch.is_tensor(emb):
            emb = torch.as_tensor(emb)
        patient_slide_embs.setdefault(pid, []).append(emb)

    # Step 3: Patient encoding.
    loaded = model._load_backend()
    patient_artifacts = []
    for pid, slide_embs_list in patient_slide_embs.items():
        stacked = torch.stack(slide_embs_list, dim=0).to(loaded.device)
        with torch.inference_mode():
            patient_emb = loaded.model.encode_patient(stacked).detach().cpu()
        artifact = write_patient_embeddings(
            pid,
            patient_emb,
            output_dir=patient_execution.output_dir,
            output_format=patient_execution.output_format,
            metadata={"encoder_name": model_name, "encoder_level": "patient"},
            num_slides=len(slide_embs_list),
        )
        patient_artifacts.append(artifact)

    return patient_artifacts
