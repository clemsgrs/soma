"""Tests for soma.extract.slide2vec_adapter — the execution options and tilings soma hands slide2vec."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import slide2vec.progress as slide2vec_progress
import torch
from slide2vec import ExecutionOptions

from soma.data import ImageManifest
from slide2vec.encoders.registry import encoder_registry
from soma.extract.slide2vec_adapter import execution_options, load_tilings


TEST_TILE = "_cutover_tile"
TEST_SLIDE = "_cutover_slide"
TEST_PATIENT = "_cutover_patient"
TEST_MULTI = "_cutover_multi_spacing"
TEST_SLIDE_VARIANTS = "_cutover_slide_variants"


def register_test_encoders() -> None:
    if TEST_TILE not in encoder_registry:
        encoder_registry.register(
            TEST_TILE,
            object,
            metadata={
                "level": "tile",
                "input_size": 224,
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": 0.5,
                "precision": "fp16",
                "source": "test/cutover-tile",
            },
        )
    if TEST_SLIDE not in encoder_registry:
        encoder_registry.register(
            TEST_SLIDE,
            object,
            metadata={
                "level": "slide",
                "tile_encoder": TEST_TILE,
                "tile_encoder_output_variant": "default",
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": 0.5,
                "precision": "fp16",
                "source": "test/cutover-slide",
            },
        )
    if TEST_PATIENT not in encoder_registry:
        encoder_registry.register(
            TEST_PATIENT,
            object,
            metadata={
                "level": "patient",
                "tile_encoder": TEST_TILE,
                "tile_encoder_output_variant": "default",
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": 0.5,
                "precision": "fp16",
                "source": "test/cutover-patient",
            },
        )
    if TEST_SLIDE_VARIANTS not in encoder_registry:
        # Mirrors PRISM2: a slide encoder with a base and a wider named variant.
        encoder_registry.register(
            TEST_SLIDE_VARIANTS,
            object,
            metadata={
                "level": "slide",
                "tile_encoder": TEST_TILE,
                "tile_encoder_output_variant": "default",
                "output_variants": {"base": {"encode_dim": 8}, "wide": {"encode_dim": 16}},
                "default_output_variant": "base",
                "supported_spacing_um": 0.5,
                "precision": "fp16",
                "source": "test/cutover-slide-variants",
            },
        )
    if TEST_MULTI not in encoder_registry:
        encoder_registry.register(
            TEST_MULTI,
            object,
            metadata={
                "level": "tile",
                "input_size": 224,
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": [0.25, 0.5],
                "precision": "fp16",
                "source": "test/cutover-multi",
            },
        )


register_test_encoders()


@pytest.fixture(autouse=True)
def _force_cpu_for_default_extraction_paths():
    with patch("soma.extract.extractor.torch.cuda.is_available", return_value=False), patch(
        "soma.extract.extractor.torch.cuda.device_count",
        return_value=1,
    ):
        yield


def _make_dataset(tmp_path: Path, *, with_mask: bool = False) -> ImageManifest:
    rows: dict[str, list[object]] = {
        "sample_id": ["s0"],
        "image_path": [str(tmp_path / "s0.svs")],
        "label": ["tumor"],
    }
    if with_mask:
        rows["mask_path"] = [str(tmp_path / "s0-mask.tif")]
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return ImageManifest.from_csv(csv_path)


def test_execution_options_use_cpu_budget_for_workers(monkeypatch, tmp_path: Path):
    """An unset worker count keeps soma's loader budget: the CPU budget split across
    GPUs, capped at 16; slide2vec caps the tiling workers at the CPU budget."""
    import slide2vec.api as slide2vec_api
    from soma.extract import slide2vec_adapter as adapter

    monkeypatch.setattr(slide2vec_api, "cpu_worker_limit", lambda: 24)
    monkeypatch.setattr(slide2vec_api, "slurm_cpu_limit", lambda: 24)
    monkeypatch.setattr("slide2vec.api.cpu_worker_limit", lambda: 24)

    execution = adapter.execution_options(
        ExecutionOptions(num_gpus=1),
        encoder_name=TEST_TILE,
        output_dir=tmp_path,
        save_tile_embeddings=False,
    )

    assert execution.num_workers_per_gpu == 16  # capped at 16
    assert execution.num_preprocessing_workers == 24
    assert execution.prefetch_factor == 4
    assert execution.output_dir == tmp_path
    # The precision the cache key was derived at: the encoder's recommendation.
    assert execution.precision == "fp16"


def test_execution_options_forward_the_held_worker_tuning(tmp_path: Path):
    from soma.extract import slide2vec_adapter as adapter

    execution = adapter.execution_options(
        ExecutionOptions(num_workers_per_gpu=6, prefetch_factor=8, precision="fp32"),
        encoder_name=TEST_TILE,
        output_dir=tmp_path,
        save_tile_embeddings=True,
        output_dtype="fp16",
    )

    assert (execution.num_workers_per_gpu, execution.prefetch_factor) == (6, 8)
    assert (execution.precision, execution.output_dtype, execution.save_tile_embeddings) == (
        "fp32",
        "fp16",
        True,
    )


def test_execution_options_write_the_pt_payloads_the_cache_reads(tmp_path: Path):
    from soma.extract import slide2vec_adapter as adapter

    execution = adapter.execution_options(
        ExecutionOptions(output_format="npz"),
        encoder_name=TEST_TILE,
        output_dir=tmp_path,
        save_tile_embeddings=False,
    )

    assert execution.output_format == "pt"


# --- Annotation-restricted merged bag forwarding (#110) ---------------------------------


def test_load_tilings_skips_tissue_value_provenance_under_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """AC6: under annotation sampling, load_tilings passes tissue_mask_tissue_value=None to
    the hs2p provenance validator (the ==1 check is bypassed — pixel_mapping governs)."""
    dataset = _make_dataset(tmp_path, with_mask=True)
    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    (tiling_dir / "process_list.csv").write_text("sample_id\ns0,success\n", encoding="utf-8")

    process_df = pd.DataFrame([{"sample_id": "s0", "tiling_status": "success", "error": ""}])
    monkeypatch.setattr("soma.extract.slide2vec_adapter.load_tiling_process_df", lambda path: process_df)
    monkeypatch.setattr(
        "soma.extract.slide2vec_adapter.load_tiling_result_from_row",
        lambda row: SimpleNamespace(
            requested_backend="auto",
            backend="openslide",
            requested_seg_downsample=64,
            seg_downsample=64,
        ),
    )
    captured: dict[str, object] = {}

    def _fake_validate(result, *, sample_id, image_path, mask_path, tissue_mask_tissue_value):
        captured["tissue_mask_tissue_value"] = tissue_mask_tissue_value

    monkeypatch.setattr("soma.extract.slide2vec_adapter.validate_tiling_result_provenance", _fake_validate)

    load_tilings(
        images=dataset,
        tiling_dir=tiling_dir,
        requested_seg_downsample=64,
        annotation_sampling=True,
    )
    assert captured["tissue_mask_tissue_value"] is None


def test_load_tilings_records_requested_and_actual_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = _make_dataset(tmp_path, with_mask=True)
    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()

    process_df = pd.DataFrame(
        [
            {
                "sample_id": "s0",
                "image_path": str(tmp_path / "s0.svs"),
                "mask_path": str(tmp_path / "s0-mask.tif"),
                "tiling_status": "success",
                "num_tiles": 1,
                "coordinates_npz_path": str(tmp_path / "s0.npz"),
                "coordinates_meta_path": str(tmp_path / "s0.meta.json"),
                "error": "",
                "traceback": "",
            }
        ]
    )
    (tiling_dir / "process_list.csv").write_text("sample_id,tiling_status\ns0,success\n", encoding="utf-8")
    monkeypatch.setattr("soma.extract.slide2vec_adapter.load_tiling_process_df", lambda path: process_df)
    monkeypatch.setattr(
        "soma.extract.slide2vec_adapter.load_tiling_result_from_row",
        lambda row: SimpleNamespace(
            requested_backend="auto",
            backend="openslide",
            requested_seg_downsample=64,
            seg_downsample=32,
        ),
    )
    monkeypatch.setattr(
        "soma.extract.slide2vec_adapter.validate_tiling_result_provenance", lambda *args, **kwargs: None
    )

    loaded = load_tilings(
        images=dataset,
        tiling_dir=tiling_dir,
        requested_seg_downsample=64,
    )

    assert loaded[0].requested_backend == "auto"
    assert loaded[0].backend == "openslide"


def test_load_tilings_consumes_merged_mode_sampling_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """S1: an hs2p annotation-sampling tiling dir produced with output_mode=MERGED has
    exactly one row per slide (annotation None), so soma's per-slide extraction path consumes
    it unchanged — one LoadedTiling per slide, no per-(slide, annotation) collapse."""
    csv_path = tmp_path / "seg-dataset.csv"
    pd.DataFrame(
        [
            {"sample_id": "s0", "image_path": str(tmp_path / "s0.svs"), "mask_path": str(tmp_path / "s0-mask.tif"), "label": "x"},
            {"sample_id": "s1", "image_path": str(tmp_path / "s1.svs"), "mask_path": str(tmp_path / "s1-mask.tif"), "label": "x"},
        ]
    ).to_csv(csv_path, index=False)
    dataset = ImageManifest.from_csv(csv_path)

    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    # MERGED sampling dir: one row per slide, with the annotation column present but
    # empty (the merged per-slide result carries annotation=None).
    process_df = pd.DataFrame(
        [
            {
                "sample_id": sid,
                "image_path": str(tmp_path / f"{sid}.svs"),
                "mask_path": str(tmp_path / f"{sid}-mask.tif"),
                "annotation": None,
                "tiling_status": "success",
                "num_tiles": 3,
                "coordinates_npz_path": str(tmp_path / f"{sid}.npz"),
                "coordinates_meta_path": str(tmp_path / f"{sid}.meta.json"),
                "error": "",
                "traceback": "",
            }
            for sid in ("s0", "s1")
        ]
    )
    (tiling_dir / "process_list.csv").write_text(
        "sample_id,annotation,tiling_status\ns0,,success\ns1,,success\n", encoding="utf-8"
    )
    monkeypatch.setattr("soma.extract.slide2vec_adapter.load_tiling_process_df", lambda path: process_df)
    monkeypatch.setattr(
        "soma.extract.slide2vec_adapter.load_tiling_result_from_row",
        lambda row: SimpleNamespace(
            sample_id=str(row["sample_id"]),
            requested_backend="auto",
            backend="openslide",
            requested_seg_downsample=64,
            seg_downsample=64,
            num_tiles=int(row["num_tiles"]),
            annotation=None,
        ),
    )
    monkeypatch.setattr("soma.extract.slide2vec_adapter.validate_tiling_result_provenance", lambda *a, **k: None)

    loaded = load_tilings(
        images=dataset,
        tiling_dir=tiling_dir,
        requested_seg_downsample=64,
    )

    # one tiling per slide (not per (slide, annotation)); the existing per-slide extraction
    # path therefore yields one feature artifact per slide.
    assert [lt.slide.sample_id for lt in loaded] == ["s0", "s1"]
    assert all(lt.tiling_result.num_tiles == 3 for lt in loaded)


def test_execution_options_split_auto_workers_across_gpus(monkeypatch, tmp_path: Path):
    import slide2vec.api as slide2vec_api
    from soma.extract import slide2vec_adapter as adapter

    monkeypatch.setattr(slide2vec_api, "cpu_worker_limit", lambda: 20)

    execution = adapter.execution_options(
        ExecutionOptions(num_gpus=1),
        encoder_name=TEST_TILE,
        output_dir=tmp_path,
        num_gpus=2,
        save_tile_embeddings=False,
    )

    assert execution.num_gpus == 2
    assert execution.num_workers_per_gpu == 10
