"""Tests for soma.extraction — slide2vec-backed extraction."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import torch
from hs2p import SlideSpec
from slide2vec import ExecutionOptions

from soma.cache import CacheConfig
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from slide2vec.encoders.registry import encoder_registry
from soma.extraction import FeatureExtractor, _run_with_coordinates
from soma.slide2vec_adapter import LoadedTiling
from soma.slide2vec_adapter import load_tilings


_TEST_TILE = "_cutover_tile"
_TEST_SLIDE = "_cutover_slide"
_TEST_MULTI = "_cutover_multi_spacing"


def _register_test_encoders() -> None:
    if _TEST_TILE not in encoder_registry:
        encoder_registry.register(
            _TEST_TILE,
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
    if _TEST_SLIDE not in encoder_registry:
        encoder_registry.register(
            _TEST_SLIDE,
            object,
            metadata={
                "level": "slide",
                "tile_encoder": _TEST_TILE,
                "tile_encoder_output_variant": "default",
                "output_variants": {"default": {"encode_dim": 8}},
                "default_output_variant": "default",
                "supported_spacing_um": 0.5,
                "precision": "fp16",
                "source": "test/cutover-slide",
            },
        )
    if _TEST_MULTI not in encoder_registry:
        encoder_registry.register(
            _TEST_MULTI,
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


_register_test_encoders()


@pytest.fixture(autouse=True)
def _force_cpu_for_default_extraction_paths():
    with patch("soma.extraction.torch.cuda.is_available", return_value=False), patch(
        "soma.extraction.torch.cuda.device_count",
        return_value=1,
    ):
        yield


def _make_dataset(tmp_path: Path, *, with_mask: bool = False) -> Dataset:
    rows: dict[str, list[object]] = {
        "sample_id": ["s0"],
        "image_path": [str(tmp_path / "s0.svs")],
        "label": ["tumor"],
    }
    if with_mask:
        rows["mask_path"] = [str(tmp_path / "s0-mask.tif")]
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _tiling(sample_id: str = "s0") -> object:
    return SimpleNamespace(
        x=torch.tensor([0, 224], dtype=torch.int64).numpy(),
        y=torch.tensor([0, 224], dtype=torch.int64).numpy(),
        tissue_fractions=torch.tensor([1.0, 1.0], dtype=torch.float32).numpy(),
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        effective_tile_size_px=224,
        effective_spacing_um=0.5,
        tile_size_lv0=224,
        read_level=0,
        use_padding=True,
        is_within_tolerance=True,
        sample_id=sample_id,
        coordinates_npz_path=Path("/tmp/coords.npz"),
        coordinates_meta_path=Path("/tmp/coords.meta.json"),
    )


def _artifact(
    *,
    sample_id: str,
    output_dir: Path,
    kind: str,
    tensor: torch.Tensor,
):
    kind_dir = output_dir / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    path = kind_dir / f"{sample_id}.pt"
    meta_path = kind_dir / f"{sample_id}.meta.json"
    torch.save(tensor, path)
    meta_path.write_text("{}", encoding="utf-8")
    if kind == "tile_embeddings":
        return SimpleNamespace(
            sample_id=sample_id,
            path=path,
            metadata_path=meta_path,
            format="pt",
            feature_dim=int(tensor.shape[1]),
            num_tiles=int(tensor.shape[0]),
        )
    if kind == "hierarchical_embeddings":
        return SimpleNamespace(
            sample_id=sample_id,
            path=path,
            metadata_path=meta_path,
            format="pt",
            feature_dim=int(tensor.shape[-1]),
            num_regions=int(tensor.shape[0]),
            tiles_per_region=int(tensor.shape[1]),
        )
    return SimpleNamespace(
        sample_id=sample_id,
        path=path,
        metadata_path=meta_path,
        format="pt",
        feature_dim=int(tensor.shape[0]),
    )


def test_build_execution_options_uses_cpu_budget_for_tiling_workers(monkeypatch, tmp_path: Path):
    import slide2vec.api as slide2vec_api
    from soma import slide2vec_adapter as adapter

    monkeypatch.setattr(slide2vec_api, "cpu_worker_limit", lambda: 24)
    monkeypatch.setattr(slide2vec_api, "slurm_cpu_limit", lambda: 24)

    execution = adapter.build_execution_options(
        EncoderConfig(name=_TEST_TILE),
        output_dir=tmp_path,
        num_gpus=None,
        save_tile_embeddings=False,
    )

    assert execution.num_workers is None
    assert execution.num_preprocessing_workers == 24


def test_build_execution_options_forwards_explicit_num_workers(tmp_path: Path):
    from soma import slide2vec_adapter as adapter

    execution = adapter.build_execution_options(
        EncoderConfig(name=_TEST_TILE, num_workers=6),
        output_dir=tmp_path,
        num_gpus=None,
        save_tile_embeddings=False,
    )

    assert execution.num_workers == 6


class _FakeInnerReporter:
    def __init__(self) -> None:
        self.events = []
        self.logs = []

    def emit(self, event) -> None:
        self.events.append(event)

    def write_log(self, message: str, *, stream=None) -> None:
        self.logs.append(message)

    def close(self) -> None:
        return None


def test_preprocess_delegates_to_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
    )
    with patch("soma.extraction.Pipeline", autospec=True) as MockPipeline:
        mock_instance = MockPipeline.return_value
        extractor.preprocess(tmp_path / "tiling")
    MockPipeline.assert_called_once()
    mock_instance.run.assert_called_once()


def test_preprocess_uses_configured_backend_by_default(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5, backend="openslide"),
    )

    captured = {}

    def _fake_build_preprocessing_config(preprocessing):
        captured["backend"] = preprocessing.backend
        return SimpleNamespace()

    with patch("soma.extraction.build_preprocessing_config", side_effect=_fake_build_preprocessing_config), patch(
        "soma.extraction.Pipeline", autospec=True
    ) as MockPipeline:
        mock_instance = MockPipeline.return_value
        extractor.preprocess(tmp_path / "tiling")

    assert captured["backend"] == "openslide"
    mock_instance.run.assert_called_once()


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
    monkeypatch.setattr("soma.slide2vec_adapter.load_tiling_process_df", lambda path: process_df)
    monkeypatch.setattr(
        "soma.slide2vec_adapter.load_tiling_result_from_row",
        lambda row: SimpleNamespace(requested_backend="auto", backend="openslide"),
    )
    monkeypatch.setattr("soma.slide2vec_adapter.validate_tiling_result_provenance", lambda *args, **kwargs: None)

    loaded = load_tilings(
        dataset=dataset,
        tiling_dir=tiling_dir,
        tissue_mask_tissue_value=1,
    )

    assert loaded[0].requested_backend == "auto"
    assert loaded[0].backend == "openslide"


def test_preprocess_forwards_live_tiling_progress_to_slide2vec_reporter(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
    )
    inner = _FakeInnerReporter()

    def _fake_pipeline_run(*args, **kwargs):
        from hs2p.progress import emit_progress

        emit_progress("tiling.started", total=2)
        emit_progress(
            "tiling.progress",
            total=2,
            completed=1,
            failed=0,
            pending=1,
            discovered_tiles=8,
        )
        emit_progress(
            "tiling.progress",
            total=2,
            completed=2,
            failed=0,
            pending=0,
            discovered_tiles=14,
        )
        emit_progress(
            "tiling.finished",
            total=2,
            completed=2,
            failed=0,
            pending=0,
            discovered_tiles=14,
            output_dir=str(tmp_path / "tiling"),
            process_list_path=str(tmp_path / "tiling" / "process_list.csv"),
            zero_tile_successes=0,
        )

    with patch("soma.extraction.Pipeline", autospec=True) as MockPipeline, patch(
        "slide2vec.progress.get_progress_reporter",
        return_value=inner,
    ):
        MockPipeline.return_value.run.side_effect = _fake_pipeline_run
        extractor.preprocess(tmp_path / "tiling")

    assert [
        (event.kind, event.payload)
        for event in inner.events
    ] == [
        (
            "tiling.progress",
            {
                "total": 2,
                "completed": 1,
                "failed": 0,
                "pending": 1,
                "discovered_tiles": 8,
            },
        ),
        (
            "tiling.progress",
            {
                "total": 2,
                "completed": 2,
                "failed": 0,
                "pending": 0,
                "discovered_tiles": 14,
            },
        ),
    ]


def test_preprocess_suppresses_cucim_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
    )
    caplog.set_level(logging.INFO)
    with patch("soma.extraction.Pipeline", autospec=True) as MockPipeline:
        mock_instance = MockPipeline.return_value

        def _fake_run(*args, **kwargs):
            logging.getLogger("cucim.core").info("decode noise")

        mock_instance.run.side_effect = _fake_run
        extractor.preprocess(tmp_path / "tiling")

    assert not any(record.name.startswith("cucim") for record in caplog.records)


def test_extract_tile_features_returns_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(*, model_name, output_variant, slides, tiling_results, preprocessing, execution):
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        for slide in slides:
            _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._embed_tiles",
        side_effect=_fake_embed_tiles,
    ) as embed_tiles:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")
    assert embed_tiles.called
    assert store.available_samples == ["s0"]
    assert store.is_slide_level is False
    assert store.load("s0").shape == (2, 8)


def test_extract_returns_manifest_aware_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling("s0"),
        ),
        LoadedTiling(
            slide=SlideSpec(sample_id="s1", image_path=Path("/tmp/s1.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                **{**_tiling("s1").__dict__, "num_tiles": 0}
            ),
        ),
    ]

    def _fake_embed_tiles(*, model_name, output_variant, slides, tiling_results, preprocessing, execution):
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        _artifact(sample_id="s0", output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._embed_tiles",
        side_effect=_fake_embed_tiles,
    ):
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert store.has_feature_manifest is True
    assert store.empty_feature_samples == ["s1"]
    assert store.expected_feature_samples == ["s0"]
    assert store.available_samples == ["s0"]


def test_extract_suppresses_cucim_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(*, model_name, output_variant, slides, tiling_results, preprocessing, execution):
        logging.getLogger("cucim.core").info("decode noise")
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        for slide in slides:
            _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))

    caplog.set_level(logging.INFO)
    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._embed_tiles",
        side_effect=_fake_embed_tiles,
    ):
        extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert not any(record.name.startswith("cucim") for record in caplog.records)


def test_soma_extraction_reporter_forwards_progress_events_to_inner_reporter():
    from slide2vec.progress import ProgressEvent
    from soma.extraction import _SomaExtractionReporter

    inner = _FakeInnerReporter()
    reporter = _SomaExtractionReporter(inner)

    reporter.emit(ProgressEvent(kind="embedding.started", payload={"slide_count": 2}))
    reporter.emit(
        ProgressEvent(
            kind="embedding.slide.started",
            payload={"sample_id": "slide-a", "total_tiles": 5, "progress_label": "cuda:0"},
        )
    )
    reporter.emit(
        ProgressEvent(
            kind="embedding.tile.progress",
            payload={
                "sample_id": "slide-a",
                "processed": 3,
                "total": 5,
                "unit": "tile",
                "progress_label": "cuda:0",
            },
        )
    )

    assert [event.kind for event in inner.events] == [
        "embedding.started",
        "embedding.slide.started",
        "embedding.tile.progress",
    ]


def test_extract_defaults_to_all_visible_gpus_for_multi_gpu_embedding(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_run_with_coordinates(*, model_name, output_variant, preprocessing, execution, tiling_dir, slides):
        tile_artifact = _artifact(
            sample_id="s0",
            output_dir=Path(execution.output_dir),
            kind="tile_embeddings",
            tensor=torch.ones(2, 8),
        )
        return SimpleNamespace(tile_artifacts=[tile_artifact], slide_artifacts=[])

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch("soma.extraction.torch.cuda.is_available", return_value=True), patch(
        "soma.extraction.torch.cuda.device_count",
        return_value=2,
    ), patch(
        "soma.extraction._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert run_with_coords.called
    assert run_with_coords.call_args.kwargs["execution"].num_gpus == 2
    assert run_with_coords.call_args.kwargs["execution"].output_dir.is_absolute()
    assert Path(run_with_coords.call_args.kwargs["tiling_dir"]).is_absolute()
    assert store.is_slide_level is False
    assert store.load("s0").shape == (2, 8)


def test_run_with_coordinates_stages_process_list_into_output_dir(tmp_path: Path):
    tiling_dir = tmp_path / "tiling"
    tiling_dir.mkdir()
    source_process_list = tiling_dir / "process_list.csv"
    source_process_list.write_text("sample_id,tiling_status\ns0,success\n", encoding="utf-8")

    execution = ExecutionOptions(
        output_dir=tmp_path / "features",
        num_gpus=2,
        output_format="pt",
    )

    updated_process_list = "sample_id,tiling_status,feature_status,feature_path\ns0,success,success,/tmp/features/s0.pt\n"

    with patch("soma.extraction.Pipeline", autospec=True) as MockPipeline, patch(
        "soma.extraction.Model.from_preset",
        return_value=object(),
    ):
        instance = MockPipeline.return_value

        def _fake_run_with_coordinates(coordinates_dir, *, slides):
            Path(coordinates_dir, "process_list.csv").write_text(updated_process_list, encoding="utf-8")
            return SimpleNamespace(tile_artifacts=[], slide_artifacts=[])

        instance.run_with_coordinates.side_effect = _fake_run_with_coordinates
        _run_with_coordinates(
            model_name=_TEST_TILE,
            output_variant="default",
            preprocessing=SimpleNamespace(),
            execution=execution,
            tiling_dir=tiling_dir,
            slides=[],
        )

    staged_process_list = execution.output_dir / "process_list.csv"
    assert source_process_list.read_text(encoding="utf-8") == updated_process_list
    assert staged_process_list.read_text(encoding="utf-8") == updated_process_list
    instance.run_with_coordinates.assert_called_once_with(tiling_dir, slides=[])


def test_extract_slide_features_returns_slide_embedding_store(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(*, model_name, output_variant, slides, tiling_results, preprocessing, execution):
        artifacts = []
        for slide in slides:
            artifacts.append(
                _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))
            )
        return artifacts

    def _fake_aggregate_tiles(*, model_name, output_variant, tile_artifacts, preprocessing, execution):
        for artifact in tile_artifacts:
            _artifact(sample_id=artifact.sample_id, output_dir=Path(execution.output_dir), kind="slide_embeddings", tensor=torch.ones(8))

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._embed_tiles",
        side_effect=_fake_embed_tiles,
    ), patch(
        "soma.extraction._aggregate_tiles",
        side_effect=_fake_aggregate_tiles,
    ):
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")
    assert store.available_samples == ["s0"]
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)


def test_cached_extract_writes_marker_file(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(*, model_name, output_variant, slides, tiling_results, preprocessing, execution):
        artifacts = []
        output_dir = Path(execution.output_dir) / "tile_embeddings"
        output_dir.mkdir(parents=True, exist_ok=True)
        for slide in slides:
            artifacts.append(
                _artifact(
                    sample_id=slide.sample_id,
                    output_dir=Path(execution.output_dir),
                    kind="tile_embeddings",
                    tensor=torch.ones(2, 8),
                )
            )
        return artifacts

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._embed_tiles",
        side_effect=_fake_embed_tiles,
    ):
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert store.available_samples == ["s0"]
    marker = tmp_path / "features" / "README.txt"
    assert marker.is_file()
    marker_text = marker.read_text(encoding="utf-8")
    assert "cache-backed feature location placeholder" in marker_text
    assert str(store.feature_dir.parent) in marker_text
    process_list = tmp_path / "features" / "process_list.csv"
    assert process_list.is_file()
    recorded = pd.read_csv(process_list).set_index("sample_id")
    assert list(recorded.index) == ["s0"]
    assert recorded.loc["s0", "feature_status"] == "success"
    assert recorded.loc["s0", "feature_path"] == str((store.feature_dir / "s0.pt").resolve())
    assert recorded.loc["s0", "num_tiles"] == 2
    assert recorded.loc["s0", "feature_rank"] == store.feature_rank
    assert recorded.loc["s0", "feature_dim"] == store.feature_dim


def test_slide_encoder_runtime_does_not_forward_output_variant_override(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_embed_tiles(*, model_name, output_variant, slides, tiling_results, preprocessing, execution):
        assert output_variant is None
        artifacts = []
        for slide in slides:
            artifacts.append(
                _artifact(sample_id=slide.sample_id, output_dir=Path(execution.output_dir), kind="tile_embeddings", tensor=torch.ones(2, 8))
            )
        return artifacts

    def _fake_aggregate_tiles(*, model_name, output_variant, tile_artifacts, preprocessing, execution):
        assert output_variant is None
        for artifact in tile_artifacts:
            _artifact(sample_id=artifact.sample_id, output_dir=Path(execution.output_dir), kind="slide_embeddings", tensor=torch.ones(8))

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ) as validate_runtime, patch(
        "soma.extraction._embed_tiles",
        side_effect=_fake_embed_tiles,
    ), patch(
        "soma.extraction._aggregate_tiles",
        side_effect=_fake_aggregate_tiles,
    ):
        extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert validate_runtime.call_args.kwargs["output_variant"] is None


def test_multi_gpu_uncached_extraction_uses_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_run_with_coordinates(*, model_name, output_variant, preprocessing, execution, tiling_dir, slides):
        tile_artifact = _artifact(
            sample_id="s0",
            output_dir=Path(execution.output_dir),
            kind="tile_embeddings",
            tensor=torch.ones(2, 8),
        )
        return SimpleNamespace(tile_artifacts=[tile_artifact], slide_artifacts=[])

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(
            tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )
    assert run_with_coords.called
    assert store.is_slide_level is False
    assert store.load("s0").shape == (2, 8)


def test_multi_gpu_slide_cache_population_uses_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_run_with_coordinates(*, model_name, output_variant, preprocessing, execution, tiling_dir, slides):
        output_dir = Path(execution.output_dir)
        tile_artifact = _artifact(sample_id="s0", output_dir=output_dir, kind="tile_embeddings", tensor=torch.ones(2, 8))
        slide_artifact = _artifact(sample_id="s0", output_dir=output_dir, kind="slide_embeddings", tensor=torch.ones(8))
        return SimpleNamespace(tile_artifacts=[tile_artifact], slide_artifacts=[slide_artifact])

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(
            tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )
    assert run_with_coords.called
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)
    assert any((cache_root / "tile").glob("*/features/s0.pt"))
    assert any((cache_root / "slide").glob("*/features/s0.pt"))


def test_hierarchical_tile_extraction_writes_native_embeddings(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(
            target_tile_size_px=224,
            target_spacing_um=0.5,
            target_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                effective_tile_size_px=224,
                effective_spacing_um=0.5,
                tile_size_lv0=224,
                read_level=0,
                use_padding=True,
                is_within_tolerance=True,
                sample_id="s0",
                x=torch.tensor([0], dtype=torch.int64).numpy(),
                y=torch.tensor([0], dtype=torch.int64).numpy(),
                tissue_fractions=torch.tensor([1.0], dtype=torch.float32).numpy(),
                coordinates_npz_path=Path("/tmp/coords.npz"),
                coordinates_meta_path=Path("/tmp/coords.meta.json"),
            ),
        )
    ]
    seen_hierarchical: list[bool] = []

    original_extract_uncached = FeatureExtractor._extract_uncached

    def _spy_extract_uncached(self, *, hierarchical=False, **kwargs):
        seen_hierarchical.append(hierarchical)
        output_dir = kwargs["output_dir"]
        _artifact(
            sample_id="s0",
            output_dir=Path(output_dir),
            kind="hierarchical_embeddings",
            tensor=torch.ones(1, 4, 8),
        )

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_extract_uncached",
        autospec=True,
        side_effect=_spy_extract_uncached,
    ) as extract_uncached:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")
    assert extract_uncached.called
    assert seen_hierarchical == [True]
    assert store.is_hierarchical is True
    assert store.load("s0").shape == (1, 4, 8)


def test_hierarchical_multi_gpu_uses_slide2vec_pipeline(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(
            target_tile_size_px=224,
            target_spacing_um=0.5,
            target_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(enabled=False),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                effective_tile_size_px=224,
                effective_spacing_um=0.5,
                tile_size_lv0=224,
                read_level=0,
                use_padding=True,
                is_within_tolerance=True,
                sample_id="s0",
                x=torch.tensor([0], dtype=torch.int64).numpy(),
                y=torch.tensor([0], dtype=torch.int64).numpy(),
                tissue_fractions=torch.tensor([1.0], dtype=torch.float32).numpy(),
                coordinates_npz_path=Path("/tmp/coords.npz"),
                coordinates_meta_path=Path("/tmp/coords.meta.json"),
            ),
        )
    ]

    def _fake_run_with_coordinates(*, model_name, output_variant, preprocessing, execution, tiling_dir, slides):
        artifact = _artifact(
            sample_id="s0",
            output_dir=Path(execution.output_dir),
            kind="hierarchical_embeddings",
            tensor=torch.ones(1, 4, 8),
        )
        return SimpleNamespace(hierarchical_artifacts=[artifact])

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch(
        "soma.extraction._run_with_coordinates",
        side_effect=_fake_run_with_coordinates,
    ) as run_with_coords:
        store = extractor.extract(
            tmp_path / "features",
            tiling_dir=tmp_path / "tiling",
            num_gpus=2,
        )
    assert run_with_coords.called
    assert store.is_hierarchical is True
    assert store.load("s0").shape == (1, 4, 8)


def test_hierarchical_cache_population_uses_native_cache(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        PreprocessingConfig(
            target_tile_size_px=224,
            target_spacing_um=0.5,
            target_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=SimpleNamespace(
                requested_tile_size_px=224,
                requested_spacing_um=0.5,
                effective_tile_size_px=224,
                effective_spacing_um=0.5,
                tile_size_lv0=224,
                read_level=0,
                use_padding=True,
                is_within_tolerance=True,
                sample_id="s0",
                x=torch.tensor([0], dtype=torch.int64).numpy(),
                y=torch.tensor([0], dtype=torch.int64).numpy(),
                tissue_fractions=torch.tensor([1.0], dtype=torch.float32).numpy(),
                coordinates_npz_path=Path("/tmp/coords.npz"),
                coordinates_meta_path=Path("/tmp/coords.meta.json"),
            ),
        )
    ]

    def _fake_populate_hierarchical_cache(
        self_,
        *,
        cache_resolution,
        loaded_tilings,
        prepared_tilings,
        tiling_dir,
        preprocessing,
        encoder_name,
        output_variant,
        num_gpus,
    ):
        cache_resolution.features_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(1, 4, 8), cache_resolution.features_dir / "s0.pt")

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_populate_hierarchical_cache",
        autospec=True,
        side_effect=_fake_populate_hierarchical_cache,
    ) as populate_hierarchical_cache:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert populate_hierarchical_cache.called
    assert store.is_hierarchical is True
    assert store.load("s0").shape == (1, 4, 8)


def test_slide_cache_population_delegates_to_cache_methods(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    cache_root = tmp_path / "shared-cache"
    extractor = FeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_SLIDE, save_tile_features=False),
        PreprocessingConfig(target_tile_size_px=224, target_spacing_um=0.5),
        cache=CacheConfig(root_dir=cache_root),
    )
    loaded = [
        LoadedTiling(
            slide=SlideSpec(sample_id="s0", image_path=Path("/tmp/s0.svs"), mask_path=None, spacing_at_level_0=None),
            tiling_result=_tiling(),
        )
    ]

    def _fake_populate_tile_cache(
        self_,
        *,
        cache_resolution,
        loaded_tilings,
        prepared_tilings,
        tiling_dir,
        preprocessing,
        encoder_name,
        output_variant,
        num_gpus,
    ):
        cache_resolution.features_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(2, 8), cache_resolution.features_dir / "s0.pt")

    def _fake_populate_slide_cache(
        self_,
        *,
        slide_cache,
        tile_cache,
        loaded_tilings,
        model_name,
        output_variant,
        num_gpus,
    ):
        slide_cache.features_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.ones(8), slide_cache.features_dir / "s0.pt")

    with patch("soma.extraction.load_tilings", return_value=loaded), patch(
        "soma.extraction._validate_runtime"
    ), patch.object(
        FeatureExtractor,
        "_populate_tile_cache",
        autospec=True,
        side_effect=_fake_populate_tile_cache,
    ) as populate_tile_cache, patch.object(
        FeatureExtractor,
        "_populate_slide_cache",
        autospec=True,
        side_effect=_fake_populate_slide_cache,
    ) as populate_slide_cache:
        store = extractor.extract(tmp_path / "features", tiling_dir=tmp_path / "tiling")

    assert populate_tile_cache.called
    assert populate_slide_cache.called
    assert store.is_slide_level is True
    assert store.load("s0").shape == (8,)


def test_multispacing_encoder_requires_explicit_spacing(tmp_path: Path):
    dataset = _make_dataset(tmp_path)
    extractor = FeatureExtractor(dataset, EncoderConfig(name=_TEST_MULTI))
    with pytest.raises(ValueError, match="supports multiple spacings"):
        extractor.preprocess(tmp_path / "tiling")
