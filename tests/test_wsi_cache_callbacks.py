"""WSI payload persistence commits usable caches before the pipeline returns."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
import torch

from soma import cache as cache_mod
from soma.config import CacheConfig, EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.extraction import orchestration
from soma.extraction.extractor import _PooledFeatureExtractor
from soma.slide2vec_adapter import LoadedTiling, build_preprocessing_config
from tests.test_extraction import (
    _TEST_TILE,
    _artifact,
    _register_test_encoders,
    _tiling,
)


@pytest.mark.parametrize("kind", ["tile", "hierarchical"])
def test_artifact_helper_forwards_callback_to_public_pipeline(
    tmp_path, monkeypatch, kind
):
    callback = Mock()
    artifact = object()
    pipeline = Mock()
    pipeline.run_with_coordinates.return_value = SimpleNamespace(
        **{f"{kind}_artifacts": [artifact]}
    )
    monkeypatch.setattr(orchestration, "Pipeline", Mock(return_value=pipeline))
    monkeypatch.setattr(orchestration, "_load_model", Mock())
    monkeypatch.setattr(orchestration, "_release_parent_cuda_state", Mock())
    helper = getattr(orchestration, f"_embed_{kind}_artifacts_with_coordinates")
    assert helper(
        model_name="test",
        output_variant="default",
        preprocessing=object(),
        execution=SimpleNamespace(output_dir=tmp_path / "out"),
        tiling_dir=tmp_path / "tiling",
        slides=["slide"],
        on_slide_persisted=callback,
    ) == [artifact]
    pipeline.run_with_coordinates.assert_called_once_with(
        tmp_path / "tiling",
        slides=["slide"],
        on_slide_persisted=callback,
    )


@pytest.fixture(params=["tile", "hierarchical"])
def population(tmp_path, request):
    _register_test_encoders()
    kind = request.param
    ids = [f"s{i}" for i in range(20)]
    csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {
                "sample_id": sid,
                "image_path": str(tmp_path / f"{sid}.svs"),
                "label": "tumor",
            }
            for sid in ids
        ]
    ).to_csv(csv, index=False)
    dataset = Dataset(csv)
    preprocessing = PreprocessingConfig(
        tissue_method="hsv",
        requested_tile_size_px=224,
        requested_spacing_um=0.5,
        requested_region_size_px=448,
        region_tile_multiple=2,
    )
    extractor = _PooledFeatureExtractor(
        dataset,
        EncoderConfig(name=_TEST_TILE),
        preprocessing,
        cache=CacheConfig(root_dir=tmp_path / "cache", commit_every=1),
        output_root=tmp_path / "out",
    )

    def resolve():
        return getattr(cache_mod, f"resolve_{kind}_cache")(
            cache_root=tmp_path / "cache",
            dataset=dataset,
            tile_encoder_name=_TEST_TILE,
            preprocessing=preprocessing,
            execution=EncoderConfig(name=_TEST_TILE),
            output_variant="default",
            backend_provenance={
                "requested_backend": "openslide",
                "backend": "openslide",
                "backend_by_sample_id": dict.fromkeys(ids, "openslide"),
            },
        )

    resolution = resolve()
    loaded = [
        LoadedTiling(slide=SimpleNamespace(sample_id=sid), tiling_result=_tiling(sid))
        for sid in ids
    ]

    def run():
        getattr(extractor, f"_populate_{kind}_cache")(
            cache_resolution=resolve(),
            loaded_tilings=loaded,
            prepared_tilings=[item.tiling_result for item in loaded],
            tiling_dir=tmp_path / "tiling",
            preprocessing=build_preprocessing_config(preprocessing),
            encoder_name=_TEST_TILE,
            output_variant="default",
            num_gpus=4,
        )

    def artifact(slide, execution):
        return _artifact(
            sample_id=slide.sample_id,
            output_dir=execution.output_dir,
            kind=f"{kind}_embeddings",
            tensor=torch.ones((2, 8) if kind == "tile" else (1, 4, 8)),
        )

    return SimpleNamespace(
        run=run,
        resolve=resolve,
        cache=resolution,
        artifact=artifact,
        ids=ids,
        kind=kind,
        target=f"soma.extraction.extractor._embed_{kind}_artifacts_with_coordinates",
    )


def test_population_commits_each_slide_before_single_call_returns(
    population, monkeypatch
):
    calls = []

    def embed(*, slides, execution, on_slide_persisted, **kwargs):
        calls.append([slide.sample_id for slide in slides])
        artifacts = []
        for slide in slides:
            artifact = population.artifact(slide, execution)
            artifacts.append(artifact)
            on_slide_persisted(artifact)
            metadata = json.loads(population.cache.metadata_path.read_text())
            assert (
                metadata["sample_identity_signature_by_id"][slide.sample_id]
                == population.cache.cache_stem_by_id[slide.sample_id]
            )
            assert population.cache.feature_path_for_id(slide.sample_id).is_file()
            assert (
                json.loads(population.cache.metadata_path.read_text())["feature_dim"]
                == 8
            )
        return artifacts

    monkeypatch.setattr(population.target, embed)
    population.run()
    assert calls == [population.ids]
    assert population.resolve().missing_sample_ids() == []


def test_interrupted_population_reuses_only_committed_slides(population, monkeypatch):
    def interrupted(*, slides, execution, on_slide_persisted, **kwargs):
        for slide in slides[:3]:
            on_slide_persisted(population.artifact(slide, execution))
        raise RuntimeError("interrupted")

    monkeypatch.setattr(population.target, interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        population.run()
    assert population.resolve().missing_sample_ids() == sorted(population.ids[3:])
    assert json.loads(population.cache.metadata_path.read_text())["feature_dim"] == 8
    calls = []

    def resumed(*, slides, execution, on_slide_persisted, **kwargs):
        calls.append([slide.sample_id for slide in slides])
        artifacts = [population.artifact(slide, execution) for slide in slides]
        for artifact in artifacts:
            on_slide_persisted(artifact)
        return artifacts

    monkeypatch.setattr(population.target, resumed)
    population.run()
    assert calls == [population.ids[3:]]
    assert population.resolve().missing_sample_ids() == []


def test_returned_artifacts_reconcile_missed_callbacks_without_damaging_payloads(
    population, monkeypatch
):
    def embed(*, slides, execution, on_slide_persisted, **kwargs):
        artifacts = [population.artifact(slide, execution) for slide in slides]
        on_slide_persisted(artifacts[0])
        return artifacts

    monkeypatch.setattr(population.target, embed)
    population.run()
    assert population.resolve().missing_sample_ids() == []
    for sid in population.ids:
        payload = torch.load(
            population.cache.feature_path_for_id(sid), weights_only=True
        )
        torch.testing.assert_close(
            payload,
            torch.ones((2, 8) if population.kind == "tile" else (1, 4, 8)),
        )


def test_callback_signature_failure_propagates(population, monkeypatch):
    def embed(*, slides, execution, on_slide_persisted, **kwargs):
        on_slide_persisted(population.artifact(slides[0], execution))
        pytest.fail("signature error was swallowed")

    monkeypatch.setattr(population.target, embed)
    monkeypatch.setattr(
        "soma.extraction.extractor.record_sample_identity_signatures",
        Mock(side_effect=OSError("signature write failed")),
    )
    with pytest.raises(OSError, match="signature write failed"):
        population.run()


def test_first_known_callback_dimension_is_persisted(population, monkeypatch):
    def embed(*, slides, execution, on_slide_persisted, **kwargs):
        first = population.artifact(slides[0], execution)
        first.feature_dim = None
        on_slide_persisted(first)
        assert (
            json.loads(population.cache.metadata_path.read_text())["feature_dim"]
            is None
        )
        on_slide_persisted(population.artifact(slides[1], execution))
        assert (
            json.loads(population.cache.metadata_path.read_text())["feature_dim"] == 8
        )
        raise RuntimeError("interrupted")

    monkeypatch.setattr(population.target, embed)
    with pytest.raises(RuntimeError, match="interrupted"):
        population.run()
