from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from soma.extraction import FeatureExtractor


def _write_scalar_dataset(path: Path) -> None:
    path.write_text("sample_id,image_path,label\ns0,tile.png,1\n", encoding="utf-8")


def test_feature_extraction_result_is_exact_immutable_four_part_contract() -> None:
    from soma.extraction import (
        ExtractionArtifacts,
        FeatureExtractionResult,
        FeatureProvenance,
    )

    source = object()
    dataset = object()
    provenance = FeatureProvenance(kind="pooled", zero_sample_ids=("empty",))
    artifacts = ExtractionArtifacts(feature_dir=Path("features"))

    result = FeatureExtractionResult(
        source=source,
        dataset=dataset,
        provenance=provenance,
        artifacts=artifacts,
    )

    assert [field.name for field in fields(result)] == [
        "source",
        "dataset",
        "provenance",
        "artifacts",
    ]
    assert (result.source, result.dataset, result.provenance, result.artifacts) == (
        source,
        dataset,
        provenance,
        artifacts,
    )
    with pytest.raises(FrozenInstanceError):
        result.source = object()  # type: ignore[misc]


def test_feature_extractor_extract_accepts_no_runtime_arguments() -> None:
    assert list(inspect.signature(FeatureExtractor.extract).parameters) == ["self"]
    assert not hasattr(FeatureExtractor, "run")
    assert not hasattr(FeatureExtractor, "preprocess")


def test_tile_manifest_loads_as_explicit_tile_dataset(tmp_path: Path) -> None:
    from soma.dataset import Dataset, TileDataset, load_manifest

    dataset_csv = tmp_path / "dataset.csv"
    _write_scalar_dataset(dataset_csv)

    tile = load_manifest(dataset_csv, "tile")
    slide = load_manifest(dataset_csv, "slide")

    assert type(tile) is TileDataset
    assert type(slide) is Dataset


def test_tile_dataset_extracts_through_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma import CacheConfig, EncoderConfig, ExecutionConfig, TileDataset

    dataset_csv = tmp_path / "dataset.csv"
    _write_scalar_dataset(dataset_csv)
    dataset = TileDataset(dataset_csv)

    class BoundaryModel:
        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            return cls()

        def embed_images(self, images, *, execution):
            assert [item.sample_id for item in images] == ["s0"]
            return [
                write_image_embedding(
                    torch.tensor([1.0, 2.0], dtype=torch.float32),
                    output_dir=execution.output_dir,
                    sample_id="s0",
                    output_format=execution.output_format,
                    metadata={"artifact_type": "image_embeddings", "sample_id": "s0"},
                )
            ]

    monkeypatch.setattr("soma.tile_extraction.Model", BoundaryModel)

    result = FeatureExtractor(
        dataset,
        EncoderConfig(name="phikon", batch_size=7),
        execution=ExecutionConfig(num_gpus=1, num_workers_per_gpu=0),
        cache=CacheConfig(enabled=False),
        output_root=tmp_path / "output",
    ).extract()

    assert result.dataset is dataset
    assert result.provenance.kind == "pooled_image"
    assert result.provenance.encoder_name == "phikon"
    assert result.source.available_samples == ["s0"]
    assert torch.equal(result.source.load("s0"), torch.tensor([1.0, 2.0]))
    assert result.artifacts.feature_dir == tmp_path / "output/features/image_embeddings"
    assert result.artifacts.provenance_json == tmp_path / "output/extraction_provenance.json"
    assert result.artifacts.provenance_json.is_file()


def test_spatial_expression_extracts_through_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma import CacheConfig, EncoderConfig, SpatialExpressionManifest

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,target_index\nspot-0,spot.png,0\n",
        encoding="utf-8",
    )
    np.save(tmp_path / "targets.npy", np.asarray([[1.5, 2.5]], dtype=np.float32))
    (tmp_path / "genes.json").write_text(json.dumps(["GENE_A", "GENE_B"]), encoding="utf-8")
    dataset = SpatialExpressionManifest(dataset_csv)

    class SpatialBoundaryModel:
        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            return cls()

        def embed_images(self, images, *, execution):
            assert [item.sample_id for item in images] == ["spot-0"]
            return [
                write_image_embedding(
                    torch.tensor([5.0, 6.0], dtype=torch.float32),
                    output_dir=execution.output_dir,
                    sample_id="spot-0",
                    output_format=execution.output_format,
                    metadata={
                        "artifact_type": "image_embeddings",
                        "sample_id": "spot-0",
                    },
                )
            ]

    monkeypatch.setattr("soma.tile_extraction.Model", SpatialBoundaryModel)

    result = FeatureExtractor(
        dataset,
        EncoderConfig(name="phikon", precision="fp32"),
        cache=CacheConfig(enabled=False),
        output_root=tmp_path / "output",
    ).extract()

    assert result.dataset is dataset
    assert result.provenance.kind == "pooled_image"
    assert result.source.available_samples == ["spot-0"]
    assert torch.equal(result.source.load("spot-0"), torch.tensor([5.0, 6.0]))


def test_unsupported_dataset_type_fails_during_construction(tmp_path: Path) -> None:
    from soma import EncoderConfig

    with pytest.raises(TypeError, match="Unsupported dataset/config combination"):
        FeatureExtractor(
            object(),
            EncoderConfig(name="phikon"),
            output_root=tmp_path,
        )


def test_splits_project_preserves_direct_ids_and_inherits_explicit_slide_id(
    tmp_path: Path,
) -> None:
    from soma import Dataset, SegmentationManifest, Splits

    parent_csv = tmp_path / "parents.csv"
    parent_csv.write_text(
        "sample_id,image_path,label\ns0,s0.svs,0\ns1,s1.svs,1\n",
        encoding="utf-8",
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "sample_id,split\ns0,train\ns1,test\n",
        encoding="utf-8",
    )
    effective_csv = tmp_path / "effective.csv"
    effective_csv.write_text(
        "sample_id,slide_id,image_path,label_mask_path,region_x,region_y\n"
        "s0,,s0.svs,s0-mask.tif,,\n"
        "roi-1,s1,s1.svs,s1-mask.tif,12,34\n",
        encoding="utf-8",
    )

    splits = Splits(splits_csv, Dataset(parent_csv))
    projected = splits.project(SegmentationManifest(effective_csv))

    assert projected.folds[0].train == ("s0",)
    assert projected.folds[0].tune == ()
    assert projected.folds[0].tests == {"test": ("roi-1",)}


def test_splits_project_rejects_unresolved_roi_ancestry(tmp_path: Path) -> None:
    from soma import Dataset, SegmentationManifest, Splits

    parent_csv = tmp_path / "parents.csv"
    parent_csv.write_text(
        "sample_id,image_path,label\ns0,s0.svs,0\n",
        encoding="utf-8",
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text("sample_id,split\ns0,test\n", encoding="utf-8")
    effective_csv = tmp_path / "effective.csv"
    effective_csv.write_text(
        "sample_id,slide_id,image_path,label_mask_path,region_x,region_y\n"
        "roi-x,missing,missing.svs,mask.tif,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unresolved split ancestry.*missing"):
        Splits(splits_csv, Dataset(parent_csv)).project(SegmentationManifest(effective_csv))


def test_splits_project_rejects_conflicting_direct_and_parent_assignments(
    tmp_path: Path,
) -> None:
    from soma import Dataset, SegmentationManifest, Splits

    parent_csv = tmp_path / "parents.csv"
    parent_csv.write_text(
        "sample_id,image_path,label\nroi,roi.png,0\nslide,slide.svs,1\n",
        encoding="utf-8",
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "sample_id,split\nroi,train\nslide,test\n",
        encoding="utf-8",
    )
    effective_csv = tmp_path / "effective.csv"
    effective_csv.write_text(
        "sample_id,slide_id,image_path,label_mask_path,region_x,region_y\n"
        "roi,slide,slide.svs,mask.tif,0,0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting split ancestry.*roi"):
        Splits(splits_csv, Dataset(parent_csv)).project(SegmentationManifest(effective_csv))


def test_given_image_segmentation_extracts_dense_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_dense_image

    from soma import (
        CacheConfig,
        EncoderConfig,
        PreprocessingConfig,
        SegmentationManifest,
    )

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\ns0,tile.png,mask.png\n",
        encoding="utf-8",
    )
    dataset = SegmentationManifest(dataset_csv)

    class DenseBoundaryModel:
        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            return cls()

        def embed_images_dense(self, images, *, dense, execution):
            assert [item.sample_id for item in images] == ["s0"]
            grid = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
            return [
                write_dense_image(
                    grid.numpy(),
                    output_dir=execution.output_dir,
                    sample_id="s0",
                    metadata={
                        "artifact_type": "dense_image_embeddings",
                        "feature_dim": 3,
                        "grid_shape": [2, 2],
                        "target_size": [32, 32],
                        "encoded_size": [32, 32],
                        "patch_size": [16, 16],
                        "pad": [0, 0],
                        "crop_box": [0, 0, 32, 32],
                        "channel_dim": 0,
                        "source_spacing_um": 0.5,
                        "effective_spacing_um": 0.5,
                    },
                )
            ]

    monkeypatch.setattr("soma.dense_extraction.Model", DenseBoundaryModel)
    monkeypatch.setattr("soma.dense_extraction.resolve_patch_size", lambda _name: (16, 16))

    result = FeatureExtractor(
        dataset,
        EncoderConfig(name="phikon", precision="fp32", batch_size=5),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=32,
            requested_spacing_um=0.5,
        ),
        cache=CacheConfig(enabled=False),
        output_root=tmp_path / "output",
    ).extract()

    assert result.dataset is dataset
    assert result.provenance.kind == "dense_image"
    assert result.source.available_samples == ["s0"]
    assert tuple(result.source.load("s0").shape) == (3, 2, 2)
    assert result.source.spacing("s0").source_spacing_um == 0.5
    assert result.artifacts.feature_dir == tmp_path / "output/features/dense_image_embeddings"


def test_given_image_detection_selects_dense_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from soma import DetectionManifest, EncoderConfig, PreprocessingConfig
    from soma.dense_extraction import _DenseImageExtractor

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,points_path\ns0,tile.png,points.csv\n",
        encoding="utf-8",
    )
    dataset = DetectionManifest(dataset_csv)
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)

    class DenseStoreBoundary:
        available_samples = ["s0"]
        feature_dim = 3
        feature_dir = tmp_path / "dense"

        def validate_coverage(self, sample_ids):
            assert sample_ids == ["s0"]

        def load(self, sample_id):
            assert sample_id == "s0"
            return expected

    def fake_run(self, feature_dir):
        assert self._dataset is dataset
        return DenseStoreBoundary()

    monkeypatch.setattr(_DenseImageExtractor, "run", fake_run)

    result = FeatureExtractor(
        dataset,
        EncoderConfig(name="phikon", precision="fp32"),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=32,
            requested_spacing_um=0.5,
        ),
        output_root=tmp_path / "output",
    ).extract()

    assert result.dataset is dataset
    assert result.provenance.kind == "dense_image"
    assert torch.equal(result.source.load("s0"), expected)


def test_dense_facade_reuses_existing_cache_key_for_explicit_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_dense_image

    from soma import (
        CacheConfig,
        EncoderConfig,
        PreprocessingConfig,
        SegmentationManifest,
    )
    from soma.dense_extraction import _DenseImageExtractor

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\ns0,tile.png,mask.png\n",
        encoding="utf-8",
    )
    dataset = SegmentationManifest(dataset_csv)

    class CacheIdentityModel:
        loads = 0

        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            cls.loads += 1
            return cls()

        def embed_images_dense(self, images, *, dense, execution):
            return [
                write_dense_image(
                    np.arange(12, dtype=np.float32).reshape(3, 2, 2),
                    output_dir=execution.output_dir,
                    sample_id="s0",
                    metadata={
                        "artifact_type": "dense_image_embeddings",
                        "feature_dim": 3,
                        "grid_shape": [2, 2],
                        "target_size": [32, 32],
                        "encoded_size": [32, 32],
                        "patch_size": [16, 16],
                        "pad": [0, 0],
                        "crop_box": [0, 0, 32, 32],
                        "channel_dim": 0,
                        "source_spacing_um": 0.5,
                        "effective_spacing_um": 0.5,
                    },
                )
            ]

    monkeypatch.setattr("soma.dense_extraction.Model", CacheIdentityModel)
    monkeypatch.setattr("soma.dense_extraction.resolve_patch_size", lambda _name: (16, 16))
    encoder = EncoderConfig(name="phikon", precision="fp32")
    preprocessing = PreprocessingConfig(
        requested_tile_size_px=32,
        requested_spacing_um=0.5,
    )
    cache = CacheConfig(enabled=True, root_dir=tmp_path / "cache")

    existing = _DenseImageExtractor(
        dataset,
        encoder,
        target_size=32,
        spacing_um=0.5,
        preprocessing=preprocessing,
        cache=cache,
    ).run(tmp_path / "legacy-output")
    result = FeatureExtractor(
        dataset,
        encoder,
        preprocessing=preprocessing,
        cache=cache,
        output_root=tmp_path / "canonical-output",
    ).extract()

    assert CacheIdentityModel.loads == 1
    assert result.artifacts.feature_dir == existing.feature_dir
    assert torch.equal(result.source.load("s0"), existing.load("s0"))


def test_annotation_sampled_wsi_returns_deterministic_effective_dataset_and_zero_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_dense_region

    from soma import (
        CacheConfig,
        EncoderConfig,
        MasksConfig,
        PreprocessingConfig,
        SamplingConfig,
        SegmentationManifest,
    )

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\n"
        "s0,s0.svs,s0-mask.tif\n"
        "s1,s1.svs,s1-mask.tif\n",
        encoding="utf-8",
    )
    dataset = SegmentationManifest(dataset_csv)

    def fake_tile_slide(slide, **_kwargs):
        coords = [(0, 0)] if slide.sample_id == "s0" else []
        return {
            None: SimpleNamespace(
                tiles=SimpleNamespace(
                    x=np.asarray([x for x, _ in coords]),
                    y=np.asarray([y for _, y in coords]),
                )
            )
        }

    class RegionBoundaryModel:
        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            return cls()

        def embed_regions_dense(self, regions, *, dense, execution):
            assert [(item.sample_id, item.coordinates) for item in regions] == [("s0", [(0, 0)])]
            grid = np.arange(16, dtype=np.float32).reshape(4, 2, 2)
            return [
                write_dense_region(
                    grid,
                    output_dir=execution.output_dir,
                    sample_id="s0",
                    annotation=None,
                    x=0,
                    y=0,
                    metadata={
                        "artifact_type": "dense_embeddings",
                        "feature_dim": 4,
                        "grid_shape": [2, 2],
                        "target_size": [32, 32],
                        "encoded_size": [32, 32],
                        "patch_size": [16, 16],
                        "pad": [0, 0],
                        "crop_box": [0, 0, 32, 32],
                        "channel_dim": 0,
                        "source_spacing_um": 0.5,
                        "effective_spacing_um": 0.5,
                    },
                )
            ]

    monkeypatch.setattr("hs2p.tile_slide", fake_tile_slide)
    monkeypatch.setattr("soma.dense_slide_extraction.Model", RegionBoundaryModel)
    preprocessing = PreprocessingConfig(
        requested_tile_size_px=32,
        requested_spacing_um=0.5,
        masks=MasksConfig(
            pixel_mapping={"background": 0, "tumor": 1},
            min_coverage={"tumor": 0.0},
        ),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
    )

    result = FeatureExtractor(
        dataset,
        EncoderConfig(name="phikon", precision="fp32"),
        preprocessing=preprocessing,
        cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
        output_root=tmp_path / "output",
    ).extract()

    assert result.dataset.sample_ids == ["s0__x0_y0"]
    roi = result.dataset.samples["s0__x0_y0"]
    assert (roi.slide_id, roi.region) == ("s0", (0, 0))
    assert result.provenance.zero_roi_sample_ids == ("s1",)
    assert result.source.available_samples == ["s0__x0_y0"]
    assert result.artifacts.dataset_csv == tmp_path / "output/segmentation_rois/roi_manifest.csv"
    expected_csv = (
        "sample_id,slide_id,image_path,mask_path,label_mask_path,patient_id,"
        "spacing_at_level_0,region_x,region_y\n"
        "s0__x0_y0,s0,s0.svs,,s0-mask.tif,,,0,0\n"
    )
    assert result.artifacts.dataset_csv.read_text(encoding="utf-8") == expected_csv


def test_partial_tile_extraction_resumes_only_missing_and_publishes_on_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma import CacheConfig, EncoderConfig, TileDataset

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label\ns0,s0.png,0\ns1,s1.png,1\n",
        encoding="utf-8",
    )
    dataset = TileDataset(dataset_csv)

    class PartialBoundaryModel:
        calls: list[list[str]] = []

        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            return cls()

        def embed_images(self, images, *, execution):
            ids = [item.sample_id for item in images]
            type(self).calls.append(ids)
            written = ids[:1] if len(type(self).calls) == 1 else ids
            return [
                write_image_embedding(
                    torch.tensor([float(index), 9.0]),
                    output_dir=execution.output_dir,
                    sample_id=sample_id,
                    output_format=execution.output_format,
                    metadata={
                        "artifact_type": "image_embeddings",
                        "sample_id": sample_id,
                    },
                )
                for index, sample_id in enumerate(written)
            ]

    monkeypatch.setattr("soma.tile_extraction.Model", PartialBoundaryModel)
    kwargs = dict(
        dataset=dataset,
        encoder=EncoderConfig(name="phikon", precision="fp32"),
        cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
        output_root=tmp_path / "output",
    )

    with pytest.raises(ValueError, match="Missing features.*s1"):
        FeatureExtractor(**kwargs).extract()
    assert not (tmp_path / "output/extraction_provenance.json").exists()

    result = FeatureExtractor(**kwargs).extract()

    assert PartialBoundaryModel.calls == [["s0", "s1"], ["s1"]]
    assert result.source.available_samples == ["s0", "s1"]
    assert (tmp_path / "output/extraction_provenance.json").is_file()


def test_encoder_failure_preserves_valid_tile_payload_for_missing_only_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma import CacheConfig, EncoderConfig, TileDataset

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label\ns0,s0.png,0\ns1,s1.png,1\n",
        encoding="utf-8",
    )
    dataset = TileDataset(dataset_csv)

    class FailingBoundaryModel:
        calls: list[list[str]] = []

        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            return cls()

        def embed_images(self, images, *, execution):
            ids = [item.sample_id for item in images]
            type(self).calls.append(ids)
            if len(type(self).calls) == 1:
                write_image_embedding(
                    torch.tensor([1.0, 2.0]),
                    output_dir=execution.output_dir,
                    sample_id="s0",
                    output_format=execution.output_format,
                    metadata={
                        "artifact_type": "image_embeddings",
                        "sample_id": "s0",
                    },
                )
                raise RuntimeError("encoder interrupted")
            return [
                write_image_embedding(
                    torch.tensor([3.0, 4.0]),
                    output_dir=execution.output_dir,
                    sample_id="s1",
                    output_format=execution.output_format,
                    metadata={
                        "artifact_type": "image_embeddings",
                        "sample_id": "s1",
                    },
                )
            ]

    monkeypatch.setattr("soma.tile_extraction.Model", FailingBoundaryModel)
    kwargs = dict(
        dataset=dataset,
        encoder=EncoderConfig(name="phikon", precision="fp32"),
        cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
        output_root=tmp_path / "output",
    )

    with pytest.raises(RuntimeError, match="encoder interrupted"):
        FeatureExtractor(**kwargs).extract()
    assert not (tmp_path / "output/extraction_provenance.json").exists()

    result = FeatureExtractor(**kwargs).extract()

    assert FailingBoundaryModel.calls == [["s0", "s1"], ["s1"]]
    assert torch.equal(result.source.load("s0"), torch.tensor([1.0, 2.0]))
    assert torch.equal(result.source.load("s1"), torch.tensor([3.0, 4.0]))


def test_complete_tile_cache_hit_does_not_load_encoder_or_rewrite_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma import CacheConfig, EncoderConfig, TileDataset

    dataset_csv = tmp_path / "dataset.csv"
    _write_scalar_dataset(dataset_csv)
    dataset = TileDataset(dataset_csv)

    class OneShotBoundaryModel:
        loads = 0

        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            cls.loads += 1
            return cls()

        def embed_images(self, images, *, execution):
            return [
                write_image_embedding(
                    torch.tensor([3.0, 4.0]),
                    output_dir=execution.output_dir,
                    sample_id="s0",
                    output_format=execution.output_format,
                    metadata={"artifact_type": "image_embeddings", "sample_id": "s0"},
                )
            ]

    monkeypatch.setattr("soma.tile_extraction.Model", OneShotBoundaryModel)
    kwargs = dict(
        dataset=dataset,
        encoder=EncoderConfig(name="phikon", precision="fp32"),
        cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
        output_root=tmp_path / "output",
    )
    first = FeatureExtractor(**kwargs).extract()
    payload = first.source.feature_dir / "s0.pt"
    before = (payload.read_bytes(), payload.stat().st_mtime_ns)

    second = FeatureExtractor(**kwargs).extract()

    assert OneShotBoundaryModel.loads == 1
    assert (payload.read_bytes(), payload.stat().st_mtime_ns) == before
    assert torch.equal(second.source.load("s0"), torch.tensor([3.0, 4.0]))


def test_pooled_slide_extracts_through_the_same_public_interface(
    tmp_path: Path,
) -> None:
    tifffile = pytest.importorskip("tifffile")

    from soma import (
        CacheConfig,
        Dataset,
        EncoderConfig,
        ExecutionConfig,
        PreprocessingConfig,
    )
    from tests.dense_literal_encoder import register_literal_encoder

    encoder_name = register_literal_encoder()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[..., 0] = 8
    image[..., 1] = 16
    image[..., 2] = 24
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[16:240, 16:240] = 1
    image_path = tmp_path / "slide.tif"
    mask_path = tmp_path / "mask.tif"
    resolution = 1e4 / 0.5
    tifffile.imwrite(
        image_path,
        image,
        photometric="rgb",
        resolution=(resolution, resolution),
        resolutionunit="CENTIMETER",
    )
    tifffile.imwrite(
        mask_path,
        mask,
        photometric="minisblack",
        resolution=(resolution, resolution),
        resolutionunit="CENTIMETER",
    )
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label,mask_path,spacing_at_level_0\n"
        f"s0,{image_path},1,{mask_path},0.5\n",
        encoding="utf-8",
    )
    dataset = Dataset(dataset_csv)

    result = FeatureExtractor(
        dataset,
        EncoderConfig(name=encoder_name, precision="fp32", batch_size=1),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=32,
            requested_spacing_um=0.5,
            tissue_method="precomputed_mask",
            min_coverage={"tissue": 0.0},
        ),
        execution=ExecutionConfig(
            num_gpus=1,
            num_workers_per_gpu=0,
            num_preprocessing_workers=0,
            precision="fp32",
        ),
        cache=CacheConfig(enabled=False),
        output_root=tmp_path / "output",
    ).extract()

    assert result.dataset is dataset
    assert result.provenance.kind == "pooled_bag"
    assert result.source.available_samples == ["s0"]
    assert tuple(result.source.load("s0").shape) == (49, 3)
    assert result.artifacts.tiling_dir == tmp_path / "output/tiling"


def test_hierarchical_extracts_through_the_same_public_interface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from soma import CacheConfig, Dataset, EncoderConfig, PreprocessingConfig
    from soma.extraction.extractor import _PooledFeatureExtractor
    from soma.features import FeatureStore

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label\ns0,slide.svs,1\n",
        encoding="utf-8",
    )
    dataset = Dataset(dataset_csv)
    expected = torch.arange(24, dtype=torch.float16).reshape(2, 3, 4)

    def fake_run(self, *, feature_dir):
        payload_dir = self._output_root / feature_dir / "hierarchical_embeddings"
        payload_dir.mkdir(parents=True)
        torch.save(expected, payload_dir / "s0.pt")
        return FeatureStore(self._output_root / feature_dir)

    monkeypatch.setattr(_PooledFeatureExtractor, "run", fake_run)

    result = FeatureExtractor(
        dataset,
        EncoderConfig(name="phikon", precision="fp32"),
        preprocessing=PreprocessingConfig(
            requested_region_size_px=448,
            region_tile_multiple=2,
        ),
        cache=CacheConfig(enabled=False),
        output_root=tmp_path / "output",
    ).extract()

    assert result.dataset is dataset
    assert result.provenance.kind == "hierarchical"
    assert result.source.is_hierarchical is True
    assert result.source.feature_dim == 4
    assert torch.equal(result.source.load("s0"), expected.float())
