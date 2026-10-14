"""``soma.extract``: the single way features enter (ADR 0021), at its public seam.

slide2vec's model is stubbed at soma's import seams; every test reads what the verb
returns (a source) and what it publishes (``extraction_provenance.json``).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from slide2vec import DenseOptions, ExecutionOptions

import soma
from soma.data import AnnotationManifest, ImageManifest
from tests.extraction_configs import annotation_masks, extraction, tissue_masks


def _write_scalar_dataset(path: Path) -> None:
    path.write_text("sample_id,image_path,label\ns0,tile.png,1\n", encoding="utf-8")


def _provenance(output_dir: Path) -> dict:
    return json.loads((output_dir / "extraction_provenance.json").read_text(encoding="utf-8"))


def test_extract_is_the_root_verb_and_the_namespace_of_the_live_encoder() -> None:
    assert callable(soma.extract) and soma.extract.extract is soma.extract.extract
    assert soma.extract.live_encoder is not None
    parameters = inspect.signature(soma.extract.extract).parameters
    assert list(parameters) == [
        "images",
        "encoder",
        "cache",
        "annotations",
        "unit",
        "rois",
        "execution",
        "extraction",
        "dense",
        "spacing_policy",
        "output_dir",
    ]
    assert all(p.kind is p.KEYWORD_ONLY for name, p in parameters.items() if name != "images")
    assert not hasattr(soma, "FeatureExtractor")


def test_extract_reads_an_image_manifest_never_records(tmp_path: Path) -> None:
    from soma import SampleRecord
    from soma.config import CacheOptions

    with pytest.raises(TypeError, match="ImageManifest"):
        soma.extract(
            [SampleRecord("s0", {"label": 1})],
            encoder="phikon",
            cache=CacheOptions(enabled=False),
            output_dir=tmp_path,
        )


def test_tile_dataset_extracts_through_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    _write_scalar_dataset(dataset_csv)
    images = ImageManifest.from_csv(dataset_csv)

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

    monkeypatch.setattr("soma.extract.tile_extraction.Model", BoundaryModel)

    source = soma.extract(
        images,
        encoder="phikon",
        execution=ExecutionOptions(num_gpus=1, num_workers_per_gpu=0, batch_size=7),
        cache=CacheOptions(enabled=False),
        output_dir=tmp_path / "output",
        unit="tile",
    )

    assert source.sample_ids == ["s0"]
    assert torch.equal(source.load("s0"), torch.tensor([1.0, 2.0]))
    assert source.feature_dir == tmp_path / "output/features/image_embeddings"
    record = _provenance(tmp_path / "output")
    assert (record["status"], record["provenance"]["kind"]) == ("completed", "pooled_image")
    assert record["provenance"]["encoder_name"] == "phikon"
    assert record["artifacts"]["feature_dir"] == str(source.feature_dir)


def test_spatial_expression_extracts_through_canonical_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,target_index\nspot-0,spot.png,0\n",
        encoding="utf-8",
    )
    np.save(tmp_path / "targets.npy", np.asarray([[1.5, 2.5]], dtype=np.float32))
    (tmp_path / "genes.json").write_text(
        json.dumps(["GENE_A", "GENE_B"]), encoding="utf-8"
    )
    images = ImageManifest.from_csv(dataset_csv)

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

    monkeypatch.setattr("soma.extract.tile_extraction.Model", SpatialBoundaryModel)

    source = soma.extract(
        images,
        encoder="phikon",
        cache=CacheOptions(enabled=False),
        output_dir=tmp_path / "output",
        unit="tile",
    )

    assert _provenance(tmp_path / "output")["provenance"]["kind"] == "pooled_image"
    assert source.sample_ids == ["spot-0"]
    assert torch.equal(source.load("spot-0"), torch.tensor([5.0, 6.0]))


def test_unsupported_extraction_unit_fails_before_anything_runs(tmp_path: Path) -> None:
    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    _write_scalar_dataset(dataset_csv)
    with pytest.raises(ValueError, match="unit must be"):
        soma.extract(
            ImageManifest.from_csv(dataset_csv),
            encoder="phikon",
            unit="blob",
            cache=CacheOptions(enabled=False),
            output_dir=tmp_path / "output",
        )
    assert not (tmp_path / "output").exists()


def test_given_image_segmentation_extracts_dense_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_dense_image

    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\ns0,tile.png,mask.png\n",
        encoding="utf-8",
    )
    images = ImageManifest.from_csv(dataset_csv)

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

    monkeypatch.setattr("soma.extract.dense_extraction.Model", DenseBoundaryModel)
    monkeypatch.setattr(
        "soma.extract.dense_extraction.resolve_patch_size", lambda _name: (16, 16)
    )

    source = soma.extract(
        images,
        encoder="phikon",
        execution=ExecutionOptions(precision="fp32", batch_size=5),
        extraction=extraction(0.5, 32),
        dense=DenseOptions(spacing_um=0.5, target_size=32),
        cache=CacheOptions(enabled=False),
        output_dir=tmp_path / "output",
        unit="tile",
    )

    assert _provenance(tmp_path / "output")["provenance"]["kind"] == "dense_image"
    assert source.sample_ids == ["s0"]
    assert tuple(source.load("s0").shape) == (3, 2, 2)
    assert source.spacing("s0") == 0.5
    assert source.geometry("s0").level0_px_per_token_px == 1.0
    assert source.feature_dir == tmp_path / "output/features/dense_image_embeddings"
    assert source.rois is None


def test_rois_are_refused_as_pre_cropped_images(tmp_path: Path) -> None:
    """ROIs are windows of whole slides: read as pre-cropped images, each would be its
    whole parent slide with its address lost, so the verb refuses it before running."""
    from soma.config import CacheOptions
    from soma.extract import RoiSet
    from soma.extract.rois import roi_entries

    dataset_csv = tmp_path / "slides.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\ns0,s0.svs,s0-mask.tif\n", encoding="utf-8"
    )
    images = ImageManifest.from_csv(dataset_csv)
    rois = RoiSet(roi_entries(images, AnnotationManifest.from_csv(dataset_csv), {"s0": [(0, 0)]}))
    with pytest.raises(TypeError, match="unit='slide'"):
        soma.extract(
            images,
            encoder="phikon",
            rois=rois,
            unit="tile",
            cache=CacheOptions(enabled=False),
            output_dir=tmp_path / "out",
        )


def test_annotation_sampled_slides_return_roi_keyed_grids_whose_rois_name_their_slide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_dense_region

    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\n"
        "s0,s0.svs,s0-mask.tif\n"
        "s1,s1.svs,s1-mask.tif\n",
        encoding="utf-8",
    )

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
            assert [(item.sample_id, item.coordinates) for item in regions] == [
                ("s0", [(0, 0)])
            ]
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

    def fake_mask_regions(path, *, regions, size, **_kwargs):
        assert (str(path), regions) == ("s0-mask.tif", [((0, 0), 0.5)])
        return [np.ones((size[1], size[0]), dtype=np.uint8)]

    monkeypatch.setattr("hs2p.tile_slide", fake_tile_slide)
    monkeypatch.setattr("soma.extract.dense_slide_extraction.Model", RegionBoundaryModel)
    monkeypatch.setattr(
        "soma.extract.dense.reader.read_mask_regions_within_slide", fake_mask_regions
    )
    preprocessing = extraction(
        0.5,
        32,
        masks=annotation_masks({"background": 0, "tumor": 1}, {"tumor": 0.0}, strategy="joint"),
    )

    source = soma.extract(
        ImageManifest.from_csv(dataset_csv),
        annotations=AnnotationManifest.from_csv(dataset_csv),
        encoder="phikon",
        extraction=preprocessing,
        dense=DenseOptions(spacing_um=0.5, target_size=32),
        cache=CacheOptions(enabled=True, root_dir=tmp_path / "cache"),
        output_dir=tmp_path / "output",
    )

    # The grids are keyed by ROI; the ROIs name their slide (and nothing about where they
    # lie in it). s1 sampled no ROI.
    assert source.sample_ids == ["s0__x0_y0"]
    assert source.rois.children == {"s0__x0_y0": "s0"}
    record = _provenance(tmp_path / "output")
    assert record["provenance"]["zero_sample_ids"] == ["s1"]
    roi_table = tmp_path / "output/segmentation_rois/dataset.csv"
    assert record["artifacts"]["dataset_csv"] == str(roi_table)
    # The label maps read the crop ROI sampling stored in its cache entry.
    label_maps = source.rois.label_maps(size=(32, 32))
    crop_path = label_maps.crops["s0__x0_y0"]
    assert crop_path.parent.parent.parent.parent == tmp_path / "cache" / "roi_sampling"
    assert crop_path.parent.parent.name == "masks" and crop_path.name == "0_0.png"
    assert label_maps.load("s0__x0_y0").tolist() == [[1] * 32] * 32
    expected_csv = (
        "sample_id,slide_id,image_path,mask_path,label_mask_path,patient_id,"
        "spacing_at_level_0,region_x,region_y,label_mask_crop_path\n"
        f"s0__x0_y0,s0,s0.svs,,s0-mask.tif,,,0,0,{crop_path}\n"
    )
    assert roi_table.read_text(encoding="utf-8") == expected_csv


def test_all_zero_roi_extraction_returns_empty_source_without_loading_encoder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\ns0,s0.svs,s0-mask.tif\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "hs2p.tile_slide",
        lambda _slide, **_kwargs: {
            None: SimpleNamespace(
                tiles=SimpleNamespace(
                    x=np.asarray([], dtype=np.int64),
                    y=np.asarray([], dtype=np.int64),
                )
            )
        },
    )

    class EncoderMustNotLoad:
        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            raise AssertionError("all-zero ROI extraction must not load the encoder")

    monkeypatch.setattr("soma.extract.dense_slide_extraction.Model", EncoderMustNotLoad)
    source = soma.extract(
        ImageManifest.from_csv(dataset_csv),
        annotations=AnnotationManifest.from_csv(dataset_csv),
        encoder="phikon",
        extraction=extraction(
            0.5, 32, masks=annotation_masks({"background": 0, "tumor": 1}, {"tumor": 0.0})
        ),
        dense=DenseOptions(spacing_um=0.5, target_size=32),
        cache=CacheOptions(enabled=False),
        output_dir=tmp_path / "output",
    )

    assert source.sample_ids == []
    assert source.rois.children == {}
    assert source.feature_dim == 0
    record = _provenance(tmp_path / "output")
    assert record["provenance"]["zero_sample_ids"] == ["s0"]
    assert record["artifacts"]["dataset_csv"] == str(tmp_path / "output/segmentation_rois/dataset.csv")


def test_annotation_sampling_samples_rois_at_the_geometry_the_grid_reads_them_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ROIs are sampled (and their masks cropped) at the tiling and read at ``dense``: an
    unset tiling takes the dense geometry, not the encoder preset's, and a tiling that
    disagrees with ``dense`` is refused."""
    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\ns0,s0.svs,s0-mask.tif\n",
        encoding="utf-8",
    )
    tilings = []

    def fake_tile_slide(_slide, *, tiling, **_kwargs):
        tilings.append(tiling)
        empty = np.asarray([], dtype=np.int64)
        return {None: SimpleNamespace(tiles=SimpleNamespace(x=empty, y=empty))}

    monkeypatch.setattr("hs2p.tile_slide", fake_tile_slide)
    masks = annotation_masks({"background": 0, "tumor": 1}, {"tumor": 0.0})

    def extract(preprocessing):
        return soma.extract(
            ImageManifest.from_csv(dataset_csv),
            annotations=AnnotationManifest.from_csv(dataset_csv),
            encoder="phikon",
            extraction=preprocessing,
            dense=DenseOptions(spacing_um=1.0, target_size=512, tolerance=0.1),
            cache=CacheOptions(enabled=False),
            output_dir=tmp_path / "output",
        )

    extract(extraction(None, None, masks=masks))
    assert [
        (t.requested_spacing_um, t.requested_tile_size_px, t.tolerance) for t in tilings
    ] == [(1.0, 512, 0.1)]
    with pytest.raises(ValueError, match="disagree"):
        extract(extraction(0.5, 224, masks=masks))


def test_tile_extraction_missing_returned_outputs_commits_nothing_and_publishes_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label\ns0,s0.png,0\ns1,s1.png,1\n",
        encoding="utf-8",
    )
    images = ImageManifest.from_csv(dataset_csv)

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
                        "compatibility": {"encoder_name": "phikon"},
                    },
                )
                for index, sample_id in enumerate(written)
            ]

    monkeypatch.setattr("soma.extract.tile_extraction.Model", PartialBoundaryModel)
    kwargs = dict(
        unit="tile",
        encoder="phikon",
        cache=CacheOptions(enabled=True, root_dir=tmp_path / "cache"),
        output_dir=tmp_path / "output",
    )

    # slide2vec returned no output for s1: nothing is signed, nothing is published.
    with pytest.raises(RuntimeError, match="nothing is committed"):
        soma.extract(images, **kwargs)
    assert not (tmp_path / "output/extraction_provenance.json").exists()

    source = soma.extract(images, **kwargs)

    # Unsigned outputs go back to slide2vec, which owns per-image resume.
    assert PartialBoundaryModel.calls == [["s0", "s1"], ["s0", "s1"]]
    assert source.sample_ids == ["s0", "s1"]
    assert (tmp_path / "output/extraction_provenance.json").is_file()


def test_complete_tile_cache_hit_does_not_load_encoder_or_rewrite_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slide2vec.artifacts import write_image_embedding

    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    _write_scalar_dataset(dataset_csv)
    images = ImageManifest.from_csv(dataset_csv)

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
                    metadata={
                        "artifact_type": "image_embeddings",
                        "sample_id": "s0",
                        "compatibility": {"encoder_name": "phikon"},
                    },
                )
            ]

    monkeypatch.setattr("soma.extract.tile_extraction.Model", OneShotBoundaryModel)
    kwargs = dict(
        unit="tile",
        encoder="phikon",
        cache=CacheOptions(enabled=True, root_dir=tmp_path / "cache"),
        output_dir=tmp_path / "output",
    )
    first = soma.extract(images, **kwargs)
    payload = first.feature_dir / "s0.pt"
    before = (payload.read_bytes(), payload.stat().st_mtime_ns)

    second = soma.extract(images, **kwargs)

    assert OneShotBoundaryModel.loads == 1
    assert (payload.read_bytes(), payload.stat().st_mtime_ns) == before
    assert torch.equal(second.load("s0"), torch.tensor([3.0, 4.0]))


def test_pooled_slide_extracts_through_the_same_public_interface(
    tmp_path: Path,
) -> None:
    tifffile = pytest.importorskip("tifffile")

    from soma.config import CacheOptions

    encoder_name = "fake"  # soma.testing's weight-free encoder
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

    source = soma.extract(
        ImageManifest.from_csv(dataset_csv),
        encoder=encoder_name,
        extraction=extraction(0.5, 32, method="precomputed_mask", masks=tissue_masks(0.0)),
        execution=ExecutionOptions(
            num_gpus=1,
            num_workers_per_gpu=0,
            num_preprocessing_workers=0,
            precision="fp32",
            batch_size=1,
        ),
        cache=CacheOptions(enabled=False),
        output_dir=tmp_path / "output",
    )

    record = _provenance(tmp_path / "output")
    assert record["provenance"]["kind"] == "pooled_bag"
    assert source.sample_ids == ["s0"]
    assert tuple(source.load("s0").shape) == (49, 3)
    assert record["artifacts"]["tiling_dir"] == str(tmp_path / "output/tiling")


def _stub_roi_sampling(monkeypatch: pytest.MonkeyPatch):
    """Stub hs2p sampling (ROIs at (0, 0) and (32, 0) of every slide), slide2vec's region
    encoder and the mask crop read; return the sampled slide ids and the model class
    (``encoded``: the coordinates of each encode call)."""
    from slide2vec.artifacts import write_dense_region

    sampled: list[str] = []

    def fake_tile_slide(slide, **_kwargs):
        sampled.append(slide.sample_id)
        xs = np.asarray([0, 32])
        return {None: SimpleNamespace(tiles=SimpleNamespace(x=xs, y=np.zeros_like(xs)))}

    class RegionModel:
        encoded: list[list[tuple[int, int]]] = []

        @classmethod
        def from_preset(cls, _name: str, **_kwargs):
            return cls()

        def embed_regions_dense(self, regions, *, dense, execution):
            type(self).encoded.append([tuple(c) for r in regions for c in r.coordinates])
            return [
                write_dense_region(
                    np.ones((4, 2, 2), dtype=np.float32),
                    output_dir=execution.output_dir,
                    sample_id=region.sample_id,
                    annotation=None,
                    x=int(x),
                    y=int(y),
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
                for region in regions
                for x, y in region.coordinates
            ]

    monkeypatch.setattr("hs2p.tile_slide", fake_tile_slide)
    monkeypatch.setattr("soma.extract.dense_slide_extraction.Model", RegionModel)
    monkeypatch.setattr(
        "soma.extract.dense.reader.read_mask_regions_within_slide",
        lambda path, *, regions, size, **_: [np.zeros((size[1], size[0]), np.uint8) for _ in regions],
    )
    return sampled, RegionModel


def test_rois_of_an_earlier_extraction_are_encoded_again_without_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``rois=`` re-encodes the ROIs an annotation-sampled extraction drew (another
    encoder or recipe over the same ROIs): no sampling, the same ROI ids."""
    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\ns0,s0.svs,s0-mask.tif\n", encoding="utf-8"
    )
    images = ImageManifest.from_csv(dataset_csv)
    sampled, RegionModel = _stub_roi_sampling(monkeypatch)
    dense = DenseOptions(spacing_um=0.5, target_size=32)
    masks = annotation_masks({"background": 0, "tumor": 1}, {"tumor": 0.0})
    sampled_source = soma.extract(
        images,
        annotations=AnnotationManifest.from_csv(dataset_csv),
        encoder="phikon",
        extraction=extraction(0.5, 32, masks=masks),
        dense=dense,
        cache=CacheOptions(enabled=False),
        output_dir=tmp_path / "sampled",
    )

    again = soma.extract(
        images,
        rois=sampled_source.rois,
        encoder="phikon",
        extraction=extraction(0.5, 32, masks=masks),
        dense=dense,
        cache=CacheOptions(enabled=False),
        output_dir=tmp_path / "again",
    )

    assert sampled == ["s0"]
    assert RegionModel.encoded == [[(0, 0), (32, 0)], [(0, 0), (32, 0)]]
    assert again.sample_ids == sampled_source.sample_ids == ["s0__x0_y0", "s0__x32_y0"]
    assert again.rois.children == {"s0__x0_y0": "s0", "s0__x32_y0": "s0"}
    assert _provenance(tmp_path / "again")["provenance"]["kind"] == "slide_region_dense"


@pytest.mark.parametrize(
    "replay_dense",
    [DenseOptions(spacing_um=1.0, target_size=32), DenseOptions(spacing_um=0.5, target_size=16)],
    ids=["other-spacing", "smaller-window"],
)
def test_rois_are_refused_at_another_geometry_than_they_were_sampled_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replay_dense: DenseOptions
) -> None:
    """An ROI's label crop is stored at the spacing and window it was sampled at: replayed
    at another one, its grid would not align with that crop, so the verb refuses. The
    binding survives a replay and a subset."""
    from soma.config import CacheOptions

    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label_mask_path\ns0,s0.svs,s0-mask.tif\n", encoding="utf-8"
    )
    images = ImageManifest.from_csv(dataset_csv)
    _, RegionModel = _stub_roi_sampling(monkeypatch)
    masks = annotation_masks({"background": 0, "tumor": 1}, {"tumor": 0.0})
    sampled_at = {
        "encoder": "phikon",
        "extraction": extraction(0.5, 32, masks=masks),
        "dense": DenseOptions(spacing_um=0.5, target_size=32),
        "cache": CacheOptions(enabled=False),
    }
    sampled_source = soma.extract(
        images, annotations=AnnotationManifest.from_csv(dataset_csv), output_dir=tmp_path / "sampled", **sampled_at
    )
    again = soma.extract(images, rois=sampled_source.rois, output_dir=tmp_path / "again", **sampled_at)

    with pytest.raises(ValueError, match="sampled at"):
        soma.extract(
            images,
            rois=again.rois.subset(["s0__x32_y0"]),
            encoder="phikon",
            extraction=extraction(replay_dense.spacing_um, replay_dense.target_size, masks=masks),
            dense=replay_dense,
            cache=CacheOptions(enabled=False),
            output_dir=tmp_path / "elsewhere",
        )
    assert len(RegionModel.encoded) == 2


@pytest.mark.parametrize(
    "replay_row",
    ["s0,s0.svs,s0-mask.tif,0.25", "s0,s0-rescanned.svs,s0-mask.tif,0.5"],
    ids=["other-spacing", "other-image-path"],
)
def test_rois_are_refused_with_a_manifest_whose_parent_row_differs(tmp_path: Path, replay_row: str) -> None:
    """An ROI is read from (and its cache identity keyed on) the slide row it was sampled
    from: replayed with a manifest whose row for that slide differs, the verb refuses
    rather than silently encode the old row."""
    from soma.config import CacheOptions
    from soma.extract import RoiSet
    from soma.extract.rois import roi_entries

    header = "sample_id,image_path,label_mask_path,spacing_at_level_0\n"
    sampled_csv = tmp_path / "sampled.csv"
    sampled_csv.write_text(header + "s0,s0.svs,s0-mask.tif,0.5\n", encoding="utf-8")
    replay_csv = tmp_path / "replay.csv"
    replay_csv.write_text(header + replay_row + "\n", encoding="utf-8")
    rois = RoiSet(
        roi_entries(ImageManifest.from_csv(sampled_csv), AnnotationManifest.from_csv(sampled_csv), {"s0": [(0, 0)]})
    )
    with pytest.raises(ValueError, match=r"\['s0'\].*differs"):
        soma.extract(
            ImageManifest.from_csv(replay_csv),
            encoder="phikon",
            rois=rois,
            cache=CacheOptions(enabled=False),
            output_dir=tmp_path / "out",
        )
