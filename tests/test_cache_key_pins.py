"""Literal cache keys and sample identities of a synthetic slide, tile and ROI.

Pinned on main before the legacy records were replaced by the manifests (slice 7a): the
cache layer must keep deriving the same directory keys and the same per-sample identity
stems from each manifest row, so every existing cache — the warm EVA bach root included —
is still a hit. Paths are literal (never opened), so the strings are stable across
machines. A failure here means existing caches would be re-extracted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import slide2vec

from soma.cache import resolve_image_cache, resolve_tile_cache, resolve_tiling_cache
from soma.config import CacheOptions, FeaturesConfig
from tests.experiment_configs import experiment
from tests.extraction_configs import annotation_masks, extraction


def _images(csv: Path):
    """The cache layer's input for the rows of ``csv``."""
    from soma.data import ImageManifest

    return ImageManifest.from_csv(csv)


# --------------------------------------------------------------------------- #
# A synthetic slide: tiling key + stems, pooled tile-feature key + stems.
# --------------------------------------------------------------------------- #

SLIDE_ROWS = (
    "sample_id,image_path,mask_path,spacing_at_level_0\n"
    "slide_a,/data/slides/slide_a.tif,/data/masks/slide_a.png,0.25\n"
    "slide_b,/data/slides/slide_b.svs,,\n"
)
PINNED_SLIDE_TILING = ("a7e9cf6d303b3c2a", {"slide_a": "339b417287a81924", "slide_b": "d13e0ee6379f82c1"})
PINNED_SLIDE_FEATURES = ("457638037afe772f", {"slide_a": "335a26b0198c7bf6", "slide_b": "98abc0e3cf921aa2"})


def test_a_synthetic_slide_keeps_its_tiling_and_feature_keys(tmp_path: Path):
    csv = tmp_path / "slides.csv"
    csv.write_text(SLIDE_ROWS)
    preprocessing = extraction(0.5, 224)

    tiling = resolve_tiling_cache(
        cache_root=tmp_path / "cache",
        images=_images(csv),
        preprocessing=preprocessing,
        backend_provenance={"requested_backend": "openslide"},
    )
    features = resolve_tile_cache(
        cache_root=tmp_path / "cache",
        images=_images(csv),
        tile_encoder_name="uni",
        preprocessing=preprocessing,
        execution=slide2vec.ExecutionOptions(precision="fp32"),
        output_variant="default",
        feature_type="bag",
        dtype="fp32",
    )

    assert (tiling.key, tiling.cache_stem_by_id) == PINNED_SLIDE_TILING
    assert (features.key, features.cache_stem_by_id) == PINNED_SLIDE_FEATURES


# --------------------------------------------------------------------------- #
# A synthetic tile: the given-image key of the EVA bach run (uni2, fp16).
# --------------------------------------------------------------------------- #

BACH_TILE = "/data/pathology/projects/fmtf-benchmark/raw/eva/bach/ICIAR2018_BACH_Challenge/test/Benign/b042.tif"
#: The warm EVA bach cache root holds ``image/a1040e43432f838f``, its sample signed so.
PINNED_TILE_FEATURES = ("a1040e43432f838f", {"bach_test_Benign_b042": "9323889883fd1fb1"})


def test_a_synthetic_tile_keeps_the_eva_bach_image_key(tmp_path: Path):
    csv = tmp_path / "tiles.csv"
    csv.write_text(f"sample_id,image_path\nbach_test_Benign_b042,{BACH_TILE}\n")

    features = resolve_image_cache(
        cache_root=tmp_path / "cache",
        images=_images(csv),
        tile_encoder_name="uni2",
        execution=slide2vec.ExecutionOptions(),
        output_variant="default",
        dtype="fp16",
    )

    assert (features.key, features.cache_stem_by_id) == PINNED_TILE_FEATURES


# --------------------------------------------------------------------------- #
# A synthetic ROI: annotation sampling through the real Pipeline, its pixel reads and
# the encoder stubbed. Pins the ROI sampling key and per-slide stems, and the dense key
# and per-ROI identity stems the extraction signs.
# --------------------------------------------------------------------------- #

ROI_TARGET = 32
ROI_PATCH = 16
ROI_FEATURE_DIM = 4
ROI_SLIDES = (
    "sample_id,image_path,mask_path,label_mask_path,spacing_at_level_0\n"
    "s0,/fake/s0.tif,/fake/s0_tissue.png,/fake/s0_mask.tif,0.5\n"
    "s1,/fake/s1.tif,,/fake/s1_mask.tif,0.25\n"
    "s2,/fake/s2.tif,,/fake/s2_mask.tif,\n"
    "s3,/fake/s3.tif,,/fake/s3_mask.tif,\n"
)
ROI_SPLITS = "sample_id,split,fold\ns0,train,0\ns1,train,0\ns2,tune,0\ns3,test,0\n"
#: The ROI sampling key and the per-slide stems naming each slide's coords and crops.
PINNED_ROI_SAMPLING = (
    "289c9fd9da4118af",
    ["0a15795fc0500d8f", "1e84924ccfdab731", "607c4934382e6b62", "cc6e9e793d75a8a6"],
)
#: The ROI grid key and each ROI's identity stem, as the extraction signs them.
PINNED_ROI_GRIDS = (
    "84972292be8cc6dc",
    {
        "s0__x0_y0": "d65dfb3725b20fe0",
        "s0__x32_y0": "0d390bc3be54c08d",
        "s1__x0_y0": "b98d04608983f6e3",
        "s1__x32_y0": "7540779b29fe0f42",
        "s2__x0_y0": "5adaf48e4c1a1025",
        "s2__x32_y0": "a01e26bf23c74a9d",
        "s3__x0_y0": "a0428cce8b2adb86",
        "s3__x32_y0": "c80df8ae7f17bb2b",
    },
)


class _FakeDenseModel:
    """``slide2vec.Model`` at soma's seam: one deterministic grid per ROI, written by
    slide2vec's own writer so the layout is upstream's."""

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "_FakeDenseModel":
        return cls()

    def embed_regions_dense(self, regions, *, dense, execution):
        from slide2vec.artifacts import write_dense_region

        from soma.dense.geometry import compute_dense_geometry

        geometry = compute_dense_geometry(target_size=int(dense.target_size), patch_size=(ROI_PATCH, ROI_PATCH))
        artifacts = []
        for region in regions:
            for x, y in region.coordinates:
                grid = np.ones((ROI_FEATURE_DIM, *geometry.grid_shape), dtype=np.float32)
                artifacts.append(
                    write_dense_region(
                        grid,
                        output_dir=execution.output_dir,
                        sample_id=region.sample_id,
                        annotation=region.annotation,
                        x=int(x),
                        y=int(y),
                        metadata={
                            "source_spacing_um": float(dense.spacing_um),
                            "effective_spacing_um": float(dense.spacing_um),
                            "artifact_type": "dense_embeddings",
                            "x": int(x),
                            "y": int(y),
                            "feature_dim": ROI_FEATURE_DIM,
                            "grid_shape": list(geometry.grid_shape),
                            "target_size": list(geometry.target_size),
                            "patch_size": list(geometry.patch_size),
                            "encoded_size": list(geometry.encoded_size),
                            "pad": list(geometry.pad),
                            "spacing_um": float(dense.spacing_um),
                            "pad_mode": dense.pad_mode,
                            "image_pad_value": dense.image_pad_value,
                            "window_size": dense.window_size,
                            "overlap": float(dense.overlap),
                            "feature_kind": dense.feature_kind,
                            "attention_blocks": [int(b) for b in dense.attention_blocks],
                            "attention_include_registers": bool(dense.attention_include_registers),
                        },
                    )
                )
        return artifacts


def _stub_roi_extraction(monkeypatch) -> None:
    import soma.dense.reader as reader
    import soma.dense_slide_extraction as dense_slide_extraction

    def sample(*args, sample_ids=None, **kwargs):
        return {str(sid): [(0, 0), (ROI_TARGET, 0)] for sid in sample_ids}

    def mask_regions(path, *, regions, size, **kwargs):
        return [np.full((size[1], size[0]), 1 if location[0] else 0, dtype=np.uint8) for location, _ in regions]

    monkeypatch.setattr(dense_slide_extraction, "sample_slide_rois", sample)
    monkeypatch.setattr(dense_slide_extraction, "Model", _FakeDenseModel)
    monkeypatch.setattr(reader, "read_mask_regions_within_slide", mask_regions)


def test_a_synthetic_roi_keeps_its_sampling_and_grid_keys(tmp_path: Path, monkeypatch):
    from soma.pipeline import Pipeline

    _stub_roi_extraction(monkeypatch)
    (tmp_path / "slides.csv").write_text(ROI_SLIDES)
    (tmp_path / "splits.csv").write_text(ROI_SPLITS)
    cache_root = tmp_path / "cache"
    config = experiment(
        tmp_path / "slides.csv",
        tmp_path / "splits.csv",
        output_dir=tmp_path / "runs",
        features=FeaturesConfig(
            encoder="uni",
            extraction=extraction(
                0.5, ROI_TARGET, masks=annotation_masks({"background": 0, "tumor": 1}, {"tumor": 0.0})
            ),
            cache=CacheOptions(root_dir=cache_root),
        ),
        decoder="lightweight_conv",
        head="segmentation",
        head_params={"classes": {"background": [0], "tumor": [1]}},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 1},
        evaluation={"metrics": ["mean_dice"], "save_overlays": False},
    )

    Pipeline(config).run()

    (sampling,) = (cache_root / "roi_sampling").iterdir()
    stems = sorted(path.stem for path in (sampling / "coords").glob("*.csv"))
    (grids,) = (cache_root / "dense").iterdir()
    signed = json.loads((grids / "cache_metadata.json").read_text())["sample_identity_signature_by_id"]

    assert (sampling.name, stems) == PINNED_ROI_SAMPLING
    assert (grids.name, signed) == PINNED_ROI_GRIDS
