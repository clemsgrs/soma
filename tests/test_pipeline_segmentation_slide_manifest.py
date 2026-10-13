"""Slide-manifest segmentation: slides + masks → hs2p sampling → ROI dense grids → train.

The genuinely-new A3a soma integration, tested deterministically and offline. The two
cross-repo extraction primitives — hs2p annotation sampling (``sample_slide_rois``) and the
slide2vec dense encode (``Model.embed_regions_dense``) — are stubbed at their import seams
(each is independently tested in its own repo: hs2p ``test_annotation_coverage``, slide2vec
``test_dense_stage``); everything in between (ROI manifest + split propagation, the dense
cache keyed on the sampling spec, explicit ROI training context, and the full cached dense
training/eval) runs for real.

The dense stub writes through slide2vec's own :func:`write_dense_region`, so the on-disk
layout and sidecar these tests read back are upstream's rather than the stub's idea of
them — the point of the migration is that soma no longer owns that schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import slide2vec

from soma.config import CacheOptions, Component, FeaturesConfig
from tests.extraction_configs import annotation_masks, extraction
from tests.experiment_configs import experiment

NUM_CLASSES = 2
# A patch-16 encoder that accepts a variable encoder input, so the tiny 32 px fixtures
# below are a geometry the contract can actually honour (ADR 0006): declaring a 32 px
# dense input on a fixed-224 encoder now raises at resolve time, as it should.
ENCODER = "uni"
TARGET = 32
PATCH = 16
GRID = TARGET // PATCH  # 2
FEATURE_DIM = 4
PIXEL_MAPPING = {"background": 0, "tumor": 1}
#: hs2p's MaskSamplingConfig holds every label's raw values as a tuple.
HELD_PIXEL_MAPPING = {"background": (0,), "tumor": (1,)}
# Re-recorded on main after annotation sampling defaulted an unset filter to a_t=0.
#: The ROI sampling key and the dense grid key(s) of the end-to-end run (pinned in slice 6c).
PINNED_ROI_CACHE_KEYS = ("289c9fd9da4118af", ["84972292be8cc6dc"])


def _roi_extraction(*, min_cov: float = 0.0, **tiling_fields) -> slide2vec.PreprocessingConfig:
    """slide2vec's extraction for slide-manifest ROIs: TARGET px at 0.5 µm/px over the
    {background, tumor} annotation vocabulary."""
    return extraction(
        0.5, TARGET, masks=annotation_masks(PIXEL_MAPPING, {"tumor": min_cov}), tiling_fields=tiling_fields
    )


# --------------------------------------------------------------------------- #
# Fixtures: a synthetic slide manifest + slide-level splits.
# --------------------------------------------------------------------------- #


def _write_slide_manifest(root: Path, slide_ids: list[str]) -> tuple[Path, Path]:
    manifest = root / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path\n"
        + "\n".join(f"{sid},/fake/{sid}.tif,/fake/{sid}_mask.tif" for sid in slide_ids)
        + "\n"
    )
    splits = root / "slide_splits.csv"
    assign = {
        slide_ids[0]: "train",
        slide_ids[1]: "train",
        slide_ids[2]: "tune",
        slide_ids[3]: "test",
    }
    splits.write_text(
        "sample_id,split,fold\n" + "\n".join(f"{sid},{s},0" for sid, s in assign.items()) + "\n"
    )
    return manifest, splits


def _coords_for(slide_id: str) -> list[tuple[int, int]]:
    # Two deterministic ROIs per slide.
    return [(0, 0), (TARGET, 0)]


class _FakeDenseModel:
    """Stands in for ``slide2vec.Model`` at soma's import seam.

    Records the ``embed_regions_dense`` contract soma states — which slides, which
    coordinates, and the ``DenseOptions``/``ExecutionOptions`` it built — and persists a
    deterministic grid per ROI through slide2vec's own writer, so the layout under test is
    upstream's.
    """

    calls: list[dict] = []

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "_FakeDenseModel":
        return cls(name, **kwargs)

    def embed_regions_dense(self, regions, *, dense, execution):
        from slide2vec.artifacts import write_dense_region

        from soma.data.geometry import compute_dense_geometry

        type(self).calls.append(
            {
                "name": self.name,
                "regions": [
                    (r.sample_id, [tuple(int(v) for v in c) for c in r.coordinates])
                    for r in regions
                ],
                "source_spacings": {r.sample_id: r.spacing_at_level_0 for r in regions},
                "dense": dense,
                "execution": execution,
            }
        )
        geometry = compute_dense_geometry(
            target_size=int(dense.target_size), patch_size=(PATCH, PATCH)
        )
        rng = np.random.default_rng(0)
        artifacts = []
        for region in regions:
            for x, y in region.coordinates:
                grid = rng.standard_normal(
                    (FEATURE_DIM, geometry.grid_shape[0], geometry.grid_shape[1])
                ).astype(np.float32)
                artifacts.append(
                    write_dense_region(
                        grid,
                        output_dir=execution.output_dir,
                        sample_id=region.sample_id,
                        annotation=region.annotation,
                        x=int(x),
                        y=int(y),
                        metadata=_dense_sidecar(
                            dense,
                            geometry,
                            grid,
                            source_spacing_um=region.spacing_at_level_0,
                            x=int(x),
                            y=int(y),
                        ),
                    )
                )
        return artifacts


#: Relative band within which the fake slide2vec reads a source natively (its read plan).
_FAKE_READ_TOLERANCE = 0.05


def _dense_sidecar(dense, geometry, grid, *, source_spacing_um=None, x=0, y=0) -> dict:
    """The geometry sidecar slide2vec writes next to every dense ROI grid.

    Like slide2vec's read plan, a source within tolerance of the request is read natively,
    so the grid's ``effective_spacing_um`` is the source's own spacing, not the request.
    """
    requested = float(dense.spacing_um)
    source = requested if source_spacing_um is None else float(source_spacing_um)
    within = abs(source - requested) / requested <= _FAKE_READ_TOLERANCE
    return {
        "source_spacing_um": source,
        "effective_spacing_um": source if within else requested,
        "artifact_type": "dense_embeddings",
        # The ROI's level-0 origin, spelled as slide2vec's ``_region_metadata`` spells it.
        "x": int(x),
        "y": int(y),
        "feature_dim": int(grid.shape[0]),
        "grid_shape": [int(geometry.grid_shape[0]), int(geometry.grid_shape[1])],
        "target_size": [int(geometry.target_size[0]), int(geometry.target_size[1])],
        "patch_size": [int(geometry.patch_size[0]), int(geometry.patch_size[1])],
        "encoded_size": [int(geometry.encoded_size[0]), int(geometry.encoded_size[1])],
        "pad": [int(geometry.pad[0]), int(geometry.pad[1])],
        "spacing_um": float(dense.spacing_um),
        "pad_mode": dense.pad_mode,
        "image_pad_value": dense.image_pad_value,
        "window_size": dense.window_size,
        "overlap": float(dense.overlap),
        "feature_kind": dense.feature_kind,
        "attention_blocks": [int(b) for b in dense.attention_blocks],
        "attention_include_registers": bool(dense.attention_include_registers),
    }


def _patch_dense_model(monkeypatch) -> type[_FakeDenseModel]:
    import soma.extract.dense_slide_extraction as dse

    _FakeDenseModel.calls = []
    monkeypatch.setattr(dse, "Model", _FakeDenseModel)
    return _FakeDenseModel


def _patch_mask_regions(monkeypatch, window, *, calls: list[dict] | None = None):
    """Stub the annotation-raster read at its seam: ``window(path, location=, spacing_um=,
    size=, **aligned_to)`` gives each ROI's labels; each slide opening is one ``calls`` entry."""
    import soma.extract.dense.reader as reader_mod

    def _fake(path, *, regions, size, **kwargs):
        if calls is not None:
            calls.append({"path": path, "regions": list(regions), "size": size, **kwargs})
        return [
            window(path, location=location, spacing_um=spacing_um, size=size, **kwargs)
            for location, spacing_um in regions
        ]

    monkeypatch.setattr(reader_mod, "read_mask_regions_within_slide", _fake)


def _patch_extraction(monkeypatch):
    """Stub hs2p sampling + slide2vec encode + the WSI mask/image region reads (offline)."""
    import soma.extract.dense_slide_extraction as dse
    import soma.extract.dense.reader as reader_mod

    # hs2p sampling → known coords per slide.
    monkeypatch.setattr(
        dse,
        "sample_slide_rois",
        lambda images, annotations, **kw: {
            sid: _coords_for(sid) for sid in (kw.get("sample_ids") or images.sample_ids)
        },
    )
    _patch_dense_model(monkeypatch)

    # Annotation raster read (ROI sampling's crop writer) → a deterministic label window
    # per ROI origin; one call per slide opening.
    mask_reads: list[dict] = []
    _patch_mask_regions(
        monkeypatch,
        lambda path, *, location, spacing_um, size, **kw: np.full(
            (size[1], size[0]), 1 if location[0] else 0, dtype=np.uint8
        ),  # ROI(0,0)→all bg, ROI(32,0)→all tumor
        calls=mask_reads,
    )

    # Image region read → a deterministic RGB window per ROI (the overlay writer reads the
    # ROI window from the whole-slide image_path, never opening the gigapixel slide).
    def _fake_image_region(
        path,
        *,
        location,
        size,
        spacing_um,
        backend,
        tolerance,
        interpolation="area",
        spacing_at_level_0=None,
    ):
        w, h = size
        return np.zeros((h, w, 3), dtype=np.uint8)

    monkeypatch.setattr(reader_mod, "read_image_region_at_spacing", _fake_image_region)
    return mask_reads


def _config(
    root: Path,
    manifest: Path,
    splits: Path,
    *,
    min_cov: float = 0.0,
    kind: str = "cached",
    spacing_policy: str = "strict",
    cache: CacheOptions | None = None,
    **tiling_fields,
):
    return experiment(
        manifest,
        splits,
        output_dir=root / 'out',
        features=FeaturesConfig(kind=kind, encoder=ENCODER, extraction=_roi_extraction(min_cov=min_cov, **tiling_fields), spacing_policy=spacing_policy, cache=cache if cache is not None else CacheOptions()),
        decoder="lightweight_conv",
        head="segmentation",
        head_params={"classes": {"background": [0], "tumor": [1]}},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 1},
        evaluation={"metrics": ["mean_dice", "mean_iou"]},
        scheduler="cosine",
    )


# --------------------------------------------------------------------------- #
# The input mode: inferred from the cohort, not from a masks block (ADR 0019).
# --------------------------------------------------------------------------- #


def test_the_slide_manifest_mode_is_inferred_from_the_cohorts_label_mask_path_column(tmp_path: Path):
    """A segmentation cohort with a ``label_mask_path`` column whose masks declare an
    annotation vocabulary samples ROIs from whole slides; a pre-cropped cohort carries
    the same column but keeps hs2p's binary tissue vocabulary, so it is read as tiles."""
    from dataclasses import replace

    from soma.pipeline.build import extraction_unit

    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    annotated = _config(tmp_path, manifest, splits)
    pre_cropped = replace(
        annotated,
        features=replace(annotated.features, extraction=extraction(0.5, TARGET)),
        learner=replace(
            annotated.learner,
            params={**annotated.learner.params, "head": Component("segmentation", {"num_classes": 2})},
        ),
    )

    assert extraction_unit(annotated) == "slide"
    assert extraction_unit(pre_cropped) == "tile"

    points_only = tmp_path / "points.csv"
    points_only.write_text("sample_id,image_path,points_path\ns0,/fake/s0.tif,/fake/s0.csv\n")
    assert extraction_unit(replace(annotated, cohort=replace(annotated.cohort, records=points_only))) == "tile"


# --------------------------------------------------------------------------- #
# End-to-end wiring through the real Pipeline.run().
# --------------------------------------------------------------------------- #


def test_slide_manifest_runs_end_to_end(tmp_path: Path, monkeypatch):
    from soma.pipeline import Pipeline

    mask_reads = _patch_extraction(monkeypatch)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    pipeline = Pipeline(_config(tmp_path, manifest, splits))
    result = pipeline.run()

    # Each slide's annotation raster is opened once, at ROI sampling, aligned to its own
    # slide and read against the sampling vocabulary; training and evaluation read the
    # stored crops instead.
    assert sorted(str(read["reference_path"]) for read in mask_reads) == [
        f"/fake/{sid}.tif" for sid in ["s0", "s1", "s2", "s3"]
    ]
    assert all(read["pixel_mapping"] == HELD_PIXEL_MAPPING for read in mask_reads)
    assert all(
        read["regions"] == [((0, 0), 0.5), ((TARGET, 0), 0.5)] for read in mask_reads
    )

    assert "test/mean_dice" in result.summary
    assert result.fold_results[0].test_evaluations["test"].metrics["mean_dice"] >= 0.0
    assert pipeline.cohort.sample_ids == ["s0", "s1", "s2", "s3"]
    assert pipeline.cohort.folds[0].train == ("s0", "s1")
    assert pipeline.cohort.folds[0].tune == ("s2",)
    assert pipeline.cohort.folds[0].tests == {"test": ("s3",)}

    provenance_files = list((tmp_path / "out").rglob("dense_source.json"))
    assert len(provenance_files) == 1
    provenance = json.loads(provenance_files[0].read_text(encoding="utf-8"))
    assert provenance["kind"] == "slide_manifest_dense_cache"
    assert provenance["parent_dataset_csv"] == str(manifest)
    assert Path(provenance["dataset_csv"]).name == "dataset.csv"
    assert "splits_csv" not in provenance
    assert "parent_splits_csv" not in provenance

    # Extraction records the ROI table it drew; the ROI cohort is derived in memory.
    roi_manifests = list((tmp_path / "out").rglob("segmentation_rois/dataset.csv"))
    assert roi_manifests, "ROI manifest not written"
    import pandas as pd

    roi_df = pd.read_csv(roi_manifests[0])
    assert len(roi_df) == 8  # 4 slides x 2 coords
    assert set(roi_df.columns) >= {
        "sample_id",
        "image_path",
        "label_mask_path",
        "region_x",
        "region_y",
    }
    assert set(roi_df["region_x"]) == {0, TARGET}
    assert not list((tmp_path / "out").rglob("roi_splits.csv"))

    # One raw-value crop per ROI under the ROI sampling entry, addressed by the dataset.
    (sampling_entry,) = (tmp_path / "out").rglob("roi_sampling/*")
    # The ROI sampling and dense grid keys derive from the features config alone; a
    # config-layout change must not move them (pinned in slice 6c).
    dense_entries = sorted(path.name for path in (tmp_path / "out").rglob("dense/*") if path.is_dir())
    assert (sampling_entry.name, dense_entries) == PINNED_ROI_CACHE_KEYS
    from PIL import Image

    for row in roi_df.itertuples():
        crop_path = Path(row.label_mask_crop_path)
        assert crop_path.parent.parent == sampling_entry / "masks"
        assert crop_path.name == f"{row.region_x}_{row.region_y}.png"
        with Image.open(crop_path) as crop:
            assert np.asarray(crop).tolist() == [[1 if row.region_x else 0] * TARGET] * TARGET


def test_slide_manifest_overlays_read_each_roi_window(tmp_path: Path, monkeypatch):
    """``save_segmentation_overlays`` (on by default) blends each ROI's prediction and
    supervision over its own window of the whole slide, read at the run spacing."""
    import soma.extract.dense.reader as reader_mod
    from soma.pipeline import Pipeline

    _patch_extraction(monkeypatch)
    reads: list[tuple[str, tuple[int, int], tuple[int, int], float]] = []

    def _window(path, *, location, size, spacing_um, backend, tolerance, interpolation="area", spacing_at_level_0=None):
        reads.append((Path(path).name, tuple(location), tuple(size), spacing_um))
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    monkeypatch.setattr(reader_mod, "read_image_region_at_spacing", _window)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    Pipeline(_config(tmp_path, manifest, splits)).run()

    (overlays,) = (tmp_path / "out").rglob("overlays/test")
    for kind in ("pred", "gt"):
        assert sorted(path.name for path in (overlays / kind).glob("*.png")) == [
            "s3__x0_y0.png",
            "s3__x32_y0.png",
        ]
    assert {read for read in reads if read[0] == "s3.tif"} == {
        ("s3.tif", (0, 0), (TARGET, TARGET), 0.5),
        ("s3.tif", (TARGET, 0), (TARGET, TARGET), 0.5),
    }


def test_slide_manifest_pipeline_reads_coarse_masks_at_native_grid_spacing(
    tmp_path: Path, monkeypatch
):
    from soma.pipeline import Pipeline

    _patch_extraction(monkeypatch)
    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path,spacing_at_level_0\n"
        "s0,/fake/s0.tif,/fake/s0_mask.tif,0.5\n"
        "s1,/fake/s1.tif,/fake/s1_mask.tif,0.4862\n"
        "s2,/fake/s2.tif,/fake/s2_mask.tif,0.5\n"
        "s3,/fake/s3.tif,/fake/s3_mask.tif,0.657476464\n"
    )
    splits = tmp_path / "slide_splits.csv"
    splits.write_text(
        "sample_id,split,fold\n" "s0,train,0\n" "s1,train,0\n" "s2,tune,0\n" "s3,test,0\n"
    )
    observed: dict[str, set[float]] = {}
    observed_backends: set[str] = set()
    observed_slide_backends: set[str] = set()

    def _fake_mask(path, *, location, size, spacing_um, backend, **kwargs):
        observed.setdefault(Path(path).stem, set()).add(spacing_um)
        observed_backends.add(backend)
        observed_slide_backends.add(kwargs["reference_backend"])
        return np.zeros((size[1], size[0]), dtype=np.uint8)

    _patch_mask_regions(monkeypatch, _fake_mask)
    config = _config(
        tmp_path,
        manifest,
        splits,
        spacing_policy="native_if_coarser",
        backend="auto",
        mask_backend="openslide",
    )

    Pipeline(config).run()

    assert observed["s0_mask"] == {0.5}
    # Within tolerance of 0.5, s1 is read natively: its grids record 0.4862 and its ROI
    # masks must cover the same area, so they are read at 0.4862 too.
    assert observed["s1_mask"] == {0.4862}
    assert observed["s3_mask"] == {0.657476464}
    assert observed_backends == {"openslide"}
    # The slide each mask is aligned to is opened with the slide reader, not the mask's.
    assert observed_slide_backends == {"auto"}


def _extract_slide_rois(tmp_path: Path, manifest: Path, **options):
    """``soma.extract`` over a slide manifest, its annotation sampling and encoder stubbed
    (``_patch_extraction`` or ``_patch_dense_model`` installed by the caller)."""
    import soma
    from soma.data import AnnotationManifest, ImageManifest

    tiling = options.setdefault("extraction", _roi_extraction()).tiling
    # The grid reads ROIs at the tiling they were sampled at, as a run's plan derives it.
    options.setdefault(
        "dense",
        slide2vec.DenseOptions(
            spacing_um=float(tiling.requested_spacing_um),
            target_size=int(tiling.requested_tile_size_px),
            tolerance=float(tiling.tolerance),
            backend=tiling.backend,
        ),
    )
    options.setdefault("cache", CacheOptions(enabled=False))
    return soma.extract(
        ImageManifest.from_csv(manifest),
        annotations=AnnotationManifest.from_csv(manifest),
        encoder=ENCODER,
        output_dir=tmp_path / "extraction",
        **options,
    )


def _slides(manifest: Path):
    """The image and annotation manifests of a slide manifest, as sampling reads them."""
    from soma.data import AnnotationManifest, ImageManifest

    return ImageManifest.from_csv(manifest), AnnotationManifest.from_csv(manifest)


def _stub_sampling(monkeypatch, coords_by_slide: dict[str, list[tuple[int, int]]]) -> None:
    import soma.extract.dense_slide_extraction as dse

    monkeypatch.setattr(
        dse,
        "sample_slide_rois",
        lambda images, annotations, **kw: {
            sid: coords_by_slide[sid] for sid in (kw.get("sample_ids") or images.sample_ids)
        },
    )
    _patch_mask_regions(
        monkeypatch, lambda path, *, location, spacing_um, size, **kw: np.zeros((size[1], size[0]), np.uint8)
    )


def test_slide_manifest_propagates_slide_splits_to_rois(tmp_path: Path, monkeypatch):
    """Every ROI inherits its parent slide's split/fold — no split creation."""
    from soma.data import Cohort

    _patch_extraction(monkeypatch)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    source = _extract_slide_rois(tmp_path, manifest)
    projected = Cohort.from_csv(manifest, splits).derived(source.rois.children)

    # s0/s1 → train, s2 → tune, s3 → test; each slide has 2 ROIs.
    assert projected.folds[0].train == (
        "s0__x0_y0",
        "s0__x32_y0",
        "s1__x0_y0",
        "s1__x32_y0",
    )
    assert projected.folds[0].tune == ("s2__x0_y0", "s2__x32_y0")
    assert projected.folds[0].tests == {"test": ("s3__x0_y0", "s3__x32_y0")}


def test_roi_dataset_rewrite_never_exposes_a_partial_file(tmp_path: Path, monkeypatch):
    """Sibling ``run.folds`` launches rewrite the ROI manifest while others read it (#482)."""
    _patch_extraction(monkeypatch)
    import os

    from soma.data import AnnotationManifest, ImageManifest
    from soma.extract.rois import roi_entries, write_roi_table

    manifest, _ = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    images = ImageManifest.from_csv(manifest)
    coords = {sid: _coords_for(sid) for sid in images.sample_ids}
    rois = roi_entries(images, AnnotationManifest.from_csv(manifest), coords)
    out_dir = tmp_path / "rois"
    complete = write_roi_table(rois, out_dir=out_dir).read_bytes()

    seen_at_swap: list[bytes] = []
    real_replace = os.replace

    def _replace(src, dst):
        seen_at_swap.append(Path(dst).read_bytes())
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", _replace)
    rewritten = write_roi_table(rois, out_dir=out_dir)

    # Until the swap a reader still sees the previous complete file.
    assert seen_at_swap == [complete]
    assert rewritten.read_bytes() == complete
    assert [p.name for p in out_dir.iterdir()] == ["dataset.csv"]


def test_slide_source_spacing_survives_roi_derivation_and_reaches_slide2vec(
    tmp_path: Path, monkeypatch
):
    slides = tmp_path / "slides.csv"
    slides.write_text(
        "sample_id,image_path,label_mask_path,spacing_at_level_0\n"
        "s0,/fake/s0.tif,/fake/s0_mask.tif,0.25\n"
    )
    _stub_sampling(monkeypatch, {"s0": [(0, 0)]})
    model = _patch_dense_model(monkeypatch)

    _extract_slide_rois(tmp_path, slides)

    assert model.calls[0]["source_spacings"] == {"s0": 0.25}


def test_slide_manifest_dense_extraction_forwards_num_gpus(tmp_path: Path, monkeypatch):
    manifest = tmp_path / "slides.csv"
    manifest.write_text("sample_id,image_path,label_mask_path\ns0,/fake/s0.tif,/fake/s0_mask.tif\n")
    _stub_sampling(monkeypatch, {"s0": [(0, 0)]})
    model = _patch_dense_model(monkeypatch)

    _extract_slide_rois(tmp_path, manifest, execution=slide2vec.ExecutionOptions(num_gpus=2))

    assert model.calls[0]["execution"].num_gpus == 2


def test_slide_manifest_dense_extraction_groups_sources_by_effective_spacing(
    tmp_path: Path, monkeypatch
):
    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path,spacing_at_level_0\n"
        "ordinary,/fake/ordinary.tif,/fake/ordinary_mask.tif,0.5\n"
        "coarse,/fake/coarse.tif,/fake/coarse_mask.tif,0.657476464\n"
    )
    _stub_sampling(monkeypatch, {"ordinary": [(0, 0)], "coarse": [(0, 0)]})
    model = _patch_dense_model(monkeypatch)
    _extract_slide_rois(
        tmp_path,
        manifest,
        extraction=_roi_extraction(tolerance=0.05),
        spacing_policy="native_if_coarser",
    )

    assert [(call["regions"], call["dense"].spacing_um) for call in model.calls] == [
        ([("ordinary", [(0, 0)])], 0.5),
        ([("coarse", [(0, 0)])], 0.657476464),
    ]


def test_slide_sampling_forwards_source_spacing_to_hs2p(tmp_path: Path, monkeypatch):
    from hs2p import SlideSpec
    from soma.extract.dense_slide_extraction import sample_slide_rois

    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path,spacing_at_level_0\n"
        "s0,/fake/s0.tif,/fake/s0_mask.tif,0.25\n"
    )
    captured: list[SlideSpec] = []

    def _tile_slide(slide, **kwargs):
        captured.append(slide)
        return {
            None: SimpleNamespace(
                tiles=SimpleNamespace(
                    x=np.asarray([0], dtype=np.int64),
                    y=np.asarray([0], dtype=np.int64),
                )
            )
        }

    monkeypatch.setattr("hs2p.tile_slide", _tile_slide)
    coords = sample_slide_rois(*_slides(manifest), preprocessing=_roi_extraction())

    assert coords == {"s0": [(0, 0)]}
    assert captured[0].spacing_at_level_0 == 0.25


def test_slide_sampling_uses_native_spacing_only_for_coarser_sources(tmp_path: Path, monkeypatch):
    from soma.extract.dense_slide_extraction import sample_slide_rois

    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path,spacing_at_level_0\n"
        "ordinary,/fake/ordinary.tif,/fake/ordinary_mask.tif,0.5\n"
        "coarse,/fake/coarse.tif,/fake/coarse_mask.tif,0.657476464\n"
    )
    observed: list[tuple[str, float]] = []

    def _tile_slide(slide, **kwargs):
        observed.append((slide.sample_id, kwargs["tiling"].requested_spacing_um))
        return {
            None: SimpleNamespace(
                tiles=SimpleNamespace(
                    x=np.asarray([0], dtype=np.int64),
                    y=np.asarray([0], dtype=np.int64),
                )
            )
        }

    monkeypatch.setattr("hs2p.tile_slide", _tile_slide)
    sample_slide_rois(
        *_slides(manifest),
        preprocessing=_roi_extraction(tolerance=0.05),
        spacing_policy="native_if_coarser",
    )

    assert observed == [("ordinary", 0.5), ("coarse", 0.657476464)]


def test_slide_sampling_forwards_the_filter_the_roi_key_records(tmp_path: Path, monkeypatch):
    """hs2p filters annotation contours (``a_t`` × ``ref_tile_size``²) and runs pixel QC with
    the ``filtering`` it is given; the roi_sampling key hashes the held filter, so the
    sampler hands hs2p that filter, and resolves an unset one as the key does: with no
    contour-area threshold (``a_t=0``), so every annotated component is sampled. Tissue
    tiling keeps hs2p's default threshold."""
    from hs2p import FilterConfig
    from soma.extract.dense_slide_extraction import sample_slide_rois
    from soma.models.encoders.validation import resolve_preprocessing_config

    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path,spacing_at_level_0\n"
        "s0,/fake/s0.tif,/fake/s0_mask.tif,0.5\n"
    )
    filters: list[FilterConfig] = []

    def _tile_slide(slide, **kwargs):
        filters.append(kwargs["filtering"])
        return {
            None: SimpleNamespace(
                tiles=SimpleNamespace(
                    x=np.asarray([0], dtype=np.int64),
                    y=np.asarray([0], dtype=np.int64),
                )
            )
        }

    monkeypatch.setattr("hs2p.tile_slide", _tile_slide)
    held = FilterConfig(ref_tile_size=TARGET, a_t=2, filter_white=True)
    preprocessing = extraction(
        0.5, TARGET, masks=annotation_masks(PIXEL_MAPPING, {"tumor": 0.0}), filter=held
    )
    sample_slide_rois(*_slides(manifest), preprocessing=preprocessing)
    unset = _roi_extraction()
    assert unset.filter is None
    sample_slide_rois(*_slides(manifest), preprocessing=unset)

    resolved = resolve_preprocessing_config(unset, encoder_name=ENCODER).filter
    assert filters == [held, resolved]
    assert (filters[1].a_t, filters[1].ref_tile_size) == (0, TARGET)
    assert resolved == FilterConfig(ref_tile_size=TARGET, a_t=0)

    tissue = resolve_preprocessing_config(extraction(0.5, TARGET), encoder_name=ENCODER).filter
    assert tissue == FilterConfig(ref_tile_size=TARGET)
    assert tissue.a_t == 4


# --------------------------------------------------------------------------- #
# Cache key folds the sampling spec.
# --------------------------------------------------------------------------- #


def _roi_dense_key(signature) -> str:
    from soma.extract.cache.keys import build_dense_cache_key

    return build_dense_cache_key(
        tile_encoder_name=ENCODER,
        patch_size=(PATCH, PATCH),
        dense=slide2vec.DenseOptions(spacing_um=0.5, target_size=TARGET),
        execution=None,
        sampling_signature=signature,
    )


def test_distinct_sampling_specs_yield_distinct_cache_keys():
    from soma.extract.cache import roi_sampling_signature

    sig_a = roi_sampling_signature(_roi_extraction(min_cov=0.1))
    sig_b = roi_sampling_signature(_roi_extraction(min_cov=0.5))
    assert _roi_dense_key(sig_a) != _roi_dense_key(sig_b)  # distinct min_coverage ⇒ distinct cache
    assert _roi_dense_key(sig_a) == _roi_dense_key(roi_sampling_signature(_roi_extraction(min_cov=0.1)))
    # Absent (pre-cropped tiles) differs from any sampled key, and is itself stable.
    assert _roi_dense_key(None) != _roi_dense_key(sig_a)
    assert _roi_dense_key(None) == _roi_dense_key(None)


def test_spacing_policy_changes_sampling_cache_key():
    from soma.extract.cache import roi_sampling_signature

    strict = roi_sampling_signature(_roi_extraction(min_cov=0.1), spacing_policy="strict")
    native = roi_sampling_signature(_roi_extraction(min_cov=0.1), spacing_policy="native_if_coarser")

    assert native["spacing_policy"] == "native_if_coarser"
    assert _roi_dense_key(strict) != _roi_dense_key(native)


# --------------------------------------------------------------------------- #
# Deferred-combo guards.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"kind": "live"}, "Live"),
    ],
)
def test_slide_manifest_deferred_combos_raise(tmp_path: Path, overrides, match):
    from soma.pipeline import Pipeline

    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    cfg = _config(tmp_path, manifest, splits, **overrides)
    with pytest.raises(NotImplementedError, match=match):
        _attach(Pipeline(cfg), tmp_path / "out" / "run")


def test_slide_manifest_declares_the_sliding_window_to_slide2vec(tmp_path: Path, monkeypatch):
    """A window smaller than the target is expressed as DenseOptions window/overlap.

    This is how a native-only encoder reaches a larger supervision tile — phikon@224 over
    512 px ROIs, the BEETLE recipe — stated here at fixture scale.

    The sliding itself — encoder slid over patch-aligned windows of each padded ROI, grids
    blended back — is slide2vec's, and is tested there. soma's remaining responsibility is
    to state the window it wants and to cache what comes back at the target grid size.
    """
    # One slide, two sampled ROIs.
    manifest = tmp_path / "slides.csv"
    manifest.write_text("sample_id,image_path,label_mask_path\ns0,/fake/s0.tif,/fake/s0_mask.tif\n")
    _stub_sampling(monkeypatch, {"s0": [(0, 0), (32, 0)]})
    model = _patch_dense_model(monkeypatch)

    WINDOW = 16  # < TARGET (32) -> genuine sliding
    store = _extract_slide_rois(
        tmp_path,
        manifest,
        dense=slide2vec.DenseOptions(spacing_um=0.5, target_size=TARGET, window_size=WINDOW, overlap=0.5),
        execution=slide2vec.ExecutionOptions(batch_size=2),
    )

    (call,) = model.calls
    assert call["dense"].window_size == WINDOW
    assert call["dense"].overlap == 0.5
    assert call["dense"].target_size == TARGET
    assert call["dense"].spacing_um == 0.5
    assert call["regions"] == [("s0", [(0, 0), (32, 0)])]
    grid = store.load("s0__x0_y0")
    assert grid.shape == (FEATURE_DIM, GRID, GRID)  # stitched back to the target grid


def test_slide_manifest_grids_are_namespaced_per_slide(tmp_path: Path, monkeypatch):
    """ROI grids live at slide2vec's ``<slide_id>/<x>_<y>.pt``, addressed from the ROI.

    The ROI id is never split back apart to find its grid: the ROI's slide and origin are
    the address (ADR 0007). A slide id that itself contains the ``__x``/``_y`` separators
    therefore resolves correctly.
    """
    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path\ns0__x1_y2,/fake/s0.tif,/fake/s0_mask.tif\n"
    )
    _stub_sampling(monkeypatch, {"s0__x1_y2": [(0, 0)]})
    _patch_dense_model(monkeypatch)

    store = _extract_slide_rois(tmp_path, manifest)

    assert (store.feature_dir / "s0__x1_y2" / "0_0.pt").is_file()
    assert store.load("s0__x1_y2__x0_y0").shape == (FEATURE_DIM, GRID, GRID)


# --------------------------------------------------------------------------- #
# Cached ROI sampling (#365): a warm relaunch never re-pays hs2p sampling.
# --------------------------------------------------------------------------- #


class _CountingSampler:
    """Counting stand-in for ``sample_slide_rois`` that honours the slide-id filter.

    Records which slide ids each call sampled, so the tests can assert "sampling is
    not re-paid" as an external behaviour: zero calls on a full hit, only the
    added/changed slides on a partial hit.
    """

    def __init__(self, coords_for=None) -> None:
        self.calls: list[list[str]] = []
        self._coords_for = coords_for or _coords_for

    def __call__(self, images, annotations, *, preprocessing, spacing_policy="strict", sample_ids=None):
        ids = list(images.sample_ids) if sample_ids is None else [str(s) for s in sample_ids]
        self.calls.append(sorted(ids))
        return {sid: self._coords_for(sid) for sid in ids}


def _install_counting_sampler(monkeypatch, sampler: _CountingSampler) -> None:
    import soma.extract.dense_slide_extraction as dse

    _patch_extraction(monkeypatch)
    monkeypatch.setattr(dse, "sample_slide_rois", sampler)


def _launch(
    tmp_path: Path,
    manifest: Path,
    splits: Path,
    run_name: str,
    *,
    cache_enabled: bool = True,
):
    """Drive the context builder as one launch (a fresh Pipeline, its own run dir)."""
    from soma.pipeline import Pipeline

    cfg = _config(tmp_path, manifest, splits, cache=CacheOptions(enabled=cache_enabled))
    return _attach(Pipeline(cfg), tmp_path / "out" / run_name)


def _attach(pipeline, run_dir: Path):
    """Build the run's features as ``Pipeline.run`` does (extraction included)."""
    return pipeline._features(run_dir)


def _roi_csv_bytes(tmp_path: Path, run_name: str) -> bytes:
    roi_dir = tmp_path / "out" / run_name / "segmentation_rois"
    return (roi_dir / "dataset.csv").read_bytes()


def test_slide_manifest_relaunch_skips_sampling_and_reproduces_manifest(
    tmp_path: Path, monkeypatch, caplog
):
    """Second launch with an unchanged config: zero sampler calls, byte-identical
    ROI manifest derived from cached coords."""
    import logging

    sampler = _CountingSampler()
    _install_counting_sampler(monkeypatch, sampler)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])

    _launch(tmp_path, manifest, splits, "run1")
    assert sampler.calls == [["s0", "s1", "s2", "s3"]]

    with caplog.at_level(logging.INFO):
        _launch(tmp_path, manifest, splits, "run2")
    assert sampler.calls == [["s0", "s1", "s2", "s3"]]  # no new calls

    assert _roi_csv_bytes(tmp_path, "run1") == _roi_csv_bytes(tmp_path, "run2")


def test_slide_manifest_partial_miss_samples_only_new_and_changed(tmp_path: Path, monkeypatch):
    """Adding a slide samples only that slide; re-annotating one (new label raster)
    re-samples only that slide. Hit slides are untouched."""
    sampler = _CountingSampler()
    _install_counting_sampler(monkeypatch, sampler)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    _launch(tmp_path, manifest, splits, "run1")

    # s1 points at a new annotation raster; s4 is a brand-new slide.
    manifest.write_text(
        "sample_id,image_path,label_mask_path\n"
        "s0,/fake/s0.tif,/fake/s0_mask.tif\n"
        "s1,/fake/s1.tif,/fake/s1_mask_v2.tif\n"
        "s2,/fake/s2.tif,/fake/s2_mask.tif\n"
        "s3,/fake/s3.tif,/fake/s3_mask.tif\n"
        "s4,/fake/s4.tif,/fake/s4_mask.tif\n"
    )
    splits.write_text(
        "sample_id,split,fold\n" "s0,train,0\ns1,train,0\ns2,tune,0\ns3,test,0\ns4,train,0\n"
    )
    _launch(tmp_path, manifest, splits, "run2")

    assert sampler.calls == [["s0", "s1", "s2", "s3"], ["s1", "s4"]]

    import pandas as pd

    roi_df = pd.read_csv(tmp_path / "out" / "run2" / "segmentation_rois" / "dataset.csv")
    assert len(roi_df) == 10  # 5 slides x 2 coords: cached + fresh merged
    assert set(roi_df["slide_id"]) == {"s0", "s1", "s2", "s3", "s4"}


def test_slide_manifest_zero_roi_slide_is_cached_and_contributes_no_rows(
    tmp_path: Path, monkeypatch
):
    """A slide that sampled zero ROIs is a cache hit on relaunch (not re-sampled) and
    contributes no manifest rows, exactly as on the fresh launch."""
    sampler = _CountingSampler(coords_for=lambda sid: [] if sid == "s1" else _coords_for(sid))
    _install_counting_sampler(monkeypatch, sampler)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])

    _launch(tmp_path, manifest, splits, "run1")
    _launch(tmp_path, manifest, splits, "run2")
    assert sampler.calls == [["s0", "s1", "s2", "s3"]]  # s1's [] answer hit the cache

    import pandas as pd

    for run_name in ("run1", "run2"):
        roi_df = pd.read_csv(tmp_path / "out" / run_name / "segmentation_rois" / "dataset.csv")
        assert len(roi_df) == 6  # 3 slides x 2 coords; s1 contributes none
        assert "s1" not in set(roi_df["slide_id"])
    assert _roi_csv_bytes(tmp_path, "run1") == _roi_csv_bytes(tmp_path, "run2")


def test_slide_manifest_cache_disabled_samples_everything_without_cache_io(
    tmp_path: Path, monkeypatch
):
    """cache.enabled=false: every launch samples every slide; no roi_sampling cache
    directory is ever created."""
    sampler = _CountingSampler()
    _install_counting_sampler(monkeypatch, sampler)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])

    _launch(tmp_path, manifest, splits, "run1", cache_enabled=False)
    _launch(tmp_path, manifest, splits, "run2", cache_enabled=False)

    all_slides = ["s0", "s1", "s2", "s3"]
    assert sampler.calls == [all_slides, all_slides]
    assert not list((tmp_path / "out").rglob("roi_sampling"))


def test_slide_manifest_relaunch_reads_no_annotation_raster(tmp_path: Path, monkeypatch):
    """A warm relaunch finds every ROI's crop in the cache: no raster is opened again."""
    sampler = _CountingSampler()
    _install_counting_sampler(monkeypatch, sampler)
    mask_reads: list[dict] = []
    _patch_mask_regions(
        monkeypatch,
        lambda path, *, size, **kw: np.zeros((size[1], size[0]), np.uint8),
        calls=mask_reads,
    )
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])

    _launch(tmp_path, manifest, splits, "run1")
    assert len(mask_reads) == 4
    _launch(tmp_path, manifest, splits, "run2")
    assert len(mask_reads) == 4


def test_slide_manifest_missing_crops_resample_the_slide_and_reuse_its_grids(
    tmp_path: Path, monkeypatch
):
    """A slide whose crops are gone (e.g. a cache from before crops existed) is re-sampled
    and re-cropped; re-sampling yields the same coords, so its dense grids still hit."""
    sampler = _CountingSampler()
    _install_counting_sampler(monkeypatch, sampler)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    _launch(tmp_path, manifest, splits, "run1")
    import pandas as pd

    roi_df = pd.read_csv(tmp_path / "out" / "run1" / "segmentation_rois" / "dataset.csv")
    s1_crops = [Path(p) for p in roi_df.loc[roi_df["slide_id"] == "s1", "label_mask_crop_path"]]
    s1_crops[0].unlink()
    _FakeDenseModel.calls.clear()

    _launch(tmp_path, manifest, splits, "run2")

    assert sampler.calls == [["s0", "s1", "s2", "s3"], ["s1"]]
    assert all(path.is_file() for path in s1_crops)
    assert all(region_ids == [] for call in _FakeDenseModel.calls for _, region_ids in call["regions"])
    assert _roi_csv_bytes(tmp_path, "run1") == _roi_csv_bytes(tmp_path, "run2")


def test_slide_manifest_cache_disabled_writes_crops_into_the_run(tmp_path: Path, monkeypatch):
    sampler = _CountingSampler()
    _install_counting_sampler(monkeypatch, sampler)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])

    _launch(tmp_path, manifest, splits, "run1", cache_enabled=False)

    import pandas as pd

    run_dir = tmp_path / "out" / "run1"
    roi_df = pd.read_csv(run_dir / "segmentation_rois" / "dataset.csv")
    crops = [Path(p) for p in roi_df["label_mask_crop_path"]]
    assert len(crops) == 8
    assert all(p.is_file() and run_dir / "segmentation_rois" / "masks" in p.parents for p in crops)


def test_sample_slide_rois_filter_samples_only_requested_slides(tmp_path: Path, monkeypatch):
    """The sampler's slide-id filter tiles only the requested slides, in manifest order."""
    from soma.extract.dense_slide_extraction import sample_slide_rois

    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path\n"
        "s0,/fake/s0.tif,/fake/s0_mask.tif\n"
        "s1,/fake/s1.tif,/fake/s1_mask.tif\n"
        "s2,/fake/s2.tif,/fake/s2_mask.tif\n"
    )
    tiled: list[str] = []

    def _tile_slide(slide, **kwargs):
        tiled.append(slide.sample_id)
        return {
            None: SimpleNamespace(
                tiles=SimpleNamespace(
                    x=np.asarray([0], dtype=np.int64),
                    y=np.asarray([0], dtype=np.int64),
                )
            )
        }

    monkeypatch.setattr("hs2p.tile_slide", _tile_slide)
    common = dict(preprocessing=_roi_extraction())
    slides = _slides(manifest)

    coords = sample_slide_rois(*slides, sample_ids=["s2", "s0"], **common)
    assert tiled == ["s0", "s2"]  # manifest order, s1 untouched
    assert coords == {"s0": [(0, 0)], "s2": [(0, 0)]}

    with pytest.raises(ValueError, match="not in the slide manifest"):
        sample_slide_rois(*slides, sample_ids=["nope"], **common)


# --------------------------------------------------------------------------- #
# Resumable dense extraction (#140): encode only the missing ROIs.
# --------------------------------------------------------------------------- #


def test_slide_manifest_resume_encodes_only_missing(tmp_path: Path, monkeypatch):
    """A resumed dense run re-encodes only the absent ROIs; a fully-cached slide is
    never opened, and already-materialized grids are left untouched."""
    from soma.extract.dense.store import DENSE_SIDECAR_SUFFIX

    # Two slides, two ROIs each.
    manifest = tmp_path / "slides.csv"
    manifest.write_text(
        "sample_id,image_path,label_mask_path\n"
        "s0,/fake/s0.tif,/fake/s0_mask.tif\n"
        "s1,/fake/s1.tif,/fake/s1_mask.tif\n"
    )
    _stub_sampling(monkeypatch, {"s0": [(0, 0), (32, 0)], "s1": [(0, 0), (32, 0)]})
    model = _patch_dense_model(monkeypatch)

    def _extract():
        return _extract_slide_rois(
            tmp_path,
            manifest,
            execution=slide2vec.ExecutionOptions(batch_size=2),
            cache=CacheOptions(enabled=True),
        )

    # Run 1: populate the whole cache.
    store = _extract()
    assert sorted(store.sample_ids) == [
        "s0__x0_y0",
        "s0__x32_y0",
        "s1__x0_y0",
        "s1__x32_y0",
    ]
    features_dir = store.feature_dir

    # Simulate a crash before s1's grids landed: drop s1's grids + sidecars.
    for x in (0, 32):
        (features_dir / "s1" / f"{x}_0.pt").unlink()
        (features_dir / "s1" / f"{x}_0{DENSE_SIDECAR_SUFFIX}").unlink()
    s0_mtimes = {x: (features_dir / "s0" / f"{x}_0.pt").stat().st_mtime_ns for x in (0, 32)}

    # Run 2: resume — only s1's ROIs are absent.
    model.calls.clear()
    resumed = _extract()

    # Exactly the absent set re-encoded; the fully-cached slide is never named, so
    # slide2vec never opens it.
    assert [call["regions"] for call in model.calls] == [[("s1", [(0, 0), (32, 0)])]]
    # The present (s0) grids are left byte-for-byte untouched.
    for x, mtime in s0_mtimes.items():
        assert (features_dir / "s0" / f"{x}_0.pt").stat().st_mtime_ns == mtime
    # And the resume produced a complete, readable store.
    assert sorted(resumed.sample_ids) == [
        "s0__x0_y0",
        "s0__x32_y0",
        "s1__x0_y0",
        "s1__x32_y0",
    ]



def test_an_extractions_roi_table_is_refused_as_cohort_records(tmp_path: Path):
    """ROIs are extraction-internal (ADR 0021): the ROI table an earlier extraction wrote
    is provenance, not a cohort. Handed to ``Pipeline`` its rows would read as whole
    slides and be sampled again, so the run is refused before anything is read."""
    from soma.pipeline import Pipeline

    manifest = tmp_path / "rois.csv"
    manifest.write_text(
        "sample_id,slide_id,image_path,label_mask_path,region_x,region_y,label_mask_crop_path\n"
        "s0__x0_y0,s0,/fake/s0.tif,/fake/s0_mask.tif,0,0,\n"
        "s0__x32_y0,s0,/fake/s0.tif,/fake/s0_mask.tif,32,0,\n"
        "s1__x0_y0,s1,/fake/s1.tif,/fake/s1_mask.tif,0,0,\n"
    )
    splits = tmp_path / "roi_splits.csv"
    splits.write_text(
        "sample_id,split,fold\ns0__x0_y0,train,0\ns0__x32_y0,tune,0\ns1__x0_y0,test,0\n"
    )

    with pytest.raises(ValueError, match="ROI table.*slide manifest"):
        Pipeline(_config(tmp_path, manifest, splits))
