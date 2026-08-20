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

from soma.config import (
    DecoderConfig,
    EvalConfig,
    ExecutionConfig,
    MasksConfig,
    PipelineConfig,
    PreprocessingConfig,
    SamplingConfig,
    TaskConfig,
    TrainingConfig,
)

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
    assign = {slide_ids[0]: "train", slide_ids[1]: "train", slide_ids[2]: "tune", slide_ids[3]: "test"}
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

        from soma.dense.geometry import compute_dense_geometry

        type(self).calls.append(
            {
                "name": self.name,
                "regions": [
                    (r.sample_id, [tuple(int(v) for v in c) for c in r.coordinates])
                    for r in regions
                ],
                "source_spacings": {
                    r.sample_id: r.spacing_at_level_0 for r in regions
                },
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
                        metadata=_dense_sidecar(dense, geometry, grid),
                    )
                )
        return artifacts


def _dense_sidecar(dense, geometry, grid) -> dict:
    """The geometry sidecar slide2vec writes next to every dense ROI grid."""
    return {
        "artifact_type": "dense_embeddings",
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
    import soma.dense_slide_extraction as dse

    _FakeDenseModel.calls = []
    monkeypatch.setattr(dse, "Model", _FakeDenseModel)
    return _FakeDenseModel


def _patch_extraction(monkeypatch):
    """Stub hs2p sampling + slide2vec encode + the WSI mask/image region reads (offline)."""
    import soma.dense_slide_extraction as dse
    import soma.dense.reader as reader_mod
    import soma.tasks.segmentation as segmod

    # hs2p sampling → known coords per slide.
    monkeypatch.setattr(dse, "sample_slide_rois", lambda dataset, **kw: {sid: _coords_for(sid) for sid in dataset.sample_ids})
    _patch_dense_model(monkeypatch)
    # Mask region read → a deterministic label window per ROI origin.
    def _fake_mask_region(path, *, location, size, spacing_um, backend, tolerance):
        x, _ = location
        w, h = size
        return np.full((h, w), 1 if x else 0, dtype=np.int64)  # ROI(0,0)→all bg, ROI(32,0)→all tumor
    monkeypatch.setattr(segmod, "read_mask_region_at_spacing", _fake_mask_region)
    # Image region read → a deterministic RGB window per ROI (the overlay writer reads the
    # ROI window from the whole-slide image_path, never opening the gigapixel slide).
    def _fake_image_region(path, *, location, size, spacing_um, backend, tolerance, interpolation="area"):
        w, h = size
        return np.zeros((h, w, 3), dtype=np.uint8)
    monkeypatch.setattr(reader_mod, "read_image_region_at_spacing", _fake_image_region)


def _config(root: Path, manifest: Path, splits: Path, *, masks: MasksConfig | None, min_cov: float = 0.0):
    return PipelineConfig(
        dataset_csv=manifest,
        splits_csv=splits,
        output_root=root / "out",
        dataset_type="segmentation",
        encoder=__import__("soma.config", fromlist=["EncoderConfig"]).EncoderConfig(name=ENCODER),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET,
            requested_spacing_um=0.5,
            masks=masks or MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": min_cov}),
            sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        ),
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=2),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )


# --------------------------------------------------------------------------- #
# End-to-end wiring through the real Pipeline.run().
# --------------------------------------------------------------------------- #


def test_slide_manifest_runs_end_to_end(tmp_path: Path, monkeypatch):
    from soma.pipeline import Pipeline

    _patch_extraction(monkeypatch)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    pipeline = Pipeline(_config(tmp_path, manifest, splits, masks=None))
    result = pipeline.run()

    assert "test/mean_dice" in result.summary
    assert result.fold_results[0].test_reports["test"].metrics["mean_dice"] >= 0.0
    assert pipeline.dataset.sample_ids == ["s0", "s1", "s2", "s3"]
    assert pipeline.splits.folds[0].train == ("s0", "s1")
    assert pipeline.splits.folds[0].tune == ("s2",)
    assert pipeline.splits.folds[0].tests == {"test": ("s3",)}

    provenance_files = list((tmp_path / "out").rglob("dense_source.json"))
    assert len(provenance_files) == 1
    provenance = json.loads(provenance_files[0].read_text(encoding="utf-8"))
    assert provenance["kind"] == "slide_manifest_dense_cache"
    assert provenance["parent_dataset_csv"] == str(manifest)
    assert provenance["parent_splits_csv"] == str(splits)
    assert Path(provenance["dataset_csv"]).name == "roi_manifest.csv"
    assert Path(provenance["splits_csv"]).name == "roi_splits.csv"

    # The derived ROI manifest + ROI splits were written, one row per (slide, coord).
    roi_manifests = list((tmp_path / "out").rglob("roi_manifest.csv"))
    assert roi_manifests, "ROI manifest not written"
    import pandas as pd

    roi_df = pd.read_csv(roi_manifests[0])
    assert len(roi_df) == 8  # 4 slides x 2 coords
    assert set(roi_df.columns) >= {"sample_id", "image_path", "label_mask_path", "region_x", "region_y"}
    assert set(roi_df["region_x"]) == {0, TARGET}


def test_slide_manifest_propagates_slide_splits_to_rois(tmp_path: Path, monkeypatch):
    """Every ROI inherits its parent slide's split/fold — no split creation."""
    _patch_extraction(monkeypatch)
    from soma.dense_slide_extraction import build_roi_manifest
    from soma.dataset import SegmentationManifest

    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    dataset = SegmentationManifest(manifest)
    coords = {sid: _coords_for(sid) for sid in dataset.sample_ids}
    roi_manifest, roi_splits = build_roi_manifest(dataset, splits, coords, out_dir=tmp_path / "rois")

    import pandas as pd

    splits_df = pd.read_csv(roi_splits)
    # s0/s1 → train, s2 → tune, s3 → test; each slide has 2 ROIs.
    by_split = splits_df.groupby("split")["sample_id"].apply(list)
    assert len(by_split["train"]) == 4 and all(r.startswith(("s0", "s1")) for r in by_split["train"])
    assert len(by_split["tune"]) == 2 and all(r.startswith("s2") for r in by_split["tune"])
    assert len(by_split["test"]) == 2 and all(r.startswith("s3") for r in by_split["test"])
    assert set(splits_df["fold"]) == {0}


def test_slide_source_spacing_survives_roi_derivation_and_reaches_slide2vec(
    tmp_path: Path, monkeypatch
):
    from soma.config import EncoderConfig
    from soma.dataset import SegmentationManifest
    from soma.dense_slide_extraction import (
        SlideManifestDenseExtractor,
        build_roi_manifest,
    )

    slides = tmp_path / "slides.csv"
    slides.write_text(
        "sample_id,image_path,label_mask_path,spacing_at_level_0\n"
        "s0,/fake/s0.tif,/fake/s0_mask.tif,0.25\n"
    )
    splits = tmp_path / "splits.csv"
    splits.write_text("sample_id,split,fold\ns0,test,0\n")
    source = SegmentationManifest(slides)
    roi_manifest, _ = build_roi_manifest(
        source, splits, {"s0": [(0, 0)]}, out_dir=tmp_path / "rois"
    )
    roi_dataset = SegmentationManifest(roi_manifest)
    assert roi_dataset.samples["s0__x0_y0"].spacing_at_level_0 == 0.25

    model = _patch_dense_model(monkeypatch)
    SlideManifestDenseExtractor(
        roi_dataset,
        EncoderConfig(name=ENCODER),
        masks=MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET,
            requested_spacing_um=0.5,
        ),
    ).run(feature_dir=tmp_path / "features")

    assert model.calls[0]["source_spacings"] == {"s0": 0.25}


def test_slide_manifest_dense_extraction_forwards_num_gpus(tmp_path: Path, monkeypatch):
    from soma.config import EncoderConfig
    from soma.dataset import SegmentationManifest
    from soma.dense_slide_extraction import SlideManifestDenseExtractor

    manifest = tmp_path / "rois.csv"
    manifest.write_text(
        "sample_id,slide_id,image_path,label_mask_path,region_x,region_y\n"
        "s0__x0_y0,s0,/fake/s0.tif,/fake/s0_mask.tif,0,0\n"
    )
    model = _patch_dense_model(monkeypatch)
    SlideManifestDenseExtractor(
        SegmentationManifest(manifest),
        EncoderConfig(name=ENCODER),
        masks=MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET,
            requested_spacing_um=0.5,
        ),
        execution=ExecutionConfig(num_gpus=2),
    ).run(feature_dir=tmp_path / "features")

    assert model.calls[0]["execution"].num_gpus == 2


def test_slide_sampling_forwards_source_spacing_to_hs2p(tmp_path: Path, monkeypatch):
    from hs2p import SlideSpec
    from soma.dataset import SegmentationManifest
    from soma.dense_slide_extraction import sample_slide_rois

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
    coords = sample_slide_rois(
        SegmentationManifest(manifest),
        masks=MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET,
            requested_spacing_um=0.5,
        ),
    )

    assert coords == {"s0": [(0, 0)]}
    assert captured[0].spacing_at_level_0 == 0.25


# --------------------------------------------------------------------------- #
# Cache key folds the sampling spec.
# --------------------------------------------------------------------------- #


def test_distinct_sampling_specs_yield_distinct_cache_keys():
    from soma.cache.keys import build_dense_cache_key
    from soma.config import EncoderConfig
    from soma.dense_slide_extraction import sampling_signature

    enc = EncoderConfig(name=ENCODER)
    pre = PreprocessingConfig(requested_tile_size_px=TARGET, requested_spacing_um=0.5)
    masks_a = MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.1})
    masks_b = MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.5})
    sampling = SamplingConfig(strategy="joint", output_mode="merged")

    def _key(sig):
        return build_dense_cache_key(
            tile_encoder_name=ENCODER, target_size=(TARGET, TARGET), patch_size=(PATCH, PATCH),
            pad_mode="reflect", execution=enc, preprocessing=pre, window_size=None, overlap=0.0,
            sampling_signature=sig,
        )

    sig_a = sampling_signature(masks_a, sampling, pre)
    sig_b = sampling_signature(masks_b, sampling, pre)
    assert _key(sig_a) != _key(sig_b)  # distinct min_coverage ⇒ distinct cache
    assert _key(sig_a) == _key(sampling_signature(masks_a, sampling, pre))  # stable
    # Absent (pre-cropped tiles) differs from any sampled key, and is itself stable.
    assert _key(None) != _key(sig_a)
    assert _key(None) == _key(None)


# --------------------------------------------------------------------------- #
# Region-aware mask target read.
# --------------------------------------------------------------------------- #


def test_extract_targets_reads_mask_region_when_record_has_region(tmp_path: Path, monkeypatch):
    from soma.dense.geometry import compute_dense_geometry
    from soma.dataset import SampleRecord
    from soma.tasks.segmentation import SegmentationHead
    import soma.tasks.segmentation as segmod

    captured = {}

    def _fake(path, *, location, size, spacing_um, backend, tolerance):
        captured.update(location=location, size=size, spacing_um=spacing_um)
        return np.zeros((size[1], size[0]), dtype=np.int64)

    monkeypatch.setattr(segmod, "read_mask_region_at_spacing", _fake)
    geometry = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    head = SegmentationHead(num_classes=NUM_CLASSES, geometry=geometry, spacing_um=0.5)
    record = SampleRecord(
        sample_id="s0__x64_y0", image_path=Path("/fake/s0.tif"), label=None,
        label_mask_path=Path("/fake/s0_mask.tif"), region=(64, 0),
    )
    targets = head.extract_targets(record)
    assert tuple(targets["mask"].shape) == (TARGET, TARGET)
    assert captured["location"] == (64, 0)
    assert captured["size"] == (TARGET, TARGET)
    assert captured["spacing_um"] == 0.5


# --------------------------------------------------------------------------- #
# Deferred-combo guards.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"feature_mode": "live"}, "Live"),
    ],
)
def test_slide_manifest_deferred_combos_raise(tmp_path: Path, overrides, match):
    from soma.pipeline import Pipeline

    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])
    cfg = _config(tmp_path, manifest, splits, masks=None)
    from dataclasses import replace

    cfg = replace(cfg, **overrides)
    pipeline = Pipeline(cfg)
    with pytest.raises(NotImplementedError, match=match):
        pipeline._build_slide_manifest_dense_context(run_dir=tmp_path / "out" / "run")


def test_slide_manifest_declares_the_sliding_window_to_slide2vec(tmp_path: Path, monkeypatch):
    """A window smaller than the target is expressed as DenseOptions window/overlap.

    This is how a native-only encoder reaches a larger supervision tile — phikon@224 over
    512 px ROIs, the BEETLE recipe — stated here at fixture scale.

    The sliding itself — encoder slid over patch-aligned windows of each padded ROI, grids
    blended back — is slide2vec's, and is tested there. soma's remaining responsibility is
    to state the window it wants and to cache what comes back at the target grid size.
    """
    from soma.config import EncoderConfig
    from soma.dense_slide_extraction import SlideManifestDenseExtractor
    from soma.dataset import SegmentationManifest

    # A manifest of ROI rows (region origins) for one slide.
    roi_manifest = tmp_path / "roi_manifest.csv"
    roi_manifest.write_text(
        "sample_id,slide_id,image_path,label_mask_path,region_x,region_y\n"
        "s0__x0_y0,s0,/fake/s0.tif,/fake/s0_mask.tif,0,0\n"
        "s0__x32_y0,s0,/fake/s0.tif,/fake/s0_mask.tif,32,0\n"
    )
    dataset = SegmentationManifest(roi_manifest)
    model = _patch_dense_model(monkeypatch)

    WINDOW = 16  # < TARGET (32) -> genuine sliding
    extractor = SlideManifestDenseExtractor(
        dataset,
        EncoderConfig(name=ENCODER, batch_size=2),
        masks=MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET, requested_spacing_um=0.5,
            dense_window_size=WINDOW, dense_window_overlap=0.5,
        ),
    )
    store = extractor.run(feature_dir=tmp_path / "features")

    (call,) = model.calls
    assert call["dense"].window_size == WINDOW
    assert call["dense"].overlap == 0.5
    assert call["dense"].target_size == TARGET
    assert call["dense"].spacing_um == 0.5
    assert call["regions"] == [("s0", [(0, 0), (32, 0)])]
    grid = store.load("s0__x0_y0")
    assert grid.shape == (FEATURE_DIM, GRID, GRID)  # stitched back to the target grid


def test_slide_manifest_grids_are_namespaced_per_slide(tmp_path: Path, monkeypatch):
    """ROI grids live at slide2vec's ``<slide_id>/<x>_<y>.pt``, addressed from the manifest.

    The ROI id is never split back apart to find its grid: the manifest's slide_id +
    region_x/region_y are the address (ADR 0007). A slide id that itself contains the
    ``__x``/``_y`` separators therefore resolves correctly.
    """
    from soma.config import EncoderConfig
    from soma.dense_slide_extraction import SlideManifestDenseExtractor
    from soma.dataset import SegmentationManifest

    roi_manifest = tmp_path / "roi_manifest.csv"
    roi_manifest.write_text(
        "sample_id,slide_id,image_path,label_mask_path,region_x,region_y\n"
        "s0__x1_y2__x0_y0,s0__x1_y2,/fake/s0.tif,/fake/s0_mask.tif,0,0\n"
    )
    dataset = SegmentationManifest(roi_manifest)
    _patch_dense_model(monkeypatch)

    store = SlideManifestDenseExtractor(
        dataset,
        EncoderConfig(name=ENCODER),
        masks=MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        preprocessing=PreprocessingConfig(requested_tile_size_px=TARGET, requested_spacing_um=0.5),
    ).run(feature_dir=tmp_path / "features")

    assert (tmp_path / "features" / "dense_embeddings" / "s0__x1_y2" / "0_0.pt").is_file()
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

    def __call__(self, dataset, *, masks, sampling, preprocessing, sample_ids=None):
        ids = list(dataset.sample_ids) if sample_ids is None else [str(s) for s in sample_ids]
        self.calls.append(sorted(ids))
        return {sid: self._coords_for(sid) for sid in ids}


def _install_counting_sampler(monkeypatch, sampler: _CountingSampler) -> None:
    import soma.dense_slide_extraction as dse

    _patch_extraction(monkeypatch)
    monkeypatch.setattr(dse, "sample_slide_rois", sampler)


def _launch(tmp_path: Path, manifest: Path, splits: Path, run_name: str, *, cache_enabled: bool = True):
    """Drive the context builder as one launch (a fresh Pipeline, its own run dir)."""
    from dataclasses import replace

    from soma.config import CacheConfig
    from soma.pipeline import Pipeline

    cfg = _config(tmp_path, manifest, splits, masks=None)
    if not cache_enabled:
        cfg = replace(cfg, cache=CacheConfig(enabled=False))
    pipeline = Pipeline(cfg)
    return pipeline._build_slide_manifest_dense_context(run_dir=tmp_path / "out" / run_name)


def _roi_csv_bytes(tmp_path: Path, run_name: str) -> tuple[bytes, bytes]:
    roi_dir = tmp_path / "out" / run_name / "segmentation_rois"
    return (
        (roi_dir / "roi_manifest.csv").read_bytes(),
        (roi_dir / "roi_splits.csv").read_bytes(),
    )


def test_slide_manifest_relaunch_skips_sampling_and_reproduces_manifest(
    tmp_path: Path, monkeypatch, caplog
):
    """Second launch with an unchanged config: zero sampler calls, byte-identical
    ROI manifest + ROI splits derived from cached coords, and a hits-vs-sampled log."""
    import logging

    sampler = _CountingSampler()
    _install_counting_sampler(monkeypatch, sampler)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])

    _launch(tmp_path, manifest, splits, "run1")
    assert sampler.calls == [["s0", "s1", "s2", "s3"]]

    with caplog.at_level(logging.INFO, logger="soma.pipeline"):
        _launch(tmp_path, manifest, splits, "run2")
    assert sampler.calls == [["s0", "s1", "s2", "s3"]]  # no new calls

    assert _roi_csv_bytes(tmp_path, "run1") == _roi_csv_bytes(tmp_path, "run2")
    assert any(
        "4 slide(s) from cache" in message and "0 slide(s) to sample" in message
        for message in caplog.messages
    )


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
        "sample_id,split,fold\n"
        "s0,train,0\ns1,train,0\ns2,tune,0\ns3,test,0\ns4,train,0\n"
    )
    _launch(tmp_path, manifest, splits, "run2")

    assert sampler.calls == [["s0", "s1", "s2", "s3"], ["s1", "s4"]]

    import pandas as pd

    roi_df = pd.read_csv(tmp_path / "out" / "run2" / "segmentation_rois" / "roi_manifest.csv")
    assert len(roi_df) == 10  # 5 slides x 2 coords: cached + fresh merged
    assert set(roi_df["slide_id"]) == {"s0", "s1", "s2", "s3", "s4"}


def test_slide_manifest_zero_roi_slide_is_cached_and_contributes_no_rows(
    tmp_path: Path, monkeypatch
):
    """A slide that sampled zero ROIs is a cache hit on relaunch (not re-sampled) and
    contributes no manifest rows, exactly as on the fresh launch."""
    sampler = _CountingSampler(
        coords_for=lambda sid: [] if sid == "s1" else _coords_for(sid)
    )
    _install_counting_sampler(monkeypatch, sampler)
    manifest, splits = _write_slide_manifest(tmp_path, ["s0", "s1", "s2", "s3"])

    _launch(tmp_path, manifest, splits, "run1")
    _launch(tmp_path, manifest, splits, "run2")
    assert sampler.calls == [["s0", "s1", "s2", "s3"]]  # s1's [] answer hit the cache

    import pandas as pd

    for run_name in ("run1", "run2"):
        roi_df = pd.read_csv(
            tmp_path / "out" / run_name / "segmentation_rois" / "roi_manifest.csv"
        )
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


def test_sample_slide_rois_filter_samples_only_requested_slides(tmp_path: Path, monkeypatch):
    """The sampler's slide-id filter tiles only the requested slides, in manifest order."""
    from soma.dataset import SegmentationManifest
    from soma.dense_slide_extraction import sample_slide_rois

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
    common = dict(
        masks=MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET, requested_spacing_um=0.5
        ),
    )
    dataset = SegmentationManifest(manifest)

    coords = sample_slide_rois(dataset, sample_ids=["s2", "s0"], **common)
    assert tiled == ["s0", "s2"]  # manifest order, s1 untouched
    assert coords == {"s0": [(0, 0)], "s2": [(0, 0)]}

    with pytest.raises(ValueError, match="not in the slide manifest"):
        sample_slide_rois(dataset, sample_ids=["nope"], **common)


# --------------------------------------------------------------------------- #
# Resumable dense extraction (#140): encode only the missing ROIs.
# --------------------------------------------------------------------------- #


def test_slide_manifest_resume_encodes_only_missing(tmp_path: Path, monkeypatch):
    """A resumed dense run re-encodes only the absent ROIs; a fully-cached slide is
    never opened, and already-materialized grids are left untouched."""
    from soma.config import CacheConfig, EncoderConfig
    from soma.dense.store import DENSE_SIDECAR_SUFFIX
    from soma.dense_slide_extraction import SlideManifestDenseExtractor
    from soma.dataset import SegmentationManifest

    # Two slides, two ROIs each.
    roi_manifest = tmp_path / "roi_manifest.csv"
    roi_manifest.write_text(
        "sample_id,slide_id,image_path,label_mask_path,region_x,region_y\n"
        "s0__x0_y0,s0,/fake/s0.tif,/fake/s0_mask.tif,0,0\n"
        "s0__x32_y0,s0,/fake/s0.tif,/fake/s0_mask.tif,32,0\n"
        "s1__x0_y0,s1,/fake/s1.tif,/fake/s1_mask.tif,0,0\n"
        "s1__x32_y0,s1,/fake/s1.tif,/fake/s1_mask.tif,32,0\n"
    )
    dataset = SegmentationManifest(roi_manifest)
    model = _patch_dense_model(monkeypatch)

    def _make_extractor():
        return SlideManifestDenseExtractor(
            dataset,
            EncoderConfig(name=ENCODER, batch_size=2),
            masks=MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0}),
            sampling=SamplingConfig(strategy="joint", output_mode="merged"),
            preprocessing=PreprocessingConfig(requested_tile_size_px=TARGET, requested_spacing_um=0.5),
            cache=CacheConfig(enabled=True),
        )

    feature_dir = tmp_path / "features"
    # Run 1: populate the whole cache.
    store = _make_extractor().run(feature_dir=feature_dir)
    assert sorted(store.available_samples) == [
        "s0__x0_y0", "s0__x32_y0", "s1__x0_y0", "s1__x32_y0"
    ]
    features_dir = store.feature_dir

    # Simulate a crash before s1's grids landed: drop s1's grids + sidecars.
    for x in (0, 32):
        (features_dir / "s1" / f"{x}_0.pt").unlink()
        (features_dir / "s1" / f"{x}_0{DENSE_SIDECAR_SUFFIX}").unlink()
    s0_mtimes = {
        x: (features_dir / "s0" / f"{x}_0.pt").stat().st_mtime_ns for x in (0, 32)
    }

    # Run 2: resume — only s1's ROIs are absent.
    model.calls.clear()
    resumed = _make_extractor().run(feature_dir=feature_dir)

    # Exactly the absent set re-encoded; the fully-cached slide is never named, so
    # slide2vec never opens it.
    assert [call["regions"] for call in model.calls] == [[("s1", [(0, 0), (32, 0)])]]
    # The present (s0) grids are left byte-for-byte untouched.
    for x, mtime in s0_mtimes.items():
        assert (features_dir / "s0" / f"{x}_0.pt").stat().st_mtime_ns == mtime
    # And the resume produced a complete, readable store.
    assert sorted(resumed.available_samples) == [
        "s0__x0_y0", "s0__x32_y0", "s1__x0_y0", "s1__x32_y0"
    ]
