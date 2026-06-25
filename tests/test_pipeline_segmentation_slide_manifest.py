"""Slide-manifest segmentation: slides + masks → hs2p sampling → ROI dense grids → train.

The genuinely-new A3a soma integration, tested deterministically and offline. The two
cross-repo extraction primitives — hs2p annotation sampling (``sample_slide_rois``) and the
slide2vec dense encode (``encode_regions_dense``) — are stubbed at their import seams (each
is independently tested in its own repo: hs2p ``test_annotation_coverage``, slide2vec
``test_dense_regions``); everything in between (ROI manifest + split propagation, the dense
cache keyed on the sampling spec, explicit ROI training context, and the full cached dense
training/eval) runs for real.
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
    MasksConfig,
    PipelineConfig,
    PreprocessingConfig,
    SamplingConfig,
    TaskConfig,
    TrainingConfig,
)

NUM_CLASSES = 2
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
        "sample_id,image_path,mask_path\n"
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


def _patch_extraction(monkeypatch):
    """Stub hs2p sampling + slide2vec encode + the WSI mask/image region reads (offline)."""
    import slide2vec.inference as s2v_inference
    import slide2vec.runtime.dense_regions as s2v_dense
    import hs2p.wsi.wsi as hs2p_wsi
    import soma.dense_slide_extraction as dse
    import soma.dense.reader as reader_mod
    import soma.tasks.segmentation as segmod

    # hs2p sampling → known coords per slide.
    monkeypatch.setattr(dse, "sample_slide_rois", lambda dataset, **kw: {sid: _coords_for(sid) for sid in dataset.sample_ids})

    # slide2vec model load → a dummy carrying only patch_size (encode is stubbed below).
    monkeypatch.setattr(
        s2v_inference, "load_model",
        lambda **kw: SimpleNamespace(model=SimpleNamespace(patch_size=(PATCH, PATCH)), device="cpu"),
    )
    # A WSI that opens any path without touching disk (encode is stubbed, so it's unused).
    monkeypatch.setattr(hs2p_wsi, "WSI", lambda *a, **kw: SimpleNamespace())
    # slide2vec dense encode → deterministic random grids of the right shape.
    rng = np.random.default_rng(0)
    monkeypatch.setattr(
        s2v_dense, "encode_regions_dense",
        lambda *, coordinates, **kw: rng.standard_normal((len(coordinates), FEATURE_DIM, GRID, GRID)).astype(np.float32),
    )
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
        encoder=__import__("soma.config", fromlist=["EncoderConfig"]).EncoderConfig(name="phikon"),
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
    assert set(roi_df.columns) >= {"sample_id", "image_path", "mask_path", "region_x", "region_y"}
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


# --------------------------------------------------------------------------- #
# Cache key folds the sampling spec.
# --------------------------------------------------------------------------- #


def test_distinct_sampling_specs_yield_distinct_cache_keys():
    from soma.cache.keys import build_dense_cache_key
    from soma.config import EncoderConfig
    from soma.dense_slide_extraction import sampling_signature

    enc = EncoderConfig(name="phikon")
    pre = PreprocessingConfig(requested_tile_size_px=TARGET, requested_spacing_um=0.5)
    masks_a = MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.1})
    masks_b = MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.5})
    sampling = SamplingConfig(strategy="joint", output_mode="merged")

    def _key(sig):
        return build_dense_cache_key(
            tile_encoder_name="phikon", target_size=(TARGET, TARGET), patch_size=(PATCH, PATCH),
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
        mask_path=Path("/fake/s0_mask.tif"), region=(64, 0),
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


def test_slide_manifest_sliding_window_encodes_via_soma_sliding(tmp_path: Path, monkeypatch):
    """A native-only encoder (window < target) slides over each padded ROI region and
    writes cached dense grids — the phikon@224 → 512-tile path the BEETLE config uses."""
    from dataclasses import replace

    from soma.config import EncoderConfig
    from soma.dense_slide_extraction import SlideManifestDenseExtractor
    from soma.dataset import SegmentationManifest
    from soma.dense.geometry import compute_dense_geometry

    # A manifest of ROI rows (region origins) for one slide.
    roi_manifest = tmp_path / "roi_manifest.csv"
    roi_manifest.write_text(
        "sample_id,image_path,mask_path,region_x,region_y\n"
        "s0__x0_y0,/fake/s0.tif,/fake/s0_mask.tif,0,0\n"
        "s0__x32_y0,/fake/s0.tif,/fake/s0_mask.tif,32,0\n"
    )
    dataset = SegmentationManifest(roi_manifest)

    import slide2vec.inference as s2v_inference
    import hs2p.wsi.wsi as hs2p_wsi

    captured: dict = {}

    class _Model:
        patch_size = (PATCH, PATCH)

        def get_dense_transform(self):
            import torchvision.transforms as T

            return T.ToTensor()

        def encode_tiles_dense(self, window):
            # Record the per-window spatial size to prove sliding (window < target).
            captured.setdefault("window_hw", tuple(int(s) for s in window.shape[-2:]))
            b = window.shape[0]
            gh = window.shape[-2] // PATCH
            gw = window.shape[-1] // PATCH
            import torch as _t

            return _t.zeros(b, FEATURE_DIM, gh, gw)

    monkeypatch.setattr(
        s2v_inference, "load_model",
        lambda **kw: SimpleNamespace(model=_Model(), device="cpu"),
    )

    class _WSI:
        def __init__(self, *a, **kw):
            pass

        def read_region_at_spacing(self, location, spacing, size, *, tolerance, interpolation):
            w, h = size
            return np.zeros((h, w, 3), dtype=np.uint8)

    monkeypatch.setattr(hs2p_wsi, "WSI", _WSI)

    WINDOW = 16  # < TARGET (32) -> genuine sliding
    extractor = SlideManifestDenseExtractor(
        dataset,
        EncoderConfig(name="phikon", batch_size=2),
        masks=MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": 0.0}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET, requested_spacing_um=0.5,
            dense_window_size=WINDOW, dense_window_overlap=0.5,
        ),
    )
    store = extractor.run(feature_dir=tmp_path / "features")
    assert captured["window_hw"] == (WINDOW, WINDOW)  # slid at the native window, not 32
    grid = store.load("s0__x0_y0")
    assert grid.shape == (FEATURE_DIM, GRID, GRID)  # stitched back to the target grid


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

    import slide2vec.inference as s2v_inference
    import slide2vec.runtime.dense_regions as s2v_dense
    import hs2p.wsi.wsi as hs2p_wsi

    # Two slides, two ROIs each.
    roi_manifest = tmp_path / "roi_manifest.csv"
    roi_manifest.write_text(
        "sample_id,image_path,mask_path,region_x,region_y\n"
        "s0__x0_y0,/fake/s0.tif,/fake/s0_mask.tif,0,0\n"
        "s0__x32_y0,/fake/s0.tif,/fake/s0_mask.tif,32,0\n"
        "s1__x0_y0,/fake/s1.tif,/fake/s1_mask.tif,0,0\n"
        "s1__x32_y0,/fake/s1.tif,/fake/s1_mask.tif,32,0\n"
    )
    dataset = SegmentationManifest(roi_manifest)

    monkeypatch.setattr(
        s2v_inference, "load_model",
        lambda **kw: SimpleNamespace(model=SimpleNamespace(patch_size=(PATCH, PATCH)), device="cpu"),
    )

    opened_paths: list[str] = []
    monkeypatch.setattr(
        hs2p_wsi, "WSI",
        lambda path, **kw: opened_paths.append(str(path)) or SimpleNamespace(),
    )

    encoded_coords: list[list] = []
    rng = np.random.default_rng(0)

    def _encode(*, coordinates, **kw):
        encoded_coords.append([tuple(int(v) for v in c) for c in coordinates])
        return rng.standard_normal((len(coordinates), FEATURE_DIM, GRID, GRID)).astype(np.float32)

    monkeypatch.setattr(s2v_dense, "encode_regions_dense", _encode)

    def _make_extractor():
        return SlideManifestDenseExtractor(
            dataset,
            EncoderConfig(name="phikon", batch_size=2),
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
    for sid in ("s1__x0_y0", "s1__x32_y0"):
        (features_dir / f"{sid}.pt").unlink()
        (features_dir / f"{sid}{DENSE_SIDECAR_SUFFIX}").unlink()
    s0_mtimes = {
        sid: (features_dir / f"{sid}.pt").stat().st_mtime_ns
        for sid in ("s0__x0_y0", "s0__x32_y0")
    }

    # Run 2: resume — only s1's ROIs are absent.
    encoded_coords.clear()
    opened_paths.clear()
    resumed = _make_extractor().run(feature_dir=feature_dir)

    # Exactly the absent set re-encoded; the fully-cached slide never opened.
    assert encoded_coords == [[(0, 0), (32, 0)]]
    assert opened_paths == ["/fake/s1.tif"]
    # The present (s0) grids are left byte-for-byte untouched.
    for sid, mtime in s0_mtimes.items():
        assert (features_dir / f"{sid}.pt").stat().st_mtime_ns == mtime
    # And the resume produced a complete, readable store.
    assert sorted(resumed.available_samples) == [
        "s0__x0_y0", "s0__x32_y0", "s1__x0_y0", "s1__x32_y0"
    ]
