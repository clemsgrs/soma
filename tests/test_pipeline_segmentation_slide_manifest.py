"""Slide-manifest segmentation: slides + masks → hs2p sampling → ROI dense grids → train.

The genuinely-new A3a soma integration, tested deterministically and offline. The two
cross-repo extraction primitives — hs2p annotation sampling (``sample_slide_rois``) and the
slide2vec dense encode (``encode_regions_dense``) — are stubbed at their import seams (each
is independently tested in its own repo: hs2p ``test_annotation_coverage``, slide2vec
``test_dense_regions``); everything in between (ROI manifest + split propagation, the dense
cache keyed on the sampling spec, the dataset/splits swap, and the full cached dense
training/eval) runs for real.
"""

from __future__ import annotations

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
    """Stub hs2p sampling + slide2vec encode + the WSI mask region read (offline determinism)."""
    import slide2vec.inference as s2v_inference
    import slide2vec.runtime.dense_regions as s2v_dense
    import hs2p.wsi.wsi as hs2p_wsi
    import soma.dense_slide_extraction as dse
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


def _config(root: Path, manifest: Path, splits: Path, *, masks: MasksConfig | None, min_cov: float = 0.0):
    return PipelineConfig(
        dataset_csv=manifest,
        splits_csv=splits,
        output_root=root / "out",
        dataset_type="segmentation",
        encoder=__import__("soma.config", fromlist=["EncoderConfig"]).EncoderConfig(name="phikon"),
        preprocessing=PreprocessingConfig(requested_tile_size_px=TARGET, requested_spacing_um=0.5),
        masks=masks or MasksConfig(pixel_mapping=PIXEL_MAPPING, min_coverage={"tumor": min_cov}),
        sampling=SamplingConfig(strategy="joint", output_mode="merged"),
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
        ({"preprocessing": PreprocessingConfig(requested_tile_size_px=TARGET, requested_spacing_um=0.5, dense_window_size=16)}, "Sliding-window"),
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
        pipeline._build_slide_manifest_dense_store(run_dir=tmp_path / "out" / "run")
