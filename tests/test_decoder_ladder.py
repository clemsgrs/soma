"""Decoder-complexity ladder — the new rungs (design §2.4, issue #233).

Rungs 1-2 (``linear`` / ``lightweight_conv``) and their per-decoder unit behaviour live in
``test_decoders.py``. This file covers the three rungs added by #233 at the level the
acceptance criteria state them:

* **rung 4 — multi-FM ensemble**: a ``d->D``-projection decoder consumes the
  ``CompositeDenseFeatureStore``'s concatenated ``(Σdᵢ, h, w)`` grid, param-matched to a
  single-encoder decoder via the projection (the composite width is just a wider ``d``).
* **rung 0 — attention-map (decoder-free)**: the frozen encoder's attention grid becomes a
  MIDOG detection heatmap end-to-end (attention -> peak -> F1@δ) with zero trained params.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from soma.decoders import HeavyConvDecoder, LightweightConvDecoder  # noqa: E402
from soma.dense.composite import CompositeDenseFeatureStore  # noqa: E402
from soma.dense.geometry import compute_dense_geometry  # noqa: E402
from soma.dense.store import DenseFeatureStore, dense_grid_metadata, write_dense_grid  # noqa: E402

TARGET = 16
SPACING = 0.2  # µm/px; source == effective spacing makes the point transform identity


def _write_member(dir_: Path, sample_ids: list[str], *, feature_dim: int, patch: int) -> DenseFeatureStore:
    """Write a one-encoder dense store (fixed target_size / spacing, own patch+dim)."""
    dir_.mkdir(parents=True, exist_ok=True)
    geom = compute_dense_geometry(target_size=TARGET, patch_size=patch)
    meta = dense_grid_metadata(geom, feature_dim=feature_dim, pad_mode="reflect", spacing_um=SPACING)
    meta.update(source_spacing_um=SPACING, effective_spacing_um=SPACING)
    rng = np.random.default_rng(feature_dim)
    for sid in sample_ids:
        write_dense_grid(dir_, sid, torch.from_numpy(rng.standard_normal((feature_dim, *geom.grid_shape)).astype("float32")), meta)
    return DenseFeatureStore(dir_)


# --------------------------------------------------------------------------- #
# Rung 4 — multi-FM ensemble (composite concat + param-matched projection)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("decoder_cls", [LightweightConvDecoder, HeavyConvDecoder])
def test_ensemble_decoder_consumes_concatenated_multi_encoder_grid(tmp_path: Path, decoder_cls):
    # Two frozen encoders with DIFFERENT embedding dims + patch sizes -> a composite store
    # that concatenates their grids to (d1 + d2, h, w) at a common token grid.
    ids = ["s0", "s1"]
    m1 = _write_member(tmp_path / "enc_a", ids, feature_dim=4, patch=4)  # grid 4x4
    m2 = _write_member(tmp_path / "enc_b", ids, feature_dim=6, patch=8)  # grid 2x2
    composite = CompositeDenseFeatureStore([m1, m2], concat_resolution="grid")

    assert composite.feature_dim == 4 + 6  # Σdᵢ
    grid = composite.load("s0")
    h, w = composite.grid_shape
    assert tuple(grid.shape) == (10, h, w)  # concatenated channels at the common grid

    # The ensemble "decoder" is any d->D-projection rung fed the Σdᵢ grid: the projection
    # absorbs the concat width, so nothing downstream changes.
    decoder = decoder_cls(input_dim=composite.feature_dim, num_classes=1, hidden_dim=32, num_upsample_blocks=1)
    out = decoder(grid.unsqueeze(0))
    assert tuple(out.shape) == (1, 1, h * 2, w * 2)


@pytest.mark.parametrize("decoder_cls", [LightweightConvDecoder, HeavyConvDecoder])
def test_ensemble_projection_param_matches_single_encoder(decoder_cls):
    # Param-matched via projection: a decoder over a Σdᵢ = 4+6 = 10 composite has the SAME
    # downstream (below-projection) capacity as one over a single d = 10 encoder — the
    # ensemble buys width for free, not extra trainable machinery.
    ensemble = decoder_cls(input_dim=10, num_classes=1, hidden_dim=32, num_upsample_blocks=1)
    single = decoder_cls(input_dim=10, num_classes=1, hidden_dim=32, num_upsample_blocks=1)

    def downstream(dec):
        return sum(p.numel() for n, p in dec.named_parameters() if not n.startswith("proj."))

    assert downstream(ensemble) == downstream(single)
    # And a genuinely different single-encoder width leaves the downstream count untouched.
    narrow = decoder_cls(input_dim=384, num_classes=1, hidden_dim=32, num_upsample_blocks=1)
    assert downstream(ensemble) == downstream(narrow)


def test_ensemble_end_to_end_detection_fold_with_heavy_decoder(tmp_path: Path):
    """A full detection fold trains the heavy decoder on the composite (Σdᵢ) grids —
    the ensemble rung end-to-end through the real pipeline path."""
    from soma.config import DecoderConfig, EvalConfig, PreprocessingConfig, TaskConfig, TrainingConfig
    from soma.dataset import DetectionManifest, Splits
    from soma.pipeline import train_one_detection_fold

    ids = ["s0", "s1", "s2", "s3"]
    m1 = _write_member(tmp_path / "enc_a", ids, feature_dim=4, patch=4)
    m2 = _write_member(tmp_path / "enc_b", ids, feature_dim=6, patch=8)
    composite = CompositeDenseFeatureStore([m1, m2], concat_resolution="grid")

    points_dir = tmp_path / "points"
    points_dir.mkdir()
    rows = []
    for sid in ids:
        pts = points_dir / f"{sid}.csv"
        pts.write_text("x,y,class\n4,4,0\n11,11,1\n")
        rows.append(f"{sid},{sid}.jpg,{pts},{SPACING}")
    (tmp_path / "manifest.csv").write_text(
        "sample_id,image_path,points_path,spacing_at_level_0\n"
        + "\n".join(rows)
        + "\n"
    )
    assign = {"s0": "train", "s1": "train", "s2": "tune", "s3": "test"}
    (tmp_path / "splits.csv").write_text(
        "sample_id,split,fold\n" + "\n".join(f"{s},{v},0" for s, v in assign.items()) + "\n"
    )
    manifest = DetectionManifest(tmp_path / "manifest.csv")
    splits = Splits(tmp_path / "splits.csv", manifest)

    result = train_one_detection_fold(
        feature_store=composite,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(
            name="detection",
            params={"num_classes": 2, "match_distance": 0.6, "sigma": 0.3},
        ),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="heavy_conv", params={"hidden_dim": 32}),
        evaluation=EvalConfig(metrics=["mean_f1"]),
        preprocessing=PreprocessingConfig(requested_spacing_um=SPACING, requested_tile_size_px=TARGET),
    )
    assert "mean_f1" in result.test_reports["test"].metrics


# --------------------------------------------------------------------------- #
# Rung 0 — attention-map (decoder-free) MIDOG detection
# --------------------------------------------------------------------------- #


def test_attention_map_detection_heatmap_end_to_end():
    """A frozen encoder's attention grid -> heatmap -> peak -> F1@δ, zero trained params.

    MIDOG-shaped: single class. A synthetic attention grid attends to exactly the token
    covering a mitosis; the decoder-free rung must recover that detection at F1 = 1.
    """
    from soma.detection import attention_to_detection_heatmap
    from soma.tasks.detection import DetectionHead

    patch = 4
    geom = compute_dense_geometry(target_size=32, patch_size=patch)  # 8x8 token grid
    grid_h, grid_w = geom.grid_shape

    # One mitosis at target pixel (x=10, y=13) -> token cell (row=13//4=3, col=10//4=2).
    gt_x, gt_y = 10.0, 13.0
    hot_row, hot_col = int(gt_y) // patch, int(gt_x) // patch

    K = 6  # attention channels (blocks x heads)
    attn = torch.zeros(K, grid_h, grid_w)
    attn[:, hot_row, hot_col] = 1.0  # the encoder attends here

    heatmap = attention_to_detection_heatmap(attn, geometry=geom, num_classes=1)
    assert tuple(heatmap.shape) == (1, 32, 32)
    assert float(heatmap.max()) == pytest.approx(1.0)  # min-max normalised to [0, 1]

    head = DetectionHead(
        num_classes=1,
        geometry=geom,
        delta_px=5.0,
        sigma_px=2.0,
        score_threshold=0.5,
        sample_spacings={},
        metrics=["mean_f1"],
    )
    gt = torch.tensor([[[gt_x, gt_y, 0.0]]])  # (1, K=1, 3)
    metrics = head.compute_metrics(heatmap.unsqueeze(0), {"gt_points": gt})
    assert metrics["mean_f1"] == pytest.approx(1.0)


def test_attention_map_is_class_agnostic_saliency():
    # The attention saliency is class-blind (design §2.4): broadcasting to C channels gives
    # identical maps — why the rung is MIDOG-scoped (strong single-class, weak multi-class).
    from soma.detection import attention_to_detection_heatmap

    geom = compute_dense_geometry(target_size=16, patch_size=4)
    attn = torch.rand(4, *geom.grid_shape)
    hm = attention_to_detection_heatmap(attn, geometry=geom, num_classes=3)
    assert hm.shape[0] == 3
    torch.testing.assert_close(hm[0], hm[1])
    torch.testing.assert_close(hm[1], hm[2])


def test_attention_map_constant_grid_yields_no_peaks():
    # A flat attention map (nothing salient) normalises to all-zeros -> no detections, the
    # correct empty answer rather than a peak everywhere.
    from soma.detection import attention_to_detection_heatmap
    from soma.detection.peaks import extract_peaks

    geom = compute_dense_geometry(target_size=16, patch_size=4)
    hm = attention_to_detection_heatmap(torch.full((4, *geom.grid_shape), 0.7), geometry=geom, num_classes=1)
    assert float(hm.max()) == pytest.approx(0.0)
    xy, cls, score = extract_peaks(hm, min_distance=3, score_threshold=0.0)
    assert xy.shape == (0, 2)
