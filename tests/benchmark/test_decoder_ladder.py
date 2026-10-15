"""Decoder-complexity ladder — the new rungs (design §2.4, issue #233).

Rungs 1-2 (``linear`` / ``lightweight_conv``) and their per-decoder unit behaviour live in
``test_decoders.py``. This file covers the three rungs added by #233 at the level the
acceptance criteria state them:

* **rung 4 — multi-FM ensemble**: a ``d->D``-projection decoder consumes the
  ``CompositeDenseFeatureStore``'s concatenated ``(Σdᵢ, h, w)`` grid, param-matched to a
  single-encoder decoder via the projection (the composite width is just a wider ``d``).
* **rung 0 — attention map**: the ``attention_map`` decoder turns the frozen encoder's
  attention grid into a detection heatmap with no parameter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from soma.models.decoders import HeavyConvDecoder, LightweightConvDecoder  # noqa: E402
from soma.extract.dense.composite import CompositeDenseFeatureStore  # noqa: E402
from soma.data.geometry import compute_dense_geometry  # noqa: E402
from soma.extract.dense.store import CachedGridSource, dense_grid_metadata, write_dense_grid  # noqa: E402

TARGET = 16
SPACING = 0.2  # µm/px; source == effective spacing makes the point transform identity


def _write_member(dir_: Path, sample_ids: list[str], *, feature_dim: int, patch: int) -> CachedGridSource:
    """Write a one-encoder dense store (fixed target_size / spacing, own patch+dim)."""
    dir_.mkdir(parents=True, exist_ok=True)
    geom = compute_dense_geometry(target_size=TARGET, patch_size=patch)
    meta = dense_grid_metadata(geom, feature_dim=feature_dim, pad_mode="reflect", spacing_um=SPACING)
    meta.update(source_spacing_um=SPACING, effective_spacing_um=SPACING)
    rng = np.random.default_rng(feature_dim)
    for sid in sample_ids:
        write_dense_grid(dir_, sid, torch.from_numpy(rng.standard_normal((feature_dim, *geom.grid_shape)).astype("float32")), meta)
    return CachedGridSource(dir_)


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
    from soma.config import CacheOptions, FeaturesConfig
    from soma.pipeline import Pipeline
    from tests.experiment_configs import experiment
    from tests.extraction_configs import extraction

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
    config = experiment(
        tmp_path / "manifest.csv",
        tmp_path / "splits.csv",
        output_dir=tmp_path / "out",
        head="detection",
        head_params={"num_classes": 2, "match_distance": 0.6, "sigma": 0.3},
        decoder={"name": "heavy_conv", "params": {"hidden_dim": 32}},
        loaders={"batch_size": 2},
        trainer={"max_epochs": 1},
        evaluation={"metrics": ["mean_f1"]},
        features=FeaturesConfig(extraction=extraction(SPACING, TARGET), cache=CacheOptions(enabled=False)),
    )
    result = Pipeline(config, source=composite).run().fold_results[0]
    assert "mean_f1" in result.test_evaluations["test"].metrics


# --------------------------------------------------------------------------- #
# Rung 0 — the attention_map decoder (the run itself: tests/e2e/test_attention_map.py)
# --------------------------------------------------------------------------- #


def _attention_head(num_classes: int = 1):
    from soma.models.heads.detection import DetectionHead

    geom = compute_dense_geometry(target_size=32, patch_size=4)  # 8x8 token grid
    return DetectionHead(
        num_classes=num_classes, geometry=geom, spacing_um=1.0, match_distance=5.0, sigma=2.0, metrics=["mean_f1"]
    )


def test_attention_map_heatmap_is_the_normalised_attention_on_every_class():
    from soma.benchmark.attention import AttentionMapDecoder

    head = _attention_head(num_classes=2)
    attn = torch.zeros(1, 6, 8, 8)
    attn[0, :, 3, 2] = 1.0  # every head attends to one token
    decoder = AttentionMapDecoder(input_dim=6, num_classes=2)
    heatmap = head(decoder(attn))[0]
    assert list(decoder.parameters()) == []
    # Normalised to [0, 1]; the head's bilinear upsampling samples between token centres.
    assert float(heatmap.max()) > 0.99 and float(heatmap.min()) < 1e-3
    torch.testing.assert_close(heatmap[0], heatmap[1])  # class-blind
    points = head.postprocess(heatmap[None])["points"][0]  # (k, 4) x, y, class, score
    # The one hot token (row 3, col 2: centre x=10, y=14), once per class channel.
    assert sorted(points[:, 2].tolist()) == [0.0, 1.0]
    np.testing.assert_allclose(points[:, :2], [[10, 14], [10, 14]], atol=1.0)


def test_attention_map_flat_grid_yields_no_peaks():
    from soma.benchmark.attention import AttentionMapDecoder

    head = _attention_head()
    heatmap = head(AttentionMapDecoder(input_dim=4, num_classes=1)(torch.full((1, 4, 8, 8), 0.7)))
    assert head.postprocess(heatmap)["points"][0].shape[0] == 0


def test_attention_map_heads_select_channels_and_params_are_checked():
    from soma.benchmark.attention import AttentionMapDecoder, AttentionMapParams
    from soma.models.decoders.registry import decoder_registry

    attn = torch.rand(2, 4, 8, 8)
    picked = AttentionMapDecoder(input_dim=4, num_classes=1, heads=[2])(attn)
    alone = AttentionMapDecoder(input_dim=1, num_classes=1)(attn[:, 2:3])
    torch.testing.assert_close(picked, alone)
    assert decoder_registry.resolve("attention_map") is AttentionMapDecoder
    with pytest.raises(ValueError, match="reduction"):
        AttentionMapParams(reduction="median")
    with pytest.raises(ValueError, match="at least one"):
        AttentionMapParams(heads=[])
    with pytest.raises(ValueError, match="out of range"):
        AttentionMapDecoder(input_dim=4, num_classes=1, heads=[4])
