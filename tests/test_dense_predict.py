"""Tests for the outer (pixel-space) sliding-window segmentation predictor.

These exercise the inference-time *assembly* — tile cover, reflect-pad, Hann blend,
spacing resample, native-dim mapping, encode-once ensembles — over real
:class:`~soma.training.model.DenseModel`\\ s whose encoder stage is
``soma.testing.FakeEncoder`` (mean colour per patch) and whose decoder turns that colour
into class logits, so a painted region must land at the right place. Also covers the
slide2vec boundary of :func:`soma.extract.live_encoder` with a stubbed kit.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


from soma.decoders.base import Decoder
from soma.dense.predict import SlidingWindowSegmentationPredictor, _hann2d
from soma.tasks.segmentation import SegmentationHead
from soma.testing import FakeEncoder
from soma.training.model import DenseModel

PATCH = 16
TILE = 64


class _CountingEncoder(FakeEncoder):
    """The fake stage, counting how often it encodes a batch."""

    def __init__(self) -> None:
        super().__init__(target_size=TILE, patch_size=PATCH)
        self.calls = 0

    def forward(self, pixels):
        self.calls += 1
        return super().forward(pixels)


class _ColourDecoder(Decoder):
    """Logits = the token's normalised RGB, so class = dominant colour (R, G, B)."""

    def forward(self, X):
        return X[:, :3] * 10.0

    @property
    def num_classes(self) -> int:
        return 3


class _ImageColourDecoder(Decoder):
    """Image-prior decoder whose logits are the tile's ``[0, 1]`` RGB pixels."""

    consumes_image = True

    def forward(self, X, image):
        return image * 10.0

    @property
    def num_classes(self) -> int:
        return 3


def _model(encoder, decoder=None) -> DenseModel:
    head = SegmentationHead(num_classes=3, geometry=encoder.geometry)
    return DenseModel(decoder or _ColourDecoder(), head, encoder=encoder).eval()


def _predictor(models, *, spacing_um=None):
    return SlidingWindowSegmentationPredictor.from_encoder(
        models[0].encoder, models, spacing_um=spacing_um
    )


def _painted(h, w):
    """Left half red, right half blue."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2, 0] = 255
    img[:, w // 2 :, 2] = 255
    return img


# --- pure window math ---------------------------------------------------------------


def test_hann2d_strictly_positive_and_shaped():
    w = _hann2d(TILE, TILE)
    assert w.shape == (TILE, TILE)
    assert (w > 0).all()  # no zero weights anywhere → wsum never zero where a tile covers
    assert _hann2d(1, 5).shape == (1, 5) and (_hann2d(1, 5) > 0).all()
    assert np.allclose(_hann2d(1, 1), 1.0)  # fully degenerate → uniform


# --- predict_array ------------------------------------------------------------------


def test_output_matches_input_dims_multi_tile():
    pred = _predictor([_model(_CountingEncoder())])
    img = np.zeros((150, 200, 3), dtype=np.uint8)
    img[..., 1] = 255  # all green → class 1 everywhere
    out = pred.predict_array(img, overlap=0.5)
    assert out.labels.shape == (150, 200)
    assert out.probs is None
    assert (out.labels == 1).all()


def test_sub_tile_image_reflect_padded_to_one_tile():
    pred = _predictor([_model(_CountingEncoder())])
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    img[..., 2] = 255  # blue → class 2
    out = pred.predict_array(img, overlap=0.0)
    assert out.labels.shape == (40, 50)
    assert (out.labels == 2).all()


def test_spatial_regions_land_in_the_right_place():
    pred = _predictor([_model(_CountingEncoder())])
    img = np.zeros((128, 256, 3), dtype=np.uint8)
    img[:, :128, 0] = 255  # left = red
    img[:, 128:, 1] = 255  # right = green
    out = pred.predict_array(img, overlap=0.5)
    assert out.labels.shape == (128, 256)
    # Away from the blend seam the classes are exact.
    assert (out.labels[:, :100] == 0).all()
    assert (out.labels[:, 156:] == 1).all()


def test_return_probs_shape_and_normalization():
    pred = _predictor([_model(_CountingEncoder())])
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[..., 0] = 200
    out = pred.predict_array(img, overlap=0.25, return_probs=True)
    assert out.probs.shape == (3, 96, 96)
    assert np.allclose(out.probs.sum(axis=0), 1.0, atol=1e-4)  # blended softmax sums to 1


def test_ensemble_averages_models():
    encoder = _CountingEncoder()
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[..., 1] = 255
    one = _predictor([_model(encoder)]).predict_array(img, overlap=0.0)
    two = _predictor([_model(encoder), _model(encoder)]).predict_array(img, overlap=0.0)
    assert np.array_equal(one.labels, two.labels)


def test_overlap_validation():
    pred = _predictor([_model(_CountingEncoder())])
    with pytest.raises(ValueError, match="overlap"):
        pred.predict_array(np.zeros((64, 64, 3), dtype=np.uint8), overlap=1.0)


# --- encode once, decode per fold -----------------------------------------------------


def test_folds_sharing_a_stage_encode_each_batch_once():
    shared = _CountingEncoder()
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[..., 1] = 255
    out = _predictor([_model(shared), _model(shared)]).predict_array(img, overlap=0.0, batch_size=8)
    assert (out.labels == 1).all()
    assert shared.calls == 1  # 4 tiles, one batch, one encode for both folds


def test_folds_with_their_own_stages_each_encode():
    a, b = _CountingEncoder(), _CountingEncoder()
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[..., 2] = 255
    out = _predictor([_model(a), _model(b)]).predict_array(img, overlap=0.0)
    assert (out.labels == 2).all()
    assert a.calls > 0 and b.calls > 0


# --- image-prior decoders ------------------------------------------------------------


@pytest.mark.parametrize("shared", [True, False], ids=["shared-stage", "per-model"])
def test_image_prior_decoder_receives_each_tiles_pixels(shared):
    stage = _CountingEncoder()
    models = [
        _model(stage if shared else _CountingEncoder(), _ImageColourDecoder()) for _ in range(2)
    ]
    out = _predictor(models).predict_array(_painted(96, 160), overlap=0.5)
    assert (out.labels[:, :70] == 0).all()
    assert (out.labels[:, 90:] == 2).all()


def test_mixed_ensemble_hands_pixels_only_to_the_image_prior_member():
    stage = _CountingEncoder()
    models = [_model(stage), _model(stage, _ImageColourDecoder())]
    out = _predictor(models).predict_array(_painted(96, 160), overlap=0.5)
    assert (out.labels[:, :70] == 0).all()
    assert (out.labels[:, 90:] == 2).all()


def test_eva_image_decoder_predicts_through_the_predictor():
    from soma.benchmarks.eva_segmentation import EvaConvWithImageDecoder

    stage = FakeEncoder(target_size=TILE, patch_size=PATCH)
    models = [_model(stage, EvaConvWithImageDecoder(input_dim=3, num_classes=3)) for _ in range(2)]
    out = _predictor(models).predict_array(_painted(80, 100), overlap=0.25)
    assert out.labels.shape == (80, 100)


# --- predict_image: spacing resample + native-dim mapping ---------------------------


def _write_png(tmp_path, h, w, channel):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., channel] = 255
    path = tmp_path / "roi.png"
    Image.fromarray(img).save(path)
    return path


def test_predict_image_downscales_finer_roi_and_returns_native_dims(tmp_path):
    path = _write_png(tmp_path, 120, 160, channel=1)  # green → class 1
    pred = _predictor([_model(_CountingEncoder())], spacing_um=0.5)
    out = pred.predict_image(path, native_spacing_um=0.46, overlap=0.5)
    assert out.labels.shape == (120, 160)  # native PNG dims, not the resampled size
    assert (out.labels == 1).all()
    assert out.native_spacing_um == 0.46
    assert out.applied_scale is not None and out.applied_scale < 1.0


def test_predict_image_coarser_roi_predicts_at_native_and_warns(tmp_path):
    path = _write_png(tmp_path, 90, 110, channel=2)  # blue → class 2
    pred = _predictor([_model(_CountingEncoder())], spacing_um=0.5)
    with pytest.warns(UserWarning, match="coarser"):
        out = pred.predict_image(path, native_spacing_um=0.53, overlap=0.0)
    assert out.labels.shape == (90, 110)
    assert (out.labels == 2).all()
    assert out.applied_scale is None


def test_predict_image_coarser_roi_upsamples_when_allowed(tmp_path):
    path = _write_png(tmp_path, 90, 110, channel=2)
    pred = _predictor([_model(_CountingEncoder())], spacing_um=0.5)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        out = pred.predict_image(path, native_spacing_um=0.53, allow_upsample=True, overlap=0.0)
    assert out.labels.shape == (90, 110)
    assert (out.labels == 2).all()
    assert out.applied_scale is not None and out.applied_scale > 1.0


def test_predict_image_no_resample_within_tolerance(tmp_path):
    path = _write_png(tmp_path, 70, 70, channel=2)
    pred = _predictor([_model(_CountingEncoder())], spacing_um=0.5)
    out = pred.predict_image(path, native_spacing_um=0.51, overlap=0.0)
    assert out.labels.shape == (70, 70)
    assert (out.labels == 2).all()
    assert out.applied_scale is None and out.native_spacing_um == 0.51


# --- the slide2vec boundary: live_encoder; fold bundles -----------------------------


class _Kit:
    """A stub ``DenseEncodeKit``: crop box in slide2vec's (left, top, right, bottom)."""

    geometry = SimpleNamespace(
        target_size=(8, 8),
        patch_size=(4, 4),
        encoded_size=(8, 8),
        grid_shape=(2, 2),
        pad=(0, 0),
        crop_box=(0, 0, 8, 8),
    )

    @staticmethod
    def preprocessor():
        return lambda pixels: pixels.to(torch.float32) / 255.0

    @staticmethod
    def encode(batch):
        return torch.zeros((batch.shape[0], 7, 2, 2), dtype=torch.float32)


def test_live_encoder_wraps_the_public_dense_kit(monkeypatch):
    import slide2vec

    from soma.extract import FrozenEncoder, live_encoder

    prepared = {}

    class _Model:
        device = torch.device("cpu")

        def prepare_dense_encoder(self, *, dense, execution):
            prepared["dense"], prepared["execution"] = dense, execution
            return _Kit()

    monkeypatch.setattr(slide2vec.Model, "from_preset", lambda *a, **k: _Model())
    stage = live_encoder("virchow2", target_size=8, spacing_um=0.5, window=8, precision="fp32")

    assert isinstance(stage, FrozenEncoder)
    assert stage.feature_dim == 7  # probed through the kit, not the preset's pooled width
    assert tuple(stage.geometry.crop_box) == (0, 0, 8, 8)
    assert sum(p.numel() for p in stage.parameters()) == 0
    assert prepared["dense"].target_size == 8 and prepared["dense"].spacing_um == 0.5
    assert prepared["dense"].window_size == 8
    # Raw uint8 pixels are preprocessed by the stage itself.
    assert tuple(stage(torch.zeros(2, 3, 8, 8, dtype=torch.uint8)).shape) == (2, 7, 2, 2)


def test_live_encoder_requests_the_indexed_cuda_device(monkeypatch):
    import slide2vec

    from soma.extract import live_encoder

    class ProbeComplete(Exception):
        pass

    requested = {}

    def from_preset(*args, **kwargs):
        requested["device"] = kwargs.get("device")
        raise ProbeComplete

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(slide2vec.Model, "from_preset", from_preset)
    with pytest.raises(ProbeComplete):
        live_encoder("virchow2", target_size=8, spacing_um=0.5, precision="fp32")
    assert requested["device"] == "cuda:2"


def _fold_bundles(tmp_path, *, folds: int = 2):
    """One bundle per fold of a model trained on cached grids (no encoder stage)."""
    import pandas as pd

    import soma
    from soma.learn import DenseLearner, FitResult
    from soma.pipeline.build import build_grid_model

    stage = FakeEncoder(target_size=TILE, patch_size=PATCH)
    bundles = []
    for fold in range(folds):
        torch.manual_seed(fold)
        model = build_grid_model(
            feature_dim=stage.feature_dim,
            decoder={"name": "linear", "params": {}},
            head={"name": "segmentation", "params": {"num_classes": 3, "geometry": stage.geometry}},
        )
        fit = FitResult(DenseLearner(model), pd.DataFrame(), 0, ("loss", 0.0), None, 0)
        path = soma.save(fit, tmp_path / f"fold_{fold}" / "bundle", preprocessing={"spacing_um": 0.5})
        bundles.append(soma.load(path))
    return stage, bundles


def test_from_bundles_ensembles_the_fold_bundles_over_one_stage(tmp_path):
    stage, bundles = _fold_bundles(tmp_path)

    predictor = SlidingWindowSegmentationPredictor.from_bundles(bundles, encoder=stage, device="cpu")

    assert len(predictor.models) == 2 and all(m.encoder is stage for m in predictor.models)
    assert predictor.spacing_um == 0.5  # the bundles' recorded read spacing
    # Each fold's own weights were loaded.
    first, second = (next(m.decoder.parameters()) for m in predictor.models)
    assert not torch.equal(first, second)
    out = predictor.predict_array(_painted(64, 64), overlap=0.0)
    assert out.labels.shape == (64, 64)


def test_from_bundles_needs_the_stage_for_bundles_trained_on_cached_grids(tmp_path):
    _, bundles = _fold_bundles(tmp_path, folds=1)
    with pytest.raises(ValueError, match="encoder="):
        SlidingWindowSegmentationPredictor.from_bundles(bundles)
    other = FakeEncoder(target_size=32, patch_size=PATCH)
    with pytest.raises(ValueError, match="frame"):
        SlidingWindowSegmentationPredictor.from_bundles(bundles, encoder=other)
