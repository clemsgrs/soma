"""Tests for the outer (pixel-space) sliding-window segmentation predictor.

These exercise the inference-time *assembly* — tile cover, reflect-pad, Hann blend,
spacing resample, native-dim mapping — with a tiny stand-in model (no real encoder), so
they are fast and deterministic. The per-tile forward itself is the trained model's and
is covered elsewhere; here we only verify the predictor stitches it correctly.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from soma.dense.geometry import compute_dense_geometry  # noqa: E402
from soma.dense.predict import (  # noqa: E402
    SlidingWindowSegmentationPredictor,
    _hann2d,
)
from soma.dense.live import build_live_segmentation_source  # noqa: E402
from soma.config import (  # noqa: E402
    AugmentationConfig,
    DecoderConfig,
    EncoderConfig,
    ExecutionConfig,
    PipelineConfig,
    PreprocessingConfig,
    TaskConfig,
)

PATCH = 16
TILE = 64


def _preprocessor(pixels):
    """Kit-style uint8 CHW -> float CHW in [0, 1]."""
    return pixels.to(torch.float32) / 255.0


def _color_logits(geometry, grid):
    """argmax over the 3 channels -> class = dominant RGB, cropped to target_size."""
    top, left, h, w = geometry.crop_box
    return grid[:, :3, top : top + h, left : left + w] * 10.0  # sharpen the argmax


class _ColorModel:
    """Plain callable model (no encode/decode split) — exercises the per-model fallback.

    Lets a test paint regions a known colour and assert they land at the right place after
    tiling/blending. Crops the (padded) input back to ``target_size`` via the geometry, so
    its output matches the real head's ``(B, C, target_h, target_w)`` contract.
    """

    def __init__(self, geometry):
        self._g = geometry
        self.task_head = SimpleNamespace(num_classes=3)

    def __call__(self, X):
        return SimpleNamespace(logits=_color_logits(self._g, X))


class _GridModel:
    """Model exposing the encode/forward_from_grid split (mirrors LiveSegmentationModel).

    Counts ``encode`` calls so a test can assert the ensemble encodes a shared tile **once**.
    ``encode`` is a pass-through (the 'grid' is the padded input), so ``forward_from_grid``
    decodes colour exactly like :class:`_ColorModel`.
    """

    def __init__(self, geometry, kit):
        self._g = geometry
        self.kit = kit
        self.task_head = SimpleNamespace(num_classes=3)
        self.encode_calls = 0

    def encode(self, X):
        self.encode_calls += 1
        return X

    def forward_from_grid(self, grid):
        return SimpleNamespace(logits=_color_logits(self._g, grid))

    def __call__(self, X):
        return self.forward_from_grid(self.encode(X))


def _predictor(models, *, spacing_um=None):
    geom = compute_dense_geometry(target_size=TILE, patch_size=PATCH)
    return SlidingWindowSegmentationPredictor(
        models=models,
        geometry=geom,
        preprocessor=_preprocessor,
        device=torch.device("cpu"),
        spacing_um=spacing_um,
    )


def _geom():
    return compute_dense_geometry(target_size=TILE, patch_size=PATCH)


def test_build_live_segmentation_source_prepares_public_dense_kit(monkeypatch):
    """Inference can rebuild a live source from a saved config without loading a Dataset."""

    class _Kit:
        geometry = SimpleNamespace(
            target_size=(8, 8),
            patch_size=(4, 4),
            encoded_size=(8, 8),
            grid_shape=(2, 2),
            pad=(0, 0, 0, 0),
            crop_box=(0, 0, 8, 8),
        )

        @staticmethod
        def preprocessor():
            return _preprocessor

        @staticmethod
        def encode(batch):
            return torch.zeros((batch.shape[0], 7, 2, 2), dtype=torch.float32)

    prepared = {}

    class _Model:
        device = torch.device("cpu")

        def prepare_dense_encoder(self, *, dense, execution):
            prepared["dense"] = dense
            prepared["execution"] = execution
            return _Kit()

    import slide2vec

    monkeypatch.setattr(
        slide2vec.Model,
        "from_preset",
        lambda *args, **kwargs: _Model(),
    )
    config = PipelineConfig(
        dataset_csv="development.csv",
        splits_csv="development_splits.csv",
        output_root="runs",
        dataset_type="segmentation",
        encoder=EncoderConfig(name="virchow2", precision="fp32", output_variant="cls"),
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": 4}),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=8,
            requested_spacing_um=0.5,
            dense_window_size=8,
            feature_kind="patch_features",
        ),
        aggregator=None,
        execution=ExecutionConfig(num_gpus=1, precision="fp32"),
        augmentation=AugmentationConfig(),
    )

    source = build_live_segmentation_source(config)

    assert source.feature_dim == 7
    assert source.spacing_um == 0.5
    assert source.geometry.target_size == (8, 8)
    assert prepared["dense"].target_size == 8
    assert prepared["dense"].spacing_um == 0.5


def test_build_live_segmentation_source_requests_indexed_cuda_device(monkeypatch):
    """Live encoding names the active CUDA device explicitly at the slide2vec boundary."""

    class ProbeComplete(Exception):
        pass

    requested = {}

    def from_preset(*args, **kwargs):
        requested["device"] = kwargs.get("device")
        raise ProbeComplete

    import slide2vec

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(slide2vec.Model, "from_preset", from_preset)
    config = PipelineConfig(
        dataset_csv="development.csv",
        splits_csv="development_splits.csv",
        output_root="runs",
        dataset_type="segmentation",
        encoder=EncoderConfig(name="virchow2", precision="fp32", output_variant="cls"),
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": 4}),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=8,
            requested_spacing_um=0.5,
            dense_window_size=8,
            feature_kind="patch_features",
        ),
        aggregator=None,
        execution=ExecutionConfig(num_gpus=1, precision="fp32"),
        augmentation=AugmentationConfig(),
    )

    with pytest.raises(ProbeComplete):
        build_live_segmentation_source(config)

    assert requested["device"] == "cuda:2"


# --- pure window math ---------------------------------------------------------------


def test_hann2d_strictly_positive_and_shaped():
    w = _hann2d(TILE, TILE)
    assert w.shape == (TILE, TILE)
    assert (w > 0).all()  # no zero weights anywhere → wsum never zero where a tile covers
    # a length-1 axis is uniform (weight 1) while the other axis keeps its Hann
    assert _hann2d(1, 5).shape == (1, 5) and (_hann2d(1, 5) > 0).all()
    assert np.allclose(_hann2d(1, 1), 1.0)  # fully degenerate → uniform


# --- predict_array ------------------------------------------------------------------


def test_output_matches_input_dims_multi_tile():
    pred = _predictor([_ColorModel(_geom())])
    img = np.zeros((150, 200, 3), dtype=np.uint8)
    img[..., 1] = 255  # all green → class 1 everywhere
    out = pred.predict_array(img, overlap=0.5)
    assert out.labels.shape == (150, 200)
    assert out.probs is None
    assert (out.labels == 1).all()


def test_sub_tile_image_reflect_padded_to_one_tile():
    pred = _predictor([_ColorModel(_geom())])
    img = np.zeros((40, 50, 3), dtype=np.uint8)
    img[..., 2] = 255  # blue → class 2
    out = pred.predict_array(img, overlap=0.0)
    assert out.labels.shape == (40, 50)
    assert (out.labels == 2).all()


def test_spatial_regions_land_in_the_right_place():
    # Left half red (class 0), right half green (class 1); the seam must split cleanly.
    pred = _predictor([_ColorModel(_geom())])
    img = np.zeros((128, 256, 3), dtype=np.uint8)
    img[:, :128, 0] = 255  # left = red
    img[:, 128:, 1] = 255  # right = green
    out = pred.predict_array(img, overlap=0.5)
    assert out.labels.shape == (128, 256)
    # Away from the blend seam the classes are exact.
    assert (out.labels[:, :100] == 0).all()
    assert (out.labels[:, 156:] == 1).all()


def test_return_probs_shape_and_normalization():
    pred = _predictor([_ColorModel(_geom())])
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[..., 0] = 200
    out = pred.predict_array(img, overlap=0.25, return_probs=True)
    assert out.probs.shape == (3, 96, 96)
    assert np.allclose(out.probs.sum(axis=0), 1.0, atol=1e-4)  # blended softmax sums to 1


def test_ensemble_averages_models():
    # Two identical models → same result as one (mean of equal softmaxes).
    geom = _geom()
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[..., 1] = 255
    one = _predictor([_ColorModel(geom)]).predict_array(img, overlap=0.0)
    two = _predictor([_ColorModel(geom), _ColorModel(geom)]).predict_array(img, overlap=0.0)
    assert np.array_equal(one.labels, two.labels)


def test_overlap_validation():
    pred = _predictor([_ColorModel(_geom())])
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="overlap"):
        pred.predict_array(img, overlap=1.0)


# --- encode-once-decode-per-fold ----------------------------------------------------


def test_shared_encoder_ensemble_encodes_once():
    # Two folds sharing one encoder object → the tile is encoded once, both decoders reused.
    geom = _geom()
    encoder = object()  # shared sentinel
    a = _GridModel(geom, encoder)
    b = _GridModel(geom, encoder)
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[..., 1] = 255
    out = _predictor([a, b]).predict_array(img, overlap=0.5)
    assert (out.labels == 1).all()
    # Encoder ran on model a only; b's decoder reused a's grid (b never encodes).
    assert a.encode_calls > 0 and b.encode_calls == 0


def test_heterogeneous_encoders_fall_back_to_per_model_encode():
    # Different encoder objects → cannot share; each model encodes (fallback path).
    geom = _geom()
    a = _GridModel(geom, object())
    b = _GridModel(geom, object())
    img = np.zeros((96, 96, 3), dtype=np.uint8)
    img[..., 2] = 255
    out = _predictor([a, b]).predict_array(img, overlap=0.0)
    assert (out.labels == 2).all()
    assert a.encode_calls > 0 and b.encode_calls > 0  # both encoded


def test_plain_callable_models_use_fallback():
    # _ColorModel has no encode/forward_from_grid → fallback path still works.
    pred = _predictor([_ColorModel(_geom()), _ColorModel(_geom())])
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[..., 0] = 255
    out = pred.predict_array(img, overlap=0.0)
    assert (out.labels == 0).all()


# --- predict_image: spacing resample + native-dim mapping ---------------------------


def _write_png(tmp_path, h, w, channel):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[..., channel] = 255
    path = tmp_path / "roi.png"
    Image.fromarray(img).save(path)
    return path


def test_predict_image_downscales_finer_roi_and_returns_native_dims(tmp_path):
    # Finer than training (0.46 < 0.5) → area downscale, prediction mapped back to native dims.
    path = _write_png(tmp_path, 120, 160, channel=1)  # green → class 1
    pred = _predictor([_ColorModel(_geom())], spacing_um=0.5)
    out = pred.predict_image(path, native_spacing_um=0.46, overlap=0.5)
    assert out.labels.shape == (120, 160)  # native PNG dims, not the resampled size
    assert (out.labels == 1).all()
    assert out.native_spacing_um == 0.46
    assert out.applied_scale is not None and out.applied_scale < 1.0  # downscaled


def test_predict_image_coarser_roi_predicts_at_native_and_warns(tmp_path):
    # Coarser than training (0.53 > 0.5), beyond tolerance, default → no upsample + warn.
    path = _write_png(tmp_path, 90, 110, channel=2)  # blue → class 2
    pred = _predictor([_ColorModel(_geom())], spacing_um=0.5)
    with pytest.warns(UserWarning, match="coarser"):
        out = pred.predict_image(path, native_spacing_um=0.53, overlap=0.0)
    assert out.labels.shape == (90, 110)
    assert (out.labels == 2).all()
    assert out.native_spacing_um == 0.53
    assert out.applied_scale is None  # predicted at native, no resample


def test_predict_image_coarser_roi_upsamples_when_allowed(tmp_path):
    # Same coarser ROI, but allow_upsample=True → linear upscale, no warning, native dims back.
    path = _write_png(tmp_path, 90, 110, channel=2)
    pred = _predictor([_ColorModel(_geom())], spacing_um=0.5)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")  # any warning would fail the test
        out = pred.predict_image(path, native_spacing_um=0.53, allow_upsample=True, overlap=0.0)
    assert out.labels.shape == (90, 110)
    assert (out.labels == 2).all()
    assert out.applied_scale is not None and out.applied_scale > 1.0  # upscaled


def test_predict_image_no_resample_within_tolerance(tmp_path):
    # native within tolerance of training spacing → no resize, exact native dims, no scale.
    path = _write_png(tmp_path, 70, 70, channel=2)
    pred = _predictor([_ColorModel(_geom())], spacing_um=0.5)
    out = pred.predict_image(path, native_spacing_um=0.51, overlap=0.0)
    assert out.labels.shape == (70, 70)
    assert (out.labels == 2).all()
    assert out.applied_scale is None and out.native_spacing_um == 0.51
