"""Sliding-window segmentation inference over an image larger than one tile.

The training/eval path consumes fixed ``target_size`` supervision tiles; this module
assembles a **full-resolution** prediction over an arbitrary-size image by sliding the
trained tile across it (reflect-padded so even a sub-tile image yields one clean tile),
running the exact training forward per tile, and blending the per-tile **softmaxes** on
a Hann-weighted canvas. ``argmax`` of the blended canvas is the class map.

Two sliding axes, deliberately distinct (do not conflate):

* **Inner / token space** — :func:`soma.dense.sliding.encode_dense_sliding` slides the
  *encoder window* inside one padded tile and blends *token grids*; it exists for
  pos-embed reasons and is part of the model forward.
* **Outer / pixel space** — *this module* slides the *whole trained tile* over the
  *image* and blends *softmaxes*. It wraps the model forward; the model is unchanged.

The two share only the cover rule (:func:`soma.dense.sliding.cover_origins`); the Hann
blend lives in each axis's own space (token-grid torch vs pixel-canvas numpy).

Spacing: the trained model expects its tiles at the **training read-spacing**. Pyramidal
inputs (multi-resolution TIFF) are read at that spacing by the dense reader directly. A
flat raster (PNG/JPEG) carries no spacing, so to honour the training scale the caller
passes its ``native_spacing_um`` and :meth:`predict_image` resamples the image to the
training spacing before tiling, then maps the prediction back to the native pixel dims.

Scope: bounded images that materialize as a single in-memory canvas (tiles/ROIs).
WSI-scale streaming (level-0 windowed reads + slide-mask stitching) is out of scope.

The models are :class:`~soma.training.model.DenseModel` models with a frozen encoder stage
(:func:`soma.extract.live_encoder`); fold models sharing one stage encode each tile once.
A finished run's fold ensemble is its bundles: :meth:`SlidingWindowSegmentationPredictor.from_bundles`
takes the list ``[soma.load(p) for p in fold_bundles]``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from soma.dense.geometry import DenseGridGeometry
from soma.dense.reader import read_image_at_spacing
from soma.dense.sliding import cover_origins

__all__ = [
    "PredictionResult",
    "SlidingWindowSegmentationPredictor",
]

_FLAT_SUFFIXES = {".png", ".jpg", ".jpeg"}


@dataclass
class PredictionResult:
    """A full-image segmentation prediction at the input's native pixel dimensions.

    ``labels`` is the ``(H, W)`` argmax class map (indices ``0..C-1``; the caller applies
    any label remap). ``probs`` is the ``(C, H, W)`` blended softmax (float32) when
    requested, else ``None`` — dropping it keeps memory bounded for large ROIs.

    Spacing provenance (so a per-ROI scale decision is auditable across a batch):
    ``native_spacing_um`` is the input's native µm/px (``None`` if not supplied);
    ``applied_scale`` is the image resize factor used to reach the training spacing
    (``None`` ⇒ no resample: either within tolerance, or a coarser-than-training input
    predicted at native scale — see :meth:`SlidingWindowSegmentationPredictor.predict_image`).
    """

    labels: np.ndarray
    probs: np.ndarray | None = None
    native_spacing_um: float | None = None
    applied_scale: float | None = None


@dataclass
class _ResamplePlan:
    """Resolved image-resample decision for one input (see ``_plan_resample``)."""

    scale: float | None  # image resize factor to reach training spacing; None ⇒ no resize
    interpolation: str | None  # "area" (downscale) | "linear" (upscale); None if no resize
    native_mismatch_pct: float | None  # set only when predicting at native despite a mismatch


def _raw_pixels(crops: Sequence[np.ndarray]) -> torch.Tensor:
    """``(H, W, 3)`` uint8 tile crops -> the ``(B, 3, H, W)`` uint8 batch a
    :class:`~soma.training.model.DenseModel` hands an image-prior decoder."""
    pixels = torch.from_numpy(np.ascontiguousarray(np.stack(crops)))
    return pixels.permute(0, 3, 1, 2).contiguous()


def _hann2d(height: int, width: int) -> np.ndarray:
    """Separable raised-cosine window, **strictly positive** everywhere (float32).

    Same family as :func:`soma.dense.sliding._hann_1d` (``0.5 - 0.5*cos(2*pi*(i+1)/(n+1))``,
    no zero endpoints) so overlapping tiles blend smoothly and the accumulated weight is
    never zero where a tile covers. A degenerate dim (length 1) is uniform.
    """

    def _w(n: int) -> np.ndarray:
        if n <= 1:
            return np.ones(n, dtype=np.float32)
        i = np.arange(1, n + 1, dtype=np.float32)
        return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / (n + 1))).astype(np.float32)

    return np.outer(_w(height), _w(width)).astype(np.float32)


@dataclass
class SlidingWindowSegmentationPredictor:
    """Assemble a full-image prediction by sliding the trained tile + blending softmaxes.

    Holds one or more eval-mode :class:`~soma.training.model.DenseModel` models with an
    encoder stage (ensembled by mean softmax — the multi-fold convention) plus the
    geometry/transform/reader settings that pin each tile to the training regime. Build it
    with :meth:`from_encoder`, which takes those settings off the frozen encoder stage so
    inference matches training by construction.

    Attributes:
        models: Eval-mode ``DenseModel`` models; each encodes a padded ``(B, 3, Henc, Wenc)``
            batch with its stage and returns ``(B, C, target_h, target_w)`` logits.
        geometry: The run's :class:`DenseGridGeometry` (tile ``target_size`` + pad/encoded).
        preprocessor: The stage's itemwise preprocessor used during training.
        device: Device on which the trainable models live.
        spacing_um: The training read-spacing (used by :meth:`predict_image`).
        backend / tolerance: hs2p reader settings for pyramidal inputs.
    """

    models: Sequence
    geometry: DenseGridGeometry
    preprocessor: object
    device: torch.device
    spacing_um: float | None = None
    backend: str = "auto"
    tolerance: float = 0.05

    @classmethod
    def from_encoder(
        cls,
        encoder,
        models: Sequence,
        *,
        spacing_um: float | None,
        backend: str = "auto",
        tolerance: float = 0.05,
    ) -> "SlidingWindowSegmentationPredictor":
        """Build from the frozen encoder stage the models share + the trained models.

        The geometry and the transform / pad contract come off the stage (the same values
        training used), so the per-tile forward here is identical to the trained one;
        ``spacing_um`` / ``backend`` / ``tolerance`` are the training read settings.
        """
        if not models:
            raise ValueError("SlidingWindowSegmentationPredictor needs at least one model")
        parameter = next(models[0].parameters(), None)
        return cls(
            models=list(models),
            geometry=encoder.geometry,
            preprocessor=encoder.preprocessor(),
            device=parameter.device if parameter is not None else torch.device("cpu"),
            spacing_um=spacing_um,
            backend=backend,
            tolerance=tolerance,
        )

    @classmethod
    def from_bundles(
        cls,
        bundles: Sequence,
        *,
        encoder=None,
        spacing_um: float | None = None,
        backend: str = "auto",
        tolerance: float = 0.05,
        device: str | torch.device | None = None,
    ) -> "SlidingWindowSegmentationPredictor":
        """Build the fold ensemble from loaded bundles (:func:`soma.load`, one per fold).

        A bundle trained live carries its frozen encoder stage (bundles loaded together
        share one stage). A bundle trained on cached grids has none, so ``encoder`` — the
        stage that extracted those grids, e.g. :func:`soma.pipeline.build.live_encoder_for`
        of the run's config — is put in front of its adaptor, decoder and head; its frame
        must match the one the bundle was trained on. ``spacing_um`` defaults to the read
        spacing the bundles record.
        """
        from soma.training.model import DenseModel

        if not bundles:
            raise ValueError("SlidingWindowSegmentationPredictor needs at least one bundle")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        models = []
        for bundle in bundles:
            model = bundle.model
            if model.encoder is None:
                if encoder is None:
                    raise ValueError(
                        "this bundle was trained on cached grids and carries no encoder stage; "
                        "pass encoder= (the stage that extracted them)"
                    )
                if encoder.geometry.layout != model.head.geometry.layout:
                    raise ValueError(
                        f"encoder frame {encoder.geometry.layout} differs from the frame the "
                        f"bundle was trained on, {model.head.geometry.layout}"
                    )
                model = DenseModel(model.decoder, model.head, encoder=encoder, adaptor=model.adaptor)
            models.append(model.to(device).eval())
        if spacing_um is None:
            spacing_um = bundles[0].preprocessing.get("spacing_um")
        return cls.from_encoder(
            models[0].encoder, models, spacing_um=spacing_um, backend=backend, tolerance=tolerance
        )

    @torch.inference_mode()
    def predict_array(
        self,
        rgb: np.ndarray,
        *,
        overlap: float = 0.5,
        batch_size: int = 8,
        return_probs: bool = False,
    ) -> PredictionResult:
        """Slide over an ``(H, W, 3)`` uint8 array **already at the training spacing**.

        Returns a prediction at the array's own resolution. ``overlap`` is the only
        test-time knob (inter-tile stitch smoothing in ``[0, 1)``); the tile *size* is the
        trained ``geometry.target_size`` and is not negotiable — the decoder/head were
        built for exactly that field of view.
        """
        if not 0.0 <= float(overlap) < 1.0:
            raise ValueError(f"overlap must be in [0, 1), got {overlap}")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"predict_array expects (H, W, 3) uint8, got shape {rgb.shape}")
        th, tw = (int(s) for s in self.geometry.target_size)
        stride_h = max(1, int(round(th * (1.0 - float(overlap)))))
        stride_w = max(1, int(round(tw * (1.0 - float(overlap)))))

        H, W = int(rgb.shape[0]), int(rgb.shape[1])
        # Reflect-pad only where the image is smaller than a tile, so even a sub-tile ROI
        # yields one full-size tile (no constant border → no out-of-distribution edge).
        pad_b, pad_r = max(0, th - H), max(0, tw - W)
        arr = rgb
        if pad_b or pad_r:
            arr = np.pad(arr, ((0, pad_b), (0, pad_r), (0, 0)), mode="reflect")
        Hp, Wp = arr.shape[0], arr.shape[1]

        num_classes = self._num_classes()
        window = _hann2d(th, tw)
        prob = np.zeros((num_classes, Hp, Wp), dtype=np.float32)
        wsum = np.zeros((Hp, Wp), dtype=np.float32)
        origins = [
            (y, x)
            for y in cover_origins(Hp, th, stride_h)
            for x in cover_origins(Wp, tw, stride_w)
        ]

        for start in range(0, len(origins), max(1, int(batch_size))):
            batch = origins[start : start + max(1, int(batch_size))]
            crops = [arr[y : y + th, x : x + tw] for (y, x) in batch]
            X = torch.stack([self._tile_to_encoded(c) for c in crops])  # stage owns transfer
            soft = self._ensemble_softmax(X, _raw_pixels(crops))  # (B, C, th, tw) cpu float32
            for k, (y, x) in enumerate(batch):
                prob[:, y : y + th, x : x + tw] += soft[k] * window
                wsum[y : y + th, x : x + tw] += window

        prob /= np.maximum(wsum, 1e-8)[None]
        prob = prob[:, :H, :W]  # drop the reflect pad → back to the input dims
        labels = prob.argmax(axis=0).astype(_label_dtype(num_classes))
        return PredictionResult(labels=labels, probs=prob if return_probs else None)

    def predict_image(
        self,
        path: str | Path,
        *,
        native_spacing_um: float | None = None,
        allow_upsample: bool = False,
        overlap: float = 0.5,
        batch_size: int = 8,
        return_probs: bool = False,
    ) -> PredictionResult:
        """Read an image, predict, and return at the input's **native** pixel dimensions.

        Routing mirrors the dense reader: a pyramidal/spacing-bearing input is read at the
        training ``spacing_um`` (so it already arrives at the right scale); a **flat** raster
        (PNG/JPEG) is read at native resolution and resampled to the training scale only when
        ``native_spacing_um`` is given and differs beyond ``tolerance`` (the same boundary as
        hs2p's ``select_level``). Resampling is direction-aware — **area** to downscale a
        finer-than-training input (anti-aliased, like the reader's pyramidal path), **linear**
        to upscale a coarser one.

        By default a **coarser**-than-training input is *not* upsampled (hs2p never upsamples):
        it is predicted at its native scale and a warning names the field-of-view mismatch.
        Pass ``allow_upsample=True`` to instead resample up to the training scale. After any
        resample the prediction is mapped back to the native pixel dims. ``native_spacing_um``
        applies only to flat inputs; leave it ``None`` for pyramidal inputs (the reader owns
        spacing there).
        """
        path = Path(path)
        is_flat = path.suffix.lower() in _FLAT_SUFFIXES
        arr = read_image_at_spacing(
            path, spacing_um=self.spacing_um, backend=self.backend, tolerance=self.tolerance
        )
        H0, W0 = int(arr.shape[0]), int(arr.shape[1])

        plan = self._plan_resample(is_flat, native_spacing_um, allow_upsample)
        if plan.native_mismatch_pct is not None:
            warnings.warn(
                f"{path.name}: native spacing {native_spacing_um:.4g} µm/px is "
                f"{plan.native_mismatch_pct:.1f}% coarser than the training spacing "
                f"{self.spacing_um:.4g} µm/px; predicting at native scale (no upsample). "
                f"Pass allow_upsample=True to resample to the training scale.",
                stacklevel=2,
            )
        if plan.scale is not None:
            new_h, new_w = max(1, round(H0 * plan.scale)), max(1, round(W0 * plan.scale))
            arr = _resize_rgb(arr, (new_h, new_w), interpolation=plan.interpolation)

        # When resampled, we need probs to map cleanly back to native dims (bilinear on the
        # canvas beats nearest on argmax labels at boundaries), so force them on internally.
        result = self.predict_array(
            arr,
            overlap=overlap,
            batch_size=batch_size,
            return_probs=return_probs or plan.scale is not None,
        )
        if plan.scale is not None:
            result = self._resize_result_to(result, (H0, W0), return_probs=return_probs)
        result.native_spacing_um = native_spacing_um
        result.applied_scale = plan.scale
        return result

    # -- internals -------------------------------------------------------------------

    def _num_classes(self) -> int:
        head = getattr(self.models[0], "head", None)
        n = getattr(head, "num_classes", None)
        if n is None:
            raise ValueError("model.head.num_classes is required to size the prediction canvas")
        return int(n)

    def _tile_to_encoded(self, crop: np.ndarray) -> torch.Tensor:
        """Tile crop (target_size px) -> normalized, pad-to-encoded ``(3, Henc, Wenc)`` tensor."""
        pixels = torch.from_numpy(np.ascontiguousarray(crop)).permute(2, 0, 1)
        return self.preprocessor(pixels)

    def _ensemble_softmax(self, X: torch.Tensor, pixels: torch.Tensor) -> np.ndarray:
        """Mean softmax across folds for a padded batch -> ``(B, C, th, tw)`` cpu float32.

        ``pixels`` are the batch's raw uint8 tiles, which a model with an image-prior
        decoder reads as its side input. Fold models sharing one encoder stage (the
        multi-fold case built by :meth:`from_bundles`) run the expensive encoder
        forward **once** and every fold's adaptor + decoder + head on the shared grid;
        otherwise each model encodes the batch itself.
        """
        models = list(self.models)
        shared = all(m.encoder is models[0].encoder for m in models[1:])
        grid = models[0].encode(X) if shared else None
        acc: torch.Tensor | None = None
        for model in models:
            encoded = grid if shared else model.encode(X)
            s = torch.softmax(model.forward_from_grid(encoded, pixels).logits, dim=1)
            acc = s if acc is None else acc + s
        return (acc / len(models)).float().cpu().numpy()

    def _plan_resample(
        self, is_flat: bool, native_spacing_um: float | None, allow_upsample: bool
    ) -> _ResamplePlan:
        """Decide whether/how to resize a flat input to the training spacing.

        Mirrors hs2p's ``select_level`` boundary: within ``tolerance`` of the training
        spacing ⇒ treat as an exact match (no resize). Beyond it, a **finer** input is
        downscaled (area); a **coarser** input is upscaled (linear) only when
        ``allow_upsample`` — otherwise it is predicted at native scale and flagged.
        """
        none = _ResamplePlan(scale=None, interpolation=None, native_mismatch_pct=None)
        # Pyramidal inputs are already read at spacing_um by the reader; nothing to do here.
        if native_spacing_um is None or self.spacing_um is None or not is_flat:
            return none
        scale = float(native_spacing_um) / float(self.spacing_um)
        if abs(scale - 1.0) <= float(self.tolerance):
            return none  # within tolerance ≡ exact level match
        if scale < 1.0:
            return _ResamplePlan(scale=scale, interpolation="area", native_mismatch_pct=None)
        # Coarser than training, beyond tolerance.
        if allow_upsample:
            return _ResamplePlan(scale=scale, interpolation="linear", native_mismatch_pct=None)
        return _ResamplePlan(scale=None, interpolation=None, native_mismatch_pct=(scale - 1.0) * 100.0)

    def _resize_result_to(
        self, result: PredictionResult, size: tuple[int, int], *, return_probs: bool
    ) -> PredictionResult:
        """Map a prediction (at training scale) back to native ``(H0, W0)`` via the probs."""
        assert result.probs is not None  # predict_image forces probs on when resampling
        probs_t = torch.from_numpy(result.probs).unsqueeze(0)  # (1, C, h, w)
        probs_t = F.interpolate(probs_t, size=size, mode="bilinear", align_corners=False)
        probs = probs_t.squeeze(0).numpy()
        labels = probs.argmax(axis=0).astype(_label_dtype(probs.shape[0]))
        return PredictionResult(labels=labels, probs=probs if return_probs else None)


def _label_dtype(num_classes: int) -> type[np.unsignedinteger]:
    return np.uint8 if num_classes <= 256 else np.uint16


def _resize_rgb(arr: np.ndarray, size: tuple[int, int], *, interpolation: str) -> np.ndarray:
    """Resize an ``(H, W, 3)`` uint8 array to ``(new_h, new_w)`` via hs2p's ``resize_array``.

    Reuses the same resize primitive (cv2-backed) as the spacing-aware reader, so a flat PNG
    resampled here and a pyramidal TIFF resampled by the reader go through one implementation.
    """
    from hs2p.wsi.wsi import resize_array

    new_h, new_w = size
    out = resize_array(arr, (new_w, new_h), interpolation=interpolation)  # (width, height)
    return np.ascontiguousarray(out.astype(np.uint8))
