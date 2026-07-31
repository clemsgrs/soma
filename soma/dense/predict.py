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
WSI-scale streaming (level-0 windowed reads + slide-mask stitching) is out of scope —
see ``design/segmentation-design.md`` §1.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from slide2vec.runtime.dense_sliding import cover_origins

from soma.dense.geometry import DenseGridGeometry
from soma.dense.reader import read_image_at_spacing

__all__ = [
    "PredictionResult",
    "SlidingWindowSegmentationPredictor",
    "build_live_segmentation_models",
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

    Holds one or more eval-mode segmentation models (ensembled by mean softmax — the
    multi-fold convention) plus the geometry/transform/reader settings that pin each tile
    to the training regime. Build it from a :class:`~soma.dense.live.LiveSegmentationSource`
    via :meth:`from_source`, which copies those settings off the source so inference matches
    training by construction.

    Attributes:
        models: Eval-mode models whose ``forward(X) -> .logits`` is the training forward
            (e.g. :class:`~soma.training.model.LiveSegmentationModel`). Each consumes a
            padded ``(B, 3, Henc, Wenc)`` batch and returns ``(B, C, target_h, target_w)``.
        geometry: The run's :class:`DenseGridGeometry` (tile ``target_size`` + pad/encoded).
        dense_transform: The encoder's normalization-only transform (tile -> CHW tensor).
        device: Device to run the models on (inputs are moved here).
        pad_mode / image_pad_value: Pad-to-encoded contract, mirroring extraction.
        spacing_um: The training read-spacing (used by :meth:`predict_image`).
        backend / tolerance: hs2p reader settings for pyramidal inputs.
    """

    models: Sequence
    geometry: DenseGridGeometry
    dense_transform: object
    device: torch.device
    pad_mode: str = "reflect"
    image_pad_value: float | None = None
    spacing_um: float | None = None
    backend: str = "auto"
    tolerance: float = 0.05

    @classmethod
    def from_source(cls, source, models: Sequence) -> "SlidingWindowSegmentationPredictor":
        """Build from a :class:`LiveSegmentationSource` + the trained models.

        The geometry, transform, pad contract, read-spacing, and reader backend all come
        off the source (the same values training used), so the per-tile forward here is
        identical to the trained one — inference cannot silently drift from training.
        """
        if not models:
            raise ValueError("SlidingWindowSegmentationPredictor needs at least one model")
        return cls(
            models=list(models),
            geometry=source.geometry,
            dense_transform=source.dense_transform,
            device=torch.device(source.device),
            pad_mode=source.pad_mode,
            image_pad_value=source.image_pad_value,
            spacing_um=source.spacing_um,
            backend=source.backend,
            tolerance=source.tolerance,
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
            tiles = [self._tile_to_encoded(arr[y : y + th, x : x + tw]) for (y, x) in batch]
            X = torch.stack(tiles).to(self.device)  # (B, 3, Henc, Wenc)
            soft = self._ensemble_softmax(X)  # (B, C, th, tw) on cpu, float32
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
        head = getattr(self.models[0], "task_head", None)
        n = getattr(head, "num_classes", None)
        if n is None:
            raise ValueError("model.task_head.num_classes is required to size the prediction canvas")
        return int(n)

    def _tile_to_encoded(self, crop: np.ndarray) -> torch.Tensor:
        """Tile crop (target_size px) -> normalized, pad-to-encoded ``(3, Henc, Wenc)`` tensor."""
        # slide2vec's helper — the same padding embed_images_dense applies during
        # extraction, so inference pads exactly as the cached grids were built.
        from slide2vec.runtime.dense_regions import pad_image_to_encoded

        t = self.dense_transform(Image.fromarray(crop))
        t = torch.as_tensor(t).as_subclass(torch.Tensor)
        return pad_image_to_encoded(
            t, self.geometry, pad_mode=self.pad_mode, image_pad_value=self.image_pad_value
        )

    def _ensemble_softmax(self, X: torch.Tensor) -> np.ndarray:
        """Mean softmax across folds for a padded batch -> ``(B, C, th, tw)`` cpu float32.

        When the fold models **share the frozen encoder** (the multi-fold case built by
        :func:`build_live_segmentation_models` — same encoder object + window/overlap), the
        expensive encoder forward is run **once** and every fold's decoder+head runs on the
        shared grid (``encode`` / ``forward_from_grid``). Otherwise (heterogeneous models, or
        plain callables that don't expose the split) it falls back to a full ``model(X)`` per
        fold — correct, just N× the encoder cost.
        """
        models = list(self.models)
        acc: torch.Tensor | None = None
        if self._models_share_encoder(models):
            grid = models[0].encode(X)  # one ViT forward, reused by every fold
            for model in models:
                s = torch.softmax(model.forward_from_grid(grid).logits, dim=1)
                acc = s if acc is None else acc + s
        else:
            for model in models:
                s = torch.softmax(model(X).logits, dim=1)
                acc = s if acc is None else acc + s
        return (acc / len(models)).float().cpu().numpy()

    @staticmethod
    def _models_share_encoder(models: list) -> bool:
        """True iff every model exposes the encode/decode split AND shares encoder+window.

        The precondition for encoding once: all models must run the *identical* encoder
        forward (same frozen encoder object, same dense window/overlap), so their grids are
        byte-identical. ``from_source``/``build_live_segmentation_models`` guarantee this;
        anything else (a future multi-encoder ensemble, a test stub) returns False → the
        safe per-model fallback.
        """
        needed = ("encode", "forward_from_grid", "encoder", "window_size", "overlap")
        if not all(all(hasattr(m, attr) for attr in needed) for m in models):
            return False
        head = models[0]
        return all(
            m.encoder is head.encoder
            and m.window_size == head.window_size
            and m.overlap == head.overlap
            for m in models[1:]
        )

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


def build_live_segmentation_models(
    source,
    *,
    decoder_name: str,
    decoder_params: dict | None,
    num_classes: int,
    ckpt_paths: Sequence[str | Path],
    normalization=None,
    projection=None,
    encoder_identity: str = "",
):
    """Reconstruct trained :class:`LiveSegmentationModel`\\ s from a source + checkpoints.

    One model per checkpoint (the fold ensemble): build the registered decoder (deriving
    ``num_upsample_blocks`` from the geometry when the decoder takes it), pair it with a
    parameter-free :class:`SegmentationHead`, load the checkpoint's ``decoder.*`` weights,
    and put it in eval on the source's device. Mirrors the training-time construction so
    the inference forward matches by construction. Returns the list of eval-mode models.

    These checkpoints are trained on the **cached** dense path and replayed live at
    whole-slide scale, so they may carry a fitted feature adaptor (issue #286). Pass the
    run's ``normalization``/``projection`` to rebuild it: without them the strict load
    below rejects the adaptor's buffer keys, *and* the decoder would be built against the
    encoder's native dim while the checkpoint carries ``target_dim`` shapes. The adaptor is
    rebuilt **unfitted** — the checkpoint's buffers are the fitted state.
    """
    from soma.decoders.registry import build_decoder_for_grid
    from soma.tasks.segmentation import SegmentationHead
    from soma.training.feature_adaptor import (
        build_feature_adaptor,
        feature_adaptor_output_dim,
    )
    from soma.training.model import LiveSegmentationModel

    ckpt_paths = [Path(p) for p in ckpt_paths]
    if not ckpt_paths:
        raise ValueError("build_live_segmentation_models needs at least one checkpoint")

    def _make_adaptor():
        return build_feature_adaptor(
            normalization,
            projection,
            num_features=source.feature_dim,
            encoder_identity=encoder_identity,
        )

    def _make_decoder(adaptor):
        return build_decoder_for_grid(
            decoder_name,
            decoder_params,
            geometry=source.geometry,
            input_dim=feature_adaptor_output_dim(
                adaptor, num_features=source.feature_dim
            ),
            num_classes=num_classes,
        )

    models = []
    for ckpt in ckpt_paths:
        head = SegmentationHead(
            num_classes=num_classes,
            geometry=source.geometry,
            spacing_um=source.spacing_um,
            backend=source.backend,
            tolerance=source.tolerance,
        )
        adaptor = _make_adaptor()
        model = LiveSegmentationModel(
            encoder=source.encoder,
            decoder=_make_decoder(adaptor),
            task_head=head,
            device=source.device,
            precision=source.precision,
            geometry=source.geometry,
            window_size=source.window_size,
            overlap=source.overlap,
            feature_adaptor=adaptor,
        )
        state = torch.load(ckpt, weights_only=True, map_location=source.device)["model_state_dict"]
        model.load_state_dict(state)  # decoder.* (+ adaptor buffers); head is parameter-free
        models.append(model.to(source.device).eval())
    return models
