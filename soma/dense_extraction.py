"""DenseTileFeatureExtractor — encode tile images into dense ``(d, h, w)`` grids.

The dense analog of :class:`soma.tile_extraction.TileFeatureExtractor`: each tile
image is loaded, run through the encoder's **normalization-only** dense transform
(``get_dense_transform`` — NOT the pooled ``get_transform``, which crops GigaPath/
Lunit), padded up to the encoder's patch multiple, encoded via
``encode_tiles_dense``, and written as a ``(d, h, w)`` grid + sidecar through
:func:`soma.dense.write_dense_grid`.

The extraction loop (:func:`extract_dense_grids`) takes an already-loaded encoder
so it is unit-testable offline with random weights, independent of the GPU/weights
needed by :meth:`DenseTileFeatureExtractor.run`.

Dense-input mode is a *derived* window-as-knob (design §5): ``window_size=None`` ⇒
``whole`` (one padded forward); a smaller ``window_size`` (+ ``overlap``) slides the
encoder over patch-aligned windows and blends the token grids (see
:func:`soma.dense.sliding.encode_dense_sliding`). Single-GPU only (multi-GPU sharding
deferred).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from slide2vec.inference import load_model
from slide2vec.runtime.slide_encode import slide_encode_autocast_ctx
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset

from soma.cache import (
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_cache_root,
    resolve_dense_cache,
)
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig, PreprocessingConfig
from soma.dataset import Dataset, SampleRecord
from soma.dense import (
    DenseFeatureStore,
    DenseGridGeometry,
    compute_dense_geometry,
    dense_grid_metadata,
    normalize_hw,
    write_dense_grid,
)
from soma.dense.reader import read_image_at_spacing
from soma.dense.sliding import describe_dense_mode, encode_dense_sliding
from soma.slide2vec_adapter import build_execution_options

logger = logging.getLogger(__name__)

_PAD_MODES = {"reflect", "constant", "zero", "replicate"}


def _pad_image_to_encoded(
    tensor: torch.Tensor,
    geometry: DenseGridGeometry,
    *,
    pad_mode: str,
    image_pad_value: float | None,
) -> torch.Tensor:
    """Pad a ``(C, H, W)`` tile (bottom/right) up to ``geometry.encoded_size``."""
    pad_bottom, pad_right = geometry.pad
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    x = tensor.unsqueeze(0)  # F.pad's 2-D modes need a batch dim
    pad = (0, pad_right, 0, pad_bottom)  # (left, right, top, bottom)
    if pad_mode in ("constant", "zero"):
        x = F.pad(x, pad, mode="constant", value=float(image_pad_value or 0.0))
    else:
        x = F.pad(x, pad, mode=pad_mode)
    return x.squeeze(0)


class _DenseTileImageDataset(TorchDataset):
    """Load tile images, apply the dense transform, and pad to ``encoded_size``."""

    def __init__(
        self,
        records: list[SampleRecord],
        dense_transform: Callable,
        geometry: DenseGridGeometry,
        *,
        pad_mode: str,
        image_pad_value: float | None,
        spacing_um: float | None = None,
        backend: str = "auto",
        tolerance: float = 0.05,
    ) -> None:
        self._records = records
        self._transform = dense_transform
        self._geometry = geometry
        self._pad_mode = pad_mode
        self._image_pad_value = image_pad_value
        self._spacing_um = spacing_um
        self._backend = backend
        self._tolerance = tolerance

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        record = self._records[idx]
        # The reader routes by format: flat (PNG/JPEG, or no spacing) → PIL with
        # spacing ignored; pyramidal/spacing-bearing → hs2p (finest level <= requested
        # spacing, downscaled; byte-identical to a page-0 PIL read at an exact match).
        array = read_image_at_spacing(
            record.image_path,
            spacing_um=self._spacing_um,
            backend=self._backend,
            tolerance=self._tolerance,
        )
        tensor = self._transform(Image.fromarray(array))
        tensor = torch.as_tensor(tensor).as_subclass(torch.Tensor)
        if tensor.ndim != 3:
            raise ValueError(
                f"dense transform for tile '{record.sample_id}' produced a "
                f"{tensor.ndim}-D tensor; expected (C, H, W)."
            )
        if tuple(int(s) for s in tensor.shape[-2:]) != self._geometry.target_size:
            raise ValueError(
                f"tile '{record.sample_id}' is {tuple(int(s) for s in tensor.shape[-2:])} "
                f"after the dense transform, but the run's target_size is "
                f"{self._geometry.target_size}. v1 assumes a fixed tile size; resize the "
                "tiles or set target_size to match."
            )
        padded = _pad_image_to_encoded(
            tensor, self._geometry, pad_mode=self._pad_mode, image_pad_value=self._image_pad_value
        )
        return padded, record.sample_id


def extract_dense_grids(
    *,
    encoder,
    device: torch.device | str,
    dense_transform: Callable,
    geometry: DenseGridGeometry,
    records: Sequence[SampleRecord],
    out_dir: Path | str,
    spacing_um: float | None = None,
    backend: str = "auto",
    tolerance: float = 0.05,
    pad_mode: str = "reflect",
    image_pad_value: float | None = None,
    mask_pad_value: int | None = None,
    window_size: int | None,
    overlap: float,
    feature_kind: str = "patch_features",
    attention_blocks: tuple[int, ...] = (-1,),
    attention_include_registers: bool = False,
    batch_size: int = 1,
    precision: str = "fp32",
    num_workers: int = 0,
    prefetch_factor: int | None = None,
) -> int | None:
    """Encode ``records`` into dense grids written under ``out_dir``; return ``d``.

    Injectable core: takes a constructed dense-capable ``encoder`` (with
    ``encode_tiles_dense`` / ``encode_tiles_attention``), so it runs offline in tests
    with random weights. ``window_size``/``overlap`` are required (no silent default):
    ``window_size=None`` is the ``whole`` path (one padded forward), a smaller
    ``window_size`` slides the encoder over patch-aligned windows — the caller must
    choose so a sliding run is never mis-keyed/mis-extracted as ``whole``.

    ``feature_kind`` picks the per-window encode: ``patch_features`` →
    ``encode_tiles_dense`` (the ViT patch grid); ``cls_attention`` →
    ``encode_tiles_attention`` with ``attention_blocks`` / ``attention_include_registers``
    (per-head prefix-token self-attention, ``K`` channels). The sliding/stitching and
    write path are identical — an attention grid is just another ``(C, gh, gw)`` grid.
    """
    dense_input_mode = "whole" if window_size is None else "sliding_window"
    if pad_mode not in _PAD_MODES:
        raise ValueError(f"unsupported pad_mode {pad_mode!r}; expected one of {sorted(_PAD_MODES)}")

    if feature_kind == "cls_attention":
        attention_blocks = tuple(int(b) for b in attention_blocks)
        attention_include_registers = bool(attention_include_registers)

        def encode_fn(window: torch.Tensor) -> torch.Tensor:
            return encoder.encode_tiles_attention(
                window, blocks=attention_blocks, include_registers=attention_include_registers
            )
    elif feature_kind == "patch_features":
        encode_fn = encoder.encode_tiles_dense
    else:
        raise ValueError(
            f"unsupported feature_kind {feature_kind!r}; expected 'patch_features' or 'cls_attention'"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = _DenseTileImageDataset(
        list(records),
        dense_transform,
        geometry,
        pad_mode=pad_mode,
        image_pad_value=image_pad_value,
        spacing_um=spacing_um,
        backend=backend,
        tolerance=tolerance,
    )
    loader_kwargs: dict = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0 and prefetch_factor is not None:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    feature_dim: int | None = None
    with torch.inference_mode(), slide_encode_autocast_ctx(device, precision):
        for batch, sample_ids in loader:
            batch = batch.to(device, non_blocking=True)
            grids = (
                encode_dense_sliding(
                    encoder,
                    batch,
                    geometry=geometry,
                    window_size=window_size,
                    overlap=overlap,
                    encode_fn=encode_fn,
                )
                .detach()
                .float()
                .cpu()
            )
            if grids.ndim != 4:
                raise ValueError(
                    f"encode_tiles_dense returned a {grids.ndim}-D tensor; expected "
                    "(B, d, grid_h, grid_w)."
                )
            if feature_dim is None:
                feature_dim = int(grids.shape[1])
            for grid, sample_id in zip(grids, sample_ids):
                metadata = dense_grid_metadata(
                    geometry,
                    feature_dim=int(grid.shape[0]),
                    pad_mode=pad_mode,
                    image_pad_value=image_pad_value,
                    mask_pad_value=mask_pad_value,
                    dense_input_mode=dense_input_mode,
                    window_size=window_size,
                    overlap=overlap,
                    spacing_um=spacing_um,
                    feature_kind=feature_kind,
                    attention_blocks=attention_blocks,
                    attention_include_registers=attention_include_registers,
                )
                write_dense_grid(out_dir, str(sample_id), grid, metadata)
    return feature_dim


class DenseTileFeatureExtractor:
    """Encode tile images into dense ``(d, h, w)`` grids (``dataset_type="segmentation"``).

    Args:
        dataset: Dataset whose ``image_path`` fields point to tile images.
        encoder: Encoder configuration. For encoders that recommend
            ``dynamic_img_size=False`` (H-optimus), set
            ``allow_non_recommended_settings=True`` to opt into the variable input
            size dense extraction needs.
        target_size: The supervision tile/mask size (int or ``(h, w)``). Fixed per
            run (v1).
        pad_mode: How to pad up to the patch multiple — ``"reflect"`` (default, no
            out-of-distribution constant region), ``"constant"``/``"zero"``, or
            ``"replicate"``.
    """

    def __init__(
        self,
        dataset: Dataset,
        encoder: EncoderConfig,
        *,
        target_size: int | tuple[int, int],
        spacing_um: float,
        backend: str = "auto",
        tolerance: float = 0.05,
        pad_mode: str = "reflect",
        window_size: int | None = None,
        overlap: float = 0.0,
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig | None = None,
        preprocessing: PreprocessingConfig | None = None,
    ) -> None:
        if pad_mode not in _PAD_MODES:
            raise ValueError(f"unsupported pad_mode {pad_mode!r}; expected one of {sorted(_PAD_MODES)}")
        if window_size is not None and int(window_size) <= 0:
            raise ValueError(f"window_size must be a positive int or None, got {window_size!r}")
        if not (0.0 <= float(overlap) < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {overlap!r}")
        self._dataset = dataset
        self._encoder = encoder
        self._target_size = normalize_hw(target_size, name="target_size")
        self._spacing_um = float(spacing_um)
        self._backend = backend
        self._tolerance = float(tolerance)
        self._pad_mode = pad_mode
        self._window_size = None if window_size is None else int(window_size)
        self._overlap = float(overlap)
        self._dense_input_mode = "whole" if window_size is None else "sliding_window"
        self._execution = execution
        self._cache = cache or CacheConfig(enabled=False)
        # The run's preprocessing identity (spacing/tolerance/backend/tile size) is
        # folded into the dense cache key — so grids read at different spacings can
        # never alias to one cache entry (mirrors the pooled/bag tile path).
        self._preprocessing = preprocessing
        # feature_kind + attention knobs (design — attention-pixel segmentation §7).
        # patch_features → encode_tiles_dense; cls_attention → encode_tiles_attention.
        self._feature_kind = (
            (preprocessing.feature_kind or "patch_features")
            if preprocessing is not None
            else "patch_features"
        )
        if preprocessing is not None and self._feature_kind == "cls_attention":
            self._attention_blocks = tuple(preprocessing.attention.blocks)
            self._attention_include_registers = bool(preprocessing.attention.include_registers)
        else:
            self._attention_blocks = (-1,)
            self._attention_include_registers = False

    def _image_pad_value(self) -> float | None:
        # Only meaningful for constant/zero padding; None (N/A) for reflect/replicate.
        return 0.0 if self._pad_mode in ("constant", "zero") else None

    def run(self, feature_dir: str | Path) -> DenseFeatureStore:
        feature_dir = Path(feature_dir).resolve()
        feature_dir.mkdir(parents=True, exist_ok=True)

        # Construct a dense-capable encoder: dynamic_img_size for non-native sizes
        # (signature-gated in load_model; a no-op for encoders that hardcode it, and
        # the opt-in for H-optimus when allow_non_recommended_settings is set).
        loaded = load_model(
            name=self._encoder.name,
            output_variant=self._encoder.output_variant,
            allow_non_recommended_settings=self._encoder.allow_non_recommended_settings,
            dynamic_img_size=True,
        )
        encoder = loaded.model
        device = loaded.device
        patch_size = encoder.patch_size  # encoder-authoritative
        geometry = compute_dense_geometry(target_size=self._target_size, patch_size=patch_size)
        dense_transform = encoder.get_dense_transform()  # NOT loaded.transforms (pooled, crops!)

        # Announce the resolved dense-input mode before the cache check, so it always shows
        # (cache hit too — extract_dense_grids only runs on a miss) regardless of logging
        # config. print, not logger, so the user never has to opt into seeing it.
        print(f"Dense extraction mode: {describe_dense_mode(self._window_size, self._overlap)}")

        cache_resolution = None
        out_dir = feature_dir
        if self._cache.enabled:
            # NOTE deferred optimization: this loads the encoder before checking the
            # cache, so a full cache hit still pays the (one-time) model load. The
            # cache key needs patch_size, which is encoder-authoritative. Matching
            # TileFeatureExtractor's check-before-load is a follow-up.
            cache_root = resolve_cache_root(self._cache, feature_dir=feature_dir)
            cache_resolution = resolve_dense_cache(
                cache_root=cache_root,
                dataset=self._dataset,
                tile_encoder_name=self._encoder.name,
                target_size=self._target_size,
                patch_size=patch_size,
                pad_mode=self._pad_mode,
                execution=self._encoder,
                preprocessing=self._preprocessing,
                dense_input_mode=self._dense_input_mode,
                window_size=self._window_size,
                overlap=self._overlap,
                feature_kind=self._feature_kind,
                attention_blocks=self._attention_blocks,
                attention_include_registers=self._attention_include_registers,
                fingerprint_files=self._cache.fingerprint_files,
                validate_payloads=self._cache.validate_payloads,
            )
            if cache_resolution.complete:
                logger.info("Reusing cached dense grids from %s", cache_resolution.features_dir)
                return DenseFeatureStore(cache_resolution.cache_dir)
            out_dir = cache_resolution.features_dir

        execution = build_execution_options(
            self._encoder,
            execution=self._execution,
            encoder_name=self._encoder.name,
            output_dir=out_dir,
            num_gpus=1,  # single-GPU path (multi-GPU sharding deferred)
            save_tile_embeddings=True,
        )
        records = list(self._dataset.samples.values())
        logger.info(
            "Encoding %d tiles into dense grids with '%s' at target_size=%s, patch=%s -> grid %s",
            len(records),
            self._encoder.name,
            self._target_size,
            patch_size,
            geometry.grid_shape,
        )
        feature_dim = extract_dense_grids(
            encoder=encoder,
            device=device,
            dense_transform=dense_transform,
            geometry=geometry,
            records=records,
            out_dir=out_dir,
            spacing_um=self._spacing_um,
            backend=self._backend,
            tolerance=self._tolerance,
            pad_mode=self._pad_mode,
            image_pad_value=self._image_pad_value(),
            mask_pad_value=None,  # ignore_index is owned by the segmentation dataset slice
            window_size=self._window_size,
            overlap=self._overlap,
            feature_kind=self._feature_kind,
            attention_blocks=self._attention_blocks,
            attention_include_registers=self._attention_include_registers,
            batch_size=self._encoder.batch_size,
            # execution.precision honors an ExecutionConfig.precision override
            # (build_execution_options falls back to the encoder's precision when unset),
            # matching TileFeatureExtractor.
            precision=execution.precision,
            num_workers=execution.resolved_num_workers_per_gpu(),
            prefetch_factor=execution.prefetch_factor,
        )

        if cache_resolution is not None and feature_dim is not None:
            cache_resolution = record_feature_dim(
                cache_resolution,
                feature_dim,
                validate_payloads=self._cache.validate_payloads,
            )
            cache_resolution = record_sample_identity_signatures(
                cache_resolution,
                [record.sample_id for record in records],
                validate_payloads=self._cache.validate_payloads,
            )
        return DenseFeatureStore(out_dir)
