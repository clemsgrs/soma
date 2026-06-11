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

Scope (v1): ``dense_input_mode="whole"`` (single forward, padded), single-GPU.
Multi-GPU sharding and the ``sliding_window`` mode are deferred.
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
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig
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
    dense_input_mode: str = "whole",
    batch_size: int = 1,
    precision: str = "fp32",
    num_workers: int = 0,
    prefetch_factor: int | None = None,
) -> int | None:
    """Encode ``records`` into dense grids written under ``out_dir``; return ``d``.

    Injectable core: takes a constructed dense-capable ``encoder`` (with
    ``encode_tiles_dense``), so it runs offline in tests with random weights.
    """
    if dense_input_mode != "whole":
        raise NotImplementedError(
            f"dense_input_mode={dense_input_mode!r} is not implemented; only 'whole' "
            "(single padded forward) is built. 'sliding_window' (overlapping native-size "
            "windows + blended overlaps) is the planned fidelity-preserving mode."
        )
    if pad_mode not in _PAD_MODES:
        raise ValueError(f"unsupported pad_mode {pad_mode!r}; expected one of {sorted(_PAD_MODES)}")

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
            grids = encoder.encode_tiles_dense(batch).detach().float().cpu()
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
        dense_input_mode: str = "whole",
        execution: ExecutionConfig = ExecutionConfig(),
        cache: CacheConfig | None = None,
    ) -> None:
        if dense_input_mode != "whole":
            raise NotImplementedError(
                f"dense_input_mode={dense_input_mode!r} is not implemented; only 'whole' is built."
            )
        if pad_mode not in _PAD_MODES:
            raise ValueError(f"unsupported pad_mode {pad_mode!r}; expected one of {sorted(_PAD_MODES)}")
        self._dataset = dataset
        self._encoder = encoder
        self._target_size = normalize_hw(target_size, name="target_size")
        self._spacing_um = float(spacing_um)
        self._backend = backend
        self._tolerance = float(tolerance)
        self._pad_mode = pad_mode
        self._dense_input_mode = dense_input_mode
        self._execution = execution
        self._cache = cache or CacheConfig(enabled=False)

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
                dense_input_mode=self._dense_input_mode,
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
            dense_input_mode=self._dense_input_mode,
            batch_size=self._encoder.batch_size,
            # execution.precision honors an ExecutionConfig.precision override
            # (build_execution_options falls back to the encoder's precision when unset),
            # matching TileFeatureExtractor.
            precision=execution.precision,
            num_workers=execution.resolved_num_workers_per_gpu(),
            prefetch_factor=execution.prefetch_factor,
        )

        if cache_resolution is not None and feature_dim is not None:
            record_feature_dim(cache_resolution, feature_dim)
            record_sample_identity_signatures(
                cache_resolution, [record.sample_id for record in records]
            )
        return DenseFeatureStore(out_dir)
