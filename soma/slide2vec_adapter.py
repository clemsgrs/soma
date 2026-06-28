"""Helpers for driving slide2vec from soma."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path

import torch
from hs2p import SlideSpec
from hs2p.preprocessing import validate_tiling_result_provenance
import slide2vec.progress as slide2vec_progress

from slide2vec import (
    ExecutionOptions,
    PreprocessingConfig as Slide2VecPreprocessingConfig,
)
import slide2vec.api as slide2vec_api
import slide2vec.progress as slide2vec_progress
from slide2vec.utils.tiling_io import load_tiling_process_df, load_tiling_result_from_row

from soma.config import EncoderConfig, ExecutionConfig, PreprocessingConfig, PreviewConfig
from soma.dataset import Dataset, SampleRecord
from soma.encoders.validation import resolve_encoder_precision

# slide2vec deep-merges a partial ``masks`` block over these shipped default labels. A
# customized annotation vocabulary that omits one of them (notably ``tissue``) must null it
# out so the merge does not re-admit it (see ``_build_masks_block``).
_SLIDE2VEC_DEFAULT_MASK_LABELS: tuple[str, ...] = tuple(
    slide2vec_api.DEFAULT_MASKS["pixel_mapping"].keys()
)


@dataclass(frozen=True)
class LoadedTiling:
    slide: SlideSpec
    tiling_result: object
    requested_backend: str | None = None
    backend: str | None = None


def to_slide_spec(record: SampleRecord) -> SlideSpec:
    return SlideSpec(
        sample_id=record.sample_id,
        image_path=record.image_path,
        mask_path=record.mask_path,
        spacing_at_level_0=record.metadata.get("spacing_at_level_0"),
    )


def build_slide_specs(dataset: Dataset) -> list[SlideSpec]:
    return [to_slide_spec(record) for record in dataset.samples.values()]


def tiling_num_tiles(tiling_result: object) -> int:
    """Return the number of tiles recorded in a tiling result."""
    num_tiles = getattr(tiling_result, "num_tiles", None)
    if num_tiles is not None:
        return int(num_tiles)
    coordinates = getattr(tiling_result, "x", None)
    if coordinates is not None:
        return int(len(coordinates))
    return 0


def validate_tiling_result_segmentation(
    tiling_result: object,
    *,
    requested_seg_downsample: int,
    sample_id: str,
) -> None:
    """Validate the caller-requested segmentation downsample.

    hs2p records both requested and effective segmentation downsamples. Cache
    compatibility is keyed to the requested value; the effective value may
    differ for SAM2 or when a requested downsample resolves to a pyramid level.
    """
    actual = getattr(tiling_result, "requested_seg_downsample", None)
    if actual is None:
        raise ValueError(
            f"Tiling result for sample_id={sample_id} is missing requested_seg_downsample"
        )
    if int(actual) != int(requested_seg_downsample):
        raise ValueError(
            "Precomputed tiles requested_seg_downsample mismatch: "
            f"expected {int(requested_seg_downsample)!r}, found {int(actual)!r} "
            f"for sample_id={sample_id}"
        )


def ensure_supported_mask_value(
    dataset: Dataset,
    preprocessing: PreprocessingConfig,
) -> None:
    # A customized masks block governs tile selection by per-class coverage over its own
    # pixel_mapping (the single source of truth, #110): arbitrary mask values like
    # {background:1, tumor:2} are honored, so the legacy tissue-value==1 guard must not fire.
    if preprocessing.masks is not None:
        return
    if int(preprocessing.tissue_mask_tissue_value) == 1:
        return
    if not any(record.mask_path is not None for record in dataset.samples.values()):
        return
    raise ValueError(
        "slide2vec-backed tiling currently supports only tissue_mask_tissue_value=1 "
        "when mask_path is provided."
    )


def _build_masks_block(
    preprocessing: PreprocessingConfig,
) -> tuple[dict[str, object], bool]:
    """Translate soma preprocessing into slide2vec's ``masks`` block + ``independent_sampling``.

    Two regimes:

    * No annotation ``masks`` block (the default tissue path): emit only the tissue threshold
      (``min_coverage.tissue``). slide2vec deep-merges this over its shipped DEFAULT_MASKS, so
      the untouched ``{background:0, tissue:1}`` / ``per_annotation`` default stays byte-for-byte
      tissue tiling. ``independent_sampling`` keeps slide2vec's default (``True``).
    * A customized annotation ``masks`` block (the annotation-restricted merged bag, #110):
      forward the FULL annotation vocabulary — ``pixel_mapping``, per-class ``min_coverage``,
      and ``colors`` — plus an EXPLICIT ``output_mode``. The explicit ``output_mode`` is
      load-bearing: slide2vec's DEFAULT_MASKS ``output_mode`` is ``per_annotation`` and is
      deep-merged, so omitting it would silently route a customized bag run to per-annotation
      tiling (which collides on sample_id in load_tilings). ``independent_sampling`` is derived
      from ``sampling.strategy`` (``independent`` → ``True``; ``joint`` → ``False``).
    """
    masks = preprocessing.masks
    if masks is None:
        tissue_only = {
            "min_coverage": {"tissue": float(preprocessing.min_coverage.get("tissue") or 0.0)}
        }
        return tissue_only, True

    sampling = preprocessing.sampling
    output_mode = sampling.output_mode if sampling is not None else "merged"
    strategy = sampling.strategy if sampling is not None else "joint"
    pixel_mapping: dict[str, object] = dict(masks.pixel_mapping)
    min_coverage: dict[str, object] = dict(masks.min_coverage)
    # Always forward a colors map covering every label in the vocabulary. hs2p validates that
    # ``colors`` carries a key for each ``pixel_mapping`` label, and slide2vec's deep-merge
    # leaves the default ``{background, tissue}`` colors in place; without an explicit entry
    # for each user class (e.g. ``tumor``) that check raises. Unspecified classes get ``None``
    # (no overlay) — cosmetic only, excluded from cache identity.
    user_colors = masks.colors or {}
    colors: dict[str, object] = {
        label: (list(user_colors[label]) if user_colors.get(label) is not None else None)
        for label in pixel_mapping
    }
    # slide2vec deep-merges this block over its DEFAULT_MASKS ``{background:0, tissue:1}``,
    # so a default label the user's vocabulary omits (notably ``tissue``) would otherwise
    # survive the merge — colliding pixel values and re-admitting tissue-only tiles. Drop
    # each such default label via hs2p's null-to-drop idiom (set its pixel value to null;
    # hs2p strips it from sampling, and from the companion maps for coherence).
    for default_label in _SLIDE2VEC_DEFAULT_MASK_LABELS:
        if default_label not in pixel_mapping:
            pixel_mapping[default_label] = None
            min_coverage[default_label] = None
            colors[default_label] = None
    return (
        {
            "output_mode": output_mode,
            "pixel_mapping": pixel_mapping,
            "min_coverage": min_coverage,
            "colors": colors,
        },
        strategy == "independent",
    )


def build_preprocessing_config(
    preprocessing: PreprocessingConfig,
) -> Slide2VecPreprocessingConfig:
    if preprocessing.requested_tile_size_px is None:
        raise ValueError("requested_tile_size_px must be resolved before extraction")
    if preprocessing.requested_spacing_um is None:
        raise ValueError("requested_spacing_um must be resolved before extraction")
    if preprocessing.tissue_method is None:
        raise ValueError(
            "tissue_method is required unless the dataset provides precomputed masks"
        )
    masks_block, independent_sampling = _build_masks_block(preprocessing)
    payload: dict[str, object] = {
        "backend": preprocessing.backend,
        "requested_spacing_um": float(preprocessing.requested_spacing_um),
        "requested_tile_size_px": int(preprocessing.requested_tile_size_px),
        "tolerance": float(preprocessing.tolerance),
        "overlap": float(preprocessing.overlap),
        # soma expresses the tissue coverage threshold as ``preprocessing.min_coverage["tissue"]``
        # (a masks-shaped map mirroring hs2p's ``TilingConfig.min_coverage``). slide2vec has no
        # top-level ``tissue_threshold`` field — the threshold lives in the ``masks`` block as
        # ``min_coverage.tissue`` (the single source of truth). slide2vec deep-merges a partial
        # ``masks`` over its shipped DEFAULT_MASKS, so we only state the override. When a
        # customized annotation masks block is active (#110), ``_build_masks_block`` instead
        # forwards the full annotation vocabulary + an explicit output_mode.
        "masks": masks_block,
        "independent_sampling": independent_sampling,
        "on_the_fly": True,
        "adaptive_batching": False,
        "use_supertiles": True,
        "segmentation": {
            "method": preprocessing.tissue_method,
            "downsample": int(preprocessing.seg_downsample),
        },
        "filtering": {
            "ref_tile_size": int(
                preprocessing.ref_tile_size_px
                if preprocessing.ref_tile_size_px is not None
                else preprocessing.requested_tile_size_px
            ),
            "a_t": int(preprocessing.a_t),
        },
        "preview": asdict(preprocessing.preview),
    }
    if preprocessing.sam2_num_workers is not None:
        payload["segmentation"]["sam2_num_workers"] = int(preprocessing.sam2_num_workers)
    if preprocessing.sam2_device is not None:
        payload["segmentation"]["sam2_device"] = str(preprocessing.sam2_device)
    if preprocessing.requested_region_size_px is not None:
        payload["requested_region_size_px"] = int(preprocessing.requested_region_size_px)
    if preprocessing.region_tile_multiple is not None:
        payload["region_tile_multiple"] = int(preprocessing.region_tile_multiple)
    if preprocessing.read_tile_size_px is not None:
        payload["read_tile_size_px"] = int(preprocessing.read_tile_size_px)
    if preprocessing.read_region_size_px is not None:
        payload["read_region_size_px"] = int(preprocessing.read_region_size_px)
    allowed_fields = set(getattr(Slide2VecPreprocessingConfig, "__dataclass_fields__", {}))
    filtered = {key: value for key, value in payload.items() if key in allowed_fields}
    return Slide2VecPreprocessingConfig(**filtered)


def build_preview_config(preview: dict[str, object] | None = None) -> PreviewConfig:
    return PreviewConfig(**({} if preview is None else dict(preview)))


def build_execution_options(
    encoder: EncoderConfig,
    *,
    execution: ExecutionConfig | None = None,
    encoder_name: str | None = None,
    output_dir: Path,
    num_gpus: int | None,
    save_tile_embeddings: bool,
    output_dtype: str | None = None,
) -> ExecutionOptions:
    execution = execution or ExecutionConfig()
    num_gpus_value = num_gpus if num_gpus is not None else execution.num_gpus
    resolved_num_gpus = (
        int(num_gpus_value)
        if num_gpus_value is not None
        else (torch.cuda.device_count() if torch.cuda.is_available() else 1)
    )
    precision = execution.precision
    if precision is None:
        precision = resolve_encoder_precision(encoder, encoder_name=encoder_name)
    prefetch_factor = 4 if execution.prefetch_factor is None else int(execution.prefetch_factor)
    if execution.num_workers_per_gpu is None:
        num_workers_per_gpu = min(16, max(1, slide2vec_api.cpu_worker_limit() // max(1, resolved_num_gpus)))
    else:
        num_workers_per_gpu = int(execution.num_workers_per_gpu)
    return ExecutionOptions(
        output_dir=output_dir,
        output_format="pt",
        batch_size=int(encoder.batch_size),
        num_workers_per_gpu=num_workers_per_gpu,
        num_preprocessing_workers=execution.num_preprocessing_workers,
        num_gpus=num_gpus_value,
        precision=precision,
        prefetch_factor=prefetch_factor,
        save_tile_embeddings=save_tile_embeddings,
        save_slide_embeddings=False,
        save_latents=False,
        # On-disk feature precision (#164). soma resolves cache.dtype → 'fp16'/'fp32' once
        # and passes the concrete value, so slide2vec casts at its artifact writer to the
        # exact dtype folded into the cache key (key and storage can never drift). None
        # would let slide2vec follow precision; soma always passes a resolved value.
        output_dtype=output_dtype,
    )


def load_tilings(
    *,
    dataset: Dataset,
    tiling_dir: Path,
    requested_seg_downsample: int,
    tissue_mask_tissue_value: int,
    masks_active: bool = False,
) -> list[LoadedTiling]:
    # Under a customized annotation masks block (#110), tile selection gates on per-class
    # coverage over the masks' own pixel_mapping — arbitrary mask values are honored — so the
    # hs2p provenance validator's tissue-value==1 check must not fire. Passing None to its
    # ``tissue_mask_tissue_value`` skips that check while keeping the rest of the provenance
    # validation (image/mask path identity) intact.
    effective_tissue_value = None if masks_active else int(tissue_mask_tissue_value)
    process_list_path = tiling_dir / "process_list.csv"
    if not process_list_path.is_file():
        raise ValueError(
            f"Tiling directory '{tiling_dir}' is missing process_list.csv"
        )
    process_df = load_tiling_process_df(process_list_path)
    rows_by_sample_id = {
        str(row["sample_id"]): row
        for row in process_df.to_dict("records")
    }
    records = list(dataset.samples.values())
    progress = _CachedTilingLoadProgress(total=len(records))
    progress.start()
    loaded: list[LoadedTiling] = []
    try:
        for index, record in enumerate(records, start=1):
            row = rows_by_sample_id.get(record.sample_id)
            if row is None:
                raise ValueError(f"No tiling result found for sample_id={record.sample_id}")
            if row["tiling_status"] != "success":
                raise RuntimeError(
                    f"Tiling failed for {record.sample_id}: {row.get('error', '')}"
                )
            tiling_result = load_tiling_result_from_row(row)
            requested_backend = str(getattr(tiling_result, "requested_backend", "auto"))
            actual_backend = str(getattr(tiling_result, "backend", requested_backend))
            validate_tiling_result_provenance(
                tiling_result,
                sample_id=record.sample_id,
                image_path=record.image_path,
                mask_path=record.mask_path,
                tissue_mask_tissue_value=(
                    effective_tissue_value if record.mask_path is not None else None
                ),
            )
            validate_tiling_result_segmentation(
                tiling_result,
                requested_seg_downsample=requested_seg_downsample,
                sample_id=record.sample_id,
            )
            loaded.append(
                LoadedTiling(
                    slide=to_slide_spec(record),
                    tiling_result=tiling_result,
                    requested_backend=requested_backend,
                    backend=actual_backend,
                )
            )
            progress.update(index)
        return loaded
    finally:
        progress.finish(len(loaded))


class _CachedTilingLoadProgress:
    """Single progress indicator for loading cached tilings."""

    def __init__(self, *, total: int) -> None:
        self._total = max(0, int(total))
        self._reporter = slide2vec_progress.get_progress_reporter()
        self._progress = getattr(self._reporter, "progress", None)
        self._task_id: int | None = None
        self._rich = hasattr(self._reporter, "console") and hasattr(self._progress, "add_task")

    def start(self) -> None:
        if self._total <= 0:
            return
        if self._rich:
            ensure_started = getattr(self._reporter, "_ensure_progress_started", None)
            if callable(ensure_started):
                ensure_started()
            else:
                self._progress.start()
            self._task_id = self._progress.add_task("Loading cached tilings", total=self._total)
            return
        slide2vec_progress.emit_progress_log(f"… loading cached tilings: 0/{self._total}")

    def update(self, completed: int) -> None:
        if self._total <= 0:
            return
        completed = max(0, min(int(completed), self._total))
        if self._rich:
            if self._task_id is not None:
                self._progress.update(self._task_id, completed=completed)
            return
        if completed % 100 == 0 or completed == self._total:
            slide2vec_progress.emit_progress_log(
                f"… loading cached tilings: {completed}/{self._total}"
            )

    def finish(self, completed: int) -> None:
        if self._total <= 0:
            return
        completed = max(0, min(int(completed), self._total))
        if self._rich:
            if self._task_id is not None:
                self._progress.remove_task(self._task_id)
                self._task_id = None
            return
        slide2vec_progress.emit_progress_log(
            f"✓ loaded cached tilings: {completed}/{self._total}"
        )
