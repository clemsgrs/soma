"""Helpers for driving slide2vec from soma."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from hs2p import SlideSpec
from hs2p.preprocessing import validate_tiling_result_provenance
from slide2vec import ExecutionOptions
import slide2vec.api as slide2vec_api
import slide2vec.progress as slide2vec_progress
from slide2vec.utils.tiling_io import load_tiling_process_df, load_tiling_result_from_row

from soma.data._legacy import LegacySamples, LegacyRecord
from soma.encoders.validation import resolve_encoder_precision


@dataclass(frozen=True)
class LoadedTiling:
    slide: SlideSpec
    tiling_result: object
    requested_backend: str | None = None
    backend: str | None = None


def to_slide_spec(record: LegacyRecord) -> SlideSpec:
    return SlideSpec(
        sample_id=record.sample_id,
        image_path=record.image_path,
        mask_path=record.mask_path,
        spacing_at_level_0=record.spacing_at_level_0,
    )


def build_slide_specs(dataset: LegacySamples) -> list[SlideSpec]:
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


def execution_options(
    execution: ExecutionOptions,
    *,
    encoder_name: str,
    output_dir: Path,
    save_tile_embeddings: bool,
    output_dtype: str | None = None,
    on_image_mismatch: str = "raise",
    num_gpus: int | None = None,
) -> ExecutionOptions:
    """The run's ``ExecutionOptions`` for one slide2vec call.

    soma states what is its to decide: where slide2vec writes (the resolved cache
    directory) and in which format (``pt``, the only one soma's caches read), the
    on-disk dtype folded into the cache key (#164), whether tile
    embeddings are kept, what to do with an image re-pointed under its ``sample_id``, and
    the precision the cache key was derived at (``execution.precision``, else the
    encoder's recommendation), so the key and the computed features cannot disagree.
    An unset ``num_workers_per_gpu`` keeps soma's loader budget: the CPU budget split
    across the GPUs, capped at 16 (slide2vec would give image routes no workers).
    """
    values: dict[str, object] = {
        "output_dir": Path(output_dir),
        "output_format": "pt",
        "precision": resolve_encoder_precision(execution.precision, encoder_name=encoder_name),
        "save_tile_embeddings": bool(save_tile_embeddings),
        "output_dtype": output_dtype,
        "on_image_mismatch": on_image_mismatch,
    }
    if num_gpus is not None:
        values["num_gpus"] = int(num_gpus)
    if execution.num_workers_per_gpu is None:
        gpus = max(1, int(values.get("num_gpus", execution.num_gpus)))
        values["num_workers_per_gpu"] = min(16, max(1, slide2vec_api.cpu_worker_limit() // gpus))
    return replace(execution, **values)


def load_tilings(
    *,
    dataset: LegacySamples,
    tiling_dir: Path,
    requested_seg_downsample: int,
    annotation_sampling: bool = False,
) -> list[LoadedTiling]:
    # Binary tissue tiling reads a precomputed tissue mask's tissue label as 1, which hs2p
    # records on the result. Under annotation sampling (#110) tile selection gates on
    # per-class coverage over the masks' own pixel_mapping and hs2p records no tissue
    # value, so the provenance check expects ``None``; the image/mask path identity checks
    # stay intact.
    effective_tissue_value = None if annotation_sampling else 1
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
            # Supplied coordinates were checked against their row when they were staged,
            # and no tissue segmentation made them: the tissue provenance checks below
            # would test settings the artifact never used.
            if record.coordinates_path is None:
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
