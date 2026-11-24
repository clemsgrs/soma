"""``soma.extract(on_progress=...)`` and ``Pipeline(on_extraction_progress=...)`` on
synthetic data with the weight-free ``fake`` encoders: a recording hook follows each
extraction mode, cold, then on the warm cache."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import slide2vec.progress
from slide2vec import DenseOptions, ExecutionOptions

from hs2p import FilterConfig

import soma
from soma.config import CacheOptions
from soma.data import AnnotationManifest, ImageManifest
from soma.extract import ExtractionProgress
from soma.testing import synthetic_cohort
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, learner_section, trainer_section
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset, make_segmentation_slides, make_tile_dataset
from tests.extraction_configs import annotation_masks, extraction, tissue_masks

TILE_PX = 32

PHASES = ["checking_cache", "tiling", "loading_model", "embedding", "aggregating", "finished"]


def _execution(**fields) -> ExecutionOptions:
    return ExecutionOptions(**{**CPU_EXECUTION, "batch_size": 4, **fields})


def _phases(snapshots: list[ExtractionProgress]) -> list[str]:
    """The phases in the order they were first reported."""
    return list(dict.fromkeys(snapshot.phase for snapshot in snapshots))


def _check_cycle(
    artifact, name: str, snapshots: list[ExtractionProgress], *, phases, total: int, cached: int
) -> None:
    """One extraction's snapshots: the phases in order, each moving forward, the end last,
    every counter within its total, and ``samples_total`` the manifest's ``total`` rows."""
    artifact.check_equal(f"{name}/phases", _phases(snapshots), phases)
    order = [PHASES.index(snapshot.phase) for snapshot in snapshots]
    artifact.check_equal(f"{name}/phases_move_forward", order == sorted(order), True)
    artifact.check_equal(f"{name}/samples_total_is_the_manifest", sorted({s.samples_total for s in snapshots}), [total])
    known = sorted({s.samples_cached for s in snapshots} - {None})
    artifact.check_equal(f"{name}/samples_cached_once_known", known, [cached])
    artifact.check_equal(f"{name}/first_samples_cached", snapshots[0].samples_cached, None)
    within = all(
        0 <= s.samples_done <= total - (s.samples_cached or 0)
        and (s.tiles_total is None) == (s.tiles_done is None)
        and (s.tiles_total is None or 0 <= s.tiles_done <= s.tiles_total)
        and (s.tiles_total is None or s.phase == "embedding")
        for s in snapshots
    )
    artifact.check_equal(f"{name}/counters_within_totals", within, True)
    last = snapshots[-1]
    artifact.check_equal(
        f"{name}/finished",
        (last.phase, last.samples_cached, last.samples_done, last.tiles_done, last.tiles_total),
        ("finished", cached, total - cached, None, None),
    )


def test_pre_cropped_images_report_progress_cold_then_from_the_warm_cache(tmp_path, encoder_name, new_artifact):
    """Each image is one sample: the cold call loads the encoder and counts the images it
    encodes (no tiles); the warm call checks the cache and finishes, every image cached."""
    artifact = new_artifact("extraction_progress_pre_cropped_images")
    dataset_csv, _splits = make_tile_dataset(tmp_path / "data", n=8)
    images = ImageManifest.from_csv(dataset_csv)
    calls = []
    for name in ("cold", "warm"):
        snapshots: list[ExtractionProgress] = []
        soma.extract(
            images,
            encoder=encoder_name,
            unit="tile",
            cache=CacheOptions(enabled=True, root_dir=tmp_path / "cache"),
            execution=_execution(),
            output_dir=tmp_path / name,
            on_progress=snapshots.append,
        )
        calls.append(snapshots)
    cold, warm = calls
    cold_phases = ["checking_cache", "loading_model", "embedding", "finished"]
    _check_cycle(artifact, "cold", cold, phases=cold_phases, total=len(images), cached=0)
    artifact.check_equal("cold/encoder", sorted({s.encoder for s in cold}), [encoder_name])
    artifact.check_equal("cold/tiles", sorted({str((s.tiles_done, s.tiles_total)) for s in cold}), ["(None, None)"])
    _check_cycle(artifact, "warm", warm, phases=["checking_cache", "finished"], total=len(images), cached=8)
    artifact.assert_passed()


def _slide_extraction():
    """The slide scenarios' tiling: 32 px tiles at 0.5 µm, Otsu tissue, half-tissue tiles."""
    return extraction(
        SPACING_UM,
        TILE_PX,
        method="otsu",
        seg_downsample=16,
        masks=tissue_masks(0.5),
        tiling_fields={"backend": "openslide"},
        filter=FilterConfig(ref_tile_size=TILE_PX, a_t=1),
    )


def _slides_source(images, encoder, root: Path, name: str, on_progress):
    return soma.extract(
        images,
        encoder=encoder,
        cache=CacheOptions(enabled=True, root_dir=root / "cache"),
        extraction=_slide_extraction(),
        execution=_execution(batch_size=16),
        output_dir=root / name,
        on_progress=on_progress,
    )


def _extract_slides(images, encoder, root: Path, name: str) -> list[ExtractionProgress]:
    snapshots: list[ExtractionProgress] = []
    _slides_source(images, encoder, root, name, snapshots.append)
    return snapshots


def _features(source) -> dict[str, np.ndarray]:
    return {sample_id: np.asarray(source.load(sample_id)) for sample_id in source.sample_ids}


def _tiles_seen(snapshots: list[ExtractionProgress]) -> list[int]:
    """The tile totals of the embedding snapshots."""
    return sorted({s.tiles_total for s in snapshots if s.phase == "embedding"})


def test_whole_slides_with_a_tile_encoder_report_tiles_and_a_partial_resume(tmp_path, encoder_name, new_artifact):
    """Three slides cold, then all six (three are cached), then all six warm."""
    artifact = new_artifact("extraction_progress_whole_slides")
    records, _splits = synthetic_cohort(tmp_path / "cohort", n=6)
    images = ImageManifest.from_csv(records)
    first = ImageManifest([images[sample_id] for sample_id in images.sample_ids[:3]])
    slide_phases = ["checking_cache", "tiling", "loading_model", "embedding", "finished"]

    part = _extract_slides(first, encoder_name, tmp_path, "part")
    _check_cycle(artifact, "part", part, phases=slide_phases, total=len(first), cached=0)
    resume = _extract_slides(images, encoder_name, tmp_path, "resume")
    _check_cycle(artifact, "resume", resume, phases=slide_phases, total=len(images), cached=3)
    warm = _extract_slides(images, encoder_name, tmp_path, "warm")
    _check_cycle(artifact, "warm", warm, phases=["checking_cache", "finished"], total=len(images), cached=6)

    # Every tile of the pending slides is counted, and the resume counts only the new slides'.
    part_tiles, resume_tiles = _tiles_seen(part), _tiles_seen(resume)
    artifact.check_equal("part/one_tile_total", len(part_tiles), 1)
    artifact.check_equal("resume/one_tile_total", len(resume_tiles), 1)
    artifact.check_at_least("part/tiles_total", part_tiles[0] if part_tiles else 0, 3)
    artifact.check_at_least("resume/tiles_total", resume_tiles[0] if resume_tiles else 0, 3)
    artifact.facts["tiles_total"] = {"part": part_tiles, "resume": resume_tiles}
    artifact.assert_passed()


def test_a_slide_encoder_embeds_then_aggregates_and_reuses_cached_tile_features(tmp_path, new_artifact):
    """``fake-slide`` runs two passes: embedding, then aggregating. A tile encoder's warm
    cache leaves nothing to embed, so the slide encoder's embedding pass starts done. The
    hook changes no feature."""
    artifact = new_artifact("extraction_progress_slide_encoder")
    records, _splits = synthetic_cohort(tmp_path / "cohort", n=4)
    images = ImageManifest.from_csv(records)
    every_phase = ["checking_cache", "tiling", "loading_model", "embedding", "aggregating", "finished"]

    cold: list[ExtractionProgress] = []
    hooked = _features(_slides_source(images, "fake-slide", tmp_path / "slide", "cold", cold.append))
    _check_cycle(artifact, "cold", cold, phases=every_phase, total=len(images), cached=0)
    plain = _features(_slides_source(images, "fake-slide", tmp_path / "plain", "cold", None))
    artifact.check_equal("same_samples", sorted(hooked), sorted(plain))
    artifact.check_equal("same_features", all(np.array_equal(hooked[k], plain[k]) for k in plain), True)
    warm = _extract_slides(images, "fake-slide", tmp_path / "slide", "warm")
    _check_cycle(artifact, "warm", warm, phases=["checking_cache", "finished"], total=len(images), cached=4)

    # The tile cache filled by ``fake`` is the one ``fake-slide`` aggregates.
    _extract_slides(images, "fake", tmp_path / "shared", "tiles")
    reused = _extract_slides(images, "fake-slide", tmp_path / "shared", "slides")
    reused_phases = ["checking_cache", "embedding", "aggregating", "finished"]
    _check_cycle(artifact, "reused", reused, phases=reused_phases, total=len(images), cached=0)
    embedding = [s for s in reused if s.phase == "embedding"]
    artifact.check_equal("reused/embedding_starts_done", embedding[0].samples_done if embedding else None, 4)
    artifact.assert_passed()


def test_dense_grids_on_pre_cropped_images_count_images_cold_then_from_the_warm_cache(
    tmp_path, encoder_name, new_artifact
):
    """A dense grid per image: the same phases as pooled images, no tiles."""
    artifact = new_artifact("extraction_progress_dense_images")
    dataset_csv, _splits = make_dense_dataset(tmp_path / "data", "segmentation", n=6)
    images = ImageManifest.from_csv(dataset_csv)
    calls = []
    for name in ("cold", "warm"):
        snapshots: list[ExtractionProgress] = []
        soma.extract(
            images,
            encoder=encoder_name,
            unit="tile",
            dense=DenseOptions(spacing_um=SPACING_UM, target_size=96),
            cache=CacheOptions(enabled=True, root_dir=tmp_path / "cache"),
            execution=_execution(),
            output_dir=tmp_path / name,
            on_progress=snapshots.append,
        )
        calls.append(snapshots)
    cold, warm = calls
    cold_phases = ["checking_cache", "loading_model", "embedding", "finished"]
    _check_cycle(artifact, "cold", cold, phases=cold_phases, total=len(images), cached=0)
    artifact.check_equal("cold/tiles", sorted({str((s.tiles_done, s.tiles_total)) for s in cold}), ["(None, None)"])
    _check_cycle(artifact, "warm", warm, phases=["checking_cache", "finished"], total=len(images), cached=6)
    artifact.assert_passed()


ROI_PX = 64


def _sample_rois(slides_csv: Path, root: Path, name: str, sample_ids=None):
    """Dense grids over 64 px ROIs sampled from the slides' annotation rasters."""
    images = ImageManifest.from_csv(slides_csv)
    if sample_ids is not None:
        images = ImageManifest([images[sample_id] for sample_id in sample_ids])
    snapshots: list[ExtractionProgress] = []
    store = soma.extract(
        images,
        annotations=AnnotationManifest.from_csv(slides_csv),
        encoder="fake",
        extraction=extraction(
            SPACING_UM,
            ROI_PX,
            masks=annotation_masks(
                {"background": 0, "light": 1, "dark": 2}, {"background": 0.0, "light": 0.0, "dark": 0.0}
            ),
            tiling_fields={"backend": "openslide", "mask_backend": "tifffile"},
        ),
        dense=DenseOptions(spacing_um=SPACING_UM, target_size=ROI_PX, backend="openslide"),
        cache=CacheOptions(enabled=True, root_dir=root / "cache"),
        execution=_execution(batch_size=8),
        output_dir=root / name,
        on_progress=snapshots.append,
    )
    return snapshots, store


def test_rois_sampled_from_whole_slides_count_slides_and_their_regions_as_tiles(tmp_path, new_artifact):
    """Two slides cold, then all four (two are cached), then all four warm. A sample is a
    slide, and each of its ROIs is a tile."""
    artifact = new_artifact("extraction_progress_sampled_rois")
    slides_csv, _splits = make_segmentation_slides(tmp_path / "slides", n=4)
    every_phase = ["checking_cache", "tiling", "loading_model", "embedding", "finished"]
    first = ["slide0", "slide1"]
    every_slide = len(ImageManifest.from_csv(slides_csv))

    part, _ = _sample_rois(slides_csv, tmp_path, "part", sample_ids=first)
    _check_cycle(artifact, "part", part, phases=every_phase, total=len(first), cached=0)
    resume, store = _sample_rois(slides_csv, tmp_path, "resume")
    _check_cycle(artifact, "resume", resume, phases=every_phase, total=every_slide, cached=2)
    warm, _ = _sample_rois(slides_csv, tmp_path, "warm")
    _check_cycle(artifact, "warm", warm, phases=["checking_cache", "finished"], total=every_slide, cached=4)

    new_rois = sum(1 for slide in store.rois.children.values() if slide in {"slide2", "slide3"})
    artifact.check_equal("resume/tiles_total", _tiles_seen(resume), [new_rois])
    artifact.assert_passed()


def _segmentation_config(tmp_path: Path, encoder: dict) -> dict:
    """One epoch of segmentation over dense grids of 96 px images."""
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "segmentation")
    return {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "kind": "cached",
            "encoder": encoder,
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": 96,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
            },
            "execution": {**CPU_EXECUTION, "batch_size": 8},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": learner_section(
            "dense",
            {
                "decoder": {"name": "lightweight_conv"},
                "head": {"name": "segmentation", "params": {"num_classes": 3}},
            },
            learning_rate=3e-3,
            normalization={"method": "none"},
        ),
        "trainer": trainer_section(1),
        "evaluation": {"metrics": ["mean_dice"]},
    }


def _cycles(snapshots: list[ExtractionProgress]) -> list[list[ExtractionProgress]]:
    """The snapshots cut after each ``finished``: one list per ``soma.extract`` call."""
    cycles, current = [], []
    for snapshot in snapshots:
        current.append(snapshot)
        if snapshot.phase == "finished":
            cycles.append(current)
            current = []
    return cycles + ([current] if current else [])


def test_a_pipeline_reports_every_extraction_it_runs_to_its_hook(tmp_path, encoder_name, new_artifact):
    """``Pipeline(on_extraction_progress=...)`` follows the run's extraction: one cycle for
    one encoder, one per member for a composite, each named by its encoder."""
    from soma.config import ExperimentConfig
    from soma.pipeline import Pipeline

    artifact = new_artifact("extraction_progress_pipeline")
    cold_phases = ["checking_cache", "loading_model", "embedding", "finished"]
    single: list[ExtractionProgress] = []
    config = _segmentation_config(tmp_path / "single", {"name": encoder_name})
    rows = len(ImageManifest.from_csv(config["cohort"]["records"]))
    Pipeline(ExperimentConfig.from_dict(config), on_extraction_progress=single.append).run()
    (cycle,) = _cycles(single)
    _check_cycle(artifact, "single", cycle, phases=cold_phases, total=rows, cached=0)

    composite: list[ExtractionProgress] = []
    members = [{"name": encoder_name}, {"name": encoder_name, "dense": {"overlap": 0.25}}]
    config = _segmentation_config(tmp_path / "composite", {"name": "composite", "params": {"members": members}})
    rows = len(ImageManifest.from_csv(config["cohort"]["records"]))
    Pipeline(ExperimentConfig.from_dict(config), on_extraction_progress=composite.append).run()
    cycles = _cycles(composite)
    artifact.check_equal("composite/cycles", len(cycles), 2)
    for index, member_cycle in enumerate(cycles):
        _check_cycle(artifact, f"composite/member_{index}", member_cycle, phases=cold_phases, total=rows, cached=0)
        encoders = sorted({s.encoder for s in member_cycle})
        artifact.check_equal(f"composite/member_{index}/encoder", encoders, [encoder_name])
    artifact.assert_passed()


def test_a_failed_extraction_raises_and_sends_no_finished_snapshot(tmp_path, encoder_name, new_artifact):
    artifact = new_artifact("extraction_progress_failed_extraction")
    dataset_csv, _splits = make_tile_dataset(tmp_path / "data", n=4)
    images = ImageManifest.from_csv(dataset_csv)
    Path(images[images.sample_ids[-1]].image_path).unlink()
    snapshots: list[ExtractionProgress] = []
    raised = False
    try:
        soma.extract(
            images,
            encoder=encoder_name,
            unit="tile",
            cache=CacheOptions(enabled=True, root_dir=tmp_path / "cache"),
            execution=_execution(),
            output_dir=tmp_path / "run",
            on_progress=snapshots.append,
        )
    except Exception:  # noqa: BLE001 - whatever the reader raises for a missing image
        raised = True
    artifact.check_equal("raised", raised, True)
    artifact.check_equal("snapshots_before_the_error", len(snapshots) > 0, True)
    artifact.check_equal("no_finished", [s.phase for s in snapshots if s.phase == "finished"], [])
    artifact.assert_passed()


def test_a_hook_that_raises_is_logged_once_and_the_extraction_is_unchanged(
    tmp_path, encoder_name, new_artifact, caplog
):
    artifact = new_artifact("extraction_progress_failing_hook")
    dataset_csv, _splits = make_tile_dataset(tmp_path / "data", n=8)
    images = ImageManifest.from_csv(dataset_csv)
    calls: list[ExtractionProgress] = []

    def failing(snapshot: ExtractionProgress) -> None:
        calls.append(snapshot)
        raise RuntimeError("display is gone")

    def run(name, on_progress):
        return soma.extract(
            images,
            encoder=encoder_name,
            unit="tile",
            cache=CacheOptions(enabled=False),
            execution=_execution(),
            output_dir=tmp_path / name,
            on_progress=on_progress,
        )

    with caplog.at_level(logging.WARNING, logger="soma"):
        failed = run("failing", failing)
    plain = run("plain", None)
    warnings = [record for record in caplog.records if "on_progress" in record.getMessage()]
    artifact.check_equal("warnings", len(warnings), 1)
    artifact.check_equal("warning_has_traceback", warnings[0].exc_info is not None if warnings else False, True)
    artifact.check_equal("hook_called_after_failing", len(calls) > 1, True)
    artifact.check_equal("last_call", calls[-1].phase if calls else None, "finished")
    a, b = _features(failed), _features(plain)
    artifact.check_equal("same_samples", sorted(a), sorted(b))
    artifact.check_equal("same_features", all(np.array_equal(a[k], b[k]) for k in a), True)
    artifact.assert_passed()


class _RecordingReporter:
    """A slide2vec reporter a caller activates around ``soma.extract``."""

    def __init__(self) -> None:
        self.kinds: list[str] = []
        self.closed = 0

    def emit(self, event) -> None:
        self.kinds.append(event.kind)

    def write_log(self, message: str, *, stream=None) -> None:
        pass

    def close(self) -> None:
        self.closed += 1


def test_a_reporter_the_caller_activated_receives_the_same_events_with_the_hook(
    tmp_path, encoder_name, new_artifact
):
    """The hook is additive: slide2vec's events still reach the caller's reporter, and
    soma's own progress events never do."""
    artifact = new_artifact("extraction_progress_caller_reporter")
    records, _splits = synthetic_cohort(tmp_path / "cohort", n=3)
    images = ImageManifest.from_csv(records)
    seen = {}
    for name, hook in (("plain", None), ("hooked", [])):
        reporter = _RecordingReporter()
        with slide2vec.progress.activate_progress_reporter(reporter):
            soma.extract(
                images,
                encoder=encoder_name,
                cache=CacheOptions(enabled=True, root_dir=tmp_path / name / "cache"),
                extraction=_slide_extraction(),
                execution=_execution(batch_size=16),
                output_dir=tmp_path / name / "run",
                on_progress=None if hook is None else hook.append,
            )
        # activate_progress_reporter closes it once on exit; soma closes it never.
        seen[name] = (reporter, hook)
    (plain, _), (hooked, snapshots) = seen["plain"], seen["hooked"]
    artifact.check_equal("same_event_kinds", sorted(set(hooked.kinds)), sorted(set(plain.kinds)))
    slides_finished = [r.kinds.count("embedding.slide.finished") for r in (plain, hooked)]
    artifact.check_equal("same_slide_events", slides_finished[1], slides_finished[0])
    artifact.check_equal("no_soma_events", [kind for kind in hooked.kinds if kind.startswith("soma.")], [])
    artifact.check_equal("closed_once", (plain.closed, hooked.closed), (1, 1))
    artifact.check_equal("hook_finished", snapshots[-1].phase if snapshots else None, "finished")
    artifact.assert_passed()
