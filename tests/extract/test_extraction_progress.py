"""The fold behind ``soma.extract(on_progress=...)``: events in, snapshots out.

Each test hands :class:`soma.extract.progress.ProgressFold` one hand-written event
stream, the one an extraction mode produces (soma's own ``soma.*`` events beside the
slide2vec and hs2p kinds), and reads the snapshots the hook received. A fake clock
drives the throttle.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
import time
from pathlib import Path

import pytest
import slide2vec.progress as slide2vec_progress
from slide2vec.progress import ProgressEvent

from soma.extract import ExtractionProgress, progress
from soma.extract.progress import ProgressFold


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _fold(clock: Clock | None = None, encoder: str = "uni2"):
    received: list[ExtractionProgress] = []
    fold = ProgressFold(received.append, encoder=encoder, clock=clock or Clock())
    return fold, received


def _feed(fold: ProgressFold, *events: tuple, clock: Clock | None = None, step: float = 1.0) -> None:
    """Emit each ``(kind, payload)`` event, ``step`` seconds of ``clock`` apart."""
    for kind, *payload in events:
        if clock is not None:
            clock.now += step
        fold.emit(ProgressEvent(kind=kind, payload=payload[0] if payload else {}))


def _counters(received: list[ExtractionProgress]) -> list[tuple]:
    return [
        (s.phase, s.samples_cached, s.samples_done, s.tiles_done, s.tiles_total) for s in received
    ]


def _phases(received: list[ExtractionProgress]) -> list[str]:
    return [snapshot.phase for snapshot in received]


def test_a_full_cache_hit_checks_the_cache_then_finishes_with_every_sample_cached() -> None:
    fold, received = _fold()
    _feed(
        fold,
        ("soma.started", {"samples_total": 3}),
        ("soma.checked", {"pending": []}),
        ("soma.finished",),
    )
    assert received == [
        ExtractionProgress(
            encoder="uni2",
            phase="checking_cache",
            samples_total=3,
            samples_cached=None,
            samples_done=0,
            tiles_done=None,
            tiles_total=None,
        ),
        ExtractionProgress(
            encoder="uni2",
            phase="finished",
            samples_total=3,
            samples_cached=3,
            samples_done=0,
            tiles_done=None,
            tiles_total=None,
        ),
    ]


def test_a_tile_encoder_on_whole_slides_tiles_loads_embeds_and_finishes() -> None:
    clock = Clock()
    fold, received = _fold(clock)
    _feed(
        fold,
        ("soma.started", {"samples_total": 2}),
        ("tiling.started", {"slide_count": 2}),
        ("tiling.progress", {"total": 2, "completed": 1, "failed": 0, "pending": 1}),
        ("tiling.summary", {"total": 2, "completed": 2, "failed": 0, "pending": 0}),
        # The cache check may load the encoder to compare its identity: not the extraction's load.
        ("model.loading", {"model_name": "uni2"}),
        ("soma.checked", {"pending": ["a", "b"], "tiles": {"a": 10, "b": 6}}),
        ("embedding.started", {"slide_count": 2}),
        ("model.loading", {"model_name": "uni2"}),
        ("model.ready", {"model_name": "uni2", "device": "cuda:0"}),
        ("embedding.slide.started", {"sample_id": "a", "total_tiles": 10}),
        ("embedding.tile.progress", {"sample_id": "a", "processed": 4, "total": 10}),
        ("embedding.tile.progress", {"sample_id": "a", "processed": 10, "total": 10}),
        ("embedding.slide.finished", {"sample_id": "a", "num_tiles": 10}),
        ("embedding.slide.started", {"sample_id": "b", "total_tiles": 6}),
        ("embedding.tile.progress", {"sample_id": "b", "processed": 6, "total": 6}),
        ("embedding.slide.finished", {"sample_id": "b", "num_tiles": 6}),
        ("soma.finished",),
        clock=clock,
    )
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("tiling", None, 0, None, None),
        ("tiling", None, 1, None, None),
        ("tiling", None, 2, None, None),
        ("tiling", 0, 2, None, None),
        ("loading_model", 0, 0, None, None),
        ("embedding", 0, 0, 0, 16),
        ("embedding", 0, 0, 4, 16),
        ("embedding", 0, 0, 10, 16),
        ("embedding", 0, 1, 10, 16),
        ("embedding", 0, 1, 16, 16),
        ("embedding", 0, 2, 16, 16),
        ("finished", 0, 2, None, None),
    ]


def test_counter_changes_are_throttled_to_one_snapshot_per_half_second_but_phases_are_not() -> None:
    clock = Clock()
    fold, received = _fold(clock)
    _feed(
        fold,
        ("soma.started", {"samples_total": 4}),
        ("soma.checked", {"pending": ["a", "b", "c", "d"], "tiles": {"a": 10, "b": 10, "c": 10, "d": 10}}),
        ("model.loading", {"model_name": "uni2"}),
        clock=clock,
        step=0.1,
    )
    # Counter-only changes 0.2 s apart: one in three gets through.
    _feed(
        fold,
        *[
            ("embedding.tile.progress", {"sample_id": "a", "processed": processed, "total": 10})
            for processed in range(1, 11)
        ],
        clock=clock,
        step=0.2,
    )
    # A phase change and the end are delivered at once, however close to the last snapshot.
    _feed(fold, ("soma.finished",), clock=clock, step=0.01)
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("loading_model", 0, 0, None, None),
        ("embedding", 0, 0, 1, 40),
        ("embedding", 0, 0, 4, 40),
        ("embedding", 0, 0, 7, 40),
        ("embedding", 0, 0, 10, 40),
        ("finished", 0, 4, None, None),
    ]


def test_nothing_is_sent_when_nothing_changed() -> None:
    clock = Clock()
    fold, received = _fold(clock)
    _feed(
        fold,
        ("soma.started", {"samples_total": 1}),
        ("soma.checked", {"pending": ["a"], "tiles": {"a": 10}}),
        ("embedding.tile.progress", {"sample_id": "a", "processed": 5, "total": 10}),
        ("embedding.batch.timing", {"sample_id": "a", "batch_index": 3}),
        ("embedding.tile.progress", {"sample_id": "a", "processed": 5, "total": 10}),
        ("backend.selected", {"sample_id": "a", "backend": "cucim"}),
        clock=clock,
        step=5.0,
    )
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("checking_cache", 0, 0, None, None),
        ("embedding", 0, 0, 5, 10),
    ]


def test_a_hook_that_raises_is_warned_about_once_and_still_called(caplog) -> None:
    calls: list[str] = []

    def failing(snapshot: ExtractionProgress) -> None:
        calls.append(snapshot.phase)
        raise RuntimeError("display gone")

    clock = Clock()
    fold = ProgressFold(failing, encoder="uni2", clock=clock)
    with caplog.at_level(logging.WARNING):
        _feed(
            fold,
            ("soma.started", {"samples_total": 1}),
            ("soma.checked", {"pending": ["a"]}),
            ("model.loading", {"model_name": "uni2"}),
            ("soma.finished",),
            clock=clock,
        )
    assert calls == ["checking_cache", "checking_cache", "loading_model", "finished"]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "on_progress" in warnings[0].getMessage()
    assert warnings[0].exc_info is not None and "display gone" in str(warnings[0].exc_info[1])


def test_an_interrupt_from_the_hook_is_not_swallowed() -> None:
    def interrupting(snapshot: ExtractionProgress) -> None:
        raise KeyboardInterrupt

    fold = ProgressFold(interrupting, encoder="uni2", clock=Clock())
    with pytest.raises(KeyboardInterrupt):
        fold.emit(ProgressEvent(kind="soma.started", payload={"samples_total": 1}))


def test_a_slide_encoder_embeds_then_aggregates_with_tile_cached_slides_done_from_the_start() -> None:
    clock = Clock()
    fold, received = _fold(clock, encoder="prism")
    _feed(
        fold,
        ("soma.started", {"samples_total": 4}),
        # "d" is in the slide cache; "a" has its tile features cached, so it is not embedded again.
        ("soma.checked", {"pending": ["a", "b", "c"], "embedded": ["a"], "tiles": {"a": 10, "b": 6, "c": 4}}),
        ("model.loading", {"model_name": "virchow"}),
        ("embedding.slide.started", {"sample_id": "b", "total_tiles": 6}),
        ("embedding.slide.finished", {"sample_id": "b", "num_tiles": 6}),
        ("embedding.slide.started", {"sample_id": "c", "total_tiles": 4}),
        ("embedding.slide.finished", {"sample_id": "c", "num_tiles": 4}),
        ("soma.aggregation.started", {}),
        # The aggregator's own load is part of aggregating.
        ("model.loading", {"model_name": "prism"}),
        ("soma.sample.aggregated", {"sample_id": "a"}),
        ("soma.sample.aggregated", {"sample_id": "b"}),
        ("soma.sample.aggregated", {"sample_id": "c"}),
        ("soma.finished",),
        clock=clock,
    )
    assert {snapshot.encoder for snapshot in received} == {"prism"}
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("checking_cache", 1, 0, None, None),
        ("loading_model", 1, 0, None, None),
        ("embedding", 1, 1, 10, 20),
        ("embedding", 1, 2, 16, 20),
        ("embedding", 1, 3, 20, 20),
        ("aggregating", 1, 0, None, None),
        ("aggregating", 1, 1, None, None),
        ("aggregating", 1, 2, None, None),
        ("aggregating", 1, 3, None, None),
        ("finished", 1, 3, None, None),
    ]


def test_a_slide_encoder_with_every_tile_feature_cached_reports_the_embedding_pass_done() -> None:
    clock = Clock()
    fold, received = _fold(clock, encoder="prism")
    _feed(
        fold,
        ("soma.started", {"samples_total": 2}),
        ("soma.checked", {"pending": ["a", "b"], "embedded": ["a", "b"], "tiles": {"a": 10, "b": 6}}),
        ("soma.embedding.started", {}),
        ("soma.aggregation.started", {}),
        ("soma.sample.aggregated", {"sample_id": "a"}),
        ("soma.sample.aggregated", {"sample_id": "b"}),
        ("soma.finished",),
        clock=clock,
    )
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("checking_cache", 0, 0, None, None),
        ("embedding", 0, 2, 16, 16),
        ("aggregating", 0, 0, None, None),
        ("aggregating", 0, 1, None, None),
        ("aggregating", 0, 2, None, None),
        ("finished", 0, 2, None, None),
    ]


def test_aggregation_on_several_gpus_counts_soma_own_embedding_events_as_aggregating() -> None:
    clock = Clock()
    fold, received = _fold(clock, encoder="prism")
    _feed(
        fold,
        ("soma.started", {"samples_total": 3}),
        # Every tile feature is cached: no embedding pass, empty slide "c" has nothing to aggregate.
        ("soma.checked", {"pending": ["a", "b", "c"], "embedded": ["a", "b", "c"], "tiles": {"a": 5, "b": 5}}),
        ("soma.aggregation.started", {"aggregated": ["c"]}),
        ("embedding.started", {"slide_count": 2}),
        ("embedding.progress", {"slide_count": 2, "slides_completed": 1}),
        ("embedding.progress", {"slide_count": 2, "slides_completed": 2}),
        ("embedding.finished", {"slide_count": 2, "slides_completed": 2}),
        ("soma.finished",),
        clock=clock,
    )
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("checking_cache", 0, 0, None, None),
        ("aggregating", 0, 1, None, None),
        ("aggregating", 0, 2, None, None),
        ("aggregating", 0, 3, None, None),
        ("finished", 0, 3, None, None),
    ]


def test_pre_cropped_images_on_one_gpu_count_images_and_no_tiles() -> None:
    clock = Clock()
    fold, received = _fold(clock)
    _feed(
        fold,
        ("soma.started", {"samples_total": 4}),
        ("soma.checked", {"pending": ["a", "b", "c"]}),
        ("images.started", {"total": 3, "skipped": 0, "encoding": 3, "num_gpus": 1}),
        ("model.loading", {"model_name": "uni2"}),
        # On one GPU the model loads inside the call, then encodes at once.
        ("model.ready", {"model_name": "uni2", "device": "cuda:0"}),
        ("images.batch.finished", {"rank": 0, "images": 2}),
        ("images.batch.finished", {"rank": 0, "images": 1}),
        ("images.finished", {"total": 3}),
        ("soma.finished",),
        clock=clock,
    )
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("checking_cache", 1, 0, None, None),
        ("loading_model", 1, 0, None, None),
        ("embedding", 1, 0, None, None),
        ("embedding", 1, 2, None, None),
        ("embedding", 1, 3, None, None),
        ("finished", 1, 3, None, None),
    ]


def test_dense_images_on_several_gpus_load_until_the_first_worker_encodes() -> None:
    clock = Clock()
    fold, received = _fold(clock)
    _feed(
        fold,
        ("soma.started", {"samples_total": 5}),
        ("soma.checked", {"pending": ["a", "b", "c", "d", "e"]}),
        # slide2vec found one image already on disk: it counts as encoded.
        ("dense.images.started", {"total": 5, "skipped": 1, "encoding": 4, "num_gpus": 2}),
        # A worker's model load does not reach the calling process; a parent load is not the workers'.
        ("model.ready", {"model_name": "uni2", "device": "cpu"}),
        ("dense.images.batch.finished", {"rank": 1, "images": 2}),
        ("dense.images.batch.finished", {"rank": 0, "images": 2}),
        ("dense.images.finished", {"total": 5}),
        ("soma.finished",),
        clock=clock,
    )
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("checking_cache", 0, 0, None, None),
        ("loading_model", 0, 0, None, None),
        ("embedding", 0, 3, None, None),
        ("embedding", 0, 5, None, None),
        ("finished", 0, 5, None, None),
    ]


def test_rois_sampled_from_whole_slides_count_slides_and_regions_as_tiles() -> None:
    clock = Clock()
    fold, received = _fold(clock)
    _feed(
        fold,
        ("soma.started", {"samples_total": 3}),
        # ROI sampling: "s0" was sampled by an earlier run, "s1" and "s2" are sampled now.
        ("soma.tiling.started", {"tiled": ["s0"]}),
        ("soma.sample.tiled", {"sample_id": "s1"}),
        ("soma.sample.tiled", {"sample_id": "s2"}),
        # Every ROI of "s0" is cached; "s1" lacks 3 ROIs and "s2" lacks 2.
        ("soma.checked", {"pending": ["s1", "s2"], "tiles": {"s1": 3, "s2": 2}}),
        # Two calls (one per read spacing), each on two GPUs.
        ("dense.regions.started", {"total": 3, "skipped": 0, "encoding": 3, "num_gpus": 2}),
        ("dense.batch.finished", {"rank": 0, "regions": 2}),
        ("dense.batch.finished", {"rank": 1, "regions": 1}),
        ("dense.regions.finished", {"total": 3}),
        ("soma.sample.embedded", {"sample_ids": ["s1"]}),
        ("dense.regions.started", {"total": 2, "skipped": 1, "encoding": 1, "num_gpus": 2}),
        ("dense.batch.finished", {"rank": 0, "regions": 1}),
        ("dense.regions.finished", {"total": 2}),
        ("soma.sample.embedded", {"sample_ids": ["s2"]}),
        ("soma.finished",),
        clock=clock,
    )
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("tiling", None, 1, None, None),
        ("tiling", None, 2, None, None),
        ("tiling", None, 3, None, None),
        # Of the two pending slides, both are tiled.
        ("tiling", 1, 2, None, None),
        ("loading_model", 1, 0, None, None),
        ("embedding", 1, 0, 2, 5),
        ("embedding", 1, 0, 3, 5),
        ("embedding", 1, 1, 3, 5),
        ("embedding", 1, 1, 4, 5),
        ("embedding", 1, 1, 5, 5),
        ("embedding", 1, 2, 5, 5),
        ("finished", 1, 2, None, None),
    ]


def test_rois_on_one_gpu_encode_from_the_model_load_and_count_regions_per_call() -> None:
    clock = Clock()
    fold, received = _fold(clock)
    _feed(
        fold,
        ("soma.started", {"samples_total": 2}),
        ("soma.checked", {"pending": ["s0", "s1"], "tiles": {"s0": 4, "s1": 4}}),
        ("dense.regions.started", {"total": 8, "skipped": 0, "encoding": 8, "num_gpus": 1}),
        ("model.loading", {"model_name": "uni2"}),
        ("model.ready", {"model_name": "uni2", "device": "cuda:0"}),
        ("dense.regions.finished", {"total": 8}),
        ("soma.sample.embedded", {"sample_ids": ["s0", "s1"]}),
        ("soma.finished",),
        clock=clock,
    )
    assert _counters(received) == [
        ("checking_cache", None, 0, None, None),
        ("checking_cache", 0, 0, None, None),
        ("loading_model", 0, 0, None, None),
        ("embedding", 0, 0, 0, 8),
        ("embedding", 0, 0, 8, 8),
        ("embedding", 0, 2, 8, 8),
        ("finished", 0, 2, None, None),
    ]


def test_the_hook_is_never_called_in_a_forked_child(tmp_path: Path) -> None:
    calls = tmp_path / "calls.txt"

    def record(snapshot: ExtractionProgress) -> None:
        with calls.open("a", encoding="utf-8") as handle:
            handle.write(f"{snapshot.phase} {snapshot.samples_cached}\n")

    clock = Clock()
    fold = ProgressFold(record, encoder="uni2", clock=clock)
    _feed(fold, ("soma.started", {"samples_total": 2}), ("soma.checked", {"pending": []}), clock=clock)
    child = multiprocessing.get_context("fork").Process(
        target=fold.emit, args=(ProgressEvent(kind="soma.finished", payload={}),)
    )
    child.start()
    child.join()
    assert child.exitcode == 0
    _feed(fold, ("soma.finished",), clock=clock)
    assert calls.read_text(encoding="utf-8").splitlines() == [
        "checking_cache None",
        "checking_cache 2",
        "finished 2",
    ]


def test_hook_calls_from_several_threads_never_overlap() -> None:
    inside = 0
    overlaps = 0
    calls = 0

    def slow(snapshot: ExtractionProgress) -> None:
        nonlocal inside, overlaps, calls
        inside += 1
        overlaps += inside > 1
        time.sleep(0.0005)
        calls += 1
        inside -= 1

    class Ticking:
        """Each reading is a second later, so no snapshot is throttled."""

        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            self.now += 1.0
            return self.now

    fold = ProgressFold(slow, encoder="uni2", clock=Ticking())
    sample_ids = ["a", "b", "c", "d"]
    fold.emit(ProgressEvent(kind="soma.started", payload={"samples_total": 4}))
    fold.emit(
        ProgressEvent(kind="soma.checked", payload={"pending": sample_ids, "tiles": dict.fromkeys(sample_ids, 100)})
    )

    def encode(sample_id: str) -> None:
        for processed in range(1, 101):
            fold.emit(
                ProgressEvent(
                    kind="embedding.tile.progress",
                    payload={"sample_id": sample_id, "processed": processed, "total": 100},
                )
            )

    threads = [threading.Thread(target=encode, args=(sample_id,)) for sample_id in sample_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert calls == 2 + 400 and overlaps == 0


def test_a_caller_reporter_without_write_log_keeps_slide2vec_printing_the_log(tmp_path: Path, capsys) -> None:
    class EmitOnly:
        def emit(self, event) -> None:
            return None

        def close(self) -> None:
            return None

    with slide2vec_progress.activate_progress_reporter(EmitOnly()):
        with progress.reporting(lambda snapshot: None, encoder="uni2", samples_total=1, output_dir=tmp_path):
            slide2vec_progress.emit_progress_log("cache entry rebuilt")
    assert "cache entry rebuilt" in capsys.readouterr().out
