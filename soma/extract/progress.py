"""Extraction progress for a person: the snapshot ``soma.extract(on_progress=...)`` hands
its hook, and the fold that builds it from the extraction's events.

The fold reads two streams. slide2vec and hs2p report what they do (``tiling.progress``,
``embedding.tile.progress``, ``images.batch.finished``, ...); soma reports what they cannot
know (``soma.*`` kinds: the cohort, the cache check, the aggregation pass, the end).
Every event updates one state, and the hook receives the whole state as an
:class:`ExtractionProgress`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

import slide2vec.progress as slide2vec_progress

__all__ = ["ExtractionProgress", "ProgressFold", "report", "reporting", "reporting_active"]

logger = logging.getLogger(__name__)

Phase = Literal["checking_cache", "tiling", "loading_model", "embedding", "aggregating", "finished"]
_ORDER = ("checking_cache", "tiling", "loading_model", "embedding", "aggregating", "finished")
#: The shortest time between two snapshots of one phase, in seconds. The first snapshot of
#: a phase is never held back.
MIN_INTERVAL_S = 0.5


@dataclass(frozen=True)
class ExtractionProgress:
    """The state of one extraction, as ``soma.extract``'s ``on_progress`` hook receives it.

    Attributes:
        encoder: The encoder this extraction runs. A composite run extracts once per
            member, so each member gets its own cycle of snapshots.
        phase: What the extraction is doing. Phases only move forward, and
            ``finished`` is always the last snapshot of a successful extraction.
        samples_total: The rows of the image manifest.
        samples_cached: The rows whose features were in the cache at the start. ``None``
            until the cache has been checked.
        samples_done: Of the rows not cached, how many have completed the current
            phase. It starts again from 0 in each phase.
        tiles_done: Tiles encoded so far, over every pending row and every GPU. Only set
            during ``embedding``, and only where the extraction counts tiles.
        tiles_total: Tiles to encode, over every pending row. Set with ``tiles_done``.
    """

    encoder: str
    phase: Phase
    samples_total: int
    samples_cached: int | None
    samples_done: int
    tiles_done: int | None
    tiles_total: int | None


#: slide2vec kinds that mean a slide's tiles are being encoded.
_SLIDE_WORK = frozenset({"embedding.slide.started", "embedding.tile.progress", "embedding.slide.finished"})
#: Tiling kinds that carry how many slides are tiled (hs2p's, as slide2vec forwards them).
_TILING_COUNTS = frozenset({"tiling.progress", "tiling.finished", "tiling.summary"})
#: The start of a slide2vec call that encodes images or regions, and its end.
_STAGE_STARTED = frozenset({"images.started", "dense.images.started", "dense.regions.started"})
_STAGE_FINISHED = frozenset({"images.finished", "dense.images.finished", "dense.regions.finished"})
#: A batch of images encoded, by one GPU.
_IMAGE_BATCHES = frozenset({"images.batch.finished", "dense.images.batch.finished"})


class ProgressFold:
    """Folds extraction events into :class:`ExtractionProgress` snapshots for ``on_progress``."""

    def __init__(
        self,
        on_progress: Callable[[ExtractionProgress], Any],
        *,
        encoder: str,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_progress = on_progress
        self._encoder = str(encoder)
        self._clock = clock
        # The hook runs in the calling process only, one call at a time.
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._phase: Phase = "checking_cache"
        self._samples_total = 0
        #: The rows not cached, once the cache is checked.
        self._pending: set[str] | None = None
        self._last: ExtractionProgress | None = None
        self._last_at = 0.0
        self._hook_failed = False
        #: Pending rows with nothing left to encode when embedding starts.
        self._embedded: set[str] = set()
        #: Tiles to encode per pending row, where the extraction counts tiles.
        self._tiles: dict[str, int] | None = None
        # The counters of the current phase.
        self._tiled = 0
        self._done: set[str] = set()
        self._count = 0
        self._tiles_finished = 0
        self._tiles_running: dict[str, int] = {}
        #: Regions encoded by the slide2vec call under way.
        self._regions_running = 0
        #: Images slide2vec found already encoded on disk.
        self._images_skipped = 0
        #: Whether the slide2vec call under way encodes in this process, after its model load.
        self._in_process = False

    def emit(self, event: Any) -> None:
        """Fold one event (anything with a ``kind`` and a ``payload``).

        An event from another process (a worker forked with the reporter active) is
        dropped: the hook is only ever called in the process that created the fold.
        """
        if os.getpid() != self._pid:
            return
        with self._lock:
            self._fold(str(event.kind), dict(event.payload or {}))
            self._deliver()

    # -- folding ---------------------------------------------------------------------------

    def _fold(self, kind: str, payload: dict[str, Any]) -> None:
        if kind == "soma.started":
            self._samples_total = int(payload["samples_total"])
        elif kind == "soma.checked":
            self._pending = {str(sample_id) for sample_id in payload["pending"]}
            self._embedded = {str(sample_id) for sample_id in payload.get("embedded", ())} & self._pending
            tiles = payload.get("tiles")
            self._tiles = None if tiles is None else {str(k): int(v) for k, v in tiles.items()}
            self._tiles_finished = sum((self._tiles or {}).get(sample_id, 0) for sample_id in self._embedded)
        elif kind == "soma.finished":
            self._enter("finished")
        elif kind == "tiling.started":
            self._enter("tiling")
        elif kind == "soma.tiling.started":
            self._enter("tiling")
            self._done |= {str(sample_id) for sample_id in payload.get("tiled", ())}
        elif kind == "soma.sample.tiled":
            if self._phase == "tiling":
                self._done.add(str(payload["sample_id"]))
        elif kind in _TILING_COUNTS:
            if self._phase == "tiling":
                self._tiled = int(payload.get("completed", 0)) + int(payload.get("failed", 0))
        elif kind == "model.loading":
            # A load before the cache check compares the encoder's identity; one after
            # embedding loads the aggregator. Neither is the extraction's model load.
            if self._pending is not None:
                self._enter("loading_model")
        elif kind in _SLIDE_WORK:
            self._fold_slide_work(kind, payload)
        elif kind in _STAGE_STARTED:
            if self._pending is not None:
                self._enter("loading_model")
            self._in_process = int(payload.get("num_gpus", 1)) <= 1
            skipped = int(payload.get("skipped", 0))
            if kind == "dense.regions.started":
                self._regions_running = skipped
            else:
                self._images_skipped += skipped
        elif kind == "model.ready":
            # On one GPU slide2vec loads the model inside the call and encodes right away.
            if self._in_process and self._phase == "loading_model":
                self._enter("embedding")
        elif kind in _IMAGE_BATCHES:
            self._enter("embedding")
            if self._phase == "embedding":
                self._count += int(payload.get("images", 0))
        elif kind == "dense.batch.finished":
            self._enter("embedding")
            self._regions_running += int(payload.get("regions", 0))
        elif kind in _STAGE_FINISHED:
            self._in_process = False
            if kind == "dense.regions.finished":
                self._enter("embedding")
                self._tiles_finished += int(payload.get("total", 0))
                self._regions_running = 0
        elif kind == "soma.embedding.started":
            # Nothing to encode: every pending row's tile features are cached.
            self._enter("embedding")
        elif kind == "soma.sample.embedded":
            self._enter("embedding")
            if self._phase == "embedding":
                self._done |= {str(sample_id) for sample_id in payload["sample_ids"]}
        elif kind == "soma.aggregation.started":
            self._enter("aggregating")
            self._done |= {str(sample_id) for sample_id in payload.get("aggregated", ())}
        elif kind == "soma.sample.aggregated":
            if self._phase == "aggregating":
                self._done.add(str(payload["sample_id"]))
        elif kind in ("embedding.progress", "embedding.finished"):
            # soma's own events for slide aggregation on several GPUs.
            if self._phase == "aggregating":
                self._count = int(payload.get("slides_completed", 0))

    def _fold_slide_work(self, kind: str, payload: dict[str, Any]) -> None:
        sample_id = str(payload.get("sample_id"))
        if self._pending is None or sample_id not in self._pending:
            return
        self._enter("embedding")
        if self._phase != "embedding":
            return
        if kind == "embedding.tile.progress":
            self._tiles_running[sample_id] = int(payload.get("processed", 0))
        elif kind == "embedding.slide.finished":
            self._tiles_running.pop(sample_id, None)
            self._tiles_finished += int(payload.get("num_tiles", 0))
            self._done.add(sample_id)

    def _enter(self, phase: Phase) -> None:
        """Move forward to ``phase`` (never back), starting its counters again."""
        if _ORDER.index(phase) <= _ORDER.index(self._phase):
            return
        self._phase = phase
        self._tiled = self._count = 0
        self._done = set()
        if phase == "embedding":
            self._done = set(self._embedded)
            self._count = self._images_skipped

    # -- snapshots -------------------------------------------------------------------------

    def _snapshot(self) -> ExtractionProgress:
        pending = None if self._pending is None else len(self._pending)
        bound = self._samples_total if pending is None else pending
        if self._phase == "finished":
            done = bound
        elif self._phase == "tiling":
            tiled = self._done if self._pending is None else self._done & self._pending
            done = self._tiled + len(tiled)
        elif self._phase in ("embedding", "aggregating"):
            done = len(self._done & (self._pending or set())) + self._count
        else:
            done = 0
        tiles_done = tiles_total = None
        if self._phase == "embedding" and self._tiles is not None:
            tiles_total = sum(self._tiles.values())
            encoded = self._tiles_finished + sum(self._tiles_running.values()) + self._regions_running
            tiles_done = min(encoded, tiles_total)
        return ExtractionProgress(
            encoder=self._encoder,
            phase=self._phase,
            samples_total=self._samples_total,
            samples_cached=None if pending is None else self._samples_total - pending,
            samples_done=max(0, min(done, bound)),
            tiles_done=tiles_done,
            tiles_total=tiles_total,
        )

    def _deliver(self) -> None:
        snapshot = self._snapshot()
        if snapshot == self._last:
            return
        now = self._clock()
        if self._last is not None and snapshot.phase == self._last.phase and now - self._last_at < MIN_INTERVAL_S:
            return
        self._last, self._last_at = snapshot, now
        try:
            self._on_progress(snapshot)
        except Exception:
            # A display must not fail the extraction: say so once, keep calling it.
            if not self._hook_failed:
                self._hook_failed = True
                logger.warning(
                    "The on_progress hook raised; the extraction goes on and keeps calling it.",
                    exc_info=True,
                )


_ACTIVE_FOLD: ContextVar[ProgressFold | None] = ContextVar("soma_extraction_progress", default=None)


def reporting_active() -> bool:
    """Whether an ``on_progress`` hook follows the extraction under way."""
    return _ACTIVE_FOLD.get() is not None


def report(kind: str, **payload: Any) -> None:
    """Hand one of soma's own events to the active fold; nothing without a hook.

    soma's events go to the fold only, never to slide2vec's reporters, so a call without
    a hook behaves as if they did not exist.
    """
    fold = _ACTIVE_FOLD.get()
    if fold is not None:
        fold.emit(slide2vec_progress.ProgressEvent(kind=kind, payload=payload))


@contextlib.contextmanager
def reporting(
    on_progress: Callable[[ExtractionProgress], Any] | None,
    *,
    encoder: str,
    samples_total: int,
    output_dir: str | Path,
) -> Iterator[None]:
    """Follow one extraction with ``on_progress``: the first snapshot on entry, ``finished``
    on a clean exit, none after an error. Without a hook, nothing is installed."""
    if on_progress is None:
        yield
        return
    fold = ProgressFold(on_progress, encoder=encoder)
    token = _ACTIVE_FOLD.set(fold)
    try:
        with slide2vec_progress.activate_progress_reporter(_folding_reporter(fold, output_dir)):
            report("soma.started", samples_total=int(samples_total))
            yield
            report("soma.finished")
    finally:
        _ACTIVE_FOLD.reset(token)


def _folding_reporter(fold: ProgressFold, output_dir: str | Path) -> _FoldingReporter:
    """The reporter that would be active without the hook, behind the fold: the caller's
    reporter, else the one slide2vec would create (Rich on a terminal or in a notebook),
    else none."""
    from soma.extract.reporters import _SomaExtractionReporter

    active = slide2vec_progress.get_progress_reporter()
    if isinstance(active, slide2vec_progress.NullProgressReporter):
        created = slide2vec_progress.create_api_progress_reporter(output_dir=output_dir)
        if not isinstance(created, slide2vec_progress.NullProgressReporter):
            return _FoldingReporter(fold, _SomaExtractionReporter(created), owns_inner=True)
    return _FoldingReporter(fold, active, owns_inner=False)


class _FoldingReporter:
    """slide2vec's active reporter while a hook listens: each event goes unchanged to the
    reporter that would have been active, then to the fold."""

    def __init__(self, fold: ProgressFold, inner: Any, *, owns_inner: bool) -> None:
        self._fold = fold
        self._inner = inner
        self._owns_inner = owns_inner
        self._fold_failed = False

    def __getattr__(self, name: str) -> Any:
        # Rich-aware helpers read the display's ``console`` and ``progress``.
        if name.startswith("__") or name in {"_fold", "_inner", "_owns_inner", "_fold_failed"}:
            raise AttributeError(name)
        return getattr(self._inner, name)

    def emit(self, event: Any) -> None:
        self._inner.emit(event)
        try:
            self._fold.emit(event)
        except Exception:
            # An event the fold cannot read must not fail the extraction.
            if not self._fold_failed:
                self._fold_failed = True
                logger.warning("Could not fold progress event %r.", getattr(event, "kind", event), exc_info=True)

    def write_log(self, message: str, *, stream=None) -> None:
        self._inner.write_log(message, stream=stream)

    def close(self) -> None:
        if self._owns_inner:
            self._inner.close()
