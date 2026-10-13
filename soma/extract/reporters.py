"""Progress reporters and logger-noise suppression for feature extraction."""

from __future__ import annotations

import contextlib
import logging

import hs2p.progress as hs2p_progress
import slide2vec.progress as slide2vec_progress


class _DeduplicateLogFilter(logging.Filter):
    """Suppress repeated log messages from the same logger."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if msg in self._seen:
            return False
        self._seen.add(msg)
        return True


@contextlib.contextmanager
def _suppress_logger_noise_ctx(*logger_names: str):
    """Temporarily raise selected logger trees to WARNING."""
    original_levels: dict[str, int] = {}
    try:
        for logger_name in logger_names:
            for name, logger in logging.root.manager.loggerDict.items():
                if not isinstance(name, str):
                    continue
                if name != logger_name and not name.startswith(f"{logger_name}."):
                    continue
                original_levels[name] = logging.getLogger(name).level
                logging.getLogger(name).setLevel(logging.WARNING)
            original_levels[logger_name] = logging.getLogger(logger_name).level
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        yield
    finally:
        for logger_name, level in original_levels.items():
            logging.getLogger(logger_name).setLevel(level)


class _SomaExtractionReporter:
    """
    Progress reporter wrapper for soma's extraction step.

    Wraps a slide2vec reporter to:
    - Deduplicate identical write_log messages
    - Delegate embedding progress tracking to slide2vec's rich UI
      (one bar per GPU with per-slide updates)
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self._seen_logs: set[str] = set()

    def __getattr__(self, name: str):
        # Forward attribute access (e.g. `progress`, `console`, `_ensure_progress_started`)
        # to the wrapped reporter so Rich-aware helpers like _CacheValidationProgress work.
        return getattr(self._inner, name)

    def write_log(self, message: str, *, stream=None) -> None:
        if message in self._seen_logs:
            return
        self._seen_logs.add(message)
        self._inner.write_log(message, stream=stream)


class _TilingProgressBridgeReporter:
    """Forward live hs2p tiling updates into slide2vec's active reporter."""

    def emit(self, event) -> None:
        if getattr(event, "kind", None) != "tiling.progress":
            return
        slide2vec_progress.get_progress_reporter().emit(
            slide2vec_progress.ProgressEvent(kind=event.kind, payload=dict(event.payload))
        )

    def close(self) -> None:
        return None

    def write_log(self, message: str, *, stream=None) -> None:
        target = stream or None
        slide2vec_progress.emit_progress_log(message, stream=target)


@contextlib.contextmanager
def _make_extraction_reporter_ctx(feature_dir):
    """
    Context manager that activates a *SomaExtractionReporter* when no
    reporter is already active, and installs a log-dedup filter on
    slide2vec's inference logger to suppress repeated warnings.
    """
    dedup_filter = _DeduplicateLogFilter()
    inference_logger = logging.getLogger("slide2vec.inference")
    inference_logger.addFilter(dedup_filter)
    try:
        with _suppress_logger_noise_ctx("cucim"):
            active = slide2vec_progress.get_progress_reporter()
            if not isinstance(active, slide2vec_progress.NullProgressReporter):
                yield
                return
            inner = slide2vec_progress.create_api_progress_reporter(output_dir=feature_dir)
            if isinstance(inner, slide2vec_progress.NullProgressReporter):
                yield
                return
            with slide2vec_progress.activate_progress_reporter(_SomaExtractionReporter(inner)):
                yield
    finally:
        inference_logger.removeFilter(dedup_filter)


@contextlib.contextmanager
def _forward_tiling_progress_ctx():
    """Forward hs2p tiling progress into the active slide2vec reporter."""
    with hs2p_progress.activate_progress_reporter(_TilingProgressBridgeReporter()):
        yield
