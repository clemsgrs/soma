"""soma's default log output for callers that never configure Python logging.

Python drops INFO records when nobody configures logging, so a script that only calls
``Pipeline(config).run()`` would see none of soma's progress messages. On first use
soma attaches one stderr handler to the ``soma`` logger, unless the process already
has a logging setup (a handler on the root or the ``soma`` logger), which soma then
leaves untouched.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


class _StderrHandler(logging.StreamHandler):
    """Write to whatever ``sys.stderr`` is at emit time.

    Binding the stream at emit time keeps records visible when ``sys.stderr`` is
    swapped after soma's first use: a Rich live display redirecting it above its
    progress bars, or a test capturing it.
    """

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass


class _UntilRootConfigured(logging.Filter):
    """Drop records once they reach root handlers of the caller's own.

    A caller who configures logging after soma's first use receives soma's records
    through their root handlers; soma's handler then stays quiet so no line prints
    twice. When the ``soma`` logger does not propagate, the root handlers never see
    soma's records, so soma's handler keeps printing them.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        reaches_root = logging.getLogger("soma").propagate
        return not (reaches_root and logging.getLogger().handlers)


def ensure_default_logging() -> None:
    """Attach soma's default stderr handler when the process has no logging setup."""
    soma_logger = logging.getLogger("soma")
    if logging.getLogger().handlers or soma_logger.handlers:
        return
    # The handler has no level of its own: the ``soma`` logger's level alone decides
    # what is shown, so a caller who sets it to DEBUG sees DEBUG records.
    handler = _StderrHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    handler.addFilter(_UntilRootConfigured())
    soma_logger.addHandler(handler)
    # Raise soma to INFO only when no level was chosen: a level set on the ``soma``
    # logger, or a root level other than Python's default WARNING (for example a
    # script that sets the root to ERROR to stay quiet), is the caller's and is kept.
    root_level_is_default = logging.getLogger().level == logging.WARNING
    if soma_logger.level == logging.NOTSET and root_level_is_default:
        soma_logger.setLevel(logging.INFO)


def configure_cli_logging() -> None:
    """Logging setup for the ``soma`` command: soma's INFO records, other libraries' warnings.

    The handler goes on the root logger, as an application's should. The root level stays
    at WARNING so third-party INFO records stay hidden; only the ``soma`` logger is raised
    to INFO. A root logger that already has handlers keeps them and gains none.
    """
    logging.basicConfig(format=_FORMAT, datefmt=_DATEFMT, handlers=[_StderrHandler()])
    logging.getLogger("soma").setLevel(logging.INFO)
