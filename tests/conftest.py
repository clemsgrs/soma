from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for sibling in (ROOT.parent / "slide2vec", ROOT.parent / "hs2p"):
    if sibling.exists():
        sys.path.insert(0, str(sibling))


@pytest.fixture(autouse=True)
def restore_soma_logger():
    """Undo the ``soma`` logger level and handlers a test's in-process CLI call sets.

    ``soma.cli.main`` raises the ``soma`` logger to INFO; without this, every test that
    runs after a CLI test would capture soma INFO records it never asked for.
    """
    soma_logger = logging.getLogger("soma")
    saved = (soma_logger.handlers[:], soma_logger.level, soma_logger.propagate)
    yield
    soma_logger.handlers[:] = saved[0]
    soma_logger.setLevel(saved[1])
    soma_logger.propagate = saved[2]
