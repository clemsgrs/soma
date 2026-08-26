"""Thin shim for the BEETLE slide-manifest curator.

The curator lives with the BEETLE project protocol in :mod:`examples.beetle.curate`
(invokable as ``python -m examples.beetle.curate``); it emits soma's unified Manifest
(``dataset.csv`` + ``splits.csv`` + ``summary.json``). This file is kept so the historical
``python examples/make_beetle_manifest.py`` entry point still works.

    python examples/make_beetle_manifest.py               # strict full 587-slide/527-patient cohort
    python examples/make_beetle_manifest.py --slides 4    # writes non_publication_smoke_manifest/
    python examples/make_beetle_manifest.py --coverage    # also run the (slow) coverage scan
"""

from __future__ import annotations

import sys
from pathlib import Path

# Prefer the in-repo soma over any stale installed copy.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.beetle.curate import main

if __name__ == "__main__":
    main()
