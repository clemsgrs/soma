"""Configs and callback factories that a fold's worker process can import.

``Pipeline(config, devices=...)`` trains each fold in a new ``spawn`` process, which
receives the config, the source and the callback factory pickled. Pickle stores a class
by its module and name, so everything here lives at module level: a worker imports this
module to rebuild it, and sees none of a test's monkeypatches or local functions.

The runs are the tile protocol of ``tests.benchmark.spawn_benchmarks``: soma.testing's
weight-free ``fake`` encoder over the synthetic tiles of
``tests.e2e.synthetic.make_tile_dataset``, a binary-classification head trained for a
few epochs, with the trainer pinned to ``cpu``.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from soma import ExperimentConfig
from soma.pipeline import FoldContext
from tests.benchmark.spawn_benchmarks import build_tile_config, cache_snapshot

__all__ = ["CACHE_SNAPSHOT", "PID_FILE", "FoldRecorder", "tile_cv_config"]

#: The file, in a fold's directory, that :class:`FoldRecorder` writes the cache into.
CACHE_SNAPSHOT = "cache_snapshot.json"
#: The file, in a fold's directory, that names the process the fold trained in.
PID_FILE = "pid"


def tile_cv_config(
    dataset_csv: str | Path, splits_csv: str | Path, output_root: str | Path, **run: Any
) -> ExperimentConfig:
    """The tile protocol on ``splits_csv``, seed 0, with ``run`` keywords (``run_id``,
    ``folds``, ``resume``) set. The feature cache is ``<output_root>/feature_cache``."""
    config = build_tile_config(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=output_root,
        seed=0,
        overrides={},
        encoder="fake",
    )
    return replace(config, run=replace(config.run, **run)) if run else config


class FoldRecorder:
    """A callback factory that records where each fold trains, then acts for it.

    As each fold starts, it writes the fold's process id (:data:`PID_FILE`) and, when
    ``cache_root`` is set, the feature cache as it then is (:data:`CACHE_SNAPSHOT`) into
    the fold's directory. ``actions`` maps a fold to what it does before it trains:
    ``raise`` a ``ValueError``, or start ``slow`` (after 8 seconds). It adds no Trainer
    callback, so it changes nothing about what a fold learns.
    """

    def __init__(self, cache_root: str | Path | None = None, actions: dict[int, str] | None = None) -> None:
        self.cache_root = None if cache_root is None else Path(cache_root)
        self.actions = dict(actions or {})

    def __call__(self, context: FoldContext) -> list[Any]:
        fold_dir = Path(context.fold_dir)
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / PID_FILE).write_text(str(os.getpid()))
        if self.cache_root is not None:
            snapshot = cache_snapshot(self.cache_root)
            (fold_dir / CACHE_SNAPSHOT).write_text(json.dumps(snapshot, sort_keys=True))
        action = self.actions.get(context.fold)
        if action == "raise":
            raise ValueError(f"fold {context.fold} failed")
        if action == "slow":
            time.sleep(8)
        return []
