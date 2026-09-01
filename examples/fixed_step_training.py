"""Train with a fixed optimizer-step budget instead of an epoch budget.

``TrainingConfig(max_steps=N, epochs=None)`` (soma 1.12.0) stops training after
exactly N optimizer updates, even when N does not divide the number of updates
per epoch — the final epoch is then partial, and its tune evaluation still runs.

Self-contained: fabricates random precomputed feature bags (no encoder, no
download) and trains an attention-MIL head on CPU. The metric values are
meaningless — the exact update count is the point.

Run with no arguments:

    python examples/fixed_step_training.py
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import warnings
from pathlib import Path

# Force CPU so the example runs anywhere, even on a machine whose GPUs are
# busy (delete this line to use your GPU).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

import pandas as pd
import torch

from soma import (
    AggregatorConfig,
    Dataset,
    EvalConfig,
    FeatureStore,
    Splits,
    TaskConfig,
    TrainingConfig,
    train,
)

# --- Scaffolding (not soma API): random precomputed feature bags ------------------
# A FeatureStore reads a directory of .pt files, one bag of tile vectors per
# sample — exactly what a real feature extraction would have written.

WORK = Path(tempfile.mkdtemp(prefix="soma-fixed-steps-"))
FEATURES = WORK / "features"
FEATURES.mkdir()
torch.manual_seed(0)

sample_ids = [f"s{i:02d}" for i in range(12)]
labels = [i % 2 for i in range(12)]
split = (["train"] * 8) + (["tune"] * 2) + (["test"] * 2)
for sid in sample_ids:
    torch.save(torch.randn(16, 384), FEATURES / f"{sid}.pt")  # 16 tiles x 384 dims

dataset_csv = WORK / "dataset.csv"
splits_csv = WORK / "splits.csv"
pd.DataFrame(
    {
        "sample_id": sample_ids,
        # Paths are provenance here: training reads the precomputed bags only.
        "image_path": [str(WORK / "slides" / f"{s}.tif") for s in sample_ids],
        "label": labels,
    }
).to_csv(dataset_csv, index=False)
pd.DataFrame({"sample_id": sample_ids, "split": split}).to_csv(splits_csv, index=False)

dataset = Dataset(dataset_csv)
splits = Splits(splits_csv, dataset)
store = FeatureStore(FEATURES)

# --- The step budget --------------------------------------------------------------
# 8 training bags at batch_size=4 give 2 optimizer updates per epoch. A budget
# of 7 steps therefore needs ceil(7 / 2) = 4 epochs, and the 4th epoch stops
# after a single update — a partial final epoch.

MAX_STEPS = 7
BATCH_SIZE = 4
n_train = split.count("train")
updates_per_epoch = math.ceil(n_train / BATCH_SIZE)
derived_epochs = math.ceil(MAX_STEPS / updates_per_epoch)
print(f"train bags:        {n_train} (batch_size={BATCH_SIZE})")
print(f"updates per epoch: {updates_per_epoch}")
print(f"step budget:       max_steps={MAX_STEPS}")
print(
    f"derived epochs:    ceil({MAX_STEPS}/{updates_per_epoch}) = {derived_epochs} "
    f"(final epoch runs {MAX_STEPS - (derived_epochs - 1) * updates_per_epoch} "
    f"of {updates_per_epoch} updates)"
)

result = train(
    feature_store=store,
    dataset=dataset,
    splits=splits,
    aggregator=AggregatorConfig(name="abmil"),
    task=TaskConfig(name="binary_classification"),
    training=TrainingConfig(
        max_steps=MAX_STEPS,
        epochs=None,  # exactly one training budget may be set
        batch_size=BATCH_SIZE,
        learning_rate=1e-3,
        seed=0,
        pin_memory=False,
        persistent_workers=False,
    ),
    evaluation=EvalConfig(metrics=["balanced_accuracy"]),
    run_dir=WORK / "run",
)

# --- The exact optimizer-update count ---------------------------------------------
train_result = result.fold_results[0].train_result
print()
print(f"optimizer updates performed: {train_result.optimizer_steps} (== max_steps)")
print(f"epochs trained:              {len(train_result.history)}")
print("per-epoch log (the tune split is evaluated every epoch, partial ones too):")
for log in train_result.history:
    updates_this_epoch = min(
        updates_per_epoch, MAX_STEPS - log.epoch * updates_per_epoch
    )
    partial = " (partial epoch)" if updates_this_epoch < updates_per_epoch else ""
    print(
        f"  epoch {log.epoch + 1}: {updates_this_epoch} update(s){partial} | "
        f"train_loss={log.train_loss:.3f} tune_loss={log.tune_loss:.3f} "
        f"tune_metrics={ {k: round(v, 3) for k, v in log.tune_metrics.items()} }"
    )

assert train_result.optimizer_steps == MAX_STEPS
assert len(train_result.history) == derived_epochs
