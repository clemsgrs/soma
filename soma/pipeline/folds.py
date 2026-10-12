"""``run_fold``: one fold in the canonical order, written once (ADR 0017).

Byte identity between a ``Pipeline`` run and a hand composition of the public verbs
depends on the order random state is consumed, so the order is fixed here and replayed
step for step by the parity scenarios in ``tests/e2e/test_parity.py``:

1. ``soma.seed_everything(seed, fold=fold)``
2. fit the feature adaptor on the train split (an explicit train-only step)
3. build the model (weight initialisation draws here) and its learner
4. build the loaders
5. ``Baseline.fit`` on the train targets (set shape)
6. ``soma.fit`` with ``Checkpoint`` (in memory, no file) and ``EarlyStopping``
7. the tune-time step (detection: ``sweep_threshold`` on tune)
8. per split: ``soma.predict`` (+ baseline rows) -> ``soma.evaluate`` -> write
   ``predictions_<split>`` and ``metrics.json``
9. ``soma.save`` the fold's bundle

Shape differences live in the :class:`~soma.pipeline.build.FoldSetup` the builders return.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from soma import evaluation as _verbs
from soma import learn
from soma.atomic_io import atomic_write_json
from soma.evaluation import Evaluation, Predictions
from soma.pipeline.aggregate import save_metrics, save_training_history
from soma.pipeline.build import FoldSetup, SplitPlan
from soma.pipeline.layout import BUNDLE_DIRNAME
from soma.training.seed import seed_everything

__all__ = ["FoldResult", "run_fold"]


@dataclass(frozen=True)
class FoldResult:
    """Result of training + evaluation for a single fold."""

    fold: int
    train_result: learn.FitResult
    tune_evaluation: Evaluation
    test_evaluations: dict[str, Evaluation]  # split_name → evaluation
    bundle: Path | None = None


def run_fold(fold_dir: str | Path, setup: FoldSetup) -> FoldResult:
    """Train, score and save one fold in the canonical order."""
    fold_dir = Path(fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(setup.seed, fold=setup.fold)  # 1
    adaptor = setup.fit_adaptor(fold_dir)  # 2
    learner = setup.build_learner(adaptor)  # 3
    loaders = setup.build_loaders(learner, fold_dir)  # 4
    baseline = setup.fit_baseline(learner.head) if setup.fit_baseline is not None else None  # 5
    # The setup's callbacks are the whole policy: fit adds no EarlyStopping or Progress of its own.
    fit = learn.fit(  # 6
        learner,
        loaders.train,
        loaders.tune,
        callbacks=setup.callbacks(),
        patience=None,
        progress=False,
        **setup.trainer,
    )
    if loaders.after_fit is not None:
        loaders.after_fit(fold_dir)
    head = fit.learner.head
    predicted: dict[str, Any] = {}

    def predict(split: SplitPlan):
        # Each split is inferred once, so the tune-time step and scoring share it.
        if split.name not in predicted:
            predictions = _verbs.predict(fit.learner, split.loader, list(split.callbacks))
            if baseline is not None:
                predictions = Predictions.concat([predictions, baseline.predict(list(split.placeholder_ids))])
            predicted[split.name] = predictions
        return predicted[split.name]

    tune_split = loaders.splits[0]
    threshold = setup.tune_step(predict(tune_split), tune_split, head) if setup.tune_step else None  # 7
    evaluations: dict[str, Evaluation] = {}
    for split in loaders.splits:  # 8
        predictions = predict(split)
        del predicted[split.name]  # only the loop holds it now: released once written
        scored = predictions if threshold is None else predictions.with_threshold(threshold)
        evaluations[split.name] = _verbs.evaluate(
            scored, split.targets, metrics=setup.metrics(head), subgroups=split.subgroups
        )
        if split.write:
            _write_predictions(predictions, fold_dir / f"predictions_{split.name}")
        if split.subgroups:
            atomic_write_json(
                fold_dir / f"subgroup_metrics_{split.name}.json", evaluations[split.name].subgroups
            )
    tune_evaluation = evaluations.pop("tune")
    save_metrics(tune_evaluation, evaluations, fold_dir / "metrics.json")
    save_training_history(fit.history, fold_dir / "training_history.json")
    bundle = learn.save(  # 9
        fit,
        fold_dir / BUNDLE_DIRNAME,
        preprocessing=setup.preprocessing(head, threshold),
        provenance={**setup.provenance, "fold": setup.fold, "num_folds": setup.num_folds},
    )
    # The result keeps the learner alive: free the accelerator for the next fold.
    fit.learner.cpu()
    return FoldResult(
        fold=setup.fold,
        train_result=fit,
        tune_evaluation=tune_evaluation,
        test_evaluations=evaluations,
        bundle=bundle,
    )


def _write_predictions(predictions, stem: Path) -> None:
    """A parquet frame (set predictions, points) or a label-map directory."""
    if hasattr(predictions, "to_parquet"):
        predictions.to_parquet(stem.with_name(stem.name + ".parquet"))
    else:
        predictions.save(stem)
