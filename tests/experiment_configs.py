"""A concise builder for :class:`soma.config.ExperimentConfig` in tests.

Tests state only what they are about; every other field keeps the section's default, as
a user constructing the dataclasses would get it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soma.config import (
    CohortConfig,
    Component,
    EvaluationConfig,
    ExperimentConfig,
    FeaturesConfig,
    LearnerConfig,
    LoadersConfig,
    RunConfig,
    TrainerConfig,
)

__all__ = ["callback", "experiment"]


def callback(config: ExperimentConfig, name: str) -> dict[str, Any] | None:
    """The params of ``config``'s trainer callback ``name`` (``None`` when absent)."""
    for leaf in config.trainer.callbacks:
        if leaf.name == name:
            return dict(leaf.params)
    return None


def experiment(
    records: str | Path = "dataset.csv",
    splits: str | Path = "splits.csv",
    *,
    head: str | dict | Component | None = "binary_classification",
    head_params: dict[str, Any] | None = None,
    aggregator: str | dict | Component | None = None,
    decoder: str | dict | Component | None = None,
    learner: str | None = None,
    features: FeaturesConfig | dict[str, Any] | None = None,
    output_dir: str | Path = "runs",
    seed: int = 0,
    run: dict[str, Any] | None = None,
    cohort: dict[str, Any] | None = None,
    loaders: dict[str, Any] | None = None,
    trainer: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    **learner_params: Any,
) -> ExperimentConfig:
    """An experiment over ``records`` / ``splits``.

    ``head`` (with ``head_params``), ``aggregator`` and ``decoder`` are the learner's
    model leaves; ``learner`` defaults to ``dense`` with a decoder, else ``mil``; any other
    keyword (``optimizer``, ``scheduler``, ``normalization``, ...) is a learner param.
    ``head=None`` builds no learner (a task-free run). Section dicts are their
    dataclass's keywords.
    """
    learner_config = None
    if head is not None:
        params: dict[str, Any] = dict(learner_params)
        head_leaf = Component.of(head)
        if head_params:
            head_leaf = Component(head_leaf.name, {**head_leaf.params, **head_params})
        params["head"] = head_leaf
        if aggregator is not None:
            params["aggregator"] = aggregator
        if decoder is not None:
            params["decoder"] = decoder
        learner_config = LearnerConfig(learner or ("dense" if decoder is not None else "mil"), params)
    if isinstance(features, dict):
        features = FeaturesConfig(**features)
    return ExperimentConfig(
        run=RunConfig(**{"output_dir": output_dir, "seed": seed, **(run or {})}),
        cohort=CohortConfig(records=records, splits=splits, **(cohort or {})),
        features=features if features is not None else FeaturesConfig(),
        loaders=LoadersConfig(**(loaders or {})),
        learner=learner_config,
        trainer=TrainerConfig(**(trainer or {})),
        evaluation=EvaluationConfig(**(evaluation or {})),
    )
