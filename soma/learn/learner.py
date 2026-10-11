"""``Learner``: the object a user subclasses to own a model and its step logic.

The :class:`~soma.learn.trainer.Trainer` is a generic loop; everything that knows about
models, losses, metrics and optimizers lives here. The contract mirrors Lightning's
``LightningModule`` so a later swap is mechanical, with one deliberate difference:
``configure_optimizers`` receives an explicit :class:`Budget` instead of a back-reference
to the trainer.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

from soma.registry import Registry

__all__ = [
    "Budget",
    "Learner",
    "SchedulerSpec",
    "StepOutput",
    "learner_registry",
    "optimizer_registry",
    "resolve_leaf",
    "scheduler_registry",
    "uncovered_parameters",
]


@dataclass(frozen=True)
class StepOutput:
    """What one ``validation_step`` hands to ``validation_summary``.

    ``outputs`` is whatever the learner reduces (logits, risks, compact stat rows); the
    Trainer never looks inside. ``loss`` is ``None`` when the loss only exists over the
    whole split (a Cox partial likelihood).
    """

    loss: Tensor | None
    outputs: Tensor
    targets: dict[str, Tensor]
    size: int


@dataclass(frozen=True)
class SchedulerSpec:
    """A scheduler and the cadence the Trainer steps it at."""

    scheduler: torch.optim.lr_scheduler.LRScheduler
    interval: Literal["epoch", "step"] = "epoch"


@dataclass(frozen=True)
class Budget:
    """What the Trainer knows before the first step; ``configure_optimizers`` reads it."""

    max_epochs: int | None
    max_steps: int | None
    updates_per_epoch: int

    @property
    def total_epochs(self) -> int:
        if self.max_epochs is not None:
            return int(self.max_epochs)
        if self.max_steps is None:
            raise ValueError("a Budget needs max_epochs or max_steps")
        return -(-int(self.max_steps) // max(int(self.updates_per_epoch), 1))

    @property
    def total_steps(self) -> int:
        if self.max_steps is not None:
            return int(self.max_steps)
        return self.total_epochs * max(int(self.updates_per_epoch), 1)


# ---------------------------------------------------------------------------
# {name, params} leaves
# ---------------------------------------------------------------------------

optimizer_registry = Registry("optimizer")
scheduler_registry = Registry("scheduler")
learner_registry = Registry("learner")

for _name, _cls in {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "adagrad": torch.optim.Adagrad,
}.items():
    optimizer_registry.register(_name, _cls)

for _name, _cls in {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "constant": torch.optim.lr_scheduler.ConstantLR,
    "linear": torch.optim.lr_scheduler.LinearLR,
    "step": torch.optim.lr_scheduler.StepLR,
    "multistep": torch.optim.lr_scheduler.MultiStepLR,
    "exponential": torch.optim.lr_scheduler.ExponentialLR,
    "one_cycle": torch.optim.lr_scheduler.OneCycleLR,
    "cosine_warm_restarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
}.items():
    scheduler_registry.register(_name, _cls)


def resolve_leaf(leaf: Any, registry: Registry) -> tuple[type, dict[str, Any], dict[str, Any]]:
    """Normalise a ``{name, params}`` leaf (or its string shorthand) to ``(cls, params, extra)``.

    ``extra`` holds the leaf's other keys (``interval`` on a scheduler leaf).
    """
    if isinstance(leaf, str):
        leaf = {"name": leaf}
    if not isinstance(leaf, Mapping) or "name" not in leaf:
        raise ValueError(
            f"a component leaf is a name or a {{name, params}} mapping, got {leaf!r}"
        )
    params = dict(leaf.get("params") or {})
    extra = {k: v for k, v in leaf.items() if k not in {"name", "params"}}
    return registry.resolve(str(leaf["name"])), params, extra


# ---------------------------------------------------------------------------
# Learner
# ---------------------------------------------------------------------------


class Learner(nn.Module):
    """Owns ``model`` and the step logic; subclass and override what should differ.

    Args:
        model: The module to train.
        optimizer: A ``{name, params}`` leaf (or a name) for the optimizer; ``None`` is
            Adam at ``lr=1e-4, weight_decay=1e-5``.
        schedulers: Leaves for the learning-rate schedulers, each stepped at its own
            ``interval`` (``"epoch"`` by default; ``cosine`` follows the budget). The
            default installs none; the config's ``scheduler`` leaf is cosine.
    """

    DEFAULT_OPTIMIZER: dict[str, Any] = {"name": "adam", "params": {"lr": 1e-4, "weight_decay": 1e-5}}

    def __init__(
        self,
        model: nn.Module,
        *,
        optimizer: Any = None,
        schedulers: Sequence[Any] = (),
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_leaf = self.DEFAULT_OPTIMIZER if optimizer is None else optimizer
        self.scheduler_leaves = tuple(schedulers)

    # -- steps ---------------------------------------------------------------

    def training_step(self, batch: Any) -> Tensor | dict[str, Tensor]:
        """One training batch to a scalar loss (or a dict carrying ``"loss"``)."""
        raise NotImplementedError

    def validation_step(self, batch: Any) -> StepOutput:
        raise NotImplementedError

    def validation_summary(self, outputs: list[StepOutput]) -> dict[str, float]:
        """Reduce a split's outputs to scalars; must contain ``"loss"``."""
        raise NotImplementedError

    # -- optimisation --------------------------------------------------------

    def configure_optimizers(
        self, budget: Budget
    ) -> torch.optim.Optimizer | tuple[torch.optim.Optimizer, list[SchedulerSpec]]:
        """Default: the leaves over every ``requires_grad`` parameter. Cosine follows the
        budget's cadence (``step`` under a step budget, else ``epoch``) unless the leaf
        sets ``interval``; an omitted ``T_max`` is the budget in that cadence's units."""
        cls, params, _ = resolve_leaf(self.optimizer_leaf, optimizer_registry)
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = cls(trainable, **params)
        specs: list[SchedulerSpec] = []
        for leaf in self.scheduler_leaves:
            if (leaf.get("name") if isinstance(leaf, Mapping) else leaf) in (None, "none"):
                continue
            scls, sparams, extra = resolve_leaf(leaf, scheduler_registry)
            cosine = scls is torch.optim.lr_scheduler.CosineAnnealingLR
            interval = extra.get("interval", "step" if cosine and budget.max_steps is not None else "epoch")
            if interval not in {"epoch", "step"}:
                raise ValueError(f"scheduler interval must be 'epoch' or 'step', got {interval!r}")
            if cosine and "T_max" not in sparams:
                sparams["T_max"] = budget.total_steps if interval == "step" else budget.total_epochs
            specs.append(SchedulerSpec(scheduler=scls(optimizer, **sparams), interval=interval))
        return optimizer, specs

    # -- plumbing ------------------------------------------------------------

    def transfer_batch(self, batch: Any, device: torch.device) -> Any:
        """Move a batch to ``device``, walking dataclasses, dicts, lists, (named) tuples and tensors."""
        return _move(batch, device)

    def trainable_state_dict(self) -> dict[str, Tensor]:
        """What a checkpoint snapshots and restores in memory: the model's ``state_dict``
        by default. A learner that owns trainable modules of its own widens this pair to
        cover them (``self.state_dict()``); :class:`~soma.learn.callbacks.Checkpoint`
        refuses an optimized parameter it could not restore. ``best_model.pt`` always
        holds the bare model's state, so it loads into the model alone."""
        return self.model.state_dict()

    def load_trainable_state_dict(self, state: Mapping[str, Tensor]) -> None:
        self.model.load_state_dict(state)


def uncovered_parameters(learner: Learner, optimizer: torch.optim.Optimizer) -> list[str]:
    """Names of the optimizer's parameters that ``learner.trainable_state_dict()`` omits.

    Coverage is by storage: ``state_dict`` hands out views of the parameters, so a
    parameter the optimizer updates but no checkpoint restores is one whose storage
    is absent from the trainable state.
    """
    optimized = {p.data_ptr() for group in optimizer.param_groups for p in group["params"]}
    covered = {t.data_ptr() for t in learner.trainable_state_dict().values() if torch.is_tensor(t)}
    return [
        name
        for name, p in learner.named_parameters()
        if p.data_ptr() in optimized and p.data_ptr() not in covered
    ]


def _move(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        moved = {f.name: _move(getattr(value, f.name), device) for f in dataclasses.fields(value)}
        return dataclasses.replace(value, **moved)
    if isinstance(value, Mapping):
        return type(value)((k, _move(v, device)) for k, v in value.items())
    if isinstance(value, tuple):
        moved = [_move(v, device) for v in value]
        return type(value)(*moved) if hasattr(value, "_fields") else tuple(moved)
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    return value
