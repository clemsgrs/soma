"""``check_learner``: the conformance suite for :class:`soma.learn.Learner` implementations."""

from __future__ import annotations

import math
from typing import Any

import torch

from soma.learn.learner import Budget, Learner, SchedulerSpec, StepOutput, uncovered_parameters
from soma.testing._errors import ConformanceError

__all__ = ["check_learner"]


def check_learner(learner: Learner, batch: Any, *, budget: Budget) -> None:
    """Check a learner on one batch on CPU.

    Asserts: every trainable parameter sits in an optimizer param group; one
    ``training_step`` gives a finite scalar loss with gradients; ``validation_step``
    twice then ``validation_summary`` gives finite floats including ``"loss"``;
    ``trainable_state_dict`` covers every optimized parameter and round-trips;
    ``transfer_batch`` preserves the batch type.
    """
    device = torch.device("cpu")
    learner = learner.to(device)

    # configure_optimizers
    try:
        configured = learner.configure_optimizers(budget)
    except Exception as error:  # noqa: BLE001
        raise ConformanceError("configure_optimizers", f"raised {error!r}") from error
    optimizer, specs = (configured, []) if isinstance(configured, torch.optim.Optimizer) else configured
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ConformanceError("configure_optimizers", f"must return an Optimizer, got {type(optimizer)!r}")
    if not all(isinstance(spec, SchedulerSpec) for spec in specs):
        raise ConformanceError("configure_optimizers", "schedulers must be SchedulerSpec instances")
    grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
    missing = [name for name, p in learner.named_parameters() if p.requires_grad and id(p) not in grouped]
    if missing:
        raise ConformanceError(
            "configure_optimizers", f"trainable parameters outside every param group: {missing[:5]}"
        )

    # transfer_batch
    try:
        moved = learner.transfer_batch(batch, device)
    except Exception as error:  # noqa: BLE001
        raise ConformanceError("transfer_batch", f"raised {error!r}") from error
    if type(moved) is not type(batch):
        raise ConformanceError(
            "transfer_batch", f"changed the batch type from {type(batch)!r} to {type(moved)!r}"
        )

    # training_step
    learner.train()
    learner.zero_grad()
    try:
        out = learner.training_step(moved)
    except Exception as error:  # noqa: BLE001
        raise ConformanceError("training_step", f"raised {error!r}") from error
    if isinstance(out, dict):
        if "loss" not in out:
            raise ConformanceError("training_step", "a dict output must carry 'loss'")
        loss = out["loss"]
    else:
        loss = out
    if not torch.is_tensor(loss) or loss.ndim != 0:
        raise ConformanceError("training_step", f"loss must be a scalar Tensor, got {loss!r}")
    if not math.isfinite(float(loss)):
        raise ConformanceError("training_step", f"loss is not finite: {float(loss)}")
    if not loss.requires_grad:
        raise ConformanceError("training_step", "loss does not require grad")
    loss.backward()
    if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in learner.parameters() if p.requires_grad):
        raise ConformanceError("training_step", "no trainable parameter received a non-zero gradient")
    learner.zero_grad()

    # validation_step x2 -> validation_summary
    learner.eval()
    outputs = []
    with torch.inference_mode():
        for _ in range(2):
            try:
                step = learner.validation_step(moved)
            except Exception as error:  # noqa: BLE001
                raise ConformanceError("validation_step", f"raised {error!r}") from error
            if not isinstance(step, StepOutput):
                raise ConformanceError("validation_step", f"must return a StepOutput, got {type(step)!r}")
            outputs.append(step)
        try:
            summary = learner.validation_summary(outputs)
        except Exception as error:  # noqa: BLE001
            raise ConformanceError("validation_summary", f"raised {error!r}") from error
    if not isinstance(summary, dict) or "loss" not in summary:
        raise ConformanceError("validation_summary", f"must return a dict with 'loss', got {summary!r}")
    for key, value in summary.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
            raise ConformanceError("validation_summary", f"'{key}' must be a finite float, got {value!r}")

    # trainable_state_dict covers every optimized parameter and round-trips
    state = learner.trainable_state_dict()
    if not isinstance(state, dict) or not all(torch.is_tensor(v) for v in state.values()):
        raise ConformanceError("trainable_state_dict", "must return a dict of tensors")
    uncovered = uncovered_parameters(learner, optimizer)
    if uncovered:
        raise ConformanceError(
            "trainable_state_dict", f"optimized parameters a checkpoint would not restore: {uncovered[:5]}"
        )
    reference = {k: v.detach().clone() for k, v in state.items()}
    with torch.no_grad():
        for p in learner.parameters():
            if p.requires_grad:
                p.add_(1.0)
    try:
        learner.load_trainable_state_dict(reference)
    except Exception as error:  # noqa: BLE001
        raise ConformanceError("load_trainable_state_dict", f"raised {error!r}") from error
    for key, value in learner.trainable_state_dict().items():
        if not torch.equal(value, reference[key]):
            raise ConformanceError("load_trainable_state_dict", f"'{key}' did not round-trip")
