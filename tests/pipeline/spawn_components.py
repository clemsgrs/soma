"""A component that only this module registers, for a fold's worker to resolve.

``Pipeline(config, devices=...)`` sends each fold's worker the config, which names its
components by registry key, not by their code. A worker started under ``python -m`` does
not import the launch's ``__main__`` module again, so a key resolves there only once the
module that registered it is imported. Nothing a worker unpickles imports this module:
a run that names :data:`OPTIMIZER` trains only if the worker imports it.
"""

from __future__ import annotations

import torch

from soma import register
from soma.learn import optimizer_registry
from soma.registry import signature_params

__all__ = ["OPTIMIZER", "WorkerAdam"]

#: The registry key :class:`WorkerAdam` is registered under when this module is imported.
OPTIMIZER = "spawn_components_adam"


class WorkerAdam(torch.optim.Adam):
    """``torch.optim.Adam`` under a key of its own."""


if OPTIMIZER not in optimizer_registry:
    register(
        "optimizers",
        OPTIMIZER,
        WorkerAdam,
        params=signature_params(torch.optim.Adam, name="WorkerAdamParams", skip=("params",)),
    )
