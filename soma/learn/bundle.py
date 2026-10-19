"""``soma.save`` / ``soma.load``: one persisted model format, the bundle (ADR 0016).

A bundle is a directory::

    bundle/
      weights.safetensors   the model's full state_dict: parameters and buffers, so a
                            fitted feature adaptor (center, scale, projection) rides here
      model.json            how to rebuild the bare module before the weights load: its
                            shape, the builder's dotted path and the builder's arguments
      preprocessing.json    what predict / evaluate need beyond the module (class scheme,
                            target schema, geometry, spacing, thresholds); no tensors
      provenance.json       soma / slide2vec / hs2p versions, the selected epoch and its
                            monitor, and whatever the caller adds (fold, run id, source,
                            the component record)

A model is rebuildable when a builder recorded itself on it (:func:`record_builder`):
the pipeline's builders do. A hand-built model still saves, but ``load`` refuses it
because nothing says how to construct the module again.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
import shutil
import uuid
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

__all__ = ["ModelBundle", "load", "record_builder", "save"]

logger = logging.getLogger(__name__)

BUNDLE_FORMAT = 1
WEIGHTS = "weights.safetensors"
MODEL = "model.json"
PREPROCESSING = "preprocessing.json"
PROVENANCE = "provenance.json"


@dataclass(frozen=True)
class ModelBundle:
    """A loaded bundle: the rebuilt module (eval mode, CPU) and its two metadata dicts."""

    model: nn.Module
    preprocessing: dict[str, Any]
    provenance: dict[str, Any]


def record_builder(model: nn.Module, builder: Callable[..., nn.Module], **params: Any) -> nn.Module:
    """Record on ``model`` that ``builder(**params)`` rebuilds it; returns ``model``.

    ``params`` must round-trip through :func:`save`'s JSON encoding: plain values, lists,
    tuples, dicts (any hashable keys), numpy arrays and dataclass instances.
    """
    model.build_recipe = {"builder": f"{builder.__module__}.{builder.__qualname__}", "params": params}
    return model


def save(fit, path: str | Path, *, preprocessing: Mapping[str, Any] | None = None,
         provenance: Mapping[str, Any] | None = None) -> Path:
    """Write ``fit``'s selected model as a bundle at ``path``, atomically; returns ``path``.

    The bundle is staged in a sibling directory and renamed into place, so a first write
    is all-or-nothing; when an incomplete fold is re-run, the old bundle is retired and
    the new one renamed in, with a brief gap only the fold's own writer can observe.
    Readers never see a partial bundle. ``preprocessing`` and
    ``provenance`` are optional JSON-ready dicts; the soma / slide2vec / hs2p versions
    and the fit's selected epoch and monitor are always recorded.
    """
    from safetensors.torch import save_file

    from soma.models.model import DenseModel

    model = fit.learner.model
    recipe = getattr(model, "build_recipe", None)
    model_json = {
        "format": BUNDLE_FORMAT,
        "shape": "grid" if isinstance(model, DenseModel) else "set",
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "builder": None if recipe is None else recipe["builder"],
        "params": None if recipe is None else _encode(recipe["params"]),
    }
    monitor_name, monitor_value = fit.selection_monitor
    provenance_json = {
        **_versions(),
        "selected_epoch": int(fit.selected_epoch),
        "selection_monitor": [str(monitor_name), float(monitor_value)],
        **dict(provenance or {}),
    }
    # Clones: safetensors refuses tensors that share storage (tied weights).
    state = {key: value.detach().to("cpu").contiguous().clone() for key, value in model.state_dict().items()}

    def write(directory: Path) -> None:
        save_file(state, str(directory / WEIGHTS))
        _write_json(directory / MODEL, model_json)
        _write_json(directory / PREPROCESSING, dict(preprocessing or {}))
        _write_json(directory / PROVENANCE, provenance_json)

    return _publish(Path(path), write)


def load(path: str | Path) -> ModelBundle:
    """Rebuild a bundle's module, load its weights and put it in eval mode.

    The module is rebuilt eagerly from ``model.json``, including a live frozen encoder
    stage, so missing (gated) encoder weights fail here rather than at predict time.
    A bundle written by another soma **major** version is refused; another minor or
    patch version loads with a warning.
    """
    from safetensors.torch import load_file

    path = Path(path)
    provenance = _read_json(path / PROVENANCE)
    _check_version(str(provenance.get("soma_version", "")), path)
    recipe = _read_json(path / MODEL)
    if not recipe.get("builder"):
        raise ValueError(
            f"bundle {path} records no builder in {MODEL}: its {recipe.get('class')} was built "
            "by hand, so soma cannot rebuild it. Build the same module and load the weights "
            f"with model.load_state_dict(safetensors.torch.load_file('{path / WEIGHTS}'))."
        )
    builder = _import(recipe["builder"])
    model = builder(**_decode(recipe["params"]))
    model.load_state_dict(load_file(str(path / WEIGHTS), device="cpu"))
    model.eval()
    return ModelBundle(model=model, preprocessing=_read_json(path / PREPROCESSING), provenance=provenance)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _publish(path: Path, write: Callable[[Path], None]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.staging-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    retired = path.parent / f".{path.name}.retired-{uuid.uuid4().hex[:8]}"
    try:
        write(staging)
        if path.exists():
            os.replace(path, retired)
        try:
            os.replace(staging, path)
        except BaseException:
            if retired.exists() and not path.exists():
                os.replace(retired, path)  # put the previous bundle back
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(retired, ignore_errors=True)
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"not a soma bundle: {path} is missing")
    return json.loads(path.read_text(encoding="utf-8"))


def _versions() -> dict[str, str | None]:
    import soma

    versions: dict[str, str | None] = {"soma_version": str(soma.__version__)}
    for package in ("slide2vec", "hs2p"):
        try:
            versions[f"{package}_version"] = version(package)
        except PackageNotFoundError:
            versions[f"{package}_version"] = None
    return versions


def _major(text: str) -> int | None:
    head = text.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _check_version(saved: str, path: Path) -> None:
    import soma

    current = str(soma.__version__)
    if saved == current:
        return
    saved_major, current_major = _major(saved), _major(current)
    if saved_major is not None and current_major is not None and saved_major != current_major:
        raise ValueError(
            f"bundle {path} was written by soma {saved}; this is soma {current}, another "
            "major version, whose bundle format may differ. Load it with soma "
            f"{saved_major}.x."
        )
    warnings.warn(
        f"bundle {path} was written by soma {saved}; loading it with soma {current}.",
        UserWarning,
        stacklevel=3,
    )


def _import(dotted: str) -> Callable[..., nn.Module]:
    module_name, _, attribute = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attribute)


# ---------------------------------------------------------------------------
# A JSON encoding that round-trips builder arguments
# ---------------------------------------------------------------------------


def _encode(value: Any) -> Any:
    """JSON-ready form of ``value``; tuples, non-string-keyed dicts, numpy arrays and
    dataclasses are tagged so :func:`_decode` restores them exactly."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return {"__ndarray__": value.tolist(), "dtype": str(value.dtype)}
    if torch.is_tensor(value):
        raise TypeError("a builder argument may not be a tensor; tensors belong in the weights")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        fields = {f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
        return {"__dataclass__": f"{cls.__module__}.{cls.__qualname__}", "fields": fields}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, Mapping):
        if all(isinstance(key, str) for key in value):
            return {key: _encode(item) for key, item in value.items()}
        return {"__items__": [[_encode(key), _encode(item)] for key, item in value.items()]}
    raise TypeError(f"cannot record a builder argument of type {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__tuple__" in value:
        return tuple(_decode(v) for v in value["__tuple__"])
    if "__items__" in value:
        return {_decode(key): _decode(item) for key, item in value["__items__"]}
    if "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=value["dtype"])
    if "__dataclass__" in value:
        return _import(value["__dataclass__"])(**{k: _decode(v) for k, v in value["fields"].items()})
    return {key: _decode(item) for key, item in value.items()}
