"""Crash-safe file writes: stage next to the target, then ``os.replace``."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _staging_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Write ``text`` to ``path`` so readers never observe a partial file."""
    path = Path(path)
    staging = _staging_path(path)
    staging.write_text(text, encoding=encoding)
    os.replace(staging, path)


def atomic_write_json(path: Path, data: Any, *, indent: int = 2, **kwargs: Any) -> None:
    atomic_write_text(Path(path), json.dumps(data, indent=indent, **kwargs))


def atomic_torch_save(obj: Any, path: Path) -> None:
    """``torch.save`` through a staging file so an interrupted save leaves the old checkpoint intact."""
    import torch

    path = Path(path)
    staging = _staging_path(path)
    torch.save(obj, staging)
    os.replace(staging, path)
