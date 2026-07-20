"""``write_dense_grid`` must not serialise a batch slice's whole parent storage.

Dense extraction encodes a ``(B, d, h, w)`` batch and iterates it, so each grid handed
to :func:`write_dense_grid` is a ``batch[i]`` VIEW that shares the full batch storage.
``torch.save`` writes a tensor's underlying storage, not just its view, so saving one
directly wrote ALL ``B`` tiles' bytes into every tile's file — an ``encoder.batch_size``
-fold bloat (observed on MIDOG at B=8: 134 MB files holding 16.8 MB of grid). ``fp16``
caches accidentally escaped it because the dtype cast allocated a fresh tensor, which is
why the bug hid in the fp32 default. ``.contiguous()`` is NOT a fix: a batch slice is
already contiguous, so it no-ops.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from soma.dense.geometry import compute_dense_geometry  # noqa: E402
from soma.dense.store import dense_grid_metadata, write_dense_grid  # noqa: E402

TARGET = 64
PATCH = 4
DIM = 64
BATCH = 8


def _meta():
    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    return geom, dense_grid_metadata(geom, feature_dim=DIM, pad_mode="reflect", spacing_um=0.5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_batch_slice_is_not_saved_with_parent_storage(tmp_path: Path, dtype):
    geom, meta = _meta()
    gh, gw = geom.grid_shape if hasattr(geom, "grid_shape") else (meta["grid_shape"][0], meta["grid_shape"][1])
    batch = torch.randn(BATCH, DIM, gh, gw, dtype=torch.float32).to(dtype)
    grid = batch[3]  # a view: contiguous, but storage spans all BATCH tiles

    # precondition: the view really does carry an oversized parent storage
    own_bytes = grid.numel() * grid.element_size()
    assert grid.is_contiguous()  # so .contiguous() would be a no-op
    assert grid.untyped_storage().nbytes() == own_bytes * BATCH

    path = write_dense_grid(tmp_path, "s0", grid, meta)

    saved = torch.load(path, map_location="cpu")
    # the saved tensor owns exactly its own bytes — no parent-batch payload rode along
    assert saved.untyped_storage().nbytes() == own_bytes
    assert path.stat().st_size < own_bytes * 2  # file ~= one grid, not BATCH grids
    assert saved.dtype == dtype
    assert torch.equal(saved, grid)


def test_already_exact_tensor_is_written_unchanged(tmp_path: Path):
    """A standalone grid needs no copy and must round-trip byte-for-byte."""
    geom, meta = _meta()
    gh, gw = (meta["grid_shape"][0], meta["grid_shape"][1])
    grid = torch.randn(DIM, gh, gw)
    assert grid.untyped_storage().nbytes() == grid.numel() * grid.element_size()

    path = write_dense_grid(tmp_path, "s1", grid, meta)
    saved = torch.load(path, map_location="cpu")
    assert torch.equal(saved, grid)
    assert saved.untyped_storage().nbytes() == grid.numel() * grid.element_size()
