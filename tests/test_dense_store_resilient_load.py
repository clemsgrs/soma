"""Resilient dense-grid loading (issue #207).

``DenseFeatureStore.load`` retries transient ``OSError`` / ``FileNotFoundError``
from the underlying ``load_array`` with bounded linear backoff, so a single
spurious read error on a flaky networked mount (zfs/NFS/CIFS) does not abort a
multi-fold dense run. The happy path adds no latency and no sleep.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import soma.dense.store as store_mod  # noqa: E402
from soma.dense.geometry import compute_dense_geometry  # noqa: E402
from soma.dense.store import DenseFeatureStore, dense_grid_metadata, write_dense_grid  # noqa: E402

TARGET = 8
PATCH = 4
DIM = 3


def _make_store(tmp_path: Path) -> DenseFeatureStore:
    """Write one real ``s0`` grid + sidecar and open a store over it."""
    geom = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    meta = dense_grid_metadata(geom, feature_dim=DIM, pad_mode="reflect", spacing_um=0.2)
    rng = np.random.default_rng(0)
    grid = torch.from_numpy(rng.standard_normal((DIM, *geom.grid_shape)).astype("float32"))
    write_dense_grid(tmp_path, "s0", grid, meta)
    return DenseFeatureStore(tmp_path)


def test_happy_path_reads_once_without_sleeping(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    real_load = store_mod.load_array
    reads = {"n": 0}
    sleeps: list[float] = []

    def counting_load(path):
        reads["n"] += 1
        return real_load(path)

    monkeypatch.setattr(store_mod, "load_array", counting_load)
    monkeypatch.setattr(store_mod.time, "sleep", sleeps.append)

    out = store.load("s0")
    assert tuple(out.shape) == (DIM, *store.grid_shape)
    assert reads["n"] == 1  # a single read
    assert sleeps == []  # no backoff when the first read succeeds


def test_retries_transient_oserror_then_succeeds(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    real_load = store_mod.load_array
    sleeps: list[float] = []
    state = {"n": 0}

    def flaky_load(path):
        state["n"] += 1
        if state["n"] <= 2:  # first two reads blip, third succeeds
            raise OSError("transient stale file handle")
        return real_load(path)

    monkeypatch.setattr(store_mod, "load_array", flaky_load)
    monkeypatch.setattr(store_mod.time, "sleep", sleeps.append)

    out = store.load("s0")
    assert tuple(out.shape) == (DIM, *store.grid_shape)
    assert state["n"] == 3  # two failures + one success
    assert sleeps == [0.5, 1.0]  # linear backoff between the two failed attempts


def test_filenotfounderror_is_retried_as_transient(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    real_load = store_mod.load_array
    state = {"n": 0}
    monkeypatch.setattr(store_mod.time, "sleep", lambda _s: None)

    def flaky_load(path):
        state["n"] += 1
        if state["n"] == 1:  # mount briefly not ready
            raise FileNotFoundError("stat: no such file (yet)")
        return real_load(path)

    monkeypatch.setattr(store_mod, "load_array", flaky_load)

    out = store.load("s0")
    assert tuple(out.shape) == (DIM, *store.grid_shape)
    assert state["n"] == 2  # one blip, then success


def test_permanent_error_raises_after_bounded_attempts(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    sleeps: list[float] = []
    state = {"n": 0}

    def always_fails(_path):
        state["n"] += 1
        raise OSError("mount gone")

    monkeypatch.setattr(store_mod, "load_array", always_fails)
    monkeypatch.setattr(store_mod.time, "sleep", sleeps.append)

    with pytest.raises(OSError, match="mount gone"):  # original error surfaces unchanged
        store.load("s0")
    assert state["n"] == store_mod._LOAD_RETRIES  # exactly N attempts — bounded
    assert len(sleeps) == store_mod._LOAD_RETRIES - 1  # one backoff between each pair
