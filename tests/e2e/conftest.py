from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.dense_literal_encoder import _LiteralPatchEncoder, register_literal_encoder
from tests.e2e.harness import Artifact


def pytest_collection_modifyitems(config, items):
    here = Path(__file__).parent
    for item in items:
        if here in Path(item.fspath).parents:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(autouse=True, scope="session")
def few_threads():
    """Cap intra-op threads: the models are tiny, and on a CPU-quota'd job (SLURM, CI)
    one thread per visible core is throttled into running several times slower."""
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """Pin every scenario to CPU so runs are deterministic and never touch a shared GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture(scope="session")
def encoder_name() -> str:
    """A weight-free encoder: the mean RGB of each tile / 8x8 patch it is given."""
    return register_literal_encoder()


@pytest.fixture
def encoded_images(monkeypatch) -> list[int]:
    """Counts images the encoder sees, so a scenario can prove a cache hit encoded nothing."""
    seen: list[int] = []
    for method in ("encode_tiles", "encode_tiles_dense"):
        original = getattr(_LiteralPatchEncoder, method)

        def counting(self, batch, _original=original):
            seen.append(int(batch.shape[0]))
            return _original(self, batch)

        monkeypatch.setattr(_LiteralPatchEncoder, method, counting)
    return seen


@pytest.fixture
def new_artifact(tmp_path_factory):
    """Start a named scenario artifact.

    Paths are recorded relative to pytest's base temp dir, which is what makes two
    sessions' artifacts comparable byte for byte.
    """
    return lambda scenario: Artifact(scenario, tmp_path_factory.getbasetemp())
