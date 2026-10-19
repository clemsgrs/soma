"""A user's tile encoder in a module file of its own, which a test edits between runs.

slide2vec fingerprints an encoder by the file that defines its class, so a test that
checks what soma does after an edit to an encoder needs that class in a real file. The
module registers itself with ``slide2vec.register_encoder`` on import, as a user's
``my_encoder.py`` does.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
from slide2vec.encoders import encoder_registry

__all__ = ["UserEncoderModule", "notebook_encoder"]

#: The module's source. A tile embeds as its mean normalised RGB times ``SCALE``; each
#: 8x8 block of a dense grid likewise. ``ENCODED`` counts the tiles and images encoded.
_SOURCE = '''"""A user's tile encoder."""

import torch
import torch.nn.functional as F
from slide2vec import TileEncoder, register_encoder
from torchvision.transforms import v2

SCALE = 1.0
ENCODED = []


@register_encoder(
    "{name}",
    encode_dim=3,
    input_size=32,
    patch_size=8,
    supports_variable_input_size=True,
    supported_spacing_um=[0.25, 0.5],
    precision="fp32",
)
class UserTileEncoder(TileEncoder):
    def __init__(self, *, output_variant=None):
        self._device = torch.device("cpu")

    @property
    def encode_dim(self):
        return 3

    @property
    def patch_size(self):
        return (8, 8)

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self

    def get_transform(self):
        return v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[0.5] * 3, std=[0.25] * 3)]
        )

    def get_normalization_transform(self):
        return self.get_transform()

    def encode_tiles(self, batch):
        ENCODED.append(int(batch.shape[0]))
        return SCALE * batch.float().mean(dim=(-2, -1))

    def encode_tiles_dense(self, batch):
        ENCODED.append(int(batch.shape[0]))
        return SCALE * F.avg_pool2d(batch.float(), kernel_size=8, stride=8)

    def encode_tiles_dense_prenorm(self, batch):
        return self.encode_tiles_dense(batch)
'''


class UserEncoderModule:
    """``<module>.UserTileEncoder`` in ``<directory>/<module>.py``, registered as ``name``.

    ``edit()`` changes ``SCALE`` in the file, so the encoder extracts other features, then
    imports the module again, as a new session would. The registry and ``sys.modules`` are
    restored when the test ends.
    """

    def __init__(self, directory: Path, request: pytest.FixtureRequest, *, module: str, name: str) -> None:
        self.name = name
        self.module_name = module
        self.object = f"{module}.UserTileEncoder"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{module}.py"
        self.source = _SOURCE.replace("{name}", name)
        self._monkeypatch = monkeypatch = request.getfixturevalue("monkeypatch")
        encoder_registry.names()  # discover the built-in encoders before copying the entries
        self._entries = dict(encoder_registry._entries)
        monkeypatch.syspath_prepend(str(directory))
        request.addfinalizer(lambda: sys.modules.pop(module, None))
        self.path.write_text(self.source)
        self._load()

    @property
    def fingerprint(self) -> str:
        """The SHA-256 of the module file, as slide2vec and the run record compute it."""
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    @property
    def encoded(self) -> int:
        """How many tiles or images the current module has encoded."""
        return sum(sys.modules[self.module_name].ENCODED)

    def edit(self) -> None:
        # Longer than the original, so Python cannot load stale bytecode for it.
        self.source = self.source.replace("SCALE = 1.0", "SCALE = 2.0  # edited")
        self.path.write_text(self.source)
        self._load()

    def _load(self) -> None:
        self._monkeypatch.setattr(encoder_registry, "_entries", dict(self._entries))
        importlib.invalidate_caches()
        module = sys.modules.get(self.module_name)
        if module is None:
            importlib.import_module(self.module_name)
        else:
            importlib.reload(module)


def notebook_encoder(request: pytest.FixtureRequest, *, module: str, name: str) -> ModuleType:
    """The same encoder defined in a notebook cell: its module has no file. Registered
    as ``name`` until the test ends."""
    monkeypatch = request.getfixturevalue("monkeypatch")
    encoder_registry.names()  # discover the built-in encoders before copying the entries
    monkeypatch.setattr(encoder_registry, "_entries", dict(encoder_registry._entries))
    cell = ModuleType(module)
    monkeypatch.setitem(sys.modules, module, cell)
    exec(compile(_SOURCE.replace("{name}", name), "<notebook cell>", "exec"), cell.__dict__)
    return cell
