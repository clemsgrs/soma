"""Two deterministic CPU presets discovered by slide2vec's public entry point."""

import numpy as np
import torch
from slide2vec.encoders import TileEncoder, register_encoder


def _transform(image):
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class _FixtureEncoder(TileEncoder):
    sign = 1.0

    def __init__(self, *, output_variant=None):
        self._device = torch.device("cpu")

    @property
    def encode_dim(self):
        return 3

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self

    def get_transform(self):
        return _transform

    def encode_tiles(self, batch):
        return self.sign * batch.mean(dim=(-2, -1))


def register_presets():
    @register_encoder(
        "private-lab-fixture",
        output_variants={"default": {"encode_dim": 3}},
        default_output_variant="default",
        input_size=8,
        supports_variable_input_size=False,
        supported_spacing_um=0.5,
        precision="fp32",
        source="private fixture",
    )
    class PrivateLabFixture(_FixtureEncoder):
        pass
