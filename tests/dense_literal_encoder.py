"""A weight-free, deterministic dense encoder for the #304 byte-identity gate.

The migration of the dense-over-given-images path onto ``Model.embed_images_dense``
replaces soma's own read → transform → pad → encode → write loop with slide2vec's. The
gate for that swap is byte identity against the pre-migration grids, which needs an
encoder whose output is a *literal* function of the pixels that reached it: a random
ViT would compare weights as much as geometry, and a mismatched read would still produce
plausible-looking floats.

``_LiteralPatchEncoder`` returns the mean RGB of each non-overlapping ``8x8`` patch, so a
grid cell is exactly the block of pixels it came from. Any difference in the reader
(pyramid level, downsample kernel, extent), the padding, or the window blending shows up
as a changed number rather than as noise. The same class backs both sides of the gate —
injected directly into the old loop when the baseline was captured, and resolved through
the encoder registry by ``Model.embed_images_dense`` afterwards — so the comparison is of
the paths, never of two encoders that merely agree in principle.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from slide2vec.encoders.base import TileEncoder

#: Registry name used by the migrated path (``Model.from_preset``).
LITERAL_ENCODER_NAME = "soma304-literal-patch"
#: Side length of the patch a grid cell averages, and the encoder's ``patch_size``.
LITERAL_PATCH_SIZE = 8
#: Channels of the literal grid: one per RGB channel.
LITERAL_ENCODE_DIM = 3


def literal_rgb_tensor(image: Image.Image) -> torch.Tensor:
    """The dense transform: raw RGB as ``(3, H, W)`` floats, no normalization, no resize.

    Deliberately not a real normalization — keeping the pixel values themselves is what
    makes a grid cell readable as "the mean of these pixels" and a reader regression
    legible as a wrong number.
    """
    pixels = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1)


class _LiteralPatchEncoder(TileEncoder):
    """Mean RGB per non-overlapping ``8x8`` patch — no weights, no randomness."""

    def __init__(self, **kwargs) -> None:
        self._device = torch.device("cpu")
        self._output_variant = kwargs.get("output_variant") or "default"

    @property
    def encode_dim(self) -> int:
        return LITERAL_ENCODE_DIM

    @property
    def patch_size(self) -> tuple[int, int]:
        return (LITERAL_PATCH_SIZE, LITERAL_PATCH_SIZE)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device) -> "_LiteralPatchEncoder":
        self._device = torch.device(device)
        return self

    def get_transform(self):
        return literal_rgb_tensor

    def get_normalization_transform(self):
        return literal_rgb_tensor

    def get_dense_transform(self):
        return literal_rgb_tensor

    def encode_tiles(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(-1, -2))

    def encode_tiles_dense(self, batch: torch.Tensor) -> torch.Tensor:
        size = LITERAL_PATCH_SIZE
        return batch.unfold(2, size, size).unfold(3, size, size).mean(dim=(-1, -2))

    def encode_tiles_attention(
        self,
        batch: torch.Tensor,
        *,
        blocks: tuple[int, ...] = (-1,),
        include_registers: bool = False,
    ) -> torch.Tensor:
        """A deterministic stand-in for per-head CLS attention: ``K`` scaled patch means.

        The real thing reads prefix-token self-attention out of chosen blocks; what the
        gate needs is only that the channel layout (``[block][cls, reg…][head]``) and the
        grid geometry survive the migration, so each requested block contributes one
        channel it can be identified by, plus a register channel when asked.
        """
        grid = self.encode_tiles_dense(batch)[:, :1]  # (B, 1, gh, gw), channel 0
        channels = []
        for block in blocks:
            channels.append(grid * float(int(block) + 2))
            if include_registers:
                channels.append(grid * float(int(block) + 2) * 0.5)
        return torch.cat(channels, dim=1)


def register_literal_encoder() -> str:
    """Register the literal encoder as a preset, idempotently; return its name.

    The migrated path reaches its encoder through ``Model.from_preset`` (and soma's cache
    key through ``resolve_patch_size``), both of which read the registry, so the gate has
    to put the same class there that the baseline was captured with.
    """
    from slide2vec.encoders.registry import encoder_registry

    if LITERAL_ENCODER_NAME not in encoder_registry:
        encoder_registry.register(
            LITERAL_ENCODER_NAME,
            _LiteralPatchEncoder,
            metadata={
                "level": "tile",
                "input_size": 32,
                "patch_size": LITERAL_PATCH_SIZE,
                "supports_variable_input_size": True,
                "output_variants": {"default": {"encode_dim": LITERAL_ENCODE_DIM}},
                "default_output_variant": "default",
                "supported_spacing_um": [0.25, 0.5],
                "default_spacing_um": 0.25,
                "precision": "fp32",
            },
        )
    return LITERAL_ENCODER_NAME
