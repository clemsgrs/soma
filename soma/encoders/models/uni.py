"""UNI and UNI2 encoder implementations."""

from __future__ import annotations

import timm
import timm.layers
import torch

from soma.encoders.base import TimmEncoder
from soma.encoders.registry import register_encoder


@register_encoder(
    "uni",
    encode_dim=1024,
    input_size=224,
    recommended_spacing_um=0.5,
    precision="fp16",
    source="MahmoodLab/UNI",
)
class UNI(TimmEncoder):
    def __init__(self, *, token: str | None = None):
        super().__init__(
            "hf-hub:MahmoodLab/UNI",
            token=token,
            init_values=1e-5,
            dynamic_img_size=True,
        )


@register_encoder(
    "uni2",
    encode_dim=1536,
    input_size=224,
    recommended_spacing_um=0.5,
    precision="fp16",
    source="MahmoodLab/UNI2-h",
)
class UNI2(TimmEncoder):
    def __init__(self, *, token: str | None = None):
        super().__init__(
            "hf-hub:MahmoodLab/UNI2-h",
            token=token,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=5.33334,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
