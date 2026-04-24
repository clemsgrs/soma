from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

import slide2vec.inference as inference


def test_slide_aggregation_uses_autocast_for_fp16_precision():
    autocast_active = False
    original_autocast = inference.torch.autocast
    original_autocast_dtype = inference._autocast_dtype
    original_uses_cuda_runtime = inference._uses_cuda_runtime

    @contextmanager
    def fake_autocast(*, device_type: str, dtype):
        nonlocal autocast_active
        assert device_type == "cuda"
        assert dtype == torch.float16
        autocast_active = True
        try:
            yield
        finally:
            autocast_active = False

    def encode_slide(tile_features, coordinates, *, tile_size_lv0: int | None = None):
        assert autocast_active is True
        assert tile_features.shape == (1, 4)
        assert coordinates.shape == (1, 2)
        assert tile_size_lv0 == 224
        return torch.ones(4, dtype=torch.float32)

    inference.torch.autocast = fake_autocast
    inference._autocast_dtype = lambda torch_module, precision: torch_module.float16
    inference._uses_cuda_runtime = lambda device: True

    try:
        loaded = SimpleNamespace(device=torch.device("cpu"), model=SimpleNamespace(encode_slide=encode_slide))
        model = SimpleNamespace(level="slide", name="titan")
        slide = SimpleNamespace(sample_id="sample-1")
        tiling_result = SimpleNamespace(
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
        )
        tile_embeddings = torch.ones((1, 4), dtype=torch.float32)
        execution = SimpleNamespace(precision="fp16")

        slide_embedding, latents = inference._aggregate_tile_embeddings_for_slide(
            loaded,
            model,
            slide,
            tiling_result,
            tile_embeddings,
            preprocessing=None,
            execution=execution,
        )

        assert torch.equal(slide_embedding, torch.ones(4))
        assert latents is None
        assert autocast_active is False
    finally:
        inference.torch.autocast = original_autocast
        inference._autocast_dtype = original_autocast_dtype
        inference._uses_cuda_runtime = original_uses_cuda_runtime
