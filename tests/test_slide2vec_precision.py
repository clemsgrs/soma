from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

import slide2vec.runtime.slide_encode as slide_encode
from slide2vec.runtime.embedding_pipeline import aggregate_tile_embeddings_for_slide


def test_slide_aggregation_uses_autocast_for_fp16_precision():
    autocast_active = False
    prepared_spacings: list[tuple[float, float]] = []
    original_autocast = slide_encode.torch.autocast
    original_autocast_dtype = slide_encode.autocast_dtype
    original_uses_cuda_runtime = slide_encode.uses_cuda_runtime

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

    def prepare_coordinates(coordinates, *, base_spacing_um: float, requested_spacing_um: float):
        # slide2vec >=5.6.0 rescales coordinates through this encoder hook (it replaced the
        # by-name ``prov-gigapath`` branch that used to live in the aggregation path), so the
        # double has to honour the contract — and record that it was reached.
        prepared_spacings.append((base_spacing_um, requested_spacing_um))
        return coordinates

    slide_encode.torch.autocast = fake_autocast
    slide_encode.autocast_dtype = lambda torch_module, precision: torch_module.float16
    slide_encode.uses_cuda_runtime = lambda device: True

    try:
        loaded = SimpleNamespace(
            device=torch.device("cpu"),
            model=SimpleNamespace(
                encode_slide=encode_slide,
                prepare_coordinates=prepare_coordinates,
            ),
        )
        model = SimpleNamespace(level="slide", name="titan")
        slide = SimpleNamespace(sample_id="sample-1")
        tiling_result = SimpleNamespace(
            x=np.array([0], dtype=np.int64),
            y=np.array([0], dtype=np.int64),
            tile_size_lv0=224,
            base_spacing_um=0.25,
            requested_spacing_um=0.5,
        )
        tile_embeddings = torch.ones((1, 4), dtype=torch.float32)
        execution = SimpleNamespace(precision="fp16")

        slide_embedding, latents = aggregate_tile_embeddings_for_slide(
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
        assert prepared_spacings == [(0.25, 0.5)]
    finally:
        slide_encode.torch.autocast = original_autocast
        slide_encode.autocast_dtype = original_autocast_dtype
        slide_encode.uses_cuda_runtime = original_uses_cuda_runtime
