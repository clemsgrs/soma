"""The ``fake`` encoder: :class:`soma.testing.FakeEncoder`'s weight-free pooling registered
with slide2vec, so extraction, caching and the live path run without a foundation model."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from slide2vec import ExecutionOptions

import soma
from soma import ImageManifest
from soma.config import CacheOptions
from soma.testing import FakeEncoder, check_encoder_stage


def test_importing_soma_registers_the_fake_encoder():
    entry = {entry["name"]: entry for entry in soma.components("encoders")}["fake"]
    assert (entry["level"], entry["patch_size"], entry["default_spacing_um"]) == ("tile", 8, 0.5)


def test_the_fake_encoder_embeds_a_tile_as_its_mean_normalised_colour(tmp_path: Path):
    # Normalised as FakeEncoder normalises its pixels: (x / 255 - 0.5) / 0.25.
    Image.new("RGB", (32, 32), (51, 102, 204)).save(tmp_path / "tile.png")
    (tmp_path / "dataset.csv").write_text(f"sample_id,image_path\nt0,{tmp_path / 'tile.png'}\n")

    source = soma.extract(
        ImageManifest.from_csv(tmp_path / "dataset.csv"),
        unit="tile",
        encoder="fake",
        execution=ExecutionOptions(num_gpus=1, num_workers_per_gpu=0, batch_size=1),
        cache=CacheOptions(enabled=False),
        output_dir=tmp_path / "output",
    )

    assert torch.allclose(source.load("t0"), torch.tensor([-1.2, -0.4, 1.2]), atol=1e-6)


def test_the_fake_encoder_is_the_fake_encoder_stage_on_the_live_path():
    stage = soma.extract.live_encoder("fake", target_size=64, spacing_um=0.5, device="cpu")
    check_encoder_stage(stage)

    pixels = torch.randint(0, 256, (2, 3, 64, 64), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    expected = FakeEncoder(target_size=64, patch_size=8)(pixels)
    with torch.inference_mode():
        assert torch.allclose(stage(pixels).cpu(), expected, atol=1e-5)
