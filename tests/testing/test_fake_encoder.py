"""The ``fake`` encoder: :class:`soma.testing.FakeEncoder`'s weight-free pooling registered
with slide2vec, so extraction, caching and the live path run without a foundation model;
and ``fake-slide``, a weight-free slide encoder over it."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from hs2p import MaskSamplingConfig
from slide2vec import ExecutionOptions, PreprocessingConfig

import soma
from soma import ImageManifest
from soma.config import CacheOptions
from soma.testing import FakeEncoder, check_encoder_stage, synthetic_cohort


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


def test_importing_soma_registers_a_fake_slide_encoder_over_the_fake_encoder():
    entry = {entry["name"]: entry for entry in soma.components("encoders")}["fake-slide"]
    assert (entry["level"], entry["tile_encoder"], entry["default_spacing_um"]) == ("slide", "fake", 0.5)


def test_the_fake_slide_encoder_embeds_a_slide_as_the_mean_of_its_tile_embeddings(tmp_path: Path):
    records, _ = synthetic_cohort(tmp_path / "cohort", n=2)
    slides = ImageManifest.from_csv(records)
    tissue = MaskSamplingConfig(pixel_mapping={"background": 0, "tissue": 1}, min_coverage={"tissue": 0.1})

    def extract(encoder: str):
        return soma.extract(
            slides,
            encoder=encoder,
            extraction=PreprocessingConfig(masks=tissue),
            execution=ExecutionOptions(num_gpus=1, num_workers_per_gpu=0, num_preprocessing_workers=0, batch_size=64),
            cache=CacheOptions(enabled=False),
            output_dir=tmp_path / encoder,
        )

    tiles, slide = extract("fake"), extract("fake-slide")
    for sample_id in ("slide00", "slide01"):
        bag = tiles.load(sample_id)
        assert bag.ndim == 2 and bag.shape[0] > 1
        assert torch.allclose(slide.load(sample_id), bag.mean(dim=0), atol=1e-5)
