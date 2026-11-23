"""The backend probe of the tiling cache, through ``soma.extract`` on real slides.

With ``tiling.backend: auto``, a cached tiling is reused only while each slide and mask
would still be read by the reader its row records. soma opens the files to check this
(the backend probe) through the resolution hs2p's tiling uses.

The slides are tiled TIFFs with a resolution tag; their tissue masks are 8-bit TIFFs
without one, which a reader asked to open the mask beside its slide (with spacing
metadata required) refuses, while hs2p resolves the mask from the mask alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hs2p import FilterConfig
from slide2vec import ExecutionOptions

import soma
from soma.config import CacheOptions
from soma.data import ImageManifest
from tests.extraction_configs import extraction, tissue_masks

SPACING_UM = 0.5
SIZE_PX = 256


def _write_cohort(root: Path, sample_ids: list[str]) -> ImageManifest:
    """A slide (tagged RGB TIFF) and an untagged 8-bit tissue mask per sample."""
    import tifffile

    root.mkdir(parents=True, exist_ok=True)
    resolution = 1e4 / SPACING_UM
    rows = []
    for index, sample_id in enumerate(sample_ids):
        image = np.full((SIZE_PX, SIZE_PX, 3), 230, dtype=np.uint8)
        image[32:224, 32:224] = (150, 60 + 10 * index, 120)
        mask = np.zeros((SIZE_PX, SIZE_PX), dtype=np.uint8)
        mask[32:224, 32:224] = 1
        slide_path, mask_path = root / f"{sample_id}.tif", root / f"{sample_id}_mask.tif"
        tifffile.imwrite(
            slide_path,
            image,
            photometric="rgb",
            tile=(64, 64),
            resolution=(resolution, resolution),
            resolutionunit="CENTIMETER",
        )
        tifffile.imwrite(mask_path, mask, tile=(64, 64))
        rows.append({"sample_id": sample_id, "image_path": str(slide_path), "mask_path": str(mask_path)})
    pd.DataFrame(rows).to_csv(root / "images.csv", index=False)
    return ImageManifest.from_csv(root / "images.csv")


@pytest.fixture
def cached_extract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Run ``soma.extract`` under ``backend: auto`` on CPU with the weight-free ``fake``
    encoder, caching under one root; return what the run printed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    kwargs = dict(
        encoder="fake",
        extraction=extraction(
            SPACING_UM,
            32,
            seg_downsample=16,
            masks=tissue_masks(0.5),
            filter=FilterConfig(ref_tile_size=32, a_t=1),
            tiling_fields={"backend": "auto", "mask_backend": "auto"},
        ),
        execution=ExecutionOptions(
            num_gpus=1, num_workers_per_gpu=0, num_preprocessing_workers=0, precision="fp32", batch_size=256
        ),
        cache=CacheOptions(enabled=True, root_dir=tmp_path / "cache"),
    )

    def run(images: ImageManifest, name: str) -> str:
        capsys.readouterr()
        soma.extract(images, output_dir=tmp_path / name, **kwargs)
        return capsys.readouterr().out

    return run


def test_a_cold_run_tiles_without_probing_the_backends(tmp_path: Path, cached_extract) -> None:
    """With no cache entry, nothing needs the probe; right after tiling, hs2p has just
    written every row in this environment, so the entry is published unprobed."""
    images = _write_cohort(tmp_path / "slides", ["s0", "s1"])

    printed = cached_extract(images, "first")

    assert "tiling cache miss" in printed
    assert "✓ tiling cache populated" in printed
    assert "probing slide backends" not in printed


def _drop_a_coordinates_artifact(tmp_path: Path) -> None:
    (artifact,) = (tmp_path / "cache" / "tiling").glob("*/artifacts/*.coordinates.npz")
    artifact.unlink()


@pytest.mark.parametrize(
    ("second_samples", "damage", "reason"),
    [
        pytest.param(["s0", "s1"], None, "invalid tiling row for s1", id="a-slide-without-a-row"),
        pytest.param(["s0"], _drop_a_coordinates_artifact, "missing artifact for s0", id="a-missing-artifact"),
    ],
)
def test_an_entry_that_is_a_miss_anyway_is_tiled_again_without_probing_the_backends(
    tmp_path: Path, cached_extract, second_samples: list[str], damage, reason: str
) -> None:
    cached_extract(_write_cohort(tmp_path / "slides", ["s0"]), "first")
    if damage is not None:
        damage(tmp_path)

    printed = cached_extract(_write_cohort(tmp_path / "slides", second_samples), "second")

    assert f"({reason})" in printed
    assert "✓ tiling cache populated" in printed
    assert "probing slide backends" not in printed


def test_a_slide_whose_mask_its_reader_cannot_open_beside_it_is_a_hit_on_the_next_run(
    tmp_path: Path, cached_extract
) -> None:
    """hs2p's tiling resolves each slide's reader from the slide alone and the mask's
    from the mask alone. The probe asks the same question, so the rows the first run's
    tiling wrote are a hit on the second run instead of a re-tiling on every run."""
    images = _write_cohort(tmp_path / "slides", ["s0", "s1"])
    cached_extract(images, "first")

    second = cached_extract(images, "second")

    assert "✓ tiling cache hit" in second
    assert "tiling cache miss" not in second


def test_a_hit_under_auto_prints_the_probe_counter(tmp_path: Path, cached_extract) -> None:
    images = _write_cohort(tmp_path / "slides", ["s0", "s1"])
    cached_extract(images, "first")

    lines = cached_extract(images, "second").splitlines()

    assert "… probing slide backends: 0/2" in lines
    assert "✓ probed slide backends: 2/2" in lines
    hit = next(index for index, line in enumerate(lines) if line.startswith("✓ tiling cache hit"))
    assert lines.index("✓ probed slide backends: 2/2") < hit
