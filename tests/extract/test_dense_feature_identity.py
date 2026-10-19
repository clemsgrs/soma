"""A dense grid cache is reused only for the feature identity it was extracted with.

Both dense kinds, grids over pre-cropped images (``dense_image``) and grids over ROIs
sampled from slides (``dense``), record slide2vec's feature identity in
``cache_metadata.json``. soma checks the encoder's code on every hit, and compares the
whole identity after a slide2vec upgrade. These scenarios run real hs2p sampling and
slide2vec extraction with weight-free encoders, and simulate an upgrade by changing the
installed slide2vec version.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch
from slide2vec import DenseOptions, ExecutionOptions

import soma
from soma.config import CacheOptions
from soma.data import AnnotationManifest, ImageManifest
from soma.extract.cache import CacheFeatureIdentityMismatch
from tests.e2e.synthetic import write_tiff
from tests.extraction_configs import annotation_masks, extraction
from tests.user_encoders import UserEncoderModule

KINDS = ("dense_image", "dense")
USER_ENCODER = "soma652-user-dense"
#: Blue and magenta: each channel normalises to -2 (0) or 2 (255) under mean 0.5, std 0.25.
_COLOURS = {"blue": (0, 0, 255), "magenta": (255, 0, 255)}
_EXECUTION = ExecutionOptions(
    num_gpus=1, num_workers_per_gpu=0, num_preprocessing_workers=0, precision="fp32", batch_size=4
)


def _write_pyramid(path: Path, array: np.ndarray, *, photometric: str) -> None:
    """A two-level tiled TIFF at 0.5 µm/px, as hs2p and slide2vec read a slide."""
    resolution = 1e4 / 0.5
    options = dict(photometric=photometric, tile=(32, 32), resolution=(resolution, resolution), resolutionunit="CENTIMETER")
    with tifffile.TiffWriter(path) as writer:
        writer.write(array, subifds=1, **options)
        writer.write(array[::2, ::2], subfiletype=1, **options)


@pytest.fixture(scope="module")
def dense_data(tmp_path_factory) -> dict[str, Path]:
    """One flat blue and one flat magenta sample of each kind: a 32 px image, and a 64 px
    slide whose annotation covers it whole (four 32 px ROIs)."""
    root = tmp_path_factory.mktemp("dense-data")
    images, slides = [], []
    for name, colour in _COLOURS.items():
        image_path = root / f"{name}.tif"
        write_tiff(image_path, np.full((32, 32, 3), colour, np.uint8))
        images.append({"sample_id": name, "image_path": str(image_path)})
        slide_path, mask_path = root / f"{name}-slide.tif", root / f"{name}-mask.tif"
        _write_pyramid(slide_path, np.full((64, 64, 3), colour, np.uint8), photometric="rgb")
        _write_pyramid(mask_path, np.ones((64, 64), np.uint8), photometric="minisblack")
        slides.append({"sample_id": name, "image_path": str(slide_path), "label_mask_path": str(mask_path)})
    pd.DataFrame(images).to_csv(root / "images.csv", index=False)
    pd.DataFrame(slides).to_csv(root / "slides.csv", index=False)
    return {"dense_image": root / "images.csv", "dense": root / "slides.csv"}


def extract_grids(kind: str, data: dict[str, Path], root: Path, *, encoder: str, **cache_settings):
    """Extract ``kind`` grids at 32 px and 0.5 µm/px into ``<root>/cache``."""
    common = dict(
        encoder=encoder,
        dense=DenseOptions(spacing_um=0.5, target_size=32),
        execution=_EXECUTION,
        cache=CacheOptions(enabled=True, root_dir=root / "cache", **cache_settings),
        output_dir=root / "run",
    )
    if kind == "dense_image":
        return soma.extract(ImageManifest.from_csv(data[kind]), unit="tile", **common)
    return soma.extract(
        ImageManifest.from_csv(data[kind]),
        annotations=AnnotationManifest.from_csv(data[kind]),
        extraction=extraction(0.5, 32, masks=annotation_masks({"background": 0, "tumor": 1}, {"tumor": 0.0})),
        **common,
    )


def cache_metadata(root: Path, kind: str) -> dict:
    (metadata_path,) = (root / "cache" / kind).glob("*/cache_metadata.json")
    return json.loads(metadata_path.read_text())


def _expected_grids(source, *, scale: float) -> dict[str, torch.Tensor]:
    """Every grid cell is its sample's normalised colour times ``scale``."""
    return {
        sample_id: scale
        * torch.tensor([value / 255 * 4 - 2 for value in _COLOURS[sample_id.split("__")[0]]])
        .view(3, 1, 1)
        .expand(3, 4, 4)
        for sample_id in source.sample_ids
    }


def _assert_grids(source, expected: dict[str, torch.Tensor]) -> None:
    assert sorted(source.sample_ids) == sorted(expected)
    for sample_id, grid in expected.items():
        assert torch.equal(source.load(sample_id), grid), sample_id


@pytest.fixture
def user_encoder(tmp_path, request, monkeypatch) -> UserEncoderModule:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    return UserEncoderModule(tmp_path / "user_code", request, module="soma652_user_dense", name=USER_ENCODER)


# --- The encoder's code, checked on every hit ---------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_dense_cache_is_reused_until_its_encoder_file_is_edited(tmp_path, dense_data, user_encoder, kind):
    extracted = extract_grids(kind, dense_data, tmp_path, encoder=USER_ENCODER)
    _assert_grids(extracted, _expected_grids(extracted, scale=1.0))
    encoded = user_encoder.encoded
    original = user_encoder.fingerprint

    reused = extract_grids(kind, dense_data, tmp_path, encoder=USER_ENCODER)

    assert user_encoder.encoded == encoded
    _assert_grids(reused, _expected_grids(reused, scale=1.0))

    user_encoder.edit()

    with pytest.raises(CacheFeatureIdentityMismatch) as refused:
        extract_grids(kind, dense_data, tmp_path, encoder=USER_ENCODER)
    message = str(refused.value)
    assert f"encoder '{USER_ENCODER}'" in message and user_encoder.object in message
    assert original[:12] in message and user_encoder.fingerprint[:12] in message
    assert user_encoder.encoded == 0
