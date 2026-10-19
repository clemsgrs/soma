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
import slide2vec
import tifffile
import torch
import torch.nn.functional as F
from slide2vec import DenseOptions, ExecutionOptions, TileEncoder
from slide2vec.encoders import encoder_registry
from torchvision.transforms import v2

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
#: One sample per batch, so an interrupted extraction can leave some grids written.
_EXECUTION = ExecutionOptions(
    num_gpus=1, num_workers_per_gpu=0, num_preprocessing_workers=0, precision="fp32", batch_size=1
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


@pytest.mark.parametrize("kind", KINDS)
def test_reextract_setting_rebuilds_a_dense_cache_after_an_encoder_edit(tmp_path, dense_data, user_encoder, kind):
    extract_grids(kind, dense_data, tmp_path, encoder=USER_ENCODER)
    encoded = user_encoder.encoded

    user_encoder.edit()
    rebuilt = extract_grids(kind, dense_data, tmp_path, encoder=USER_ENCODER, on_identity_mismatch="reextract")

    assert user_encoder.encoded == encoded
    _assert_grids(rebuilt, _expected_grids(rebuilt, scale=2.0))
    recorded = cache_metadata(tmp_path, kind)["feature_identity"]
    assert recorded["slide2vec_version"] == slide2vec.__version__
    assert recorded["identity"]["encoder_code"] == {
        "object": user_encoder.object,
        "fingerprint": user_encoder.fingerprint,
        "source": "",
    }


# --- After a slide2vec upgrade ------------------------------------------------------------

SWITCHABLE_ENCODER = "soma652-switchable-dense"
_ORIGINAL_MEAN = [0.5, 0.5, 0.5]
_CHANGED_MEAN = [0.25, 0.5, 0.5]
_UPGRADED_VERSION = "999.0.0"


class _SwitchableDenseEncoder(TileEncoder):
    """Each 8x8 block of a grid is its mean normalised RGB; the test changes the mean."""

    mean: list[float] = list(_ORIGINAL_MEAN)
    instances = 0
    encoded = 0

    def __init__(self, **kwargs) -> None:
        self._device = torch.device("cpu")
        type(self).instances += 1

    @property
    def encode_dim(self) -> int:
        return 3

    @property
    def patch_size(self) -> tuple[int, int]:
        return (8, 8)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self

    def get_transform(self):
        return v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=list(type(self).mean), std=[0.25] * 3)]
        )

    def get_normalization_transform(self):
        return self.get_transform()

    def encode_tiles(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.float().mean(dim=(-2, -1))

    def encode_tiles_dense(self, batch: torch.Tensor) -> torch.Tensor:
        type(self).encoded += int(batch.shape[0])
        return F.avg_pool2d(batch.float(), kernel_size=8, stride=8)


@pytest.fixture
def switchable(monkeypatch) -> type[_SwitchableDenseEncoder]:
    """The switchable encoder, with its original transform and fresh counters, on CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    encoder_registry.names()  # discover the built-in encoders before copying the entries
    monkeypatch.setattr(encoder_registry, "_entries", dict(encoder_registry._entries))
    encoder_registry.register(
        SWITCHABLE_ENCODER,
        _SwitchableDenseEncoder,
        metadata={
            "level": "tile",
            "input_size": 32,
            "patch_size": 8,
            "supports_variable_input_size": True,
            "output_variants": {"default": {"encode_dim": 3}},
            "default_output_variant": "default",
            "supported_spacing_um": [0.25, 0.5],
            "default_spacing_um": 0.5,
            "precision": "fp32",
            "source": "",
        },
    )
    monkeypatch.setattr(_SwitchableDenseEncoder, "mean", list(_ORIGINAL_MEAN))
    monkeypatch.setattr(_SwitchableDenseEncoder, "instances", 0)
    monkeypatch.setattr(_SwitchableDenseEncoder, "encoded", 0)
    return _SwitchableDenseEncoder


def _upgrade_slide2vec(monkeypatch, encoder, *, mean: list[float]) -> None:
    """Install a newer slide2vec in which the encoder normalizes with ``mean``."""
    monkeypatch.setattr(slide2vec, "__version__", _UPGRADED_VERSION)
    monkeypatch.setattr(encoder, "mean", list(mean))


@pytest.mark.parametrize("kind", KINDS)
def test_dense_cache_is_refused_after_an_upgrade_changed_the_transform(
    tmp_path, dense_data, switchable, monkeypatch, kind
):
    extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)
    encoded = switchable.encoded

    _upgrade_slide2vec(monkeypatch, switchable, mean=_CHANGED_MEAN)

    with pytest.raises(CacheFeatureIdentityMismatch, match=r"transform\.normalize\.mean"):
        extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)
    assert switchable.encoded == encoded


@pytest.mark.parametrize("kind", KINDS)
def test_reextract_setting_rebuilds_a_dense_cache_after_an_upgrade_changed_the_transform(
    tmp_path, dense_data, switchable, monkeypatch, kind
):
    extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)

    _upgrade_slide2vec(monkeypatch, switchable, mean=_CHANGED_MEAN)
    rebuilt = extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER, on_identity_mismatch="reextract")

    # Red normalises with mean 0.25: 0 -> -1, 255 -> 3.
    expected = _expected_grids(rebuilt, scale=1.0)
    for grid in expected.values():
        grid[0] = grid[0] + 1
    _assert_grids(rebuilt, expected)
    recorded = cache_metadata(tmp_path, kind)["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _CHANGED_MEAN


@pytest.mark.parametrize("kind", KINDS)
def test_dense_cache_verified_after_an_upgrade_hits_cheaply_from_then_on(
    tmp_path, dense_data, switchable, monkeypatch, kind
):
    extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)
    encoded = switchable.encoded
    instances = switchable.instances

    _upgrade_slide2vec(monkeypatch, switchable, mean=_ORIGINAL_MEAN)
    reused = extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)

    # The upgrade kept the identity: the cache is reused, at the cost of one encoder load.
    assert switchable.encoded == encoded
    assert switchable.instances == instances + 1
    _assert_grids(reused, _expected_grids(reused, scale=1.0))
    assert cache_metadata(tmp_path, kind)["feature_identity"]["slide2vec_version"] == _UPGRADED_VERSION

    extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)

    assert switchable.instances == instances + 1


# --- The record's states ----------------------------------------------------------------


def _drop_record(root: Path, kind: str) -> None:
    """Rewrite a dense cache's metadata as soma wrote it before dense caches recorded one."""
    (metadata_path,) = (root / "cache" / kind).glob("*/cache_metadata.json")
    metadata = json.loads(metadata_path.read_text())
    del metadata["feature_identity"]
    metadata_path.write_text(json.dumps(metadata))


def interrupt(extract, *, written_batches: int) -> None:
    """Run ``extract`` with an encoder that fails after ``written_batches`` batches."""
    encode = _SwitchableDenseEncoder.encode_tiles_dense
    batches = 0

    def fail_later(self, batch):
        nonlocal batches
        batches += 1
        if batches > written_batches:
            raise RuntimeError("interrupted")
        return encode(self, batch)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_SwitchableDenseEncoder, "encode_tiles_dense", fail_later)
        with pytest.raises(Exception, match="interrupted"):
            extract()


@pytest.mark.parametrize("kind", KINDS)
def test_dense_cache_records_the_identity_slide2vec_wrote(tmp_path, dense_data, switchable, kind):
    extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)

    recorded = cache_metadata(tmp_path, kind)["feature_identity"]

    assert recorded["slide2vec_version"] == slide2vec.__version__
    assert recorded["identity"]["encoder_name"] == SWITCHABLE_ENCODER
    assert recorded["identity"]["transform"]["normalize"] == {"mean": _ORIGINAL_MEAN, "std": [0.25] * 3}


@pytest.mark.parametrize("kind", KINDS)
def test_dense_cache_with_grids_and_no_record_is_refused(tmp_path, dense_data, switchable, kind):
    extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)
    _drop_record(tmp_path, kind)
    encoded = switchable.encoded

    with pytest.raises(CacheFeatureIdentityMismatch) as refused:
        extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)
    message = str(refused.value)
    (cache_dir,) = (tmp_path / "cache" / kind).iterdir()
    assert str(cache_dir) in message
    assert "predates the record" in message
    # Never deleted silently: the grids stay for the user to decide.
    assert switchable.encoded == encoded
    assert any(cache_dir.rglob("*.pt"))


@pytest.mark.parametrize("kind", KINDS)
def test_reextract_setting_rebuilds_a_dense_cache_with_no_record(tmp_path, dense_data, switchable, kind):
    extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)
    _drop_record(tmp_path, kind)
    encoded = switchable.encoded

    rebuilt = extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER, on_identity_mismatch="reextract")

    assert switchable.encoded == 2 * encoded
    _assert_grids(rebuilt, _expected_grids(rebuilt, scale=1.0))
    recorded = cache_metadata(tmp_path, kind)["feature_identity"]
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _ORIGINAL_MEAN


@pytest.mark.parametrize("kind", KINDS)
def test_empty_dense_cache_without_a_record_starts_one_its_first_commit_completes(
    tmp_path, dense_data, switchable, kind
):
    interrupt(lambda: extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER), written_batches=0)
    _drop_record(tmp_path, kind)

    extracted = extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)

    _assert_grids(extracted, _expected_grids(extracted, scale=1.0))
    recorded = cache_metadata(tmp_path, kind)["feature_identity"]
    assert recorded["slide2vec_version"] == slide2vec.__version__
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _ORIGINAL_MEAN


@pytest.mark.parametrize("kind", KINDS)
def test_dense_extraction_interrupted_before_an_upgrade_restarts_with_the_current_recipe(
    tmp_path, dense_data, switchable, monkeypatch, kind
):
    interrupt(lambda: extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER), written_batches=1)
    assert cache_metadata(tmp_path, kind)["feature_identity"]["identity"] is None

    _upgrade_slide2vec(monkeypatch, switchable, mean=_CHANGED_MEAN)
    resumed = extract_grids(kind, dense_data, tmp_path, encoder=SWITCHABLE_ENCODER)

    expected = _expected_grids(resumed, scale=1.0)
    for grid in expected.values():
        grid[0] = grid[0] + 1
    _assert_grids(resumed, expected)
    recorded = cache_metadata(tmp_path, kind)["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _CHANGED_MEAN
