"""A pooled feature cache is reused only for the feature identity it was extracted with.

The cache key does not cover the encoder's image transform, so a slide2vec upgrade that
changes an encoder's preprocessing used to leave the key unchanged and the old features in
use (issue #512). These scenarios run real slide2vec extraction with a weight-free encoder
whose transform the test can change, and simulate the upgrade by changing the installed
slide2vec version.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import slide2vec
import torch
from PIL import Image
from slide2vec.encoders.base import PatientEncoder, SlideEncoder, TileEncoder
from slide2vec.encoders.registry import encoder_registry
from torchvision.transforms import v2

from soma.cache import CacheFeatureIdentityMismatch
from soma.config import CacheConfig, EncoderConfig, ExecutionConfig, PreprocessingConfig
from soma.dataset import Dataset, TileDataset
from soma.extraction import FeatureExtractor
from tests.e2e.synthetic import write_tiff

STUB_ENCODER = "soma512-switchable-transform"
STUB_SLIDE_ENCODER = "soma512-slide"
STUB_PATIENT_ENCODER = "soma512-patient"
_ORIGINAL_MEAN = [0.5, 0.5, 0.5]
_CHANGED_MEAN = [0.25, 0.5, 0.5]
_STD = [0.5, 0.5, 0.5]
_UPGRADED_VERSION = "999.0.0"

#: Whole-slide cache kinds, and the encoder whose extraction fills each.
_SLIDE_KINDS = {
    "tile": STUB_ENCODER,
    "hierarchical": STUB_ENCODER,
    "slide": STUB_SLIDE_ENCODER,
    "patient": STUB_PATIENT_ENCODER,
}


class _StubEncoder:
    def __init__(self, **kwargs) -> None:
        self._device = torch.device("cpu")
        self._output_variant = kwargs.get("output_variant") or "default"

    @property
    def encode_dim(self) -> int:
        return 3

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device):
        self._device = torch.device(device)
        return self


class _SwitchableTransformEncoder(_StubEncoder, TileEncoder):
    """Mean RGB of the normalized image; the normalization mean is the test's to change."""

    mean: list[float] = list(_ORIGINAL_MEAN)
    instances = 0
    encoded_images = 0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        type(self).instances += 1

    def get_transform(self):
        return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=list(type(self).mean), std=list(_STD)),
            ]
        )

    def get_normalization_transform(self):
        return self.get_transform()

    def encode_tiles(self, batch: torch.Tensor) -> torch.Tensor:
        type(self).encoded_images += int(batch.shape[0])
        return batch.mean(dim=(-1, -2))


class _MeanSlideEncoder(_StubEncoder, SlideEncoder):
    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0=None):
        return tile_features.mean(dim=0)


class _MeanPatientEncoder(_StubEncoder, PatientEncoder):
    def encode_slide(self, tile_features, coordinates=None, *, tile_size_lv0=None):
        return tile_features.mean(dim=0)

    def encode_patient(self, slide_embeddings):
        return slide_embeddings.mean(dim=0)


def _register_stub_encoders() -> None:
    common = {
        "output_variants": {"default": {"encode_dim": 3}},
        "default_output_variant": "default",
        "supported_spacing_um": [0.5],
        "default_spacing_um": 0.5,
        "precision": "fp32",
    }
    aggregates = {"tile_encoder": STUB_ENCODER, "tile_encoder_output_variant": "default"}
    for name, cls, metadata in (
        (STUB_ENCODER, _SwitchableTransformEncoder, {"level": "tile", "input_size": 32}),
        (STUB_SLIDE_ENCODER, _MeanSlideEncoder, {"level": "slide", **aggregates}),
        (STUB_PATIENT_ENCODER, _MeanPatientEncoder, {"level": "patient", **aggregates}),
    ):
        if name not in encoder_registry:
            encoder_registry.register(name, cls, metadata={**metadata, **common})


@pytest.fixture
def encoder(monkeypatch) -> type[_SwitchableTransformEncoder]:
    """The stub tile encoder, with its original transform and fresh counters, on CPU."""
    _register_stub_encoders()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(_SwitchableTransformEncoder, "mean", list(_ORIGINAL_MEAN))
    monkeypatch.setattr(_SwitchableTransformEncoder, "instances", 0)
    monkeypatch.setattr(_SwitchableTransformEncoder, "encoded_images", 0)
    return _SwitchableTransformEncoder


def _upgrade_slide2vec(monkeypatch, encoder, *, mean: list[float]) -> None:
    """Install a newer slide2vec in which the stub encoder normalizes with ``mean``."""
    monkeypatch.setattr(slide2vec, "__version__", _UPGRADED_VERSION)
    monkeypatch.setattr(encoder, "mean", list(mean))


def _image_colour(index: int) -> tuple[int, int, int]:
    return (40 * index, 90, 160)


def _expected_image_features(n: int, *, mean: list[float]) -> dict[str, torch.Tensor]:
    """Features of the first ``n`` images under ``mean``, worked out from their colours.

    An image is one flat colour, so its mean normalized RGB is that colour normalized.
    """
    return {
        f"s{index}": torch.tensor(
            [(value / 255 - m) / std for value, m, std in zip(_image_colour(index), mean, _STD)]
        )
        for index in range(n)
    }


def _image_dataset(root: Path, n: int = 4) -> TileDataset:
    """The first ``n`` images of one cohort: datasets of different ``n`` share samples."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(n):
        image_path = root / f"s{index}.png"
        Image.new("RGB", (32, 32), color=_image_colour(index)).save(image_path)
        rows.append({"sample_id": f"s{index}", "image_path": str(image_path), "label": index % 2})
    dataset_csv = root / f"dataset_{n}.csv"
    pd.DataFrame(rows).to_csv(dataset_csv, index=False)
    return TileDataset(dataset_csv)


def _extract_images(dataset: TileDataset, root: Path, **cache_settings):
    return FeatureExtractor(
        dataset,
        EncoderConfig(name=STUB_ENCODER, precision="fp32", batch_size=2),
        execution=ExecutionConfig(num_gpus=1, num_workers_per_gpu=0),
        cache=CacheConfig(enabled=True, root_dir=root / "cache", **cache_settings),
        output_root=root / "run",
    ).extract()


def _cache_dir(root: Path, kind: str) -> Path:
    (cache_dir,) = (root / "cache" / kind).iterdir()
    return cache_dir


def _cache_metadata(root: Path, kind: str) -> dict:
    return json.loads((_cache_dir(root, kind) / "cache_metadata.json").read_text())


def _forget_identity(root: Path, kind: str) -> None:
    """Rewrite a cache as soma 1.17 and slide2vec < 6.3 left it: no identity anywhere."""
    cache_dir = _cache_dir(root, kind)
    metadata_path = cache_dir / "cache_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["feature_identity"]
    metadata_path.write_text(json.dumps(metadata))
    for sidecar_path in cache_dir.glob("*/*.meta.json"):
        sidecar = json.loads(sidecar_path.read_text())
        del sidecar["compatibility"]
        sidecar_path.write_text(json.dumps(sidecar))


def _features(result) -> dict[str, torch.Tensor]:
    return {
        sample_id: result.source.load(sample_id)
        for sample_id in sorted(result.source.available_samples)
    }


def _assert_same_features(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]):
    assert actual.keys() == expected.keys()
    for sample_id, feature in expected.items():
        assert torch.equal(actual[sample_id], feature), sample_id


def _assert_expected_features(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]):
    """Compare with features worked out by hand, up to float32 rounding."""
    assert actual.keys() == expected.keys()
    for sample_id, feature in expected.items():
        assert actual[sample_id].shape[-1] == feature.shape[-1], sample_id
        assert torch.allclose(actual[sample_id], feature.expand_as(actual[sample_id]), atol=1e-5), (
            sample_id
        )


def _expected_slide_features(kind: str, *, changed: bool) -> dict[str, torch.Tensor]:
    """Literal mean RGB features of blue, magenta, green, and yellow slides.

    Each 256px slide contains 64 tiles of 32px, or 16 regions of four tiles.
    Patients p0/p1 average the first/last pair of slides. No output from
    extraction determines the expected feature values or tensor shapes.
    """
    if kind == "patient":
        values = (
            {"p0": [0.5, -1.0, 1.0], "p1": [0.5, 1.0, -1.0]}
            if changed
            else {"p0": [0.0, -1.0, 1.0], "p1": [0.0, 1.0, -1.0]}
        )
    elif changed:
        values = {
            "slide00": [-0.5, -1.0, 1.0],
            "slide01": [1.5, -1.0, 1.0],
            "slide02": [-0.5, 1.0, -1.0],
            "slide03": [1.5, 1.0, -1.0],
        }
    else:
        values = {
            "slide00": [-1.0, -1.0, 1.0],
            "slide01": [1.0, -1.0, 1.0],
            "slide02": [-1.0, 1.0, -1.0],
            "slide03": [1.0, 1.0, -1.0],
        }
    shape = {"tile": (64, 3), "hierarchical": (16, 4, 3), "slide": (3,), "patient": (3,)}[kind]
    return {sample_id: torch.tensor(vector).expand(shape) for sample_id, vector in values.items()}


def _unverifiable_warnings(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if "records no feature identity" in record.getMessage()
    ]


# --- Pre-cropped images (the path issue #512 measured) ---------------------------------


def test_image_extraction_records_the_identity_slide2vec_wrote(tmp_path, encoder):
    _extract_images(_image_dataset(tmp_path / "data"), tmp_path)

    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]

    assert recorded["slide2vec_version"] == slide2vec.__version__
    assert recorded["identity"]["encoder_name"] == STUB_ENCODER
    assert recorded["identity"]["transform"]["normalize"] == {
        "mean": _ORIGINAL_MEAN,
        "std": [0.5, 0.5, 0.5],
    }


@pytest.mark.parametrize("cached_samples", [4, 3], ids=["complete", "partial"])
def test_image_cache_is_refused_after_an_upgrade_changed_the_transform(
    tmp_path, encoder, monkeypatch, cached_samples
):
    _extract_images(_image_dataset(tmp_path / "data", n=cached_samples), tmp_path)
    encoded = encoder.encoded_images

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)

    with pytest.raises(CacheFeatureIdentityMismatch, match=r"transform\.normalize\.mean"):
        _extract_images(_image_dataset(tmp_path / "data", n=4), tmp_path)
    # Refused before anything is encoded: a partial cache is not completed either.
    assert encoder.encoded_images == encoded


def test_image_cache_whose_record_lacks_a_required_field_is_refused_after_an_upgrade(
    tmp_path, encoder, monkeypatch
):
    dataset = _image_dataset(tmp_path / "data")
    _extract_images(dataset, tmp_path)
    encoded = encoder.encoded_images
    metadata_path = _cache_dir(tmp_path, "image") / "cache_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["feature_identity"]["identity"]["feature_dtype"]
    metadata_path.write_text(json.dumps(metadata))

    _upgrade_slide2vec(monkeypatch, encoder, mean=_ORIGINAL_MEAN)

    # slide2vec 7 reports a required field the record lacks as a difference rather than
    # as something it cannot verify, so the cache is refused like any other mismatch.
    with pytest.raises(
        CacheFeatureIdentityMismatch, match=r"feature_dtype \(recorded MISSING_FIELD"
    ):
        _extract_images(dataset, tmp_path)
    assert encoder.encoded_images == encoded


def test_cache_hit_on_the_same_slide2vec_version_does_not_load_the_encoder(tmp_path, encoder):
    dataset = _image_dataset(tmp_path / "data")
    _extract_images(dataset, tmp_path)
    instances = encoder.instances

    _extract_images(dataset, tmp_path)

    assert encoder.instances == instances


def test_cache_verified_after_an_upgrade_hits_cheaply_from_then_on(
    tmp_path, encoder, monkeypatch
):
    dataset = _image_dataset(tmp_path / "data")
    _extract_images(dataset, tmp_path)
    encoded = encoder.encoded_images
    instances = encoder.instances

    _upgrade_slide2vec(monkeypatch, encoder, mean=_ORIGINAL_MEAN)
    _extract_images(dataset, tmp_path)

    # The upgrade kept the transform: the cache is reused, at the cost of one encoder load.
    assert encoder.encoded_images == encoded
    assert encoder.instances == instances + 1

    _extract_images(dataset, tmp_path)

    assert encoder.instances == instances + 1
    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION


@pytest.mark.parametrize("cached_samples", [4, 3], ids=["complete", "partial"])
def test_reextract_setting_rebuilds_an_image_cache_with_the_current_recipe(
    tmp_path, encoder, monkeypatch, cached_samples
):
    _extract_images(_image_dataset(tmp_path / "data", n=cached_samples), tmp_path)
    encoded = encoder.encoded_images
    dataset = _image_dataset(tmp_path / "data", n=4)

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    rebuilt = _extract_images(dataset, tmp_path, on_identity_mismatch="reextract")

    # Every sample is encoded again, not only the ones the cache was missing.
    assert encoder.encoded_images == encoded + 4
    _assert_expected_features(
        _features(rebuilt), _expected_image_features(4, mean=_CHANGED_MEAN)
    )
    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _CHANGED_MEAN


@pytest.mark.parametrize("cached_samples", [4, 3], ids=["complete", "partial"])
def test_image_cache_without_a_recorded_identity_is_reused_with_one_warning_and_never_stamped(
    tmp_path, encoder, monkeypatch, caplog, cached_samples
):
    _extract_images(_image_dataset(tmp_path / "data", n=cached_samples), tmp_path)
    _forget_identity(tmp_path, "image")
    encoded = encoder.encoded_images

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    with caplog.at_level("WARNING"):
        _extract_images(_image_dataset(tmp_path / "data", n=4), tmp_path)

    # Only the samples the cache was missing are encoded.
    assert encoder.encoded_images == encoded + (4 - cached_samples)
    (warning,) = _unverifiable_warnings(caplog)
    assert "slide2vec release notes" in warning
    assert "feature_identity" not in _cache_metadata(tmp_path, "image")


def test_strict_setting_extracts_an_image_cache_without_a_recorded_identity_again(
    tmp_path, encoder, monkeypatch
):
    dataset = _image_dataset(tmp_path / "data")
    _extract_images(dataset, tmp_path)
    _forget_identity(tmp_path, "image")
    encoded = encoder.encoded_images

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    rebuilt = _extract_images(dataset, tmp_path, on_unrecorded_identity="reextract")

    assert encoder.encoded_images == encoded + 4
    _assert_expected_features(
        _features(rebuilt), _expected_image_features(4, mean=_CHANGED_MEAN)
    )
    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION


def _interrupt_before_first_commit(extract, *, written_batches: int) -> None:
    """Run ``extract`` with an encoder that fails after ``written_batches`` batches.

    soma commits the images of one run together, so the batches slide2vec wrote before
    the failure stay on disk uncommitted.
    """
    encode = _SwitchableTransformEncoder.encode_tiles
    batches = 0

    def fail_later(self, batch):
        nonlocal batches
        batches += 1
        if batches > written_batches:
            raise RuntimeError("interrupted")
        return encode(self, batch)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_SwitchableTransformEncoder, "encode_tiles", fail_later)
        with pytest.raises(Exception, match="interrupted"):
            extract()


@pytest.mark.parametrize("written_batches", [0, 1])
def test_extraction_interrupted_before_its_first_commit_still_records_its_identity(
    tmp_path, encoder, caplog, written_batches
):
    dataset = _image_dataset(tmp_path / "data")
    _interrupt_before_first_commit(
        lambda: _extract_images(dataset, tmp_path), written_batches=written_batches
    )

    with caplog.at_level("WARNING"):
        _extract_images(dataset, tmp_path)

    assert not _unverifiable_warnings(caplog)
    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _ORIGINAL_MEAN


def test_empty_cache_without_a_recorded_identity_records_one_with_its_first_features(
    tmp_path, encoder, caplog
):
    dataset = _image_dataset(tmp_path / "data")
    _interrupt_before_first_commit(lambda: _extract_images(dataset, tmp_path), written_batches=0)
    _forget_identity(tmp_path, "image")

    with caplog.at_level("WARNING"):
        _extract_images(dataset, tmp_path)

    # soma 1.17 created the cache and extracted nothing into it: there is nothing to verify.
    assert not _unverifiable_warnings(caplog)
    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _ORIGINAL_MEAN


def test_features_written_by_a_run_killed_before_its_commit_are_verified_like_any_other(
    tmp_path, encoder, monkeypatch
):
    dataset = _image_dataset(tmp_path / "data")
    with pytest.MonkeyPatch.context() as patch:
        # Every image is written, then the process dies before soma commits them.
        patch.setattr(
            "soma.tile_extraction.record_sample_identity_signatures",
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            _extract_images(dataset, tmp_path)
    encoded = encoder.encoded_images

    _extract_images(dataset, tmp_path)

    # The same slide2vec wrote them: the next run reuses them and records their identity.
    assert encoder.encoded_images == encoded
    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _ORIGINAL_MEAN

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)

    with pytest.raises(CacheFeatureIdentityMismatch, match=r"transform\.normalize\.mean"):
        _extract_images(dataset, tmp_path)


def test_extraction_interrupted_before_an_upgrade_restarts_with_the_current_recipe(
    tmp_path, encoder, monkeypatch, caplog
):
    dataset = _image_dataset(tmp_path / "data")
    _interrupt_before_first_commit(lambda: _extract_images(dataset, tmp_path), written_batches=1)

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    with caplog.at_level("WARNING"):
        resumed = _extract_images(dataset, tmp_path)

    assert not _unverifiable_warnings(caplog)
    _assert_expected_features(
        _features(resumed), _expected_image_features(4, mean=_CHANGED_MEAN)
    )
    recorded = _cache_metadata(tmp_path, "image")["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _CHANGED_MEAN


@pytest.mark.parametrize("setting", ["on_identity_mismatch", "on_unrecorded_identity"])
def test_cache_config_rejects_an_unknown_identity_policy(setting):
    with pytest.raises(ValueError, match=f"cache.{setting}"):
        CacheConfig(**{setting: "ignore"})


# --- Whole-slide caches: tile bags, hierarchical, slide-level and patient-level -------


@pytest.fixture(scope="module")
def slide_manifests(tmp_path_factory) -> dict[int, Path]:
    """Flat blue, magenta, green, yellow slides, paired into two patients.

    Nonzero saturation lets real Otsu preprocessing mark the whole slide as tissue;
    every tile has the same known RGB, with no random noise or boundary mixtures.
    """
    root = tmp_path_factory.mktemp("slides")
    rows = []
    for index, colour in enumerate(((0, 0, 255), (255, 0, 255), (0, 255, 0), (255, 255, 0))):
        sample_id = f"slide{index:02d}"
        image_path = root / f"{sample_id}.tif"
        pixels = np.full((256, 256, 3), colour, dtype=np.uint8)
        write_tiff(image_path, pixels)
        rows.append({
            "sample_id": sample_id,
            "image_path": str(image_path),
            "patient_id": f"p{index // 2}",
            "label": index % 2,
        })
    frame = pd.DataFrame(rows)
    manifests = {}
    for n in (3, 4):
        manifests[n] = root / f"dataset_{n}.csv"
        frame.head(n).to_csv(manifests[n], index=False)
    return manifests


def _extract_slides(dataset_csv: Path, root: Path, kind: str, **cache_settings):
    """Extract into ``<root>/cache``; slides are tiled into ``<root>/tiling_cache``."""
    return FeatureExtractor(
        Dataset(dataset_csv),
        EncoderConfig(name=_SLIDE_KINDS[kind], precision="fp32", batch_size=64),
        PreprocessingConfig(
            backend="openslide",
            requested_tile_size_px=32,
            requested_spacing_um=0.5,
            tissue_method="otsu",
            seg_downsample=16,
            a_t=1,
            min_coverage={"tissue": 0.5},
            region_tile_multiple=2 if kind == "hierarchical" else None,
        ),
        execution=ExecutionConfig(num_gpus=1, num_workers_per_gpu=0, num_preprocessing_workers=0),
        cache=CacheConfig(enabled=True, root_dir=root / "cache", **cache_settings),
        output_root=root / "run",
    ).extract()


def test_patient_level_extraction_tiles_the_slides_into_an_empty_tiling_cache(
    tmp_path, encoder, slide_manifests
):
    assert not (tmp_path / "tiling_cache").exists()

    extracted = _extract_slides(slide_manifests[4], tmp_path, "patient")

    assert (tmp_path / "tiling_cache").is_dir()
    _assert_same_features(_features(extracted), _expected_slide_features("patient", changed=False))


@pytest.mark.parametrize("kind", _SLIDE_KINDS)
def test_slide_extraction_records_the_transform_its_features_went_through(
    tmp_path, encoder, slide_manifests, kind
):
    _extract_slides(slide_manifests[4], tmp_path, kind)

    recorded = _cache_metadata(tmp_path, kind)["feature_identity"]

    assert recorded["slide2vec_version"] == slide2vec.__version__
    assert recorded["identity"]["encoder_name"] == _SLIDE_KINDS[kind]
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _ORIGINAL_MEAN


@pytest.mark.parametrize("cached_samples", [4, 3], ids=["complete", "partial"])
@pytest.mark.parametrize("kind", _SLIDE_KINDS)
def test_slide_cache_is_refused_after_an_upgrade_changed_the_transform(
    tmp_path, encoder, slide_manifests, monkeypatch, kind, cached_samples
):
    _extract_slides(slide_manifests[cached_samples], tmp_path, kind)
    encoded = encoder.encoded_images

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)

    with pytest.raises(CacheFeatureIdentityMismatch, match=r"transform\.normalize\.mean"):
        _extract_slides(slide_manifests[4], tmp_path, kind)
    assert encoder.encoded_images == encoded


@pytest.mark.parametrize("kind", _SLIDE_KINDS)
def test_slide_cache_is_reused_after_an_upgrade_kept_the_identity(
    tmp_path, encoder, slide_manifests, monkeypatch, kind
):
    _extract_slides(slide_manifests[4], tmp_path, kind)
    encoded = encoder.encoded_images

    _upgrade_slide2vec(monkeypatch, encoder, mean=_ORIGINAL_MEAN)
    reused = _extract_slides(slide_manifests[4], tmp_path, kind)

    assert encoder.encoded_images == encoded
    _assert_same_features(_features(reused), _expected_slide_features(kind, changed=False))
    recorded = _cache_metadata(tmp_path, kind)["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION


@pytest.mark.parametrize("kind", _SLIDE_KINDS)
def test_reextract_setting_rebuilds_slide_caches_and_what_is_aggregated_from_them(
    tmp_path, encoder, slide_manifests, monkeypatch, kind
):
    _extract_slides(slide_manifests[4], tmp_path, kind)

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    rebuilt = _extract_slides(
        slide_manifests[4], tmp_path, kind, on_identity_mismatch="reextract"
    )

    _assert_same_features(_features(rebuilt), _expected_slide_features(kind, changed=True))
    recorded = _cache_metadata(tmp_path, kind)["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _CHANGED_MEAN


@pytest.mark.parametrize("kind", _SLIDE_KINDS)
def test_strict_setting_rebuilds_slide_caches_without_a_recorded_identity(
    tmp_path, encoder, slide_manifests, monkeypatch, kind
):
    _extract_slides(slide_manifests[4], tmp_path, kind)
    for cache_kind in sorted({"tile", kind} - ({"tile"} if kind == "hierarchical" else set())):
        _forget_identity(tmp_path, cache_kind)

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    rebuilt = _extract_slides(
        slide_manifests[4], tmp_path, kind, on_unrecorded_identity="reextract"
    )

    _assert_same_features(_features(rebuilt), _expected_slide_features(kind, changed=True))
    recorded = _cache_metadata(tmp_path, kind)["feature_identity"]
    assert recorded["slide2vec_version"] == _UPGRADED_VERSION


@pytest.mark.parametrize("kind", ["slide", "patient"])
def test_cache_aggregated_from_unverifiable_tiles_is_unverifiable_too(
    tmp_path, encoder, slide_manifests, monkeypatch, kind
):
    _extract_slides(slide_manifests[4], tmp_path, kind)
    _forget_identity(tmp_path, "tile")
    shutil.rmtree(_cache_dir(tmp_path, kind))

    _extract_slides(slide_manifests[4], tmp_path, kind)

    # Aggregated from tiles nothing vouches for: no identity is recorded for them either.
    assert "feature_identity" not in _cache_metadata(tmp_path, kind)

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    rebuilt = _extract_slides(
        slide_manifests[4], tmp_path, kind, on_unrecorded_identity="reextract"
    )

    # The strict setting then rebuilds the tiles and what was aggregated from them.
    _assert_same_features(_features(rebuilt), _expected_slide_features(kind, changed=True))


def _interrupt_slide_aggregation_from_unverifiable_tiles(
    root: Path, dataset_csv: Path, monkeypatch, encoder
) -> None:
    """Leave an uncommitted slide cache aggregated from tiles with no recorded identity.

    The tiles are extracted with the original recipe and lose their record, as a cache
    soma 1.17 wrote. After an upgrade that changed the recipe, a run accepts them,
    aggregates every slide, and dies before soma commits the slide cache.
    """
    _extract_slides(dataset_csv, root, "slide")
    _forget_identity(root, "tile")
    shutil.rmtree(_cache_dir(root, "slide"))

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "soma.extraction.extractor.record_sample_identity_signatures",
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        with pytest.raises(KeyboardInterrupt):
            _extract_slides(dataset_csv, root, "slide")


def test_strict_setting_rebuilds_a_slide_cache_interrupted_while_aggregating_unverifiable_tiles(
    tmp_path, encoder, slide_manifests, monkeypatch
):
    _interrupt_slide_aggregation_from_unverifiable_tiles(
        tmp_path, slide_manifests[4], monkeypatch, encoder
    )

    rebuilt = _extract_slides(
        slide_manifests[4], tmp_path, "slide", on_unrecorded_identity="reextract"
    )

    # Nothing vouches for what the interrupted run left: it is not taken for a hit.
    _assert_same_features(_features(rebuilt), _expected_slide_features("slide", changed=True))
    recorded = _cache_metadata(tmp_path, "slide")["feature_identity"]
    assert recorded["identity"]["transform"]["normalize"]["mean"] == _CHANGED_MEAN


def test_slide_cache_interrupted_while_aggregating_unverifiable_tiles_is_reused_with_a_warning(
    tmp_path, encoder, slide_manifests, monkeypatch, caplog
):
    _interrupt_slide_aggregation_from_unverifiable_tiles(
        tmp_path, slide_manifests[4], monkeypatch, encoder
    )
    encoded = encoder.encoded_images
    caplog.clear()

    with caplog.at_level("WARNING"):
        resumed = _extract_slides(slide_manifests[4], tmp_path, "slide")

    # The interrupted run reported the tile cache; this one reports the slide cache.
    (warning,) = _unverifiable_warnings(caplog)
    assert str(_cache_dir(tmp_path, "slide")) in warning
    assert encoder.encoded_images == encoded
    _assert_same_features(_features(resumed), _expected_slide_features("slide", changed=False))
    assert "feature_identity" not in _cache_metadata(tmp_path, "slide")


def test_completing_a_slide_cache_without_a_recorded_identity_warns_once(
    tmp_path, encoder, slide_manifests, monkeypatch, caplog
):
    _extract_slides(slide_manifests[3], tmp_path, "tile")
    _forget_identity(tmp_path, "tile")

    _upgrade_slide2vec(monkeypatch, encoder, mean=_CHANGED_MEAN)
    with caplog.at_level("WARNING"):
        _extract_slides(slide_manifests[4], tmp_path, "tile")

    # The run resolves the cache before and after completing it, and warns once.
    assert len(_unverifiable_warnings(caplog)) == 1
    assert "feature_identity" not in _cache_metadata(tmp_path, "tile")
