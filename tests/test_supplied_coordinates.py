"""User-supplied tile coordinates: cache identity and the checks that fail loudly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soma.cache.keys import _sample_identity_payload, sample_identity_signature
from soma.config import MasksConfig, PreprocessingConfig
from soma.data._legacy import legacy_samples_from_csv
from soma.preprocessing.supplied_coordinates import stage_supplied_coordinates
from tests.e2e.synthetic import write_coordinates_artifact

TILE_PX = 32
SPACING_UM = 0.5


def _artifact(root: Path, sample_id: str, image_path: Path, x=(0, 32, 64), **kwargs) -> Path:
    return write_coordinates_artifact(
        root,
        sample_id=sample_id,
        image_path=image_path,
        x=np.array(x),
        y=np.zeros(len(x), dtype=int),
        tile_size_px=kwargs.pop("tile_size_px", TILE_PX),
        spacing_um=kwargs.pop("spacing_um", SPACING_UM),
        slide_size=384,
    )


def _manifest(tmp_path: Path, coordinates: dict[str, Path | None]) -> Path:
    frame = pd.DataFrame(
        {
            "sample_id": list(coordinates),
            "image_path": [str(tmp_path / f"{s}.tif") for s in coordinates],
            "label": [0, 1, 0, 1][: len(coordinates)],
            "coordinates_path": [None if p is None else str(p) for p in coordinates.values()],
        }
    )
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)
    return path


def _preprocessing(**overrides) -> PreprocessingConfig:
    return PreprocessingConfig(
        requested_tile_size_px=TILE_PX,
        requested_spacing_um=SPACING_UM,
        tissue_method="otsu",
        **overrides,
    )


def _resolve_tile_cache(cache_root: Path, dataset: Dataset):
    from soma.cache import resolve_tile_cache
    from soma.config import EncoderConfig

    return resolve_tile_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="virchow",
        preprocessing=_preprocessing(),
        execution=EncoderConfig(name="virchow", precision="fp16"),
    )


def _commit(resolution, sample_id: str, tensor):
    """Write one sample's features the way extraction commits them."""
    import torch

    from soma.cache import record_feature_dim, record_sample_identity_signatures

    payload_path = resolution.feature_path_for_id(sample_id)
    torch.save(tensor, payload_path)
    # The sidecar slide2vec publishes with the payload carries the feature identity.
    payload_path.with_name(f"{payload_path.stem}.meta.json").write_text(
        '{"compatibility": {"encoder_name": "virchow"}}'
    )
    record_feature_dim(resolution, int(tensor.shape[-1]))
    return record_sample_identity_signatures(resolution, [sample_id])


def test_changing_one_coordinate_changes_the_cache_key(tmp_path: Path):
    artifacts = tmp_path / "coordinates"
    paths = {s: _artifact(artifacts, s, tmp_path / f"{s}.tif") for s in ("a", "b")}
    dataset = legacy_samples_from_csv(_manifest(tmp_path, paths))
    before = _sample_identity_payload(dataset)

    _artifact(artifacts, "a", tmp_path / "a.tif")  # same tiles, rewritten
    assert _sample_identity_payload(dataset) == before

    _artifact(artifacts, "a", tmp_path / "a.tif", x=(0, 32, 96))
    after = _sample_identity_payload(dataset)
    assert after["a"] != before["a"]
    assert after["b"] == before["b"]


def test_soma_tiled_identities_are_unchanged(tmp_path: Path):
    """A slide without supplied coordinates keeps the identity it always had."""
    # The signature soma recorded for this sample before supplied coordinates existed.
    historical = "2a28f51893731bf0"
    assert sample_identity_signature(sample_id="a", image_path="/a.tif", mask_path=None) == historical
    assert (
        sample_identity_signature(
            sample_id="a", image_path="/a.tif", mask_path=None, coordinates_digest=None
        )
        == historical
    )


def test_staging_lists_a_snapshot_of_the_supplied_artifacts(tmp_path: Path):
    """The run keeps the tiles it validated, even if the user's artifact changes later."""
    from hs2p.artifacts import load_tiling_result

    paths = {s: _artifact(tmp_path / "coordinates", s, tmp_path / f"{s}.tif") for s in ("a", "b")}
    tiling_dir = tmp_path / "tiling"
    stage_supplied_coordinates(legacy_samples_from_csv(_manifest(tmp_path, paths)), tiling_dir, _preprocessing())
    rows = pd.read_csv(tiling_dir / "process_list.csv").set_index("sample_id")
    assert rows["num_tiles"].tolist() == [3, 3]
    assert set(rows["tiling_status"]) == {"success"}

    _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif", x=(0, 32, 96))
    staged = load_tiling_result(
        Path(rows.loc["a", "coordinates_npz_path"]), Path(rows.loc["a", "coordinates_meta_path"])
    )
    assert Path(rows.loc["a", "coordinates_npz_path"]).is_relative_to(tiling_dir.resolve())
    assert staged.x.tolist() == [0, 32, 64]


def test_partially_filled_column_is_rejected(tmp_path: Path):
    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    with pytest.raises(ValueError, match="every row or for none.*'b'"):
        legacy_samples_from_csv(_manifest(tmp_path, {"a": path, "b": None}))


@pytest.mark.parametrize("dataset_type", ["tile", "segmentation", "detection"])
def test_column_is_rejected_outside_slide_datasets(tmp_path: Path, dataset_type: str):
    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    manifest = _manifest(tmp_path, {"a": path})
    frame = pd.read_csv(manifest)
    frame["label_mask_path"] = "/m.png"
    frame["points_path"] = "/p.csv"
    frame.to_csv(manifest, index=False)
    from soma.config import EncoderConfig
    from soma.extraction import FeatureExtractor

    shape = "set" if dataset_type == "tile" else "grid"
    with pytest.raises(ValueError, match="whole slides"):
        FeatureExtractor(
            legacy_samples_from_csv(manifest),
            EncoderConfig(name="phikon"),
            shape=shape,
            unit="tile",
            output_root=tmp_path / "out",
        )


def test_column_is_rejected_on_the_annotation_sampling_path(tmp_path: Path):
    """A dense grid over whole slides samples its own ROIs; supplied tiles would be
    silently ignored, so the facade refuses them up front."""
    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    manifest = _manifest(tmp_path, {"a": path})
    frame = pd.read_csv(manifest)
    frame["label_mask_path"] = "/m.png"
    frame.to_csv(manifest, index=False)
    from soma.config import EncoderConfig
    from soma.extraction import FeatureExtractor

    masks = MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5})
    with pytest.raises(ValueError, match="coordinates_path.*shape='set', unit='slide'"):
        FeatureExtractor(
            legacy_samples_from_csv(manifest),
            EncoderConfig(name="phikon"),
            _preprocessing(masks=masks),
            shape="grid",
            unit="slide",
            output_root=tmp_path / "out",
        )


@pytest.mark.parametrize(
    ("artifact_kwargs", "message"),
    [
        ({"sample_id": "other"}, "belong to sample_id 'other'"),
        ({"image_path": "elsewhere.tif"}, "were made for image .*elsewhere.tif"),
        ({"spacing_um": 1.0}, r"requested_spacing_um 1.0 \(preprocessing: 0.5\)"),
        ({"tile_size_px": 64}, r"requested_tile_size_px 64 \(preprocessing: 32\)"),
    ],
)
def test_mismatched_artifact_fails_staging(tmp_path: Path, artifact_kwargs: dict, message: str):
    sample_id = artifact_kwargs.pop("sample_id", "a")
    image_path = tmp_path / artifact_kwargs.pop("image_path", "a.tif")
    written = _artifact(tmp_path / "coordinates", sample_id, image_path, **artifact_kwargs)
    dataset = legacy_samples_from_csv(_manifest(tmp_path, {"a": written}))
    with pytest.raises(ValueError, match=message):
        stage_supplied_coordinates(dataset, tmp_path / "tiling", _preprocessing())


def test_artifact_made_for_another_source_spacing_fails_staging(tmp_path: Path):
    """The row declares 1.0 µm/px at level 0; tiles read for a 0.5 µm/px slide would
    cover another physical field."""
    written = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    manifest = _manifest(tmp_path, {"a": written})
    frame = pd.read_csv(manifest)
    frame["spacing_at_level_0"] = 1.0
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match=r"spacing_at_level_0 None .*manifest: 1.0"):
        stage_supplied_coordinates(legacy_samples_from_csv(manifest), tmp_path / "tiling", _preprocessing())


def test_missing_artifact_fails_staging(tmp_path: Path):
    dataset = legacy_samples_from_csv(_manifest(tmp_path, {"a": tmp_path / "a.coordinates.npz"}))
    with pytest.raises(FileNotFoundError, match="'a'.*does not exist"):
        stage_supplied_coordinates(dataset, tmp_path / "tiling", _preprocessing())


def test_annotation_masks_cannot_reselect_supplied_tiles(tmp_path: Path):
    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    dataset = legacy_samples_from_csv(_manifest(tmp_path, {"a": path}))
    masks = MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5})
    with pytest.raises(ValueError, match="cannot be combined with preprocessing.masks"):
        stage_supplied_coordinates(dataset, tmp_path / "tiling", _preprocessing(masks=masks))


def test_unsigned_legacy_cache_is_not_adopted_for_supplied_coordinates(tmp_path: Path):
    """A cache that predates per-sample signatures cannot prove which tiles it holds."""
    import json

    import torch

    paths = {s: _artifact(tmp_path / "coordinates", s, tmp_path / f"{s}.tif") for s in ("a", "b")}
    manifest = _manifest(tmp_path, paths)
    soma_tiled = pd.read_csv(manifest).drop(columns="coordinates_path")
    soma_tiled.to_csv(tmp_path / "soma_tiled.csv", index=False)

    def resolve(dataset):
        return _resolve_tile_cache(tmp_path / "cache", dataset)

    legacy = resolve(legacy_samples_from_csv(tmp_path / "soma_tiled.csv"))
    metadata = json.loads(legacy.metadata_path.read_text())
    metadata.pop("sample_identity_signature_by_id", None)
    metadata["feature_dim"] = 16
    legacy.metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    torch.save(torch.zeros(4, 16), legacy.feature_path_for_id("a"))

    supplied = resolve(legacy_samples_from_csv(manifest))
    assert supplied.missing_sample_ids() == ["a", "b"]


def test_tissue_mask_value_guard_does_not_apply_to_supplied_coordinates(tmp_path: Path):
    """The tiler that guard protects never runs: supplied tiles are not mask-sampled."""
    from soma.slide2vec_adapter import ensure_supported_mask_value

    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    manifest = _manifest(tmp_path, {"a": path})
    frame = pd.read_csv(manifest)
    frame["mask_path"] = str(tmp_path / "a_mask.tif")
    frame.to_csv(manifest, index=False)
    ensure_supported_mask_value(legacy_samples_from_csv(manifest), _preprocessing(tissue_mask_tissue_value=255))


def test_staged_dataset_is_keyed_on_the_tiles_the_run_embeds(tmp_path: Path):
    """Once staged, the cache identity follows the run's copies, not the user's files."""
    paths = {s: _artifact(tmp_path / "coordinates", s, tmp_path / f"{s}.tif") for s in ("a", "b")}
    staged = stage_supplied_coordinates(
        legacy_samples_from_csv(_manifest(tmp_path, paths)), tmp_path / "tiling", _preprocessing()
    )
    before = _sample_identity_payload(staged)

    _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif", x=(0, 32, 96))
    paths["b"].unlink()
    assert _sample_identity_payload(staged) == before
    restaged = stage_supplied_coordinates(staged, tmp_path / "tiling", _preprocessing())
    assert _sample_identity_payload(restaged) == before


def test_each_supplied_tile_set_keeps_its_own_cached_features(tmp_path: Path):
    """Two tile sets of one slide share a cache without overwriting each other's features,
    and each reuses its own on a later run."""
    import torch

    tile_sets = {"first": (0, 32, 64), "second": (0, 96)}
    datasets = {}
    for name, x in tile_sets.items():
        written = _artifact(tmp_path / name, "a", tmp_path / "a.tif", x=x)
        datasets[name] = legacy_samples_from_csv(_manifest(tmp_path / name, {"a": written}))
    features = {name: torch.full((len(x), 4), float(len(x))) for name, x in tile_sets.items()}

    for name, dataset in datasets.items():
        resolution = _resolve_tile_cache(tmp_path / "cache", dataset)
        assert resolution.missing_sample_ids() == ["a"]
        _commit(resolution, "a", features[name])

    for name, dataset in datasets.items():
        resolution = _resolve_tile_cache(tmp_path / "cache", dataset)
        assert resolution.complete, resolution.validation.reason
        assert resolution.missing_sample_ids() == []
        loaded = torch.load(resolution.feature_path_for_id("a"), weights_only=True)
        assert torch.equal(loaded, features[name])


def test_soma_tiled_payloads_keep_their_flat_layout(tmp_path: Path):
    """Content addressing applies to supplied coordinates only: existing caches stay valid."""
    paths = {"a": _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")}
    frame = pd.read_csv(_manifest(tmp_path, paths)).drop(columns="coordinates_path")
    frame.to_csv(tmp_path / "soma_tiled.csv", index=False)
    resolution = _resolve_tile_cache(tmp_path / "cache", legacy_samples_from_csv(tmp_path / "soma_tiled.csv"))
    assert resolution.feature_path_for_id("a").name == "a.pt"


@pytest.mark.parametrize("change", ["image_path", "coordinates"])
def test_a_stale_empty_marker_does_not_hide_a_sample(tmp_path: Path, change: str):
    """A slide recorded empty under one identity is extracted again under another."""
    import torch

    from soma.cache import record_empty_sample_ids

    frame = pd.read_csv(
        _manifest(tmp_path, {"a": _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")})
    ).drop(columns="coordinates_path")
    frame.to_csv(tmp_path / "soma_tiled.csv", index=False)
    empty = _resolve_tile_cache(tmp_path / "cache", legacy_samples_from_csv(tmp_path / "soma_tiled.csv"))
    record_empty_sample_ids(empty, ["a"])

    if change == "image_path":
        frame["image_path"] = str(tmp_path / "a_rescanned.tif")
        frame.to_csv(tmp_path / "changed.csv", index=False)
        changed = legacy_samples_from_csv(tmp_path / "changed.csv")
    else:
        written = _artifact(tmp_path / "supplied", "a", tmp_path / "a.tif")
        changed = legacy_samples_from_csv(_manifest(tmp_path / "supplied", {"a": written}))
    resolution = _resolve_tile_cache(tmp_path / "cache", changed)
    assert resolution.empty_sample_ids == set()
    assert resolution.missing_sample_ids() == ["a"]

    committed = _commit(resolution, "a", torch.zeros(3, 4))
    assert committed.complete, committed.validation.reason
    assert committed.empty_sample_ids == set()


def test_resume_keeps_the_run_snapshot_when_the_artifact_is_gone(tmp_path: Path):
    """A resume reloads the original manifest; the run's snapshot outlives the source."""
    from hs2p.artifacts import load_tiling_result

    paths = {"a": _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")}
    manifest = _manifest(tmp_path, paths)
    tiling_dir = tmp_path / "tiling"
    stage_supplied_coordinates(legacy_samples_from_csv(manifest), tiling_dir, _preprocessing())

    paths["a"].unlink()
    resumed = stage_supplied_coordinates(legacy_samples_from_csv(manifest), tiling_dir, _preprocessing())
    snapshot = resumed.samples["a"].coordinates_path
    meta = snapshot.with_name("a.coordinates.meta.json")
    assert load_tiling_result(snapshot, meta).x.tolist() == [0, 32, 64]


def test_resume_rejects_an_artifact_changed_since_the_run_started(tmp_path: Path):
    """Finished folds used the snapshot's tiles; pending ones must not use other tiles."""
    from hs2p.artifacts import load_tiling_result

    paths = {"a": _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")}
    manifest = _manifest(tmp_path, paths)
    tiling_dir = tmp_path / "tiling"
    staged = stage_supplied_coordinates(legacy_samples_from_csv(manifest), tiling_dir, _preprocessing())

    stage_supplied_coordinates(legacy_samples_from_csv(manifest), tiling_dir, _preprocessing())  # unchanged
    _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif", x=(0, 96))
    with pytest.raises(ValueError, match="'a'.*changed after this run copied them"):
        stage_supplied_coordinates(legacy_samples_from_csv(manifest), tiling_dir, _preprocessing())
    snapshot = staged.samples["a"].coordinates_path
    meta = snapshot.with_name("a.coordinates.meta.json")
    assert load_tiling_result(snapshot, meta).x.tolist() == [0, 32, 64]


@pytest.mark.parametrize("encoder", ["prism", "moozy"])
def test_supplied_coordinates_need_a_tile_encoder(tmp_path: Path, encoder: str):
    """Supplied tiles form bags of tile features; slide and patient encoders are refused."""
    from soma import EncoderConfig, FeatureExtractor
    from soma.config import CacheConfig

    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    extractor = FeatureExtractor(
        legacy_samples_from_csv(_manifest(tmp_path, {"a": path})),
        EncoderConfig(name=encoder),
        preprocessing=_preprocessing(),
        cache=CacheConfig(enabled=False),
        output_root=tmp_path / "output",
    )
    with pytest.raises(ValueError, match="support tile encoders only"):
        extractor.extract()
