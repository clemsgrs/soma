"""User-supplied tile coordinates: cache identity and the checks that fail loudly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from soma.cache.keys import _sample_identity_payload, sample_identity_signature
from soma.config import MasksConfig, PreprocessingConfig
from soma.dataset import Dataset, load_manifest
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


def test_changing_one_coordinate_changes_the_cache_key(tmp_path: Path):
    artifacts = tmp_path / "coordinates"
    paths = {s: _artifact(artifacts, s, tmp_path / f"{s}.tif") for s in ("a", "b")}
    dataset = Dataset(_manifest(tmp_path, paths))
    before = _sample_identity_payload(dataset)

    _artifact(artifacts, "a", tmp_path / "a.tif")  # same tiles, rewritten
    assert _sample_identity_payload(dataset) == before

    _artifact(artifacts, "a", tmp_path / "a.tif", x=(0, 32, 96))
    after = _sample_identity_payload(dataset)
    assert after["a"] != before["a"]
    assert after["b"] == before["b"]


def test_soma_tiled_identities_are_unchanged(tmp_path: Path):
    """A slide without supplied coordinates keeps the identity it always had."""
    signature = sample_identity_signature(sample_id="a", image_path="/a.tif", mask_path=None)
    assert signature == sample_identity_signature(
        sample_id="a", image_path="/a.tif", mask_path=None, coordinates_digest=None
    )


def test_staging_lists_the_supplied_artifacts(tmp_path: Path):
    paths = {s: _artifact(tmp_path / "coordinates", s, tmp_path / f"{s}.tif") for s in ("a", "b")}
    stage_supplied_coordinates(Dataset(_manifest(tmp_path, paths)), tmp_path / "tiling", _preprocessing())
    rows = pd.read_csv(tmp_path / "tiling" / "process_list.csv").set_index("sample_id")
    assert rows.loc["a", "coordinates_npz_path"] == str(paths["a"].resolve())
    assert rows["num_tiles"].tolist() == [3, 3]
    assert set(rows["tiling_status"]) == {"success"}


def test_partially_filled_column_is_rejected(tmp_path: Path):
    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    with pytest.raises(ValueError, match="every row or for none.*'b'"):
        Dataset(_manifest(tmp_path, {"a": path, "b": None}))


@pytest.mark.parametrize("dataset_type", ["tile", "segmentation", "detection"])
def test_column_is_rejected_outside_slide_datasets(tmp_path: Path, dataset_type: str):
    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    manifest = _manifest(tmp_path, {"a": path})
    frame = pd.read_csv(manifest)
    frame["label_mask_path"] = "/m.png"
    frame["points_path"] = "/p.csv"
    frame.to_csv(manifest, index=False)
    with pytest.raises(ValueError, match="only supported for slide-level datasets"):
        load_manifest(manifest, dataset_type)


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
    dataset = Dataset(_manifest(tmp_path, {"a": written}))
    with pytest.raises(ValueError, match=message):
        stage_supplied_coordinates(dataset, tmp_path / "tiling", _preprocessing())


def test_missing_artifact_fails_staging(tmp_path: Path):
    dataset = Dataset(_manifest(tmp_path, {"a": tmp_path / "a.coordinates.npz"}))
    with pytest.raises(FileNotFoundError, match="'a'.*does not exist"):
        stage_supplied_coordinates(dataset, tmp_path / "tiling", _preprocessing())


def test_annotation_masks_cannot_reselect_supplied_tiles(tmp_path: Path):
    path = _artifact(tmp_path / "coordinates", "a", tmp_path / "a.tif")
    dataset = Dataset(_manifest(tmp_path, {"a": path}))
    masks = MasksConfig(pixel_mapping={"background": 0, "tumor": 1}, min_coverage={"tumor": 0.5})
    with pytest.raises(ValueError, match="cannot be combined with preprocessing.masks"):
        stage_supplied_coordinates(dataset, tmp_path / "tiling", _preprocessing(masks=masks))
