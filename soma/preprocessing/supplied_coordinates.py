"""User-supplied tile coordinates: a manifest's ``coordinates_path`` in place of soma's tiling.

A slide-level ``Dataset`` may name, per slide, an hs2p tiling artifact — the
``<name>.coordinates.npz`` + ``<name>.coordinates.meta.json`` pair that
``hs2p.artifacts.save_tiling_result`` writes. soma then skips hs2p tiling for the dataset:
it copies each artifact into the run's tiling directory, checks it against its manifest
row and the run's preprocessing, and lists it there, unchanged, for slide2vec to embed.

The artifact's content joins the slide's cache identity (:func:`coordinates_digest`), so
a different tile set never reuses another set's features.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from hs2p.artifacts import load_tiling_result
from hs2p.tiling.io import normalize_artifact_path

from soma.config import PreprocessingConfig
from soma.dataset import Dataset, SampleRecord

_NPZ_SUFFIX = ".coordinates.npz"
_META_SUFFIX = ".coordinates.meta.json"


def coordinates_meta_path(coordinates_path: Path) -> Path:
    """The ``.coordinates.meta.json`` that sits beside a ``.coordinates.npz``."""
    name = Path(coordinates_path).name
    if not name.endswith(_NPZ_SUFFIX):
        raise ValueError(
            f"coordinates_path must name an hs2p tiling artifact ending in '{_NPZ_SUFFIX}' "
            f"(as written by hs2p.artifacts.save_tiling_result), got {str(coordinates_path)!r}."
        )
    return Path(coordinates_path).with_name(name[: -len(_NPZ_SUFFIX)] + _META_SUFFIX)


def _require_files(coordinates_path: Path, *, sample_id: str) -> Path:
    meta_path = coordinates_meta_path(coordinates_path)
    for path in (Path(coordinates_path), meta_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Supplied coordinates for sample {sample_id!r}: {str(path)!r} does not exist. "
                f"coordinates_path names the '{_NPZ_SUFFIX}' half of an hs2p tiling "
                f"artifact; its '{_META_SUFFIX}' must sit beside it."
            )
    return meta_path


def coordinates_digest(coordinates_path: Path, *, sample_id: str) -> str:
    """Content hash of a supplied tiling artifact: its metadata and every coordinate array.

    Hashes what the artifact holds, not its file bytes: an ``.npz`` zip carries write
    timestamps, so rewriting the same tile set must not change the digest.
    """
    meta_path = _require_files(coordinates_path, sample_id=sample_id)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    with np.load(coordinates_path, allow_pickle=False) as arrays:
        for name in sorted(arrays.files):
            array = np.ascontiguousarray(arrays[name])
            digest.update(f"{name}|{array.dtype.str}|{array.shape}|".encode("utf-8"))
            digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _check_supported(preprocessing: PreprocessingConfig) -> None:
    if preprocessing.masks is not None:
        raise ValueError(
            "Supplied coordinates (manifest column 'coordinates_path') cannot be combined with "
            "preprocessing.masks: the masks block selects tiles, and the supplied artifacts "
            "already fix them. Remove one of the two."
        )
    if preprocessing.region_tile_multiple is not None:
        raise ValueError(
            "Supplied coordinates (manifest column 'coordinates_path') support tile-level "
            "bags only, not hierarchical region tiling (preprocessing.region_tile_multiple)."
        )


def _snapshot(record: SampleRecord, snapshot_dir: Path) -> tuple[Path, Path]:
    """Copy ``record``'s artifact into the run, so the run keeps the tiles it checked."""
    assert record.coordinates_path is not None
    meta_path = _require_files(record.coordinates_path, sample_id=record.sample_id)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    npz_copy = snapshot_dir / f"{record.sample_id}{_NPZ_SUFFIX}"
    meta_copy = snapshot_dir / f"{record.sample_id}{_META_SUFFIX}"
    shutil.copyfile(record.coordinates_path, npz_copy)
    shutil.copyfile(meta_path, meta_copy)
    return npz_copy.resolve(), meta_copy.resolve()


def _load_checked(
    record: SampleRecord,
    preprocessing: PreprocessingConfig,
    *,
    npz_path: Path,
    meta_path: Path,
):
    """Load ``record``'s artifact and check it against the row and the run.

    The artifact must name the row's ``sample_id``, ``image_path`` and
    ``spacing_at_level_0``, and must have been tiled at the run's requested spacing and
    tile size: slide2vec embeds the tiles as given, so a mismatch would silently feed the
    encoder another geometry.
    """
    source = f"supplied coordinates {str(record.coordinates_path)!r}"
    result = load_tiling_result(npz_path, meta_path)
    if str(result.sample_id) != record.sample_id:
        raise ValueError(
            f"Sample {record.sample_id!r}: {source} belong to sample_id "
            f"{str(result.sample_id)!r}. Point coordinates_path at this sample's artifact."
        )
    expected_image = normalize_artifact_path(record.image_path)
    actual_image = normalize_artifact_path(result.image_path)
    if actual_image != expected_image:
        raise ValueError(
            f"Sample {record.sample_id!r}: {source} were made for image {actual_image!r}, "
            f"but the manifest row names {expected_image!r}."
        )
    if result.spacing_at_level_0 != record.spacing_at_level_0:
        # The artifact's level-0 read geometry assumes its own source spacing; under
        # another declaration the same reads would cover another physical field.
        raise ValueError(
            f"Sample {record.sample_id!r}: {source} were made with spacing_at_level_0 "
            f"{result.spacing_at_level_0} (manifest: {record.spacing_at_level_0}). "
            "Declare the same level-0 spacing in both, or regenerate the artifact."
        )
    mismatches = []
    if float(result.requested_spacing_um) != float(preprocessing.requested_spacing_um):
        mismatches.append(
            f"requested_spacing_um {float(result.requested_spacing_um)} "
            f"(preprocessing: {float(preprocessing.requested_spacing_um)})"
        )
    if int(result.requested_tile_size_px) != int(preprocessing.requested_tile_size_px):
        mismatches.append(
            f"requested_tile_size_px {int(result.requested_tile_size_px)} "
            f"(preprocessing: {int(preprocessing.requested_tile_size_px)})"
        )
    if mismatches:
        raise ValueError(
            f"Sample {record.sample_id!r}: {source} were tiled with "
            + ", ".join(mismatches)
            + ". Set preprocessing.requested_spacing_um / requested_tile_size_px to the "
            "artifacts' geometry, or regenerate the artifacts."
        )
    if result.num_tiles == 0:
        raise ValueError(f"Sample {record.sample_id!r}: {source} list no tiles.")
    return result


def stage_supplied_coordinates(
    dataset: Dataset,
    tiling_dir: Path,
    preprocessing: PreprocessingConfig,
) -> None:
    """Copy each slide's supplied artifact into ``tiling_dir`` and list it there.

    It is the tiling directory hs2p would have written, so slide2vec embeds from it
    unchanged. The copies, not the user's files, are checked and listed: replacing an
    artifact later leaves this run's tiles (and the heatmaps drawn over them) intact.
    """
    _check_supported(preprocessing)
    rows = []
    for record in dataset.samples.values():
        npz_path, meta_path = _snapshot(record, tiling_dir / "coordinates")
        result = _load_checked(record, preprocessing, npz_path=npz_path, meta_path=meta_path)
        rows.append(
            {
                "sample_id": record.sample_id,
                "annotation": "tissue",
                "output_mode": None,
                "image_path": str(record.image_path),
                "mask_path": str(record.mask_path) if record.mask_path is not None else None,
                "requested_backend": result.requested_backend,
                "backend": result.backend,
                "requested_mask_backend": None,
                "mask_backend": None,
                "spacing_at_level_0": record.spacing_at_level_0,
                "tiling_status": "success",
                "num_tiles": int(result.num_tiles),
                "coordinates_npz_path": str(npz_path),
                "coordinates_meta_path": str(meta_path),
                "tiles_tar_path": None,
                "mask_preview_path": None,
                "tiling_preview_path": None,
                "error": None,
                "traceback": None,
            }
        )
    tiling_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tiling_dir / "process_list.csv", index=False)
