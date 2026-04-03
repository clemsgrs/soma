"""Thin SOMA wrappers over hs2p's shared tiling-artifact IO."""

from __future__ import annotations

from pathlib import Path

from hs2p import (
    load_tiling_result as _load_core_tiling_result,
    save_tiling_result as _save_core_tiling_result,
)
from hs2p.preprocessing import (
    normalize_artifact_path,
    validate_tiling_result_provenance as _validate_core_tiling_result_provenance,
)

from soma.preprocessing.tiling import (
    TilingResult,
    _from_core_tiling_result,
    _to_core_tiling_result,
)


def save_tiling_result(
    result: TilingResult, output_dir: Path, slide_id: str
) -> dict[str, Path]:
    return _save_core_tiling_result(
        _to_core_tiling_result(result),
        Path(output_dir),
        slide_id,
    )


def load_tiling_result(npz_path: Path, meta_path: Path) -> TilingResult:
    return _from_core_tiling_result(_load_core_tiling_result(npz_path, meta_path))


def validate_tiling_result_provenance(
    result: TilingResult,
    *,
    sample_id: str,
    image_path: str | Path,
    tissue_mask_path: str | Path | None,
    tissue_mask_tissue_value: int | None,
) -> None:
    _validate_core_tiling_result_provenance(
        _to_core_tiling_result(result),
        sample_id=sample_id,
        image_path=image_path,
        tissue_mask_path=tissue_mask_path,
        tissue_mask_tissue_value=tissue_mask_tissue_value,
    )


__all__ = [
    "load_tiling_result",
    "normalize_artifact_path",
    "save_tiling_result",
    "validate_tiling_result_provenance",
]
