"""A tiling cache written by hs2p 4.x still loads under hs2p 5.

``fixtures/hs2p_4_4_3_tiling`` is the real output of ``hs2p.tile_slides`` 4.4.3 on a
256x256 synthetic slide (0.5 µm/px) with a precomputed ``{0, 1}`` tissue mask, four
64 px tiles. Its absolute paths are templated as ``@ROOT@``. hs2p 5 keeps every artifact
and ``process_list.csv`` field name, and soma's tiling cache keys are config-derived, so
a run resumed after the upgrade reuses this cache instead of re-tiling.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from soma.dataset import Dataset
from soma.slide2vec_adapter import load_tilings

FIXTURE = Path(__file__).parent / "fixtures" / "hs2p_4_4_3_tiling"


def _materialize(root: Path) -> Path:
    tiling_dir = root / "tiling"
    shutil.copytree(FIXTURE, tiling_dir)
    for templated in (
        tiling_dir / "process_list.csv",
        tiling_dir / "tiles" / "s0.coordinates.meta.json",
    ):
        templated.write_text(
            templated.read_text(encoding="utf-8").replace("@ROOT@", str(root)),
            encoding="utf-8",
        )
    return tiling_dir


def test_hs2p_4_tiling_cache_loads_under_hs2p_5(tmp_path: Path):
    tiling_dir = _materialize(tmp_path)
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "sample_id": ["s0"],
            "image_path": [str(tmp_path / "slide.tif")],
            "label": ["a"],
            "mask_path": [str(tmp_path / "tissue.tif")],
        }
    ).to_csv(dataset_csv, index=False)

    (loaded,) = load_tilings(
        dataset=Dataset(dataset_csv),
        tiling_dir=tiling_dir,
        requested_seg_downsample=64,
        tissue_mask_tissue_value=1,
    )

    result = loaded.tiling_result
    assert loaded.slide.sample_id == "s0"
    assert result.num_tiles == 4
    assert sorted(zip(result.x.tolist(), result.y.tolist())) == [
        (0, 0), (0, 64), (64, 0), (64, 64)
    ]
    np.testing.assert_allclose(result.tissue_fractions, np.ones(4))
    assert result.mask_level == 0
    assert result.mask_spacing_um == 0.5
