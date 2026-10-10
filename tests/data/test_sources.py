"""Sources and adapters: the only way features enter soma (design §4.2)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from soma.data import (
    Cohort,
    GridGeometry,
    SetSource,
    covers,
    from_arrays,
    from_directory,
    group_by,
    require_coverage,
)
from soma.testing import ConformanceError, check_set_source


def test_from_arrays_declares_rank_from_the_tensors() -> None:
    bags = {"a": torch.ones(5, 8), "b": torch.zeros(3, 8)}
    coords = {"a": torch.arange(10).reshape(5, 2), "b": torch.arange(6).reshape(3, 2)}
    source = from_arrays(bags, coords=coords)

    assert isinstance(source, SetSource)
    assert source.sample_ids == ["a", "b"]
    assert source.feature_dim == 8
    assert source.rank == 2
    assert torch.equal(source.load("b"), torch.zeros(3, 8))
    assert torch.equal(source.coords("a"), coords["a"])
    check_set_source(source)


def test_from_arrays_accepts_numpy_vectors_and_hierarchical_tensors() -> None:
    vectors = from_arrays({"a": np.arange(4, dtype=np.float32)})
    assert vectors.rank == 1 and vectors.feature_dim == 4
    assert vectors.coords("a") is None
    check_set_source(vectors)

    hipt = from_arrays({"a": torch.zeros(2, 3, 6), "b": torch.zeros(1, 3, 6)})
    assert hipt.rank == 3 and hipt.feature_dim == 6
    check_set_source(hipt)


def test_from_arrays_rejects_mixed_ranks_and_dims() -> None:
    with pytest.raises(ValueError, match="rank"):
        from_arrays({"a": torch.zeros(4), "b": torch.zeros(2, 4)})
    with pytest.raises(ValueError, match="feature_dim"):
        from_arrays({"a": torch.zeros(2, 4), "b": torch.zeros(2, 5)})


def test_unknown_sample_raises_key_error() -> None:
    source = from_arrays({"a": torch.zeros(2, 4)})
    with pytest.raises(KeyError, match="ghost"):
        source.load("ghost")


def test_from_directory_reads_every_supported_format_with_sibling_coords(tmp_path: Path) -> None:
    torch.save(torch.ones(3, 4), tmp_path / "a.pt")
    np.save(tmp_path / "b.npy", np.full((2, 4), 2.0, dtype=np.float32))
    np.savez(tmp_path / "c.npz", features=np.full((1, 4), 3.0, dtype=np.float32))
    with h5py.File(tmp_path / "d.h5", "w") as handle:
        handle.create_dataset("features", data=np.full((4, 4), 4.0, dtype=np.float32))
    np.save(tmp_path / "a.coords.npy", np.arange(6).reshape(3, 2))

    source = from_directory(tmp_path)

    assert source.sample_ids == ["a", "b", "c", "d"]
    assert source.rank == 2 and source.feature_dim == 4
    assert torch.equal(source.load("c"), torch.full((1, 4), 3.0))
    assert torch.equal(source.load("d"), torch.full((4, 4), 4.0))
    assert torch.equal(source.coords("a"), torch.arange(6).reshape(3, 2))
    assert source.coords("b") is None
    check_set_source(source)


def test_from_directory_maps_ids_explicitly(tmp_path: Path) -> None:
    torch.save(torch.ones(4), tmp_path / "slide_001.pt")
    torch.save(torch.zeros(4), tmp_path / "slide_002.pt")
    source = from_directory(tmp_path, ids={"first": "slide_001", "second": "slide_002"})
    assert source.sample_ids == ["first", "second"]
    assert torch.equal(source.load("second"), torch.zeros(4))
    with pytest.raises(FileNotFoundError, match="slide_003"):
        from_directory(tmp_path, ids={"third": "slide_003"})


def test_from_directory_checks_rank_across_files(tmp_path: Path) -> None:
    torch.save(torch.ones(4), tmp_path / "a.pt")
    torch.save(torch.ones(2, 4), tmp_path / "b.pt")
    source = from_directory(tmp_path)
    with pytest.raises(ConformanceError) as info:
        check_set_source(source)
    assert info.value.method == "load" and info.value.sample_id == "b"


def test_covers_and_require_coverage() -> None:
    source = from_arrays({"a": torch.zeros(4), "b": torch.zeros(4)})
    assert covers(source, ["a", "b"])
    assert not covers(source, ["a", "zzz"])
    with pytest.raises(ValueError, match="zzz"):
        require_coverage(source, ["a", "zzz"])


def _patient_cohort() -> Cohort:
    frame = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s2", "s3"],
            "patient_id": ["p0", "p0", "p1", "p1"],
            "label": [1, 1, 0, 0],
            "split": ["train", "train", "test", "test"],
        }
    )
    return Cohort.from_frames(frame, targets=["label"])


def test_group_by_concat_builds_patient_bags_from_tile_bags() -> None:
    tiles = from_arrays(
        {
            "s0": torch.full((2, 3), 0.0),
            "s1": torch.full((1, 3), 1.0),
            "s2": torch.full((3, 3), 2.0),
            "s3": torch.full((1, 3), 3.0),
        },
        coords={sid: torch.zeros(n, 2) for sid, n in {"s0": 2, "s1": 1, "s2": 3, "s3": 1}.items()},
    )
    patients = group_by(tiles, _patient_cohort(), unit="patient_id", how="concat")

    assert patients.sample_ids == ["p0", "p1"]
    assert patients.rank == 2 and patients.feature_dim == 3
    bag = patients.load("p0")
    assert bag.shape == (3, 3)
    assert bag[:, 0].tolist() == [0.0, 0.0, 1.0]
    assert patients.coords("p1").shape == (4, 2)
    check_set_source(patients)


def test_group_by_stack_builds_patient_embeddings_from_slide_vectors() -> None:
    slides = from_arrays({f"s{i}": torch.full((5,), float(i)) for i in range(4)})
    patients = group_by(slides, _patient_cohort(), unit="patient_id", how="stack")
    assert patients.rank == 2 and patients.feature_dim == 5
    assert patients.load("p1")[:, 0].tolist() == [2.0, 3.0]
    assert patients.coords("p1") is None
    check_set_source(patients)


def test_group_by_stack_of_bags_is_rank_three_and_needs_equal_bag_sizes() -> None:
    bags = from_arrays({f"s{i}": torch.zeros(2, 3) for i in range(4)})
    patients = group_by(bags, _patient_cohort(), unit="patient_id", how="stack")
    assert patients.rank == 3 and patients.load("p0").shape == (2, 2, 3)
    check_set_source(patients)

    ragged = from_arrays({"s0": torch.zeros(2, 3), "s1": torch.zeros(1, 3), "s2": torch.zeros(1, 3), "s3": torch.zeros(1, 3)})
    with pytest.raises(ValueError, match="p0"):
        group_by(ragged, _patient_cohort(), unit="patient_id", how="stack").load("p0")


def test_group_by_sample_id_is_the_identity() -> None:
    slides = from_arrays({f"s{i}": torch.zeros(5) for i in range(4)})
    assert group_by(slides, _patient_cohort(), unit="sample_id") is slides


def test_grid_geometry_maps_tokens_to_level0_pixels() -> None:
    geometry = GridGeometry.from_sizes(
        target_size=(100, 60),
        patch_size=14,
        origin_level0=(1000, 2000),
        level0_px_per_token_px=2.0,
    )
    assert geometry.grid_shape == (8, 5)
    assert geometry.encoded_size == (112, 70)
    # Token (i=1, j=2): centre at (2.5 * 14, 1.5 * 14) token-frame px, scaled by 2 and offset.
    assert geometry.token_to_level0((1, 2)) == (1070.0, 2042.0)
    xy = geometry.token_to_level0(np.array([[0, 0], [7, 4]]))
    np.testing.assert_allclose(xy, [[1014.0, 2014.0], [1126.0, 2210.0]])
