"""The dense grid metadata soma writes beside each extracted grid (soma.extract.dense.store)."""

from __future__ import annotations


def test_dense_grid_metadata_treats_prenorm_as_patch_tokens_not_attention():
    from soma.data.geometry import compute_dense_geometry
    from soma.extract.dense.store import dense_grid_metadata

    geometry = compute_dense_geometry(target_size=16, patch_size=8)
    pre = dense_grid_metadata(
        geometry, feature_dim=4, pad_mode="reflect", feature_kind="patch_features_prenorm"
    )
    post = dense_grid_metadata(geometry, feature_dim=4, pad_mode="reflect", feature_kind="patch_features")
    assert pre["feature_kind"] == "patch_features_prenorm"
    assert {k: v for k, v in pre.items() if k != "feature_kind"} == {
        k: v for k, v in post.items() if k != "feature_kind"
    }
