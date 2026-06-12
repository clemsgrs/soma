"""CompositeDenseFeatureStore — load-time multi-encoder channel concatenation.

The paper's headline (arXiv:2602.18747): concatenating per-head attention from several
frozen FMs gives a richer per-pixel vector (+7.95% mean Dice). Per-pixel resolution makes
this **resolution-agnostic** — each member encoder's ``(K_i, gh_i, gw_i)`` grid is
upsampled to the shared mask ``target_size`` (via its *own* geometry / ``crop_box``), then
channels stack into ``(ΣK_i, H, W)``. No feature-space grid resampling: heterogeneous
patch sizes / token grids just land on the common target.

This is a thin **read-time** view over independent per-member
:class:`~soma.dense.store.DenseFeatureStore`s (each with its own cache); there is no
separate composite cache. It presents the same surface the segmentation folds consume
(``load`` / ``geometry`` / ``metadata`` / ``feature_dim`` / ``validate_coverage``), but the
grids it returns are already at **target pixel resolution**, so the composite geometry is
trivial (``patch_size=(1, 1)`` ⇒ the head's interpolate-to-encoded + crop is an identity).

v1 constraint: all members must share the same ``target_size`` (the supervision size) and
read-``spacing_um`` per sample — heterogeneous *per-member native spacing* is deferred
(design §8). Members may still differ freely in patch size and token grid ``(gh_i, gw_i)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from soma.dense.geometry import DenseGridGeometry, compute_dense_geometry, normalize_hw
from soma.dense.store import DenseFeatureStore

__all__ = ["resample_grid_to_target", "CompositeDenseFeatureStore"]


def resample_grid_to_target(grid: torch.Tensor, geometry: DenseGridGeometry) -> torch.Tensor:
    """Upsample a token grid ``(C, gh, gw)`` to per-pixel features ``(C, H, W)``.

    Mirrors :meth:`SegmentationHead.forward` geometry (bilinear-interpolate to the padded
    ``encoded_size``, crop ``crop_box`` to ``target_size``) so a member's attention maps
    land on the supervision pixel grid exactly where the head/decoder would place logits.
    """
    up = F.interpolate(
        grid.unsqueeze(0).float(), size=geometry.encoded_size, mode="bilinear", align_corners=False
    )
    top, left, height, width = geometry.crop_box
    return up[0, :, top : top + height, left : left + width]


class CompositeDenseFeatureStore:
    """Read-time channel-concat over several per-member dense stores."""

    def __init__(self, members: list[DenseFeatureStore]) -> None:
        if not members:
            raise ValueError("CompositeDenseFeatureStore requires at least one member store.")
        self._members = members
        # Sample coverage = intersection across members (a sample must be present in all).
        common = set(members[0].available_samples)
        for member in members[1:]:
            common &= set(member.available_samples)
        self._available = sorted(common)
        if not self._available:
            raise ValueError(
                "CompositeDenseFeatureStore members share no common samples — each sample "
                "must be extracted by every member encoder."
            )
        self._feature_dim = sum(int(m.feature_dim) for m in members)
        self._target_size: tuple[int, int] | None = None

    # --- shape / geometry (uniform across the cohort, like DenseFeatureStore) ---

    @property
    def available_samples(self) -> list[str]:
        return list(self._available)

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    def _resolve_target_size(self, sample_id: str) -> tuple[int, int]:
        # All members must agree on the supervision target_size for this sample.
        sizes = {tuple(int(v) for v in m.metadata(sample_id)["target_size"]) for m in self._members}
        if len(sizes) != 1:
            raise ValueError(
                f"composite members disagree on target_size for '{sample_id}': {sorted(sizes)}. "
                "v1 multi-encoder concat requires a shared supervision size across members."
            )
        target = normalize_hw(next(iter(sizes)), name="target_size")
        if self._target_size is None:
            self._target_size = target
        elif target != self._target_size:
            raise ValueError(
                f"composite target_size for '{sample_id}' ({target}) differs from the cohort "
                f"target ({self._target_size}); v1 requires a uniform tile/grid size."
            )
        return target

    def geometry(self, sample_id: str) -> DenseGridGeometry:
        # Composite grids are already at target pixel resolution → patch_size (1, 1) makes
        # encoded_size == grid_shape == target_size and crop_box the full frame, so the
        # head's interpolate+crop is an identity on the concatenated grid.
        target = self._resolve_target_size(sample_id)
        return compute_dense_geometry(target_size=target, patch_size=(1, 1))

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.geometry(self._available[0]).grid_shape

    def metadata(self, sample_id: str) -> dict:
        target = self._resolve_target_size(sample_id)
        spacings = {m.metadata(sample_id).get("spacing_um") for m in self._members}
        if len(spacings) != 1:
            raise ValueError(
                f"composite members disagree on read-spacing for '{sample_id}': {sorted(spacings)}. "
                "v1 requires all members read at the same µm/px (per-member native spacing is "
                "deferred)."
            )
        return {
            "feature_type": "dense_grid",
            "feature_kind": "composite",
            "feature_dim": self._feature_dim,
            "channel_dim": 0,
            "grid_shape": [int(target[0]), int(target[1])],
            "target_size": [int(target[0]), int(target[1])],
            "patch_size": [1, 1],
            "spacing_um": next(iter(spacings)),
            "members": [
                {
                    "feature_dim": int(m.feature_dim),
                    "feature_kind": m.metadata(sample_id).get("feature_kind"),
                    "attention_blocks": m.metadata(sample_id).get("attention_blocks"),
                    "attention_include_registers": m.metadata(sample_id).get(
                        "attention_include_registers"
                    ),
                    "channel_order": m.metadata(sample_id).get("channel_order"),
                }
                for m in self._members
            ],
        }

    def load(self, sample_id: str) -> torch.Tensor:
        """Concatenate every member's upsampled grid → ``(ΣK_i, H, W)`` at target res.

        Channel order is list order × each member's own ``[block][cls, reg…][head]``.
        """
        self._resolve_target_size(sample_id)  # validate agreement before stacking
        parts = [
            resample_grid_to_target(m.load(sample_id), m.geometry(sample_id))
            for m in self._members
        ]
        return torch.cat(parts, dim=0)

    def validate_coverage(self, sample_ids: list[str]) -> None:
        available = set(self._available)
        missing = sorted(set(sample_ids) - available)
        if missing:
            raise ValueError(
                f"Missing composite dense features for {len(missing)} samples: {missing} "
                "(a sample must be extracted by every member encoder)."
            )

    def __len__(self) -> int:
        return len(self._available)
