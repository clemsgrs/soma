"""CompositeDenseFeatureStore — load-time multi-encoder channel concatenation.

The paper's headline (arXiv:2602.18747): concatenating per-pixel features from several
frozen FMs gives a richer per-pixel vector (+7.95% mean Dice). This store is a thin
**read-time** view over independent per-member :class:`~soma.dense.store.DenseFeatureStore`s
(each with its own cache); there is no separate composite cache. It presents the same
surface the dense folds consume (``load`` / ``geometry`` / ``metadata`` / ``feature_dim`` /
``validate_coverage``).

It serves two consumers with opposite spatial needs via ``concat_resolution`` (design §7):

- ``"target"`` *(the decoder-free per-pixel classifier)*: each member's ``(K_i, gh_i,
  gw_i)`` grid is upsampled to the shared mask ``target_size`` via its **own** geometry
  (:func:`resample_grid_to_target`), then channels stack into ``(ΣK_i, H, W)``. Per-pixel
  resolution makes this resolution-agnostic; the composite geometry is trivial
  (``patch_size=(1, 1)`` ⇒ the head's interpolate+crop is an identity).
- ``"grid"`` *(the trained decoder / detection)*: each member's **native** grid is
  bilinearly resampled to a common token grid ``(h, w)`` (decision B: pad fraction
  ignored), then channels stack into ``(ΣK_i, h, w)``. The decoder upsamples from
  ``(h, w)`` to the mask as usual, so it runs at token resolution (not full mask
  resolution). The reported geometry has ``grid_shape=(h, w)``, ``encoded_size=target``,
  ``crop_box`` = full frame — so the head's ``interpolate→encoded→crop`` maps the decoder
  output to the mask and the auto-upsample-depth (``ceil(log2(encoded/grid))``) tells the
  decoder to learn the ``(h, w)→target`` upsampling.

``member_norm`` (per member) is applied at load time **after** the resample, before
concat, so a large-magnitude encoder does not dominate the decoder.

v1 constraint: all members must share the same ``target_size`` and read-``spacing_um`` per
sample — heterogeneous *per-member native spacing* is deferred (design §7). Members may
still differ freely in patch size and token grid ``(gh_i, gw_i)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from soma.dense.geometry import DenseGridGeometry, compute_dense_geometry, normalize_hw
from soma.dense.source import DenseSampleSpacing
from soma.dense.store import DenseFeatureStore

__all__ = ["resample_grid_to_target", "apply_member_norm", "CompositeDenseFeatureStore"]

_VALID_RESOLUTIONS = {"grid", "target"}
_VALID_NORMS = {"none", "l2", "layernorm"}


def resample_grid_to_target(grid: torch.Tensor, geometry: DenseGridGeometry) -> torch.Tensor:
    """Upsample a token grid ``(C, gh, gw)`` to per-pixel features ``(C, H, W)``.

    Mirrors :meth:`SegmentationHead.forward` geometry (bilinear-interpolate to the padded
    ``encoded_size``, crop ``crop_box`` to ``target_size``) so a member's maps land on the
    supervision pixel grid exactly where the head/decoder would place logits.
    """
    up = F.interpolate(
        grid.unsqueeze(0).float(), size=geometry.encoded_size, mode="bilinear", align_corners=False
    )
    top, left, height, width = geometry.crop_box
    return up[0, :, top : top + height, left : left + width]


def _resample_grid_to_size(grid: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resample a ``(C, gh, gw)`` grid to ``(C, h, w)`` (decision B).

    Ignores each member's few-px pad fraction — the native grid is treated as spanning the
    target FOV. Cheap, no full-resolution intermediate.
    """
    return F.interpolate(
        grid.unsqueeze(0).float(), size=size, mode="bilinear", align_corners=False
    )[0]


def apply_member_norm(grid: torch.Tensor, norm: str) -> torch.Tensor:
    """Per-pixel channel normalization over channel dim 0 of a ``(C, ·, ·)`` grid.

    ``l2`` rescales each pixel's feature vector to unit L2; ``layernorm`` is affine-free
    per-pixel channel standardization; ``none`` is the identity.
    """
    if norm == "none":
        return grid
    if norm == "l2":
        return F.normalize(grid, p=2.0, dim=0, eps=1e-8)
    if norm == "layernorm":
        mean = grid.mean(dim=0, keepdim=True)
        std = grid.std(dim=0, keepdim=True, unbiased=False)
        return (grid - mean) / (std + 1e-8)
    raise ValueError(f"Unknown member_norm {norm!r}; must be one of {sorted(_VALID_NORMS)}.")


class CompositeDenseFeatureStore:
    """Read-time channel-concat over several per-member dense stores."""

    def __init__(
        self,
        members: list[DenseFeatureStore],
        *,
        concat_resolution: str = "target",
        concat_grid_size: tuple[int, int] | None = None,
        member_norms: list[str] | None = None,
    ) -> None:
        if not members:
            raise ValueError("CompositeDenseFeatureStore requires at least one member store.")
        if concat_resolution not in _VALID_RESOLUTIONS:
            raise ValueError(
                f"concat_resolution must be one of {sorted(_VALID_RESOLUTIONS)}, "
                f"got {concat_resolution!r}."
            )
        if member_norms is None:
            member_norms = ["none"] * len(members)
        if len(member_norms) != len(members):
            raise ValueError(
                f"member_norms ({len(member_norms)}) must align with members ({len(members)})."
            )
        for norm in member_norms:
            if norm not in _VALID_NORMS:
                raise ValueError(
                    f"member_norm must be one of {sorted(_VALID_NORMS)}, got {norm!r}."
                )
        self._members = members
        self._concat_resolution = concat_resolution
        self._member_norms = list(member_norms)
        self._concat_grid_size = (
            normalize_hw(concat_grid_size, name="concat_grid_size")
            if concat_grid_size is not None
            else None
        )
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

    def _common_grid_size(self) -> tuple[int, int]:
        """Common ``(h, w)`` for ``grid`` mode: configured, else the largest member grid.

        Member grids are cohort-uniform (fixed ``target_size`` + per-encoder patch size),
        so the largest grid is constant across samples ⇒ batchable.
        """
        if self._concat_grid_size is not None:
            return self._concat_grid_size
        grids = [m.grid_shape for m in self._members]
        return (max(g[0] for g in grids), max(g[1] for g in grids))

    def geometry(self, sample_id: str) -> DenseGridGeometry:
        target = self._resolve_target_size(sample_id)
        if self._concat_resolution == "target":
            # Composite grids are already at target pixel resolution → patch_size (1, 1)
            # makes encoded_size == grid_shape == target_size and crop_box the full frame,
            # so the head's interpolate+crop is an identity on the concatenated grid.
            return compute_dense_geometry(target_size=target, patch_size=(1, 1))
        # grid mode: report the real (h, w) decoder-input grid spanning the target FOV.
        # encoded_size = target + crop = full frame (pad ignored, decision B); grid_shape =
        # (h, w) so the head upsamples the decoder output to target and the auto
        # num_upsample_blocks = ceil(log2(target/(h,w))) is correct. patch_size is the
        # nominal per-axis stride (cosmetic; the head uses encoded_size/crop_box).
        h, w = self._common_grid_size()
        target_h, target_w = target
        patch = (max(1, round(target_h / h)), max(1, round(target_w / w)))
        return DenseGridGeometry(
            target_size=(target_h, target_w),
            patch_size=patch,
            encoded_size=(target_h, target_w),
            grid_shape=(h, w),
            pad=(0, 0),
            crop_box=(0, 0, target_h, target_w),
        )

    @property
    def grid_shape(self) -> tuple[int, int]:
        return self.geometry(self._available[0]).grid_shape

    def metadata(self, sample_id: str) -> dict:
        target = self._resolve_target_size(sample_id)
        # Ask each member for its spacing rather than reading the raw sidecar key: slide2vec
        # spells the field differently per dense writer, and ``spacing_um()`` is the one
        # place that knows both spellings. Reading the key directly would make every
        # image-sourced member report ``None``, and this disagreement check would then pass
        # vacuously on a set of unknowns instead of comparing real spacings.
        spacings = {m.spacing_um(sample_id) for m in self._members}
        if len(spacings) != 1:
            raise ValueError(
                f"composite members disagree on read-spacing for '{sample_id}': {sorted(spacings)}. "
                "v1 requires all members read at the same µm/px (per-member native spacing is "
                "deferred)."
            )
        geom = self.geometry(sample_id)
        return {
            "feature_type": "dense_grid",
            "feature_kind": "composite",
            "feature_dim": self._feature_dim,
            "channel_dim": 0,
            "concat_resolution": self._concat_resolution,
            "grid_shape": [int(geom.grid_shape[0]), int(geom.grid_shape[1])],
            "target_size": [int(target[0]), int(target[1])],
            "patch_size": [int(geom.patch_size[0]), int(geom.patch_size[1])],
            "spacing_um": next(iter(spacings)),
            "members": [
                {
                    "feature_dim": int(m.feature_dim),
                    "feature_kind": m.metadata(sample_id).get("feature_kind"),
                    "member_norm": norm,
                    "attention_blocks": m.metadata(sample_id).get("attention_blocks"),
                    "attention_include_registers": m.metadata(sample_id).get(
                        "attention_include_registers"
                    ),
                    "channel_order": m.metadata(sample_id).get("channel_order"),
                }
                for m, norm in zip(self._members, self._member_norms)
            ],
        }

    def spacing_um(self, sample_id: str) -> float | None:
        value = self.metadata(sample_id).get("spacing_um")
        return None if value is None else float(value)

    def spacing(self, sample_id: str) -> DenseSampleSpacing:
        member_spacings = [member.spacing(sample_id) for member in self._members]
        for field in ("source_spacing_um", "effective_spacing_um"):
            values = [getattr(spacing, field) for spacing in member_spacings]
            if len(set(values)) != 1:
                raise ValueError(
                    f"composite members disagree on {field} for '{sample_id}': {values}."
                )
        return member_spacings[0]

    def load(self, sample_id: str) -> torch.Tensor:
        """Concatenate every member's resampled+normalized grid along the channel axis.

        ``target`` mode → ``(ΣK_i, H, W)`` at target pixel resolution; ``grid`` mode →
        ``(ΣK_i, h, w)`` at the common token resolution. Channel order is list order ×
        each member's own ``[block][cls, reg…][head]``.
        """
        self._resolve_target_size(sample_id)  # validate agreement before stacking
        if self._concat_resolution == "target":
            parts = [
                resample_grid_to_target(m.load(sample_id), m.geometry(sample_id))
                for m in self._members
            ]
        else:
            size = self._common_grid_size()
            parts = [_resample_grid_to_size(m.load(sample_id), size) for m in self._members]
        parts = [apply_member_norm(part, norm) for part, norm in zip(parts, self._member_norms)]
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
