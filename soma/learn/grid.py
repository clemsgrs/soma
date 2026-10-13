"""The grid shape's batch, dataset and loaders (design §7.2).

A :class:`GridBatch` carries the model's input — cached feature grids from a
:class:`~soma.data.GridSource`, or pixels from an :class:`~soma.data.ImageSource` that a
frozen encoder stage inside :class:`~soma.models.model.DenseModel` re-encodes — plus
the raw physical-unit supervision of each sample. Heads encode that supervision against
their bound geometry at loss time, so one annotation serves any encoder stride.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from soma.data import Points, require_coverage
from soma.models.detection_targets import points_on_valid

__all__ = ["GridBatch", "GridDataset", "grid_collate", "grid_loaders", "ignore_to_frame"]

_PAD = float("nan")
#: Ignore-mask value for a pixel nobody annotated (0 = supervised).
_IGNORED = 255


@dataclass(frozen=True)
class GridBatch:
    """One collated grid batch.

    ``features`` ``(B, D, h, w)`` are cached token grids; ``pixels`` ``(B, 3, H, W)`` are
    image pixels. Exactly one of them is the model's input: ``features`` on the cached
    path, ``pixels`` on the live path, where the model's encoder stage turns them into
    grids. An image-prior decoder reads ``pixels`` as a named side input, so a cached
    batch for such a decoder carries both. ``targets`` holds supervision only:
    ``{"mask": (B, H, W) long}`` or ``{"points": (B, K, 3) level-0 (x, y, class) NaN-padded,
    "ignore": (B, H, W) bool}`` (``ignore`` absent when no sample has an ignore mask).
    """

    features: Tensor | None
    pixels: Tensor | None
    targets: dict[str, Tensor]
    sample_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.features is None and self.pixels is None:
            raise ValueError("a GridBatch needs features (cached grids) or pixels (live)")


def ignore_to_frame(ignore_mask: np.ndarray, geometry) -> Tensor:
    """Map a level-0 ignore raster (255 = ignored) onto the geometry's target frame.

    Target pixel ``(u, v)`` reads the nearest level-0 pixel under the geometry's level-0
    anchor; pixels that fall outside the raster are ignored. Returns ``(H, W)`` bool,
    ``True`` where nothing is supervised.
    """
    mask = np.asarray(ignore_mask)
    if mask.ndim != 2:
        raise ValueError(f"ignore mask must be 2-D, got shape {mask.shape}")
    height, width = (int(v) for v in geometry.target_size)
    top, left = int(geometry.crop_box[0]), int(geometry.crop_box[1])
    scale = float(geometry.level0_px_per_token_px)
    origin_x, origin_y = (float(v) for v in geometry.origin_level0)

    def axis(n: int, offset: int, origin: float, extent: int) -> tuple[np.ndarray, np.ndarray]:
        position = (np.arange(n) + float(offset)) * scale + origin
        inside = (position >= 0) & (position < extent)
        return inside, np.clip(np.rint(position[inside]).astype(np.int64), 0, extent - 1)

    row_in, rows = axis(height, top, origin_y, mask.shape[0])
    col_in, cols = axis(width, left, origin_x, mask.shape[1])
    ignored = np.ones((height, width), dtype=bool)
    ignored[np.ix_(row_in, col_in)] = mask[np.ix_(rows, cols)] == _IGNORED
    return torch.from_numpy(ignored)


def _uint8_chw(pixels: Any, sample_id: str) -> Tensor:
    if not torch.is_tensor(pixels) or pixels.ndim != 3 or int(pixels.shape[0]) != 3:
        raise ValueError(
            f"image source returned {getattr(pixels, 'shape', type(pixels))} for "
            f"'{sample_id}'; an ImageSource serves (3, H, W) uint8 pixels"
        )
    if pixels.dtype != torch.uint8:
        raise ValueError(f"image source returned {pixels.dtype} pixels for '{sample_id}'; want uint8")
    return pixels


class GridDataset(Dataset):
    """One split's grid samples: the model input plus the raw supervision.

    Args:
        sample_ids: The split's ids, in order.
        source: A :class:`~soma.data.GridSource` (``inputs="features"``) or an
            :class:`~soma.data.ImageSource` (``inputs="pixels"``).
        targets: The :class:`~soma.data.TargetSource` serving each sample's label map or
            points.
        inputs: ``"features"`` or ``"pixels"``.
        geometry: The head's bound geometry; label maps must match its target size and
            ignore rasters are mapped onto its frame.
        target_kind: ``"label_map"`` or ``"points"``.
        images: An :class:`~soma.data.ImageSource` whose pixels ride along with cached
            features as an image-prior decoder's side input.
        augment: ``(pixels, target) -> (pixels, target)``, applied before ``preprocess``.
        preprocess: The encoder stage's itemwise preprocessor, applied to the pixels.
        check: ``(sample_id, target)`` validation run on each raw target (the head's
            ``check_target``), so bad supervision fails naming its sample.
    """

    def __init__(
        self,
        sample_ids: Sequence[str],
        source,
        targets,
        *,
        inputs: str,
        geometry,
        target_kind: str,
        images=None,
        augment: Callable | None = None,
        preprocess: Callable | None = None,
        check: Callable[[str, Any], None] | None = None,
    ) -> None:
        self._ids = [str(sid) for sid in sample_ids]
        self._check = check
        self._source = source
        self._targets = targets
        self._inputs = inputs
        self._geometry = geometry
        self._target_kind = target_kind
        self._images = images
        self._augment = augment
        self._preprocess = preprocess

    @property
    def sample_ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def _encode(self, sample_id: str, target: Any) -> dict[str, Tensor]:
        if self._target_kind == "label_map":
            if not torch.is_tensor(target) or target.ndim != 2:
                raise ValueError(
                    f"target source returned {type(target).__name__} for '{sample_id}'; "
                    "a segmentation head needs a (H, W) label map"
                )
            size = tuple(int(v) for v in target.shape)
            if size != tuple(int(v) for v in self._geometry.target_size):
                raise ValueError(
                    f"label map for '{sample_id}' is {size} but the head's target size is "
                    f"{tuple(self._geometry.target_size)}"
                )
            return {"mask": target.to(torch.long)}
        if not isinstance(target, Points):
            raise ValueError(
                f"target source returned {type(target).__name__} for '{sample_id}'; "
                "a detection head needs Points"
            )
        rows = torch.cat(
            [target.xy.to(torch.float32), target.classes.to(torch.float32).unsqueeze(1)], dim=1
        )
        encoded = {"points": rows.reshape(-1, 3)}
        if target.ignore_mask is not None:
            # Drop points on an ignored level-0 pixel (the rule ``evaluate`` applies): the
            # framed map alone misses them once a frame pixel spans several level-0 pixels.
            ignored = np.asarray(target.ignore_mask) == _IGNORED
            on_ignored = points_on_valid(target.xy.detach().cpu().numpy(), ignored)
            encoded["points"] = encoded["points"][torch.from_numpy(~on_ignored)]
            encoded["ignore"] = ignore_to_frame(target.ignore_mask, self._geometry)
        return encoded

    def __getitem__(self, index: int):
        sample_id = self._ids[index]
        features = pixels = None
        if self._inputs == "features":
            features = self._source.load(sample_id)
            if self._images is not None:
                pixels = _uint8_chw(self._images.load(sample_id), sample_id)
                size = tuple(int(v) for v in self._geometry.target_size)
                if tuple(int(v) for v in pixels.shape[1:]) != size:
                    raise ValueError(
                        f"image for '{sample_id}' is {pixels.shape[1]}x{pixels.shape[2]} but its "
                        f"mask is {size[0]}x{size[1]}; an image-prior decoder needs them aligned."
                    )
        else:
            pixels = _uint8_chw(self._source.load(sample_id), sample_id)
        target = self._targets.load(sample_id)
        if self._check is not None:
            self._check(sample_id, target)
        if self._augment is not None:
            pixels, target = self._augment(pixels, target)
        if self._preprocess is not None:
            pixels = self._preprocess(pixels.contiguous())
        return features, pixels, self._encode(sample_id, target), sample_id


def _stack(tensors: Sequence[Tensor], what: str) -> Tensor:
    try:
        return torch.stack(list(tensors))
    except RuntimeError as error:
        shapes = sorted({tuple(t.shape) for t in tensors})
        raise ValueError(f"{what} in a batch must share one shape; got {shapes}") from error


def grid_collate(items: Sequence[tuple]) -> GridBatch:
    """Stack features / pixels / label maps; NaN-pad points to the batch's largest count."""
    features, pixels, targets, sample_ids = zip(*items)
    batch_features = None if features[0] is None else _stack(features, "dense grids")
    batch_pixels = None if pixels[0] is None else _stack(pixels, "pixels")
    if "mask" in targets[0]:
        collated = {"mask": _stack([t["mask"] for t in targets], "label maps")}
    else:
        counts = [int(t["points"].shape[0]) for t in targets]
        padded = torch.full((len(targets), max(counts, default=0), 3), _PAD, dtype=torch.float32)
        for row, (target, count) in enumerate(zip(targets, counts)):
            if count:
                padded[row, :count] = target["points"]
        collated = {"points": padded}
        ignores = [t.get("ignore") for t in targets]
        if any(mask is not None for mask in ignores):
            shape = next(mask.shape for mask in ignores if mask is not None)
            collated["ignore"] = _stack(
                [torch.zeros(shape, dtype=torch.bool) if m is None else m for m in ignores],
                "ignore maps",
            )
    return GridBatch(
        features=batch_features,
        pixels=batch_pixels,
        targets=collated,
        sample_ids=tuple(str(sid) for sid in sample_ids),
    )


def _target_kind_of(targets, probe_id: str) -> str:
    declared = getattr(targets, "target_kind", None)
    if declared is not None:
        return str(declared)
    return "points" if isinstance(targets.load(probe_id), Points) else "label_map"


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-9)


def _check_geometry(source, head, sample_ids: Sequence[str], *, inputs: str, target_kind: str) -> None:
    """The head's geometry is bound at construction: every sample must share it."""
    bound = head.geometry
    spacing_um = getattr(head, "spacing_um", None)
    for sample_id in sample_ids:
        geometry = source.geometry(sample_id)
        if inputs == "features":
            if geometry.layout != bound.layout:
                raise ValueError(
                    f"grid '{sample_id}' has layout {geometry.layout} but the head is bound to "
                    f"{bound.layout}; a grid fold needs one tile/grid layout"
                )
        elif tuple(geometry.target_size) != tuple(bound.target_size):
            raise ValueError(
                f"image '{sample_id}' is served at {tuple(geometry.target_size)} but the "
                f"head's target size is {tuple(bound.target_size)}"
            )
        if target_kind != "points":
            continue
        anchor = (*geometry.origin_level0, geometry.level0_px_per_token_px)
        expected = (*bound.origin_level0, bound.level0_px_per_token_px)
        if not all(_close(a, b) for a, b in zip(anchor, expected)):
            raise ValueError(
                f"sample '{sample_id}' maps level-0 pixels to its frame with origin "
                f"{geometry.origin_level0} and scale {geometry.level0_px_per_token_px}, but "
                f"the head is bound to origin {bound.origin_level0} and scale "
                f"{bound.level0_px_per_token_px}; point targets need one level-0 anchor "
                "per fold (bind the head with GridGeometry.from_dense(..., "
                "level0_px_per_token_px=...))"
            )
        if spacing_um is not None and not _close(source.spacing(sample_id), spacing_um):
            raise ValueError(
                f"sample '{sample_id}' is read at {source.spacing(sample_id)} µm/px but the "
                f"head converts its µm distances at {spacing_um} µm/px"
            )


def grid_loaders(
    source,
    *,
    head,
    train: Sequence[str],
    tune: Sequence[str] | None,
    tests: dict[str, Sequence[str]],
    batch_size: int = 1,
    num_workers: int = 0,
    targets=None,
    augment: Callable | None = None,
    preprocess: Callable | None = None,
    sampler=None,
    images=None,
    loader_kwargs: dict[str, Any] | None = None,
):
    """The grid branch of :func:`soma.learn.loaders` over explicit split ids.

    ``tune`` may repeat ``train`` (a pipeline fold without a tune split); see
    :func:`soma.learn.loaders` for the arguments.
    """
    from soma.learn.loaders import Loaders

    loader_kwargs = dict(loader_kwargs or {})

    target_kind = getattr(head, "target_kind", None)
    if target_kind not in ("label_map", "points"):
        raise ValueError(
            f"{type(head).__name__} is not a grid head: a grid source needs a head declaring "
            "target_kind 'label_map' or 'points'"
        )
    if targets is None:
        raise ValueError("the grid shape reads supervision from targets=: pass a TargetSource")
    inputs = "features" if hasattr(source, "feature_dim") else "pixels"
    if inputs == "features":
        if augment is not None:
            raise ValueError(
                "cached grids cannot be augmented: the features are frozen; augment an "
                "ImageSource re-encoded by the model's encoder stage instead"
            )
        if preprocess is not None:
            raise ValueError("preprocess= prepares pixels for an encoder stage; cached grids take none")
    elif images is not None:
        raise ValueError("images= rides along with cached grids; a live source already serves pixels")
    if target_kind == "points" and augment is not None:
        raise ValueError(
            "point targets cannot be augmented yet: joint point augmentation waits for "
            "torchvision 0.23 tv_tensors.KeyPoints (soma#586)"
        )
    ids = list(dict.fromkeys([*train, *(tune or ()), *(sid for split in tests.values() for sid in split)]))
    require_coverage(source, ids, what="pixels" if inputs == "pixels" else "features")
    require_coverage(targets, ids, what="targets")
    if images is not None:
        require_coverage(images, ids, what="pixels")
    served = _target_kind_of(targets, ids[0])
    if served != target_kind:
        raise ValueError(
            f"{type(head).__name__} reads {target_kind} targets but the target source serves "
            f"{served} targets"
        )
    _check_geometry(source, head, ids, inputs=inputs, target_kind=target_kind)

    def dataset(split_ids, augmentation) -> GridDataset:
        return GridDataset(
            split_ids,
            source,
            targets,
            inputs=inputs,
            geometry=head.geometry,
            target_kind=target_kind,
            images=images,
            augment=augmentation,
            preprocess=preprocess,
            check=getattr(head, "check_target", None),
        )

    def make(split_ids, *, shuffle: bool, augmentation=None, batch_sampler=None) -> DataLoader:
        if batch_sampler is not None:
            return DataLoader(
                dataset(split_ids, augmentation),
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                collate_fn=grid_collate,
                **loader_kwargs,
            )
        return DataLoader(
            dataset(split_ids, augmentation),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=grid_collate,
            **loader_kwargs,
        )

    return Loaders(
        train=make(train, shuffle=True, augmentation=augment, batch_sampler=sampler),
        tune=make(tune, shuffle=False) if tune else None,
        tests={name: make(split_ids, shuffle=False) for name, split_ids in tests.items()},
    )
