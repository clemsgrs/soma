"""DenseFeatureStore — load cached dense ``(d, h, w)`` feature grids.

Unlike :class:`soma.features.FeatureStore`, which infers a feature's shape from
its tensor *rank* (1-D slide / 2-D bag / 3-D hierarchical), a dense grid is also
3-D ``(channels, grid_h, grid_w)`` and would be mis-read as hierarchical with
``feature_dim`` taken from the last axis (``w``) instead of the channel axis
(``d``). The dense store therefore reads geometry from a **per-sample
``.meta.json`` sidecar**, never from rank.

The sidecar is the single source of truth for shape, written next to every
``.pt`` regardless of whether the feature cache is enabled — so the store behaves
identically on cached, uncached, and (later) live-re-encode paths. ``cache_metadata.json``
is used only by the cache *validator*, not here.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
from slide2vec.artifacts import load_array

from soma.dataset import ensure_filename_safe_id
from soma.dense.geometry import DenseGridGeometry, compute_dense_geometry
from soma.dense.source import DenseSampleSpacing, dense_sample_spacing_from_metadata

logger = logging.getLogger(__name__)

__all__ = [
    "DENSE_SIDECAR_SUFFIX",
    "DENSE_ARTIFACT_TYPE",
    "DENSE_PAYLOAD_SUBDIR",
    "resolve_dense_payload_dir",
    "dense_grid_metadata",
    "write_dense_grid",
    "DenseFeatureStore",
]

DENSE_SIDECAR_SUFFIX = ".meta.json"
# slide2vec's name for a dense grid over a pre-cropped image. soma's remaining local
# writer — now only the test/composite fixture path — emits upstream's artifact_type so
# every dense sidecar the cache validator sees speaks one vocabulary, whoever wrote it.
DENSE_ARTIFACT_TYPE = "dense_image_embeddings"
# slide2vec writes its two dense artifact kinds to two directories, and soma's cache
# adopts them verbatim rather than translating (ADR 0007): ROI grids over a slide go to
# ``dense_embeddings/<slide>/<x>_<y>.pt``, grids over caller-supplied images to a flat
# ``dense_image_embeddings/<sample_id>.pt``. One cache key only ever holds one of them.
DENSE_PAYLOAD_SUBDIR = "dense_embeddings"
DENSE_IMAGE_PAYLOAD_SUBDIR = "dense_image_embeddings"

# Networked/pooled mounts (zfs/NFS/CIFS) intermittently raise OSError /
# FileNotFoundError on a grid read that succeeds moments later. A bare
# load_array turns one such blip into an aborted multi-fold dense run, so the
# store retries a small bounded number of times with linear backoff.
_LOAD_RETRIES = 5
_LOAD_BACKOFF_SECONDS = 0.5


def _load_array_resilient(path: Path):
    """Load a cached array, retrying transient filesystem errors with backoff.

    Retries up to ``_LOAD_RETRIES`` times on ``OSError`` (which includes
    ``FileNotFoundError``), sleeping ``_LOAD_BACKOFF_SECONDS * attempt`` between
    tries and logging each retry. The happy path — first read succeeds — adds no
    latency and no sleep. A genuinely-permanent error still surfaces (unchanged
    type) once the retries are exhausted.
    """
    last_error: OSError | None = None
    for attempt in range(1, _LOAD_RETRIES + 1):
        try:
            return load_array(path)
        except OSError as error:  # FileNotFoundError is a subclass of OSError
            last_error = error
            if attempt == _LOAD_RETRIES:
                break
            delay = _LOAD_BACKOFF_SECONDS * attempt
            logger.warning(
                "Transient error reading dense grid %s (attempt %d/%d): %s; retrying in %.1fs",
                path,
                attempt,
                _LOAD_RETRIES,
                error,
                delay,
            )
            time.sleep(delay)
    assert last_error is not None  # loop only exits via return or a caught error
    raise last_error


def resolve_dense_payload_dir(path: Path | str) -> Path:
    """Resolve the directory holding dense ``.pt`` grids.

    Dense-specific on purpose: it descends into whichever of slide2vec's two dense
    payload dirs is present — ``dense_embeddings/`` (ROI grids over a slide) or
    ``dense_image_embeddings/`` (grids over caller-supplied images) — else treats
    ``path`` as a plain payload dir. A single cache key holds one artifact kind, so the
    two are never both present and the probe order carries no meaning. Unlike the generic
    pooled ``resolve_feature_payload_dir``, it never falls through to a sibling
    ``tile_embeddings``/``slide_embeddings`` dir — so a cache root that happens to
    hold both pooled and dense payloads still resolves to the dense grids.
    """
    root = Path(path)
    for subdir in (DENSE_PAYLOAD_SUBDIR, DENSE_IMAGE_PAYLOAD_SUBDIR):
        candidate = root / subdir
        if candidate.is_dir():
            return candidate
    return root


def dense_grid_metadata(
    geometry: DenseGridGeometry,
    *,
    feature_dim: int,
    pad_mode: str,
    image_pad_value: float | None = None,
    mask_pad_value: int | None = None,
    dense_input_mode: str = "whole",
    window_size: int | None = None,
    overlap: float = 0.0,
    spacing_um: float | None = None,
    feature_kind: str = "patch_features",
    attention_blocks: tuple[int, ...] | None = None,
    attention_include_registers: bool = False,
    channel_dim: int = 0,
) -> dict:
    """Build the self-describing sidecar payload for one dense grid.

    Carries everything the decoder/head and evaluation need to map the grid back
    to the mask: the channel/grid layout plus the §5 padding+crop geometry.

    ``image_pad_value`` is meaningful only for constant/zero padding (``None`` for
    reflect). ``mask_pad_value`` (the mask's ``ignore_index``) is forward-looking:
    the feature-grid extractor does not own mask semantics, so it is left ``None``
    here and set by the segmentation dataset/collate slice, which pads masks with
    ``ignore_index`` using this same geometry.

    ``spacing_um`` is the µm/px the tile was *read* at (``None`` = flat page-0 read).
    It is recorded so the segmentation fold can assert the mask is read at the same
    spacing the grid was extracted at — otherwise the supervision silently shifts
    against the features. (It is also part of the cache *key*, via the plumbed
    ``PreprocessingConfig``, so two spacings can never alias to one cache entry.)

    ``feature_kind`` records what the channels mean: ``patch_features`` (ViT patch
    tokens) or ``cls_attention`` (per-head prefix-token self-attention). For the latter
    the ``attention_*`` fields and ``channel_order`` describe the channel layout
    (``[block][cls, reg…][head]``) so a loader / multi-encoder concat can interpret it.
    """
    is_attention = feature_kind != "patch_features"
    return {
        "artifact_type": DENSE_ARTIFACT_TYPE,
        "feature_type": DENSE_ARTIFACT_TYPE,
        "feature_kind": str(feature_kind),
        "attention_blocks": (
            [int(b) for b in (attention_blocks or ())] if is_attention else None
        ),
        "attention_include_registers": bool(attention_include_registers) if is_attention else None,
        "channel_order": "[block][cls, reg…][head]" if is_attention else None,
        "feature_dim": int(feature_dim),
        "channel_dim": int(channel_dim),
        "grid_shape": [int(geometry.grid_shape[0]), int(geometry.grid_shape[1])],
        "target_size": [int(geometry.target_size[0]), int(geometry.target_size[1])],
        "encoded_size": [int(geometry.encoded_size[0]), int(geometry.encoded_size[1])],
        "patch_size": [int(geometry.patch_size[0]), int(geometry.patch_size[1])],
        "pad": [int(geometry.pad[0]), int(geometry.pad[1])],
        "pad_mode": str(pad_mode),
        "crop_box": [int(v) for v in geometry.crop_box],
        "image_pad_value": None if image_pad_value is None else float(image_pad_value),
        "mask_pad_value": None if mask_pad_value is None else int(mask_pad_value),
        "dense_input_mode": str(dense_input_mode),
        "window_size": None if window_size is None else int(window_size),
        "overlap": float(overlap),
        "spacing_um": None if spacing_um is None else float(spacing_um),
    }


def write_dense_grid(
    out_dir: Path | str,
    sample_id: str,
    grid: torch.Tensor,
    metadata: dict,
) -> Path:
    """Write a ``(d, h, w)`` grid plus its sidecar, validating shape vs metadata.

    Fails loud if the tensor's channel/grid axes disagree with the metadata — a
    silent mismatch here would train the decoder on the wrong layout.
    """
    if grid.ndim != 3:
        raise ValueError(
            f"dense grid for '{sample_id}' must be 3-D (channels, grid_h, grid_w), "
            f"got shape {tuple(grid.shape)}"
        )
    channel_dim = int(metadata["channel_dim"])
    if channel_dim != 0:
        raise ValueError(
            f"write_dense_grid only supports channel_dim=0, got {channel_dim}"
        )
    feature_dim = int(metadata["feature_dim"])
    grid_h, grid_w = (int(v) for v in metadata["grid_shape"])
    if tuple(grid.shape) != (feature_dim, grid_h, grid_w):
        raise ValueError(
            f"dense grid for '{sample_id}' has shape {tuple(grid.shape)} but metadata "
            f"declares (feature_dim={feature_dim}, grid_shape=({grid_h}, {grid_w}))"
        )
    sample_id = ensure_filename_safe_id(sample_id)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = out_dir / f"{sample_id}.pt"
    sidecar_path = out_dir / f"{sample_id}{DENSE_SIDECAR_SUFFIX}"
    # torch.save serialises a tensor's whole underlying STORAGE, not just its view. Dense
    # grids arrive as ``batch[i]`` slices of a ``(B, d, h, w)`` batch, so saving one
    # directly writes ALL B tiles' bytes into every tile's file — a silent
    # ``encoder.batch_size``-fold bloat (at B=8: 134 MB files holding 16.8 MB of grid).
    # ``.contiguous()`` does NOT help: a batch slice is already contiguous, so it no-ops.
    # An fp16 cache only escaped this because the dtype cast allocated a fresh tensor.
    if grid.untyped_storage().nbytes() > grid.numel() * grid.element_size():
        grid = grid.clone()
    torch.save(grid, feature_path)
    sidecar_path.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    return feature_path


class DenseFeatureStore:
    """Index and load dense ``(d, h, w)`` grids written by slide2vec.

    Every ``.pt`` must have a matching ``.meta.json`` sidecar; shape is read from the
    sidecar, never inferred from tensor rank.

    Two layouts are read, both slide2vec's own. Grids over pre-cropped images are flat —
    ``<sample_id>.pt`` — and are discovered by globbing. ROI grids over a slide are
    namespaced per slide — ``<slide_id>/<x>_<y>.pt`` — and cannot be discovered that way:
    the ROI's identity is the manifest's, and recovering ``sample_id`` from the path would
    mean parsing ``<slide>__x<X>_y<Y>`` back apart. So the caller passes ``payload_stems``,
    the ``sample_id → relative stem`` mapping it already holds (ADR 0007).
    """

    def __init__(
        self,
        feature_dir: Path | str,
        *,
        payload_stems: dict[str, str] | None = None,
    ) -> None:
        # Accept a cache dir and descend into dense_embeddings/, or a plain dir of
        # .pt files as-is. Dense-specific resolver: never falls through to a pooled
        # sibling dir (e.g. tile_embeddings) if both happen to exist.
        self._feature_dir = resolve_dense_payload_dir(feature_dir)
        self._payload_stems = (
            None if payload_stems is None else {str(k): str(v) for k, v in payload_stems.items()}
        )
        self._index: dict[str, Path] = {}
        self._meta_cache: dict[str, dict] = {}
        self._feature_dim: int | None = None
        self._grid_shape: tuple[int, int] | None = None
        self._build_index()

    def _build_index(self) -> None:
        if self._payload_stems is not None:
            for sample_id, stem in self._payload_stems.items():
                path = self._feature_dir / f"{stem}.pt"
                if path.is_file():
                    self._index[sample_id] = path
            return
        for path in sorted(self._feature_dir.glob("*.pt")):
            self._index[path.stem] = path

    def _sidecar_path(self, sample_id: str) -> Path:
        stem = sample_id if self._payload_stems is None else self._payload_stems[sample_id]
        return self._feature_dir / f"{stem}{DENSE_SIDECAR_SUFFIX}"

    def metadata(self, sample_id: str) -> dict:
        """Return the sidecar metadata for ``sample_id`` (cached)."""
        if sample_id in self._meta_cache:
            return self._meta_cache[sample_id]
        if sample_id not in self._index:
            raise KeyError(
                f"Sample '{sample_id}' not found in dense feature store. "
                f"Available: {sorted(self._index)}"
            )
        sidecar = self._sidecar_path(sample_id)
        if not sidecar.is_file():
            raise FileNotFoundError(
                f"Dense feature '{sample_id}' is missing its required sidecar {sidecar.name}; "
                "shape cannot be read from rank for dense grids."
            )
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        self._meta_cache[sample_id] = meta
        return meta

    def _ensure_shape(self) -> None:
        if self._feature_dim is not None:
            return
        if not self._index:
            raise ValueError("Cannot determine dense feature_dim: no grids found")
        first_id = next(iter(self._index))
        meta = self.metadata(first_id)
        self._feature_dim = int(meta["feature_dim"])
        gh, gw = (int(v) for v in meta["grid_shape"])
        self._grid_shape = (gh, gw)

    @property
    def available_samples(self) -> list[str]:
        return list(self._index.keys())

    @property
    def feature_dim(self) -> int:
        """Channel dimensionality ``d`` — from the sidecar, not the last axis."""
        self._ensure_shape()
        assert self._feature_dim is not None
        return self._feature_dim

    @property
    def grid_shape(self) -> tuple[int, int]:
        self._ensure_shape()
        assert self._grid_shape is not None
        return self._grid_shape

    def geometry(self, sample_id: str) -> DenseGridGeometry:
        """Reconstruct the sample's :class:`DenseGridGeometry` from its sidecar.

        Recomputed via :func:`compute_dense_geometry` from the persisted
        ``target_size`` + ``patch_size`` — the exact function the extractor used, so
        the head's crop geometry is byte-identical to what the grid was built with
        (single source of truth, no field-copy drift).
        """
        meta = self.metadata(sample_id)
        return compute_dense_geometry(
            target_size=tuple(int(v) for v in meta["target_size"]),
            patch_size=tuple(int(v) for v in meta["patch_size"]),
        )

    def spacing_um(self, sample_id: str) -> float | None:
        """Read-spacing in µm/px recorded for ``sample_id`` (``None`` for flat reads).

        slide2vec spells this field differently in its two dense writers: the ROI writer
        records ``spacing_um``, the image writer records ``declared_spacing_um`` (plus the
        resolved ``read_``/``effective_`` pair). ``declared_spacing_um`` is the one that
        matches — it is the spacing the run *asked* for, which is exactly what soma's own
        writer recorded here before the migration, for a flat raster (where the request is
        an assertion about pixels read unchanged) as much as for a pyramidal source. Taking
        ``effective_spacing_um`` instead would silently change what the segmentation fold's
        grid-vs-mask spacing guard compares. Reported upstream as clemsgrs/slide2vec#266;
        until it is reconciled, reading both spellings here is what keeps that guard armed —
        falling through to ``None`` would disarm it without a word.
        """
        metadata = self.metadata(sample_id)
        value = metadata.get("spacing_um", metadata.get("declared_spacing_um"))
        return None if value is None else float(value)

    def spacing(self, sample_id: str) -> DenseSampleSpacing:
        """Resolved source and effective grid spacing persisted by slide2vec."""
        return dense_sample_spacing_from_metadata(
            self.metadata(sample_id), sample_id=sample_id
        )

    @property
    def feature_dir(self) -> Path:
        return self._feature_dir

    def load(self, sample_id: str) -> torch.Tensor:
        """Load the dense grid for ``sample_id`` as a float32 ``(d, h, w)`` tensor."""
        if sample_id not in self._index:
            raise KeyError(
                f"Sample '{sample_id}' not found in dense feature store. "
                f"Available: {sorted(self._index)}"
            )
        tensor = _load_array_resilient(self._index[sample_id])
        if not torch.is_tensor(tensor):
            tensor = torch.as_tensor(tensor)
        meta = self.metadata(sample_id)
        feature_dim = int(meta["feature_dim"])
        grid_h, grid_w = (int(v) for v in meta["grid_shape"])
        if tuple(tensor.shape) != (feature_dim, grid_h, grid_w):
            raise ValueError(
                f"Dense grid '{sample_id}' on disk has shape {tuple(tensor.shape)} but its "
                f"sidecar declares (d={feature_dim}, grid=({grid_h}, {grid_w}))."
            )
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            return tensor.float()
        return tensor

    def validate_coverage(self, sample_ids: list[str]) -> None:
        missing = sorted(set(sample_ids) - set(self._index))
        if missing:
            raise ValueError(f"Missing dense features for {len(missing)} samples: {missing}")

    def __len__(self) -> int:
        return len(self._index)
