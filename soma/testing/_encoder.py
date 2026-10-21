"""``FakeEncoder``, the ``fake`` and ``fake-slide`` encoders, and the conformance suites for
the live grid path's stage and pixels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from slide2vec.encoders.base import SlideEncoder, TileEncoder
from torch import Tensor, nn

from soma.data.geometry import GridGeometry
from soma.testing._errors import ConformanceError

__all__ = [
    "FAKE_ENCODER",
    "FAKE_SLIDE_ENCODER",
    "FakeEncoder",
    "check_encoder_stage",
    "check_image_source",
    "register_fake_encoder",
]

#: The slide2vec encoder name :func:`register_fake_encoder` gives :class:`FakeTileEncoder`.
FAKE_ENCODER = "fake"
#: The slide2vec encoder name :func:`register_fake_encoder` gives :class:`FakeSlideEncoder`.
FAKE_SLIDE_ENCODER = "fake-slide"
#: How both fakes normalise pixels, as a real encoder normalises them: mean 0.5, std 0.25.
_MEAN, _STD = 0.5, 0.25
#: The side of the pixel block one token of the ``fake`` encoder averages.
_FAKE_PATCH_SIZE = 8


@dataclass(frozen=True)
class _FakePreprocessor:
    """Pickle-safe: uint8 ``(3, H, W)`` -> floats normalised to mean 0.5 / std 0.25 (as a
    real stage normalises its pixels), zero-padded to the encoded size."""

    target_size: tuple[int, int]
    encoded_size: tuple[int, int]

    def __call__(self, item: Tensor) -> Tensor:
        if item.dtype != torch.uint8 or item.ndim != 3 or int(item.shape[0]) != 3:
            raise ValueError(f"FakeEncoder preprocesses (3, H, W) uint8 pixels, got {item.dtype} {tuple(item.shape)}")
        if tuple(int(v) for v in item.shape[-2:]) != self.target_size:
            raise ValueError(f"FakeEncoder expects pixels at {self.target_size}, got {tuple(item.shape[-2:])}")
        pad_h = self.encoded_size[0] - self.target_size[0]
        pad_w = self.encoded_size[1] - self.target_size[1]
        return F.pad((item.float() / 255.0 - _MEAN) / _STD, (0, pad_w, 0, pad_h))


class FakeEncoder(nn.Module):
    """A deterministic, parameter-free stand-in for :class:`~soma.extract.FrozenEncoder`.

    Each token is the mean normalised RGB of its ``patch_size`` pixel block, so a grid
    cell is a literal function of the pixels that reached it. It honours the frozen
    stage contract — ``geometry``, ``feature_dim``, ``input_device``, ``preprocessor()``,
    ``forward`` over preprocessed float or raw uint8 pixels — and passes
    :func:`check_encoder_stage`, so live-path code is testable without a foundation
    model.
    """

    input_device = "cpu"

    def __init__(self, *, target_size: int | tuple[int, int], patch_size: int = 8) -> None:
        super().__init__()
        self.geometry = GridGeometry.from_sizes(target_size=target_size, patch_size=patch_size)
        self.feature_dim = 3
        self._patch = tuple(int(v) for v in self.geometry.patch_size)

    def preprocessor(self) -> _FakePreprocessor:
        return _FakePreprocessor(
            target_size=tuple(int(v) for v in self.geometry.target_size),
            encoded_size=tuple(int(v) for v in self.geometry.encoded_size),
        )

    def forward(self, pixels: Tensor) -> Tensor:
        if pixels.dtype == torch.uint8:
            preprocess = self.preprocessor()
            pixels = torch.stack([preprocess(item) for item in pixels])
        return F.avg_pool2d(pixels.float(), kernel_size=self._patch, stride=self._patch)



class FakeTileEncoder(TileEncoder):
    """:class:`FakeEncoder` as a slide2vec tile encoder, registered as ``fake``.

    Its transform normalises pixels as :class:`FakeEncoder` does; a tile embeds as its mean
    normalised RGB (``encode_tiles``) and its dense grid is the mean of each 8x8 block
    (``encode_tiles_dense``, also the pre-norm tap), so the cached and the live paths see
    one encoder. Tiles default to 32 px at 0.5 µm/px (:func:`soma.testing.synthetic_cohort`'s
    spacing); any size divisible by 8 is accepted.
    """

    def __init__(self, **kwargs) -> None:
        self._device = torch.device("cpu")

    @property
    def encode_dim(self) -> int:
        return 3

    @property
    def patch_size(self) -> tuple[int, int]:
        return (_FAKE_PATCH_SIZE, _FAKE_PATCH_SIZE)

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device) -> FakeTileEncoder:
        self._device = torch.device(device)
        return self

    def get_transform(self):
        from torchvision.transforms import v2

        return v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=[_MEAN] * 3, std=[_STD] * 3)]
        )

    def get_normalization_transform(self):
        return self.get_transform()

    def encode_tiles(self, batch: Tensor) -> Tensor:
        return batch.float().mean(dim=(-2, -1))

    def encode_tiles_dense(self, batch: Tensor) -> Tensor:
        return F.avg_pool2d(batch.float(), kernel_size=_FAKE_PATCH_SIZE, stride=_FAKE_PATCH_SIZE)

    def encode_tiles_dense_prenorm(self, batch: Tensor) -> Tensor:
        """The fake has no final norm to skip: the pre-norm grid is the same grid."""
        return F.avg_pool2d(batch.float(), kernel_size=_FAKE_PATCH_SIZE, stride=_FAKE_PATCH_SIZE)


class FakeSlideEncoder(SlideEncoder):
    """A weight-free slide encoder, registered as ``fake-slide``: a slide's embedding is the
    mean of its tiles' ``fake`` embeddings (so the mean normalised RGB of its tissue).

    Its tile encoder is ``fake``, so a slide is tiled as for ``fake`` (32 px tiles at
    0.5 µm/px by default), and the slide-encoder path (tile features cached, then pooled
    into one cached vector per slide) runs with no weights, no GPU and no network.
    """

    def __init__(self, **kwargs) -> None:
        self._device = torch.device("cpu")

    @property
    def encode_dim(self) -> int:
        return 3

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device) -> FakeSlideEncoder:
        self._device = torch.device(device)
        return self

    def encode_slide(self, tile_features: Tensor, coordinates: Tensor | None = None, *, tile_size_lv0=None) -> Tensor:
        return tile_features.float().mean(dim=0)


#: What the two fakes share: three features, fp32, 0.25 or 0.5 µm/px (0.5 by default) and
#: no weights to download.
_FAKE_METADATA = {
    "output_variants": {"default": {"encode_dim": 3}},
    "default_output_variant": "default",
    "supported_spacing_um": [0.25, 0.5],
    "default_spacing_um": 0.5,
    "precision": "fp32",
    "source": "",
}


def register_fake_encoder() -> str:
    """Register :class:`FakeTileEncoder` as ``fake`` and :class:`FakeSlideEncoder` as
    ``fake-slide`` with slide2vec (once); return the tile encoder's name.

    :mod:`soma.testing` calls it on import, so ``encoder="fake"`` and
    ``encoder="fake-slide"`` work wherever soma does.
    """
    from slide2vec.encoders.registry import encoder_registry

    tile = {"level": "tile", "input_size": 32, "patch_size": _FAKE_PATCH_SIZE, "supports_variable_input_size": True}
    slide = {"level": "slide", "tile_encoder": FAKE_ENCODER, "tile_encoder_output_variant": "default"}
    for name, cls, metadata in (
        (FAKE_ENCODER, FakeTileEncoder, tile),
        (FAKE_SLIDE_ENCODER, FakeSlideEncoder, slide),
    ):
        if name not in encoder_registry:  # registered already: this module was imported again
            encoder_registry.register(name, cls, metadata={**metadata, **_FAKE_METADATA})
    return FAKE_ENCODER

def _fail(method: str, message: str, sample_id: str | None = None) -> ConformanceError:
    return ConformanceError(method, message, sample_id=sample_id) if sample_id else ConformanceError(method, message)


def check_encoder_stage(stage, *, batch_size: int = 2) -> None:
    """Check a frozen encoder stage (:class:`~soma.extract.FrozenEncoder` or a stand-in).

    Asserts: no parameters (the stage never enters an optimizer or a checkpoint); a
    :class:`~soma.data.GridGeometry` ``geometry`` and a positive int ``feature_dim``; an
    ``input_device``; ``preprocessor()`` maps a ``(3, H, W)`` uint8 tile at
    ``target_size`` to ``(3, Henc, Wenc)`` floats at ``encoded_size``; ``forward`` on the
    collated batch returns ``(B, feature_dim, *grid_shape)``, deterministically, and the
    same grid from the raw uint8 batch.
    """
    parameters = sum(p.numel() for p in stage.parameters())
    if parameters:
        raise _fail("parameters", f"a frozen encoder stage is parameter-free, found {parameters}")
    geometry = getattr(stage, "geometry", None)
    if not isinstance(geometry, GridGeometry):
        raise _fail("geometry", f"must be a GridGeometry, got {type(geometry)!r}")
    dim = getattr(stage, "feature_dim", None)
    if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
        raise _fail("feature_dim", f"must be a positive int, got {dim!r}")
    if getattr(stage, "input_device", None) is None:
        raise _fail("input_device", "must name the device the stage takes pixels on")
    try:
        preprocess = stage.preprocessor()
    except Exception as error:  # noqa: BLE001
        raise _fail("preprocessor", f"raised {error!r}") from error
    generator = torch.Generator().manual_seed(0)
    raw = torch.randint(
        0, 256, (batch_size, 3, *geometry.target_size), dtype=torch.uint8, generator=generator
    )
    try:
        items = [preprocess(item) for item in raw]
    except Exception as error:  # noqa: BLE001
        raise _fail("preprocessor", f"raised {error!r} on a uint8 tile at target_size") from error
    if any(tuple(int(v) for v in item.shape) != (3, *geometry.encoded_size) for item in items):
        raise _fail(
            "preprocessor",
            f"must return (3, {geometry.encoded_size[0]}, {geometry.encoded_size[1]}), got "
            f"{tuple(items[0].shape)}",
        )
    if not items[0].is_floating_point():
        raise _fail("preprocessor", f"must return floats, got {items[0].dtype}")
    batch = torch.stack(items)
    try:
        with torch.inference_mode():
            first = stage(batch).detach().cpu()
            second = stage(batch).detach().cpu()
            from_raw = stage(raw).detach().cpu()
    except Exception as error:  # noqa: BLE001
        raise _fail("forward", f"raised {error!r}") from error
    expected = (batch_size, dim, *geometry.grid_shape)
    if tuple(first.shape) != expected:
        raise _fail("forward", f"must return {expected}, got {tuple(first.shape)}")
    if not torch.equal(first, second):
        raise _fail("forward", "is not deterministic: two calls on one batch differ")
    if not torch.allclose(first, from_raw, atol=1e-5):
        raise _fail("forward", "raw uint8 pixels and their preprocessed batch encode differently")


def check_image_source(source, *, sample_ids: Sequence[str] | None = None, limit: int | None = None) -> None:
    """Check an :class:`~soma.data.ImageSource`: unique ids; ``load`` returns ``(3, H, W)``
    uint8 pixels at ``geometry(...).target_size``; a positive ``spacing``; an unknown id
    raises ``KeyError``."""
    from soma.testing import _call, _check_ids, _check_unknown_id_raises_key_error, _select

    ids = _check_ids(source)
    for sample_id in _select(ids, sample_ids, limit):
        pixels = _call(source, "load", sample_id)
        if not torch.is_tensor(pixels) or pixels.ndim != 3 or int(pixels.shape[0]) != 3:
            raise _fail("load", f"must return a (3, H, W) Tensor, got {getattr(pixels, 'shape', pixels)!r}", sample_id)
        if pixels.dtype != torch.uint8:
            raise _fail("load", f"must return uint8 pixels, got {pixels.dtype}", sample_id)
        geometry = _call(source, "geometry", sample_id)
        if not isinstance(geometry, GridGeometry):
            raise _fail("geometry", f"must return a GridGeometry, got {type(geometry)!r}", sample_id)
        if tuple(int(v) for v in pixels.shape[1:]) != tuple(int(v) for v in geometry.target_size):
            raise _fail(
                "load",
                f"pixels are {tuple(pixels.shape[1:])} but geometry declares {tuple(geometry.target_size)}",
                sample_id,
            )
        spacing = _call(source, "spacing", sample_id)
        if isinstance(spacing, bool) or not isinstance(spacing, (int, float)) or not spacing > 0:
            raise _fail("spacing", f"must be a positive µm/px float, got {spacing!r}", sample_id)
    _check_unknown_id_raises_key_error(source, "load")
