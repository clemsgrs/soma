"""The seam whole-slide prediction takes its per-tile results from
(:mod:`soma.evaluation.whole_slide`): an ordered stream of batch results.

A :class:`~soma.evaluation.whole_slide.SlideTiles` holds a slide's kept tiles in the order
the map writer takes them. A :class:`~soma.evaluation.whole_slide.TileOutputs` yields, for
a batch size, each batch of those tiles with every bundle's output for it, in order.
:class:`~soma.evaluation.whole_slide.DeviceTileOutputs` runs the bundles on one device of
this process. The slides are synthetic TIFFs and the bundles carry the weight-free
``fake`` encoder's recipe, so everything runs on the CPU.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from soma.data.manifests import ImageEntry
from soma.evaluation.whole_slide import (
    DealtTileOutputs,
    DeviceTileOutputs,
    SlideFrame,
    SlideTiles,
    TileBatch,
    predict_slide_label_maps,
)
from soma.evaluation.whole_slide_points import predict_slide_points
from soma.models.model import DenseModelOutput
from tests.whole_slides import (
    SPACING_UM,
    TILE,
    MarkerDecoder,
    MarkerParams,
    blocky,
    marker_slide,
    model_of,
    save_bundle,
    save_detection_bundle,
    tile_pixels,
    write_slide,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


#: Nine kept tiles of a 120 x 150 slide, out of the writer's order.
ORIGINS = [(96, 64), (32, 0), (64, 32), (96, 0), (32, 64), (64, 0), (32, 32), (96, 32), (64, 64)]


def tile_logits(bundle_path, slide, origins) -> dict[tuple[int, int], torch.Tensor]:
    """Each tile's ``(C, H, W)`` logits from the bundle's model, one tile at a time."""
    model = model_of(bundle_path)
    source = tile_pixels(slide, {f"{x}_{y}": (x, y) for x, y in origins})
    with torch.no_grad():
        return {(x, y): model(pixels=source.load(f"{x}_{y}")[None]).logits[0] for x, y in origins}


def test_the_seam_yields_each_batch_of_the_kept_tiles_in_the_writer_s_order_with_every_bundle_s_output(tmp_path):
    slide = write_slide(tmp_path / "s1.tif", blocky(120, 150))
    bundles = [save_bundle(tmp_path / name / "bundle", seed=seed) for name, seed in (("a", 0), ("b", 1))]
    expected = [tile_logits(bundle, slide, ORIGINS) for bundle in bundles]
    tiles = SlideTiles(
        image=ImageEntry(sample_id="s1", image_path=slide), origins=ORIGINS, spacing_um=SPACING_UM, backend="tifffile"
    )
    outputs = DeviceTileOutputs.load(bundles)

    # Every bundle's output is read before the next batch, as the seam asks.
    batches = [(list(b.origins), [o.logits for o in b.outputs]) for b in outputs.batches(tiles, batch_size=4)]

    # Row by row, left to right: chunks of four, the last one shorter.
    assert [origins for origins, _ in batches] == [
        [(32, 0), (64, 0), (96, 0), (32, 32)],
        [(64, 32), (96, 32), (32, 64), (64, 64)],
        [(96, 64)],
    ]
    for origins, logits in batches:
        assert len(logits) == 2  # one output per bundle, in the bundles' order
        for bundle, batch_logits in enumerate(logits):
            assert batch_logits.shape == (len(origins), 3, TILE, TILE)
            for origin, tile in zip(origins, batch_logits):
                np.testing.assert_allclose(tile.numpy(), expected[bundle][origin].numpy(), atol=1e-5)

    # A subset of the bundles yields only theirs, over the same batches.
    (second,) = outputs.subset([1]).heads
    assert second is outputs.heads[1]
    alone = [(list(b.origins), [o.logits for o in b.outputs]) for b in outputs.subset([1]).batches(tiles, batch_size=4)]
    assert [origins for origins, _ in alone] == [origins for origins, _ in batches]
    for (_, (logits,)), (_, both) in zip(alone, batches):
        np.testing.assert_array_equal(logits.numpy(), both[1].numpy())


# --- several devices: the chunks dealt round-robin ---------------------------------------- #


class Recording:
    """A model-free device: it cuts the tiles it is given into batches, as a device does,
    and logs each batch when it computes it. Each output holds its tiles' level-0 x."""

    def __init__(self, name, log: list, heads=("head",)) -> None:
        self.name, self._log, self.heads = name, log, list(heads)

    def subset(self, indices):
        return Recording(self.name, self._log, [self.heads[index] for index in indices])

    def batches(self, tiles: SlideTiles, *, batch_size: int):
        for start in range(0, len(tiles.origins), batch_size):
            origins = tiles.origins[start : start + batch_size]
            self._log.append((self.name, origins))
            yield TileBatch(origins, [DenseModelOutput(logits=torch.tensor([x for x, _ in origins])) for _ in self.heads])


def slide_tiles(origins) -> SlideTiles:
    return SlideTiles(image=ImageEntry(sample_id="s1", image_path="s1.tif"), origins=origins, spacing_um=SPACING_UM)


#: :data:`ORIGINS` in the writer's order, cut into chunks of four: the last one shorter.
CHUNKS_OF_FOUR = [
    ((32, 0), (64, 0), (96, 0), (32, 32)),
    ((64, 32), (96, 32), (32, 64), (64, 64)),
    ((96, 64),),
]


@pytest.mark.parametrize(
    ("origins", "num_devices", "chunks", "dealt"),
    [
        (ORIGINS, 2, CHUNKS_OF_FOUR, {0: [0, 2], 1: [1]}),
        (ORIGINS, 3, CHUNKS_OF_FOUR, {0: [0], 1: [1], 2: [2]}),
        # Two chunks for three devices: the third device is given nothing.
        ([(64, 32), (32, 0), (96, 0), (32, 32), (64, 0)], 3, CHUNKS_OF_FOUR[:1] + [((64, 32),)], {0: [0], 1: [1]}),
        ([], 2, [], {}),
    ],
    ids=["last-chunk-short", "one-chunk-each", "fewer-chunks-than-devices", "no-tile"],
)
def test_dealt_outputs_give_chunk_i_to_device_i_mod_n_and_collect_the_batches_of_one_device_in_order(
    origins, num_devices, chunks, dealt
):
    log: list = []
    outputs = DealtTileOutputs([Recording(k, log) for k in range(num_devices)])

    batches = [(b.origins, [o.logits.tolist() for o in b.outputs]) for b in outputs.batches(slide_tiles(origins), batch_size=4)]

    # The batches one device would form, in its order, each with its own results.
    assert [batch_origins for batch_origins, _ in batches] == chunks
    assert [results for _, results in batches] == [[[x for x, _ in chunk]] for chunk in chunks]
    # Each device computed exactly its chunks, in order, as the batches of its own tiles.
    computed = {name: [chunks.index(batch_origins) for device, batch_origins in log if device == name] for name, _ in log}
    assert computed == dealt


def test_dealt_outputs_subset_every_device_and_take_the_heads_of_the_first():
    log: list = []
    outputs = DealtTileOutputs([Recording(k, log, heads=("a", "b")) for k in range(2)])

    second = outputs.subset([1])

    assert outputs.heads == ["a", "b"]
    assert second.heads == ["b"]
    batches = [[o.logits.tolist() for o in b.outputs] for b in second.batches(slide_tiles(ORIGINS), batch_size=4)]
    assert batches == [[[x for x, _ in chunk]] for chunk in CHUNKS_OF_FOUR]  # one output: the one bundle's


def test_dealt_outputs_hold_one_chunk_at_a_time():
    """Chunk ``i`` is asked of its device only once chunk ``i - 1`` was read: however
    fast the devices are, the collection holds one chunk's results at a time."""
    log: list = []
    outputs = DealtTileOutputs([Recording(k, log) for k in range(3)])

    held = []
    for consumed, batch in enumerate(outputs.batches(slide_tiles(ORIGINS), batch_size=1)):
        held.append(len(log) - consumed)
        list(batch.outputs)

    assert held == [1] * len(ORIGINS)


# --- the writers read the seam alone ------------------------------------------------------ #


class Replayed:
    """A seam implementation that holds no model: the results another implementation gave
    for one slide and batch size, every batch's outputs whole and on the CPU, as a device
    worker in another process would send them."""

    def __init__(self, heads, tiles: SlideTiles, batch_size: int, recorded) -> None:
        self.heads, self._tiles, self._batch_size, self._recorded = list(heads), tiles, batch_size, recorded

    @classmethod
    def of(cls, outputs, tiles: SlideTiles, *, batch_size: int) -> "Replayed":
        recorded = [
            (batch.origins, [output.logits.detach().cpu().clone() for output in batch.outputs])
            for batch in outputs.batches(tiles, batch_size=batch_size)
        ]
        return cls(outputs.heads, tiles, batch_size, recorded)

    def subset(self, indices):
        recorded = [(origins, [logits[index] for index in indices]) for origins, logits in self._recorded]
        return Replayed([self.heads[index] for index in indices], self._tiles, self._batch_size, recorded)

    def batches(self, tiles: SlideTiles, *, batch_size: int):
        assert (tiles, batch_size) == (self._tiles, self._batch_size), "replayed for another slide or batch size"
        for origins, logits in self._recorded:
            yield TileBatch(origins, [DenseModelOutput(logits=tensor) for tensor in logits])


#: The 32 px tiles at overlap 0.5 on a 120 x 150 slide at its level 0: a 16 px lattice.
STEP_PX = 16
LATTICE = [(x, y) for y in range(0, 96, STEP_PX) for x in range(0, 128, STEP_PX)]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("stitch", "ensemble"),
    [("pick_one", False), ("average", False), ("pick_one", True), ("average", True)],
    ids=["pick_one", "average", "ensemble-pick_one", "ensemble-average"],
)
def test_a_seam_holding_no_model_writes_the_maps_one_device_writes(tmp_path, stitch, ensemble):
    slide = write_slide(tmp_path / "s1.tif", blocky(120, 150))
    bundles = [save_bundle(tmp_path / name / "bundle", seed=seed) for name, seed in (("a", 0), ("b", 1))]
    tiles = SlideTiles(
        image=ImageEntry(sample_id="s1", image_path=slide), origins=LATTICE, spacing_um=SPACING_UM, backend="tifffile"
    )
    frame = SlideFrame.from_grid(
        tile_px=TILE, overlap=0.5, step_lv0=STEP_PX, slide_size=(150, 120), level0_spacing_um=SPACING_UM
    )
    on_device = DeviceTileOutputs.load(bundles)
    replayed = Replayed.of(on_device, tiles, batch_size=5)

    def maps(outputs, directory):
        paths = [tmp_path / directory / f"{k}.tif" for k in range(1 if ensemble else 2)]
        paths[0].parent.mkdir()
        fractions = predict_slide_label_maps(
            outputs,
            tiles=tiles,
            frame=frame,
            paths=paths,
            metadata=[{"stitch": stitch}] * len(paths),
            ensemble=ensemble,
            stitch=stitch,
            batch_size=5,
        )
        return [sha256(path) for path in paths], fractions

    written = maps(on_device, "device")

    assert maps(replayed, "replayed") == written
    assert all(0 < fraction < 1 for fraction in written[1])  # the lattice leaves the slide's edges out


@pytest.fixture
def marker(monkeypatch):
    """The weight-free marker decoder, registered for one test."""
    import soma
    from soma.models import decoder_registry

    monkeypatch.setattr(decoder_registry, "_entries", dict(decoder_registry._entries))
    soma.register("decoders", "marker", MarkerDecoder, params=MarkerParams)
    return {"name": "marker", "params": {}}


@pytest.mark.parametrize("ensemble", [False, True], ids=["each", "ensemble"])
def test_a_seam_holding_no_model_writes_the_points_one_device_writes(tmp_path, marker, ensemble):
    # Markers inside tiles and on their seams, in both classes.
    markers = [(20, 20, 120), (40, 24, 90, 1), (47, 47, 140), (72, 40, 100, 1), (100, 66, 80)]
    slide = write_slide(tmp_path / "s1.tif", marker_slide(120, 150, markers))
    bundles = [save_detection_bundle(tmp_path / name / "bundle", decoder=marker, num_classes=2) for name in "ab"]
    tiles = SlideTiles(
        image=ImageEntry(sample_id="s1", image_path=slide), origins=LATTICE, spacing_um=SPACING_UM, backend="tifffile"
    )
    on_device = DeviceTileOutputs.load(bundles)
    replayed = Replayed.of(on_device, tiles, batch_size=5)

    def points(outputs, directory):
        paths = [tmp_path / directory / f"{k}.parquet" for k in range(1 if ensemble else 2)]
        paths[0].parent.mkdir()
        counts = predict_slide_points(
            outputs,
            tiles=tiles,
            tile_px=TILE,
            step_px=float(STEP_PX),
            overlap=0.5,
            slide_size=(150, 120),
            level0_spacing_um=SPACING_UM,
            paths=paths,
            metadata=[{"classes": {"class_0": [0], "class_1": [1]}, "drop": []}] * len(paths),
            ensemble=ensemble,
            batch_size=5,
        )
        return [sha256(path) for path in paths], counts

    written = points(on_device, "device")

    assert points(replayed, "replayed") == written
    assert all(count == len(markers) for count in written[1])
