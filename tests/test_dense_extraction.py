"""Offline tests for dense extraction over given images (soma.dense_extraction).

The encode itself is slide2vec's (``Model.embed_images_dense``, tested upstream and gated
on byte-identity against the pre-migration grids), so it is stubbed here at soma's import
seam — the same shape ``test_pipeline_segmentation_slide_manifest`` uses for the ROI path.
What runs for real is everything soma still owns: the ``DenseImageOptions`` /
``ExecutionOptions`` contract it states, the cache key and resume decision, the payload
layout, and the ``DenseFeatureStore`` read-back.

The stub persists through slide2vec's own :func:`write_dense_image`, so the on-disk layout
and sidecar these tests read back are upstream's rather than the stub's idea of them —
which is the point of the migration.
"""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from soma.config import CacheConfig, EncoderConfig, ExecutionConfig  # noqa: E402
from soma.dataset import Dataset, SampleRecord  # noqa: E402
from soma.dense import DenseFeatureStore, compute_dense_geometry  # noqa: E402
from soma.dense.store import DENSE_SIDECAR_SUFFIX  # noqa: E402
from soma.dense_extraction import DenseTileFeatureExtractor  # noqa: E402

FEATURE_DIM = 8
PATCH = 16


def _make_tiles(tmp_path: Path, n: int, size: int) -> list[SampleRecord]:
    records = []
    for i in range(n):
        path = tmp_path / f"tile{i}.png"
        Image.fromarray(
            (torch.rand(size, size, 3) * 255).to(torch.uint8).numpy()
        ).save(path)
        records.append(SampleRecord(sample_id=f"s{i}", image_path=path, label="x"))
    return records


def _dataset(tmp_path: Path, records: list[SampleRecord]) -> Dataset:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [{"sample_id": r.sample_id, "image_path": str(r.image_path), "label": "x"} for r in records]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _dense_sidecar(dense, geometry, grid) -> dict:
    """The geometry sidecar slide2vec writes next to every dense image grid.

    Deliberately spells the read spacing ``declared_spacing_um``: that is upstream's name
    on the image path (the ROI path says ``spacing_um``), and reading both is what keeps
    the store's spacing accessor — and the fold's grid-vs-mask guard — working.
    """
    return {
        "artifact_type": "dense_image_embeddings",
        "feature_dim": int(grid.shape[0]),
        "grid_shape": [int(geometry.grid_shape[0]), int(geometry.grid_shape[1])],
        "target_size": [int(geometry.target_size[0]), int(geometry.target_size[1])],
        "patch_size": [int(geometry.patch_size[0]), int(geometry.patch_size[1])],
        "encoded_size": [int(geometry.encoded_size[0]), int(geometry.encoded_size[1])],
        "pad": [int(geometry.pad[0]), int(geometry.pad[1])],
        "declared_spacing_um": None if dense.spacing_um is None else float(dense.spacing_um),
        "effective_spacing_um": None if dense.spacing_um is None else float(dense.spacing_um),
        "reader_regime": "raster",
        "pad_mode": dense.pad_mode,
        "image_pad_value": dense.image_pad_value,
        "window_size": dense.window_size,
        "overlap": float(dense.overlap),
        "feature_kind": dense.feature_kind,
        "attention_blocks": [int(b) for b in dense.attention_blocks],
        "attention_include_registers": bool(dense.attention_include_registers),
    }


class _FakeDenseImageModel:
    """Stands in for ``slide2vec.Model`` at soma's import seam.

    Records the ``embed_images_dense`` contract soma states — which images, and the
    ``DenseImageOptions``/``ExecutionOptions`` it built — and persists a deterministic grid
    per image through slide2vec's own writer.
    """

    calls: list[dict] = []

    def __init__(self, name: str, **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "_FakeDenseImageModel":
        return cls(name, **kwargs)

    def embed_images_dense(self, images, *, dense, execution):
        from slide2vec.artifacts import write_dense_image

        type(self).calls.append(
            {
                "name": self.name,
                "sample_ids": [image.sample_id for image in images],
                "dense": dense,
                "execution": execution,
            }
        )
        target = dense.target_size
        geometry = compute_dense_geometry(
            target_size=int(target) if isinstance(target, int) else tuple(target),
            patch_size=(PATCH, PATCH),
        )
        rng = np.random.default_rng(0)
        artifacts = []
        for image in images:
            grid = rng.standard_normal(
                (FEATURE_DIM, geometry.grid_shape[0], geometry.grid_shape[1])
            ).astype(np.float32)
            artifacts.append(
                write_dense_image(
                    grid,
                    output_dir=execution.output_dir,
                    sample_id=image.sample_id,
                    metadata=_dense_sidecar(dense, geometry, grid),
                )
            )
        return artifacts


@pytest.fixture
def fake_model(monkeypatch) -> type[_FakeDenseImageModel]:
    import soma.dense_extraction as de

    _FakeDenseImageModel.calls = []
    monkeypatch.setattr(de, "Model", _FakeDenseImageModel)
    # patch_size is read from the registry without loading weights (#165); pin it so the
    # stub's grid geometry and soma's cache key agree without naming a real encoder.
    monkeypatch.setattr(de, "resolve_patch_size", lambda name: (PATCH, PATCH))
    return _FakeDenseImageModel


def _extractor(dataset: Dataset, tmp_path: Path, **overrides) -> DenseTileFeatureExtractor:
    kwargs = dict(
        target_size=32,
        spacing_um=0.5,
        cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache"),
    )
    kwargs.update(overrides)
    return DenseTileFeatureExtractor(
        dataset, EncoderConfig(name="uni", precision="fp32", batch_size=2), **kwargs
    )


@pytest.mark.parametrize(
    ("source_spacing_um", "target_size", "expected_grid"),
    [
        (
            0.5,
            16,
            [
                [[10.0, 20.0], [30.0, 40.0]],
                [[11.0, 21.0], [31.0, 41.0]],
                [[12.0, 22.0], [32.0, 42.0]],
            ],
        ),
        (0.25, 8, [[[25.0]], [[26.0]], [[27.0]]]),
    ],
)
def test_real_flat_raster_extraction_respects_manifest_source_spacing(
    tmp_path: Path,
    source_spacing_um: float,
    target_size: int,
    expected_grid: list[list[list[float]]],
):
    """The public dense path preserves exact reads and supports coarser reads."""
    from tests.dense_literal_encoder import register_literal_encoder

    encoder_name = register_literal_encoder()
    pixels = np.empty((16, 16, 3), dtype=np.uint8)
    pixels[:8, :8] = [10, 11, 12]
    pixels[:8, 8:] = [20, 21, 22]
    pixels[8:, :8] = [30, 31, 32]
    pixels[8:, 8:] = [40, 41, 42]
    image_path = tmp_path / "literal.png"
    Image.fromarray(pixels).save(image_path)
    manifest = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {
                "sample_id": "literal",
                "image_path": image_path,
                "label": 0,
                "spacing_at_level_0": source_spacing_um,
            }
        ]
    ).to_csv(manifest, index=False)

    store = DenseTileFeatureExtractor(
        Dataset(manifest),
        EncoderConfig(name=encoder_name, precision="fp32", batch_size=1),
        target_size=target_size,
        spacing_um=0.5,
        execution=ExecutionConfig(num_workers_per_gpu=0, precision="fp32"),
    ).run(tmp_path / "features")

    expected = torch.tensor(expected_grid)
    torch.testing.assert_close(store.load("literal"), expected, rtol=0, atol=0)
    metadata = store.metadata("literal")
    assert metadata["spacing_at_level_0"] == source_spacing_um
    assert metadata["source_spacing_um"] == source_spacing_um
    assert metadata["declared_spacing_um"] == 0.5
    assert metadata["effective_spacing_um"] == 0.5


def test_run_writes_grids_into_slide2vecs_image_payload_dir(tmp_path: Path, fake_model):
    """Grids land in ``dense_image_embeddings/`` — upstream's flat layout for image
    sources, under their own cache kind rather than the ROI path's ``dense`` cache."""
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=3, size=32))
    extractor = _extractor(dataset, tmp_path)
    store = extractor.run(tmp_path / "features")

    cache_dir = extractor.cache_dir(tmp_path / "features")
    assert cache_dir is not None
    assert cache_dir.parent.name == "dense_image"
    assert store.feature_dir == cache_dir / "dense_image_embeddings"
    assert store.feature_dir.name == "dense_image_embeddings"
    assert sorted(store.available_samples) == ["s0", "s1", "s2"]
    assert store.feature_dim == FEATURE_DIM
    assert store.grid_shape == (2, 2)
    assert tuple(store.load("s0").shape) == (FEATURE_DIM, 2, 2)


def test_run_states_the_read_and_encode_recipe_it_wants(tmp_path: Path, fake_model):
    """Every knob soma folds into its cache key is also stated to slide2vec — otherwise a
    key could describe a run that was never asked for."""
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=1, size=32))
    _extractor(
        dataset,
        tmp_path,
        backend="openslide",
        tolerance=0.1,
        pad_mode="constant",
    ).run(tmp_path / "features")

    dense = fake_model.calls[0]["dense"]
    assert dense.target_size == 32  # square target stays an int
    assert dense.spacing_um == 0.5
    assert dense.tolerance == 0.1
    assert dense.backend == "openslide"
    assert dense.pad_mode == "constant"
    assert dense.image_pad_value == 0.0  # meaningful only for constant/zero padding
    assert dense.window_size is None and dense.overlap == 0.0
    assert dense.feature_kind == "patch_features"


def test_run_passes_a_non_square_target_as_a_pair(tmp_path: Path, fake_model):
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=1, size=32))
    _extractor(dataset, tmp_path, target_size=(32, 48)).run(tmp_path / "features")

    assert fake_model.calls[0]["dense"].target_size == (32, 48)


def test_run_pins_single_gpu_and_the_resolved_storage_dtype(tmp_path: Path, fake_model):
    """num_gpus is pinned to 1 (sharding changes batch composition, and dense grids are
    batch-size sensitive — #305), and the write dtype is the one folded into the key."""
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=1, size=32))
    _extractor(
        dataset,
        tmp_path,
        cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache", dtype="fp16"),
    ).run(tmp_path / "features")

    execution = fake_model.calls[0]["execution"]
    assert execution.num_gpus == 1
    assert execution.output_dtype == "fp16"
    assert execution.batch_size == 2


def test_sliding_window_is_stated_rather_than_derived_downstream(tmp_path: Path, fake_model):
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=1, size=64))
    _extractor(dataset, tmp_path, target_size=64, window_size=32, overlap=0.5).run(
        tmp_path / "features"
    )

    dense = fake_model.calls[0]["dense"]
    assert dense.window_size == 32 and dense.overlap == 0.5


def test_attention_run_forwards_the_channel_layout_knobs(tmp_path: Path, fake_model):
    from soma.config import AttentionConfig, PreprocessingConfig, PreviewConfig

    preprocessing = PreprocessingConfig(
        requested_spacing_um=0.5,
        min_coverage={},
        feature_kind="cls_attention",
        attention=AttentionConfig(blocks=(-1, -2), include_registers=True),
        preview=PreviewConfig(),
    )
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=1, size=32))
    _extractor(dataset, tmp_path, preprocessing=preprocessing).run(tmp_path / "features")

    dense = fake_model.calls[0]["dense"]
    assert dense.feature_kind == "cls_attention"
    assert tuple(dense.attention_blocks) == (-1, -2)
    assert dense.attention_include_registers is True


def test_store_reads_the_spacing_upstream_spells_declared(tmp_path: Path, fake_model):
    """The image writer records ``declared_spacing_um`` where the ROI writer records
    ``spacing_um``; the store must answer either, or the fold's grid-vs-mask spacing
    guard silently compares ``None`` to ``None``."""
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=1, size=32))
    store = _extractor(dataset, tmp_path).run(tmp_path / "features")

    assert "spacing_um" not in store.metadata("s0")  # upstream really does omit it
    assert store.spacing_um("s0") == 0.5


def test_resume_encodes_only_the_missing_images(tmp_path: Path, fake_model):
    """Only the absent images are re-encoded; already-materialized grids are left
    untouched (#140). soma's cache decides this — slide2vec never sees the survivors."""
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=3, size=32))
    feature_dir = tmp_path / "features"
    store = _extractor(dataset, tmp_path).run(feature_dir)
    features_dir = store.feature_dir

    # Crash window: s1's grid + sidecar never landed.
    (features_dir / "s1.pt").unlink()
    (features_dir / f"s1{DENSE_SIDECAR_SUFFIX}").unlink()
    survivor_mtimes = {
        sid: (features_dir / f"{sid}.pt").stat().st_mtime_ns for sid in ("s0", "s2")
    }

    fake_model.calls.clear()
    resumed = _extractor(dataset, tmp_path).run(feature_dir)

    assert fake_model.calls[0]["sample_ids"] == ["s1"]
    for sid, mtime in survivor_mtimes.items():
        assert (features_dir / f"{sid}.pt").stat().st_mtime_ns == mtime
    assert sorted(resumed.available_samples) == ["s0", "s1", "s2"]


def test_cache_hit_constructs_no_model(tmp_path: Path, fake_model):
    """Check-before-load (#165): a full dense cache hit resolves via static patch_size
    registry metadata and never builds the encoder."""
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=3, size=32))
    feature_dir = tmp_path / "features"

    _extractor(dataset, tmp_path).run(feature_dir)  # populate: a miss, so the model builds
    assert len(fake_model.calls) == 1

    store = _extractor(dataset, tmp_path).run(feature_dir)
    assert len(fake_model.calls) == 1  # unchanged — nothing constructed on the hit path
    assert sorted(store.available_samples) == ["s0", "s1", "s2"]


def test_cache_resolves_complete_and_validates_the_upstream_sidecar(tmp_path: Path, fake_model):
    """complete ⇒ readable: the cache validator and the store agree on grids written by
    slide2vec's own writer, sidecar schema included."""
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=3, size=32))
    extractor = _extractor(
        dataset,
        tmp_path,
        cache=CacheConfig(enabled=True, root_dir=tmp_path / "cache", validate_payloads=True),
    )
    extractor.run(tmp_path / "features")

    cache_dir = extractor.cache_dir(tmp_path / "features")
    store = DenseFeatureStore(cache_dir)  # cache dir, descends into the payload subdir
    assert tuple(store.load("s0").shape) == (FEATURE_DIM, 2, 2)
    assert extractor.cache_dir(tmp_path / "features") == cache_dir  # side-effect-free


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pad_mode": "bilinear"}, "unsupported pad_mode"),
        ({"window_size": 0}, "window_size must be a positive int"),
        ({"window_size": 16, "overlap": 1.0}, r"overlap must be in \[0, 1\)"),
    ],
)
def test_constructor_rejects_an_unencodable_recipe(tmp_path: Path, kwargs, match):
    """Rejected in soma, before any model is built — the run is unencodable regardless of
    what slide2vec would say about it later."""
    dataset = _dataset(tmp_path, _make_tiles(tmp_path, n=1, size=32))
    with pytest.raises(ValueError, match=match):
        _extractor(dataset, tmp_path, **kwargs)
