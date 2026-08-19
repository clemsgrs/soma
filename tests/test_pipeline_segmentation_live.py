"""Offline tests for the live re-encode segmentation path (``feature_mode='live'``).

Uses a random-weights ``vit_tiny_patch16_224`` (no downloads, CPU) as the frozen
encoder so the whole live path runs offline: ``LiveSegmentationSource`` →
``LiveSegmentationDataset`` (joint read + optional v2 augmentation + normalize + pad) →
``LiveSegmentationModel`` (no_grad+autocast encoder → decoder → head) → ``Trainer.fit``
→ streaming ``_evaluate_segmentation``.

The anchor test is **live-no-aug ≈ cached**: with the same encoder weights the live
re-encode produces byte-identical grids to the cached extractor, so a live-no-aug fold
must reproduce the cached fold's metrics exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")
from PIL import Image  # noqa: E402

from slide2vec.encoders.base import TimmTileEncoder  # noqa: E402

from soma.config import AugmentationConfig, DecoderConfig, EvalConfig, TaskConfig, TrainingConfig  # noqa: E402
from soma.dataset import SegmentationManifest, Splits  # noqa: E402
from soma.dense import DenseFeatureStore, compute_dense_geometry  # noqa: E402
from soma.dense.live import LiveSegmentationSource  # noqa: E402
from soma.pipeline import train_one_segmentation_fold  # noqa: E402
from soma.training.model import LiveSegmentationModel  # noqa: E402

NUM_CLASSES = 2
TARGET = 32
PATCH = 16  # vit_tiny_patch16 -> 2x2 grid at target 32


def _encoder() -> TimmTileEncoder:
    torch.manual_seed(0)
    return TimmTileEncoder("vit_tiny_patch16_224", pretrained=False, dynamic_img_size=True)


def _build_run(root: Path, sample_ids: list[str]) -> tuple[SegmentationManifest, Splits]:
    """Real image + mask PNGs on disk (the live path re-reads them) + manifest/splits."""
    images_dir = root / "images"
    masks_dir = root / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for sid in sample_ids:
        img = (rng.random((TARGET, TARGET, 3)) * 255).astype(np.uint8)
        img_path = images_dir / f"{sid}.png"
        Image.fromarray(img).save(img_path)
        mask = rng.integers(0, NUM_CLASSES, size=(TARGET, TARGET), dtype=np.uint8)
        label_mask_path = masks_dir / f"{sid}.png"
        Image.fromarray(mask, mode="L").save(label_mask_path)
        rows.append((sid, str(img_path), str(label_mask_path)))

    manifest_csv = root / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,label_mask_path\n"
        + "\n".join(f"{sid},{img},{mask}" for sid, img, mask in rows)
        + "\n"
    )
    splits_csv = root / "splits.csv"
    split_assign = {sample_ids[0]: "train", sample_ids[1]: "train", sample_ids[2]: "tune", sample_ids[3]: "test"}
    splits_csv.write_text(
        "sample_id,split,fold\n"
        + "\n".join(f"{sid},{split},0" for sid, split in split_assign.items())
        + "\n"
    )
    return SegmentationManifest(manifest_csv), Splits(splits_csv, SegmentationManifest(manifest_csv))


def _live_source(
    encoder: TimmTileEncoder,
    *,
    augmentation: AugmentationConfig | None = None,
    window_size: int | None = None,
    overlap: float = 0.0,
) -> LiveSegmentationSource:
    from types import SimpleNamespace

    from slide2vec import DenseImageOptions, ExecutionOptions, Model

    loaded = SimpleNamespace(
        model=encoder,
        transforms=encoder.get_dense_transform(),
        device=torch.device("cpu"),
        feature_dim=encoder.encode_dim,
    )
    model = Model(name="uni", device="cpu")
    model._load_backend = lambda: loaded
    kit = model.prepare_dense_encoder(
        dense=DenseImageOptions(
            target_size=TARGET,
            window_size=window_size,
            overlap=overlap,
        ),
        execution=ExecutionOptions(num_gpus=1, precision="fp32", output_dtype="fp32"),
    )
    return LiveSegmentationSource(
        kit=kit,
        device="cpu",
        feature_dim=encoder.encode_dim,
        augmentation=augmentation or AugmentationConfig(),
        spacing_um=None,  # flat PNG read (spacing ignored)
        backend="auto",
        tolerance=0.05,
    )


def _extract_cached_grids(encoder, records, out_dir: Path, *, geometry, batch_size: int) -> None:
    """Mint the cached grids the live path is measured against.

    Encoded through slide2vec's own ``DenseGridEncoder`` — the core
    ``Model.embed_images_dense`` runs, read→normalize→pad→encode included — so this anchor
    compares the live re-encode against the real cached path rather than against a test
    reimplementation of it. Only the *write* uses soma's local fixture writer; the store
    reads either layout back the same way.
    """
    from slide2vec.runtime.dense_regions import DenseGridEncoder

    from soma.dense import dense_grid_metadata, write_dense_grid

    grid_encoder = DenseGridEncoder.resolve(
        encoder,
        target_size=TARGET,
        target_size_origin="the declared target_size",
        precision="fp32",
        dense_transform=encoder.get_dense_transform(),
    )
    records = list(records)
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        batch = torch.stack(
            [
                grid_encoder.transform_and_pad(
                    Image.open(record.image_path).convert("RGB"), origin=record.image_path
                )
                for record in chunk
            ]
        )
        grids = torch.as_tensor(np.asarray(grid_encoder.encode_batch(batch)))
        for record, grid in zip(chunk, grids):
            write_dense_grid(
                out_dir,
                record.sample_id,
                grid,
                dense_grid_metadata(
                    geometry, feature_dim=int(grid.shape[0]), pad_mode="reflect"
                ),
            )


def test_live_no_aug_grids_match_cached_bit_for_bit(tmp_path: Path):
    """The parity anchor: live-no-aug re-encoding reproduces the cached grids exactly.

    With the same encoder weights and the same batch composition, the live path's
    read→normalize→pad→encode (no_grad+autocast+.float()) is byte-identical to the
    cached extractor's (inference_mode+autocast+.float()) — so anything built on the
    grids (decoder/head/metrics) is identical up to encoder batching float noise.
    """
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, _ = _build_run(tmp_path, sample_ids)
    encoder = _encoder()
    geom = compute_dense_geometry(target_size=TARGET, patch_size=encoder.patch_size)
    records = list(manifest.samples.values())

    _extract_cached_grids(
        encoder, records, tmp_path / "dense", geometry=geom, batch_size=len(records)
    )
    cached_store = DenseFeatureStore(tmp_path / "dense")

    from soma.training.segmentation_dataset import LiveSegmentationDataset

    source = _live_source(encoder)
    ds = LiveSegmentationDataset(
        records,
        geometry=geom,
        preprocessor=source.preprocessor,
        spacing_um=None,
        backend="auto",
        tolerance=0.05,
        num_classes=NUM_CLASSES,
        ignore_index=255,
        augment=None,
    )
    batch = torch.stack([ds[i][0] for i in range(len(records))])
    live_grids = source.kit.encode(batch).float()
    for i, record in enumerate(records):
        assert torch.equal(cached_store.load(record.sample_id), live_grids[i])


def test_live_dataset_hands_augmented_uint8_pixels_to_kit_preprocessor(tmp_path: Path):
    from soma.training.segmentation_dataset import LiveSegmentationDataset

    manifest, _ = _build_run(tmp_path, ["s0", "s1", "s2", "s3"])
    record = manifest.samples["s0"]
    geometry = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)
    seen = []

    def preprocessor(item):
        seen.append(item)
        return torch.full((3, TARGET, TARGET), 7.0)

    dataset = LiveSegmentationDataset(
        [record],
        geometry=geometry,
        preprocessor=preprocessor,
        spacing_um=None,
        backend="auto",
        tolerance=0.05,
        num_classes=NUM_CLASSES,
        ignore_index=255,
    )

    image, _targets, _sample_id = dataset[0]

    assert len(seen) == 1
    assert seen[0].shape == (3, TARGET, TARGET)
    assert seen[0].dtype == torch.uint8
    assert seen[0].device.type == "cpu"
    assert torch.equal(image, torch.full((3, TARGET, TARGET), 7.0))


def test_live_no_aug_metrics_match_cached(tmp_path: Path):
    """End-to-end: a live-no-aug fold reproduces the cached fold's metrics (≈, since
    eval batches the splits differently, the grids drift by encoder float noise)."""
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits = _build_run(tmp_path, sample_ids)
    encoder = _encoder()
    geom = compute_dense_geometry(target_size=TARGET, patch_size=encoder.patch_size)
    _extract_cached_grids(
        encoder,
        manifest.samples.values(),
        tmp_path / "dense",
        geometry=geom,
        batch_size=2,
    )
    cached_store = DenseFeatureStore(tmp_path / "dense")

    common = dict(
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=2, batch_size=2),
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )
    cached_result = train_one_segmentation_fold(
        feature_store=cached_store, fold_dir=tmp_path / "cached_fold", **common
    )
    live_result = train_one_segmentation_fold(
        feature_store=_live_source(encoder), fold_dir=tmp_path / "live_fold", **common
    )
    # Approximate, not exact: the cached grids were extracted batched (all records at
    # once), while live re-encodes each eval split on its own (different batch
    # composition ⇒ ~1e-6 encoder float noise, amplified over 2 epochs of a random-init
    # decoder). The exact invariant is the grid bit-parity test above; here we only
    # assert the two paths land in the same place.
    c = cached_result.tune_report.metrics
    live = live_result.tune_report.metrics
    assert pytest.approx(c["mean_dice"], abs=1e-2) == live["mean_dice"]
    assert pytest.approx(c["mean_iou"], abs=1e-2) == live["mean_iou"]


def test_live_fold_with_augmentation_trains_and_evaluates(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits = _build_run(tmp_path, sample_ids)
    source = _live_source(
        _encoder(),
        augmentation=AugmentationConfig(
            horizontal_flip=0.5, vertical_flip=0.5, rotation_degrees=15.0, brightness=0.2
        ),
    )
    result = train_one_segmentation_fold(
        feature_store=source,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=2, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )
    assert set(result.tune_report.metrics) >= {"mean_dice", "mean_iou"}
    assert 0.0 <= result.test_reports["test"].metrics["mean_dice"] <= 1.0
    # Dense artifacts still land for the live path.
    assert (tmp_path / "fold" / "predictions_test.csv").is_file()
    assert (tmp_path / "fold" / "preds" / "test" / "s3.png").is_file()


def test_live_fold_with_sliding_window_trains_and_evaluates(tmp_path: Path):
    """The live model routes through encode_dense_sliding: a 16px window over the 32px
    tile (patch-16 -> 2x2 grid) tiles into two windows per dim, exercising the stitch
    path end-to-end through the trainer + streaming eval."""
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits = _build_run(tmp_path, sample_ids)
    source = _live_source(_encoder(), window_size=PATCH, overlap=0.0)
    result = train_one_segmentation_fold(
        feature_store=source,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice", "mean_iou"]),
    )
    assert 0.0 <= result.test_reports["test"].metrics["mean_dice"] <= 1.0


def test_live_checkpoint_excludes_encoder_and_reloads(tmp_path: Path):
    sample_ids = ["s0", "s1", "s2", "s3"]
    manifest, splits = _build_run(tmp_path, sample_ids)
    encoder = _encoder()
    train_one_segmentation_fold(
        feature_store=_live_source(encoder),
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=2),
        fold_dir=tmp_path / "fold",
        decoder=DecoderConfig(name="lightweight_conv"),
        evaluation=EvalConfig(metrics=["mean_dice"]),
    )
    ckpt = torch.load(tmp_path / "fold" / "best_model.pt", weights_only=True, map_location="cpu")
    state = ckpt["model_state_dict"]
    assert state, "checkpoint state_dict is empty"
    assert not any(k.startswith("encoder.") for k in state), "encoder must not be in the checkpoint"

    # Reload into a fresh model (encoder reconstructed separately, as the pipeline does).
    from soma.dense.geometry import compute_dense_geometry
    from soma.decoders.registry import decoder_registry
    from soma.tasks.segmentation import SegmentationHead

    geom = compute_dense_geometry(target_size=TARGET, patch_size=encoder.patch_size)
    head = SegmentationHead(num_classes=NUM_CLASSES, geometry=geom)
    # Match the trained decoder's auto-injected upsample depth (ceil(log2(32/2)) = 4),
    # else the architecture differs and the state_dict won't load.
    decoder = decoder_registry.get("lightweight_conv")(
        input_dim=encoder.encode_dim, num_classes=NUM_CLASSES, num_upsample_blocks=4
    )
    model = LiveSegmentationModel(
        kit=_live_source(encoder).kit, decoder=decoder, task_head=head,
    )
    model.load_state_dict(state)  # strict=False under the hood: encoder already built


def test_live_source_validate_coverage_is_noop():
    source = _live_source(_encoder())
    assert source.validate_coverage(["s0", "s1"]) is None


def test_live_model_delegates_cpu_batch_to_public_dense_kit():
    """The live model leaves device transfer and frozen encoding to DenseEncodeKit."""
    from soma.decoders.registry import decoder_registry
    from soma.tasks.segmentation import SegmentationHead

    geometry = compute_dense_geometry(target_size=TARGET, patch_size=PATCH)

    class LiteralKit:
        def __init__(self):
            self.inputs = []

        def encode(self, batch):
            self.inputs.append(batch)
            return torch.ones((batch.shape[0], 4, 2, 2), dtype=torch.float16)

    kit = LiteralKit()
    head = SegmentationHead(num_classes=NUM_CLASSES, geometry=geometry)
    decoder = decoder_registry.get("lightweight_conv")(
        input_dim=4, num_classes=NUM_CLASSES, num_upsample_blocks=4
    )
    model = LiveSegmentationModel(kit=kit, decoder=decoder, task_head=head)
    batch = torch.zeros((2, 3, TARGET, TARGET), dtype=torch.float32)

    output = model(batch)

    assert kit.inputs == [batch]
    assert kit.inputs[0].device.type == "cpu"
    assert output.logits.shape == (2, NUM_CLASSES, TARGET, TARGET)


def test_build_live_source_probes_feature_width_through_public_kit(tmp_path: Path, monkeypatch):
    """The width probe uses the kit and therefore honors its resolved sliding window."""
    from types import SimpleNamespace

    from slide2vec import Model
    from soma.config import DecoderConfig, EncoderConfig, PipelineConfig, PreprocessingConfig
    from soma.pipeline import Pipeline

    _build_run(tmp_path, ["s0", "s1", "s2", "s3"])

    encoder = _encoder()
    probe_shapes: list[tuple[int, int]] = []
    original = encoder.encode_tiles_dense

    def _spy(x):
        probe_shapes.append((int(x.shape[-2]), int(x.shape[-1])))
        return original(x)

    encoder.encode_tiles_dense = _spy  # TileEncoder is a plain object: instance attr shadows
    loaded = SimpleNamespace(
        model=encoder,
        transforms=encoder.get_dense_transform(),
        device=torch.device("cpu"),
        feature_dim=encoder.encode_dim,
    )
    public_model = Model(name="uni", device="cpu")
    public_model._load_backend = lambda: loaded
    public_model._load_backend_without_transform = lambda: loaded
    monkeypatch.setattr(Model, "from_preset", lambda *args, **kwargs: public_model)

    config = PipelineConfig(
        dataset_csv=str(tmp_path / "manifest.csv"),
        splits_csv=str(tmp_path / "splits.csv"),
        output_root=str(tmp_path / "out"),
        dataset_type="segmentation",
        feature_mode="live",
        encoder=EncoderConfig(name="uni", precision="fp32"),
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET,
            requested_spacing_um=0.5,
            dense_window_size=PATCH,  # 16px window over the 32px tile
            dense_window_overlap=0.0,
        ),
    )
    source = Pipeline(config)._build_live_segmentation_source()

    assert probe_shapes == [(PATCH, PATCH)] * 4
    assert source.feature_dim == encoder.encode_dim
    assert source.geometry.target_size == (TARGET, TARGET)
    assert source.kit is not None


def test_build_live_source_forwards_output_variant_and_attention_recipe(tmp_path: Path, monkeypatch):
    """Soma forwards model/feature choices; the public kit resolves their execution."""
    from types import SimpleNamespace

    from slide2vec import Model
    from soma.config import (
        AttentionConfig,
        DecoderConfig,
        EncoderConfig,
        PipelineConfig,
        PreprocessingConfig,
    )
    from soma.pipeline import Pipeline

    _build_run(tmp_path, ["s0", "s1", "s2", "s3"])
    calls = {}
    geometry = SimpleNamespace(
        target_size=(TARGET, TARGET),
        patch_size=(PATCH, PATCH),
        encoded_size=(TARGET, TARGET),
        grid_shape=(2, 2),
        pad=(0, 0),
        crop_box=(0, 0, TARGET, TARGET),
    )
    kit = SimpleNamespace(
        geometry=geometry,
        preprocessor=lambda: (lambda item: item.float()),
        encode=lambda batch: torch.zeros((batch.shape[0], 6, 2, 2)),
    )

    class PublicModel:
        device = torch.device("cpu")
        feature_dim = 24

        def prepare_dense_encoder(self, *, dense, execution):
            calls["dense"] = dense
            calls["execution"] = execution
            return kit

    def from_preset(name, **kwargs):
        calls["preset"] = (name, kwargs)
        return PublicModel()

    monkeypatch.setattr(Model, "from_preset", from_preset)
    config = PipelineConfig(
        dataset_csv=str(tmp_path / "manifest.csv"),
        splits_csv=str(tmp_path / "splits.csv"),
        output_root=str(tmp_path / "out"),
        dataset_type="segmentation",
        feature_mode="live",
        encoder=EncoderConfig(name="uni", precision="fp32", output_variant="tokens"),
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": NUM_CLASSES}),
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=TARGET,
            requested_spacing_um=0.5,
            feature_kind="cls_attention",
            attention=AttentionConfig(blocks=(-1, -2), include_registers=True),
        ),
    )

    source = Pipeline(config)._build_live_segmentation_source()

    assert calls["preset"] == (
        "uni",
        {"output_variant": "tokens", "allow_non_recommended_settings": False},
    )
    assert calls["dense"].feature_kind == "cls_attention"
    assert calls["dense"].attention_blocks == (-1, -2)
    assert calls["dense"].attention_include_registers is True
    assert calls["execution"].precision == "fp32"
    assert source.kit is kit
    assert source.feature_dim == 6
