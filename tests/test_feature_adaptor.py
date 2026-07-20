"""Tests for soma.training.feature_adaptor — the fitted feature adaptor (issue #283).

The adaptor is the one genuinely new seam this work adds: a buffer-carrying front
module inserted ahead of the aggregator/head. These tests assert its *external*
behavior — what the fitted buffers are, that they come from the Support split only,
that the optimizer never sees them, and that they survive a checkpoint round-trip.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from soma.aggregators.pooling import MeanPool
from soma.config import AggregatorConfig, TaskConfig, TrainingConfig
from soma.dataset import Dataset, Splits
from soma.features import FeatureStore
from soma.pipeline import train, train_one_fold
from soma.config import NormalizationConfig
from soma.tasks.classification import BinaryClassificationHead
from soma.training.feature_adaptor import build_feature_adaptor
from soma.training.model import MILModel


def _train_features() -> torch.Tensor:
    """(N, 3) tiles with deliberately unequal per-feature location and scale."""
    torch.manual_seed(0)
    base = torch.randn(512, 3)
    return base * torch.tensor([1.0, 10.0, 100.0]) + torch.tensor([-5.0, 0.0, 20.0])


def test_zscore_standardizes_the_features_it_was_fit_on():
    features = _train_features()
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=3
    )
    adaptor.fit([features])

    out = adaptor(features)

    assert torch.allclose(out.mean(dim=0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(out.std(dim=0, unbiased=False), torch.ones(3), atol=1e-4)


def test_no_adaptor_is_built_when_normalization_is_off():
    """"Omitted section = absent module" — the byte-parity anchor for legacy runs."""
    assert build_feature_adaptor(NormalizationConfig(), num_features=3) is None
    assert build_feature_adaptor(None, num_features=3) is None


def test_fitted_state_is_buffers_not_parameters():
    """The optimizer must never see the fitted statistics — they are an estimate of the
    Support distribution, not weights to learn."""
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=3
    )
    adaptor.fit([_train_features()])

    assert list(adaptor.parameters()) == []
    buffer_names = {name for name, _ in adaptor.named_buffers()}
    assert {"center", "scale"} <= buffer_names


def test_zscore_floors_constant_channels_at_eps():
    """A constant channel has zero spread; eps keeps it from blowing up, and the count
    of such channels is reported for QC."""
    features = torch.stack(
        [
            torch.arange(8, dtype=torch.float32),
            torch.full((8,), 3.0),  # constant → floored
            torch.full((8,), -1.0),  # constant → floored
        ],
        dim=1,
    )
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore", eps=1e-3), num_features=3
    )
    adaptor.fit([features])

    out = adaptor(features)

    assert adaptor.num_eps_floored == 2
    assert torch.isfinite(out).all()
    assert torch.allclose(out[:, 1:], torch.zeros(8, 2), atol=1e-6)


def test_zscore_refuses_to_transform_before_it_is_fit():
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=3
    )

    with pytest.raises(RuntimeError, match="before it was fit"):
        adaptor(_train_features())


def test_l2_normalizes_each_feature_vector_without_fitting():
    features = _train_features()
    adaptor = build_feature_adaptor(NormalizationConfig(method="l2"), num_features=3)

    out = adaptor(features)

    assert adaptor.requires_fit is False
    assert torch.allclose(out.norm(dim=-1), torch.ones(features.shape[0]), atol=1e-5)


def test_layernorm_standardizes_across_the_feature_axis_without_fitting():
    features = _train_features()
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="layernorm"), num_features=3
    )

    out = adaptor(features)

    assert adaptor.requires_fit is False
    assert torch.allclose(out.mean(dim=-1), torch.zeros(features.shape[0]), atol=1e-5)


def test_adaptor_preserves_shape_on_batched_bags():
    """The same module serves (N, D) tiles and (B, N, D) MIL bags."""
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=3
    )
    adaptor.fit([_train_features()])

    bags = torch.randn(4, 7, 3)
    assert adaptor(bags).shape == (4, 7, 3)


# ---------------------------------------------------------------------------
# MIL path — composition into the model
# ---------------------------------------------------------------------------


def _mil_model(adaptor=None) -> MILModel:
    torch.manual_seed(0)
    return MILModel(
        aggregator=MeanPool(input_dim=3),
        task_head=BinaryClassificationHead(input_dim=3, num_classes=2),
        feature_adaptor=adaptor,
    )


def test_model_without_adaptor_is_structurally_identical_to_today():
    """`normalization: none` leaves no trace on the model: same state_dict, same
    parameters — the model half of the byte-parity anchor."""
    torch.manual_seed(0)
    legacy = MILModel(
        aggregator=MeanPool(input_dim=3),
        task_head=BinaryClassificationHead(input_dim=3, num_classes=2),
    )

    with_none = _mil_model(adaptor=None)

    assert list(with_none.state_dict()) == list(legacy.state_dict())
    assert [name for name, _ in with_none.named_parameters()] == [
        name for name, _ in legacy.named_parameters()
    ]
    X = torch.randn(2, 5, 3)
    assert torch.equal(with_none(X).logits, legacy(X).logits)


def test_adaptor_transforms_features_before_the_aggregator():
    """The adaptor is a *front* module: the model's output equals running the head on
    pre-transformed features."""
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=3
    ).fit([_train_features()])
    model = _mil_model(adaptor=adaptor)
    bare = _mil_model(adaptor=None)

    X = torch.randn(2, 5, 3)

    assert torch.allclose(model(X).logits, bare(adaptor(X)).logits, atol=1e-6)


def test_adaptor_state_is_not_among_the_models_parameters():
    """The optimizer builds from model.parameters(); the fitted statistics must not be
    there, or they would be trained rather than estimated."""
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=3
    ).fit([_train_features()])
    model = _mil_model(adaptor=adaptor)

    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any(name.startswith("feature_adaptor") for name in parameter_names)
    assert {name for name, _ in _mil_model(adaptor=None).named_parameters()} == (
        parameter_names
    )


def test_adaptor_state_round_trips_through_the_checkpoint(tmp_path: Path):
    """The final-checkpoint test pass re-applies the exact fitted transform, so the
    buffers have to ride in the saved state_dict."""
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=3
    ).fit([_train_features()])
    trained = _mil_model(adaptor=adaptor)
    checkpoint_path = tmp_path / "best_model.pt"
    torch.save({"model_state_dict": trained.state_dict()}, checkpoint_path)

    restored = _mil_model(
        adaptor=build_feature_adaptor(
            NormalizationConfig(method="zscore"), num_features=3
        )
    )
    restored.load_state_dict(
        torch.load(checkpoint_path, weights_only=True)["model_state_dict"]
    )

    assert restored.feature_adaptor.is_fitted
    assert torch.equal(restored.feature_adaptor.center, adaptor.center)
    assert torch.equal(restored.feature_adaptor.scale, adaptor.scale)
    X = torch.randn(2, 5, 3)
    assert torch.equal(restored(X).logits, trained(X).logits)


# ---------------------------------------------------------------------------
# MIL path — end to end through train_one_fold
# ---------------------------------------------------------------------------

_TILE_DIM = 4


def _synthetic_fold(tmp_path: Path) -> tuple[Dataset, Splits, FeatureStore]:
    """4 samples — 2 train, 1 tune, 1 test — whose feature scales differ wildly per
    split, so a leaked tune/test row would visibly move the fitted statistics."""
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label\n"
        "s0,/slides/s0.svs,tumor\n"
        "s1,/slides/s1.svs,normal\n"
        "s2,/slides/s2.svs,tumor\n"
        "s3,/slides/s3.svs,normal\n",
        encoding="utf-8",
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "fold,sample_id,split\n0,s0,train\n0,s1,train\n0,s2,tune\n0,s3,test\n",
        encoding="utf-8",
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(exist_ok=True)
    torch.manual_seed(0)
    scales = {"s0": 1.0, "s1": 2.0, "s2": 500.0, "s3": 900.0}
    for sample_id, scale in scales.items():
        torch.save(torch.randn(6, _TILE_DIM) * scale, feature_dir / f"{sample_id}.pt")

    dataset = Dataset(dataset_csv)
    return dataset, Splits(splits_csv, dataset), FeatureStore(feature_dir)


def _run_fold(tmp_path: Path, normalization: NormalizationConfig | None):
    dataset, splits, store = _synthetic_fold(tmp_path)
    fold_dir = tmp_path / "fold_0"
    result = train_one_fold(
        feature_store=store,
        dataset=dataset,
        fold_split=splits.folds[0],
        aggregator=AggregatorConfig(name="mean_pool"),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=2, learning_rate=1e-3, batch_size=1),
        fold_dir=fold_dir,
        normalization=normalization,
    )
    checkpoint = torch.load(
        result.train_result.checkpoint_path, weights_only=True, map_location="cpu"
    )
    return fold_dir, checkpoint["model_state_dict"], store


def test_zscore_is_fit_from_the_support_split_only(tmp_path: Path):
    """K means K: the statistics come from the train samples' tiles and nothing else."""
    fold_dir, state_dict, store = _run_fold(
        tmp_path, NormalizationConfig(method="zscore")
    )

    support = torch.cat([store.load("s0"), store.load("s1")], dim=0).to(torch.float64)
    assert torch.allclose(
        state_dict["feature_adaptor.center"], support.mean(dim=0).float(), atol=1e-5
    )
    assert torch.allclose(
        state_dict["feature_adaptor.scale"],
        support.std(dim=0, unbiased=False).float(),
        atol=1e-4,
    )
    # The held-out splits are ~2 orders of magnitude wider; had either leaked in, the
    # fitted scale could not match the Support-only estimate above.
    everything = torch.cat([store.load(f"s{i}") for i in range(4)], dim=0)
    assert not torch.allclose(
        state_dict["feature_adaptor.scale"],
        everything.std(dim=0, unbiased=False),
        atol=1e-4,
    )


def test_evaluating_held_out_rows_never_moves_the_fitted_buffers(tmp_path: Path):
    """train_one_fold evaluates tune and test after fitting; the buffers it saved must
    be exactly the ones it fit — the transform is frozen, not running."""
    dataset, splits, store = _synthetic_fold(tmp_path)
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=_TILE_DIM
    ).fit([store.load("s0"), store.load("s1")])
    expected_center = adaptor.center.clone()
    expected_scale = adaptor.scale.clone()

    _, state_dict, _ = _run_fold(tmp_path, NormalizationConfig(method="zscore"))

    assert torch.equal(state_dict["feature_adaptor.center"], expected_center)
    assert torch.equal(state_dict["feature_adaptor.scale"], expected_scale)


def test_fold_writes_a_feature_adapter_qc_sidecar(tmp_path: Path):
    fold_dir, _, _ = _run_fold(tmp_path, NormalizationConfig(method="zscore", eps=1e-4))

    sidecar = json.loads(
        (fold_dir / "feature_adapter.json").read_text(encoding="utf-8")
    )
    assert sidecar["normalization"]["method"] == "zscore"
    assert sidecar["normalization"]["eps"] == 1e-4
    assert sidecar["normalization"]["eps_floored_channels"] == 0


def test_normalization_off_leaves_no_adaptor_in_the_run(tmp_path: Path):
    fold_dir, state_dict, _ = _run_fold(tmp_path, None)

    assert not any(key.startswith("feature_adaptor") for key in state_dict)
    assert not (fold_dir / "feature_adapter.json").exists()


def test_unsupported_path_refuses_normalization_rather_than_ignoring_it(tmp_path: Path):
    """The adaptor fits on the tile-encoder MIL path today. A request another path
    cannot honor must fail loudly — a silently-ignored transform is a false result."""
    dataset, splits, store = _synthetic_fold(tmp_path)

    with pytest.raises(ValueError, match="not yet supported"):
        train_one_fold(
            feature_store=store,
            dataset=dataset,
            fold_split=splits.folds[0],
            dataset_type="tile",
            task=TaskConfig(name="binary_classification"),
            training=TrainingConfig(epochs=1, batch_size=1),
            fold_dir=tmp_path / "fold_tile",
            normalization=NormalizationConfig(method="zscore"),
        )


def test_train_refuses_normalization_on_the_dense_paths(tmp_path: Path):
    """Same guard at the `train` entry point, which dispatches segmentation/detection
    folds that never reach train_one_fold."""
    dataset, splits, store = _synthetic_fold(tmp_path)

    with pytest.raises(ValueError, match="not yet supported"):
        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            dataset_type="segmentation",
            task=TaskConfig(name="segmentation"),
            training=TrainingConfig(epochs=1),
            run_dir=tmp_path / "run",
            normalization=NormalizationConfig(method="zscore"),
        )
