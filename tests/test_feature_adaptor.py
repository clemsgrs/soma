"""Tests for soma.training.feature_adaptor — the fitted feature adaptor.

The adaptor is the one genuinely new seam this work adds: a buffer-carrying front
module inserted ahead of the aggregator/head. These tests assert its *external*
behavior — what the fitted buffers are, that they come from the Support split only,
that the optimizer never sees them, and that they survive a checkpoint round-trip.

Sections: the module itself, then each path it is wired into — the tile-encoder MIL
path (issues #283, #284) and the slide-encoder embedding path (issue #285).
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
from soma.config import NormalizationConfig, ProjectionConfig
from soma.tasks.classification import BinaryClassificationHead
from soma.training.feature_adaptor import build_feature_adaptor
from soma.training.model import EmbeddingModel, MILModel


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


def test_pca_projects_to_the_target_width():
    """The tracer: a fitted PCA maps D-dim features onto `target_dim` columns."""
    features = _train_features()
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="pca", target_dim=2), num_features=3
    )
    adaptor.fit([features])

    out = adaptor(features)

    assert adaptor.output_dim == 2
    assert out.shape == (512, 2)


def test_pca_keeps_the_highest_variance_directions_first():
    """PCA is not any old rotation: component k captures the k-th most variance, and the
    projected columns are decorrelated."""
    features = _train_features()
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="pca", target_dim=3), num_features=3
    ).fit([features])

    out = adaptor(features).to(torch.float64)

    variances = out.var(dim=0, unbiased=True)
    assert torch.all(variances[:-1] >= variances[1:])
    # The (3,) feature was built with a 100x scale, so component 0 must dominate.
    assert variances[0] / variances.sum() > 0.9
    covariance = torch.cov(out.T)
    off_diagonal = covariance - torch.diag(torch.diagonal(covariance))
    assert torch.allclose(off_diagonal, torch.zeros_like(off_diagonal), atol=1e-3)


def test_pca_centers_on_its_own_fitted_mean():
    """`pca` centers intrinsically — the projected Support set has zero mean even with
    no normalization stage in front of it."""
    features = _train_features()
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="pca", target_dim=2), num_features=3
    ).fit([features])

    out = adaptor(features)

    assert torch.allclose(out.mean(dim=0), torch.zeros(2), atol=1e-3)


def test_repeated_pca_fits_are_byte_identical():
    """The pinned sign convention: an eigenvector is only defined up to sign, so two fits
    on identical data must not differ by a flip, or the run is not reproducible."""
    features = _train_features()
    projection = ProjectionConfig(method="pca", target_dim=2)

    first = build_feature_adaptor(None, projection, num_features=3).fit([features])
    second = build_feature_adaptor(None, projection, num_features=3).fit([features])

    assert torch.equal(first.projection_matrix, second.projection_matrix)
    assert torch.equal(first.projection_mean, second.projection_mean)
    assert torch.equal(first(features), second(features))


def test_pca_sign_convention_pins_the_largest_entry_positive():
    """The convention itself, stated as behavior: whatever LAPACK hands back, every
    component leaves this module with its largest-magnitude entry positive."""
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="pca", target_dim=3), num_features=3
    ).fit([_train_features()])

    components = adaptor.projection_matrix
    pivots = components.abs().argmax(dim=1)
    assert torch.all(components.gather(1, pivots.unsqueeze(1)) > 0)


def test_pca_is_fit_only_on_the_rows_it_is_given():
    """Leak-free by construction: the basis comes from the batches handed to fit() and
    nothing else."""
    support = _train_features()
    held_out = support * 50.0 + 1000.0
    projection = ProjectionConfig(method="pca", target_dim=2)

    support_only = build_feature_adaptor(None, projection, num_features=3).fit([support])
    leaked = build_feature_adaptor(None, projection, num_features=3).fit(
        [support, held_out]
    )

    assert support_only.n_fit_samples == support.shape[0]
    assert not torch.allclose(
        support_only.projection_mean, leaked.projection_mean, atol=1e-3
    )


def test_random_projection_is_reproducible_from_the_seed():
    """Same seed + same encoder + same dims ⇒ the same matrix, drawn without ever
    consulting the global RNG, so it is constant across training trajectories."""
    projection = ProjectionConfig(method="random", target_dim=2, seed=11)

    torch.manual_seed(1234)
    first = build_feature_adaptor(
        None, projection, num_features=3, encoder_identity="uni2"
    )
    torch.manual_seed(4321)  # a different trajectory seed must not move the matrix
    torch.randn(97)
    second = build_feature_adaptor(
        None, projection, num_features=3, encoder_identity="uni2"
    )

    assert torch.equal(first.projection_matrix, second.projection_matrix)
    assert first.projection_matrix.abs().sum() > 0


def test_random_projection_differs_by_seed_and_by_encoder():
    """The seed is combined with the encoder identity, so two encoders in one roster
    never share a matrix."""
    base = build_feature_adaptor(
        None,
        ProjectionConfig(method="random", target_dim=2, seed=11),
        num_features=3,
        encoder_identity="uni2",
    )
    reseeded = build_feature_adaptor(
        None,
        ProjectionConfig(method="random", target_dim=2, seed=12),
        num_features=3,
        encoder_identity="uni2",
    )
    other_encoder = build_feature_adaptor(
        None,
        ProjectionConfig(method="random", target_dim=2, seed=11),
        num_features=3,
        encoder_identity="virchow2",
    )

    assert not torch.equal(base.projection_matrix, reseeded.projection_matrix)
    assert not torch.equal(base.projection_matrix, other_encoder.projection_matrix)


def test_random_projection_approximately_preserves_inner_products():
    """The 1/sqrt(target_dim) scaling is what makes the map an ablation of *width* rather
    than a rescaling of the features."""
    torch.manual_seed(0)
    features = torch.nn.functional.normalize(torch.randn(64, 256), dim=-1)
    adaptor = build_feature_adaptor(
        None,
        ProjectionConfig(method="random", target_dim=512, seed=3),
        num_features=256,
        encoder_identity="uni2",
    )

    projected = adaptor(features)

    # Squared norms survive the map (this is what the 1/sqrt(target_dim) scaling buys)...
    squared_norms = projected.pow(2).sum(dim=-1)
    assert abs(float(squared_norms.mean()) - 1.0) < 0.05
    # ...and so do pairwise inner products, to the O(1/sqrt(target_dim)) JL error.
    original = features @ features.T
    preserved = projected @ projected.T
    assert float((preserved - original).abs().mean()) < 0.05


def test_random_projection_needs_no_fitting():
    """It is label-free *and* data-free: it never touches the Support set."""
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="random", target_dim=8), num_features=3
    )

    assert adaptor.requires_fit is False
    assert adaptor(_train_features()).shape == (512, 8)


def test_random_projection_may_expand_beyond_the_native_dim():
    """Unlike PCA, random is unconstrained in target_dim."""
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="random", target_dim=64), num_features=3
    )

    assert adaptor.output_dim == 64
    assert adaptor(_train_features()).shape == (512, 64)


def test_pca_preflight_rejects_target_dim_above_the_feature_dim():
    with pytest.raises(ValueError, match="exceeds the encoder's feature dimension"):
        build_feature_adaptor(
            None, ProjectionConfig(method="pca", target_dim=8), num_features=3
        )


def test_pca_preflight_rejects_too_few_fit_samples():
    """`n_fit_samples >= target_dim`, with the shortfall named."""
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="pca", target_dim=3), num_features=3
    )

    with pytest.raises(ValueError, match="only 2 feature row"):
        adaptor.fit([torch.randn(2, 3)])


def test_pca_refuses_to_transform_before_it_is_fit():
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="pca", target_dim=2), num_features=3
    )

    with pytest.raises(RuntimeError, match="before it was fit"):
        adaptor(_train_features())


def test_projection_is_applied_after_normalization():
    """Order is normalize → project: the composed adaptor equals projecting a
    separately-normalized copy of the same features."""
    features = _train_features()
    normalization = NormalizationConfig(method="zscore")
    projection = ProjectionConfig(method="pca", target_dim=2)

    composed = build_feature_adaptor(
        normalization, projection, num_features=3
    ).fit([features])
    normalize_only = build_feature_adaptor(normalization, num_features=3).fit([features])
    project_only = build_feature_adaptor(None, projection, num_features=3).fit(
        [normalize_only(features)]
    )

    assert torch.allclose(
        composed(features), project_only(normalize_only(features)), atol=1e-5
    )
    # ...and the reverse order is a genuinely different transform, so the assertion above
    # is not vacuous.
    assert not torch.allclose(
        composed(features), normalize_only.center.new_zeros(1), atol=1e-5
    )


def test_projection_buffers_are_frozen_state_not_parameters():
    """The projection must not be a learned layer: learning it would relocate the very
    capacity confound the ablation exists to remove."""
    adaptor = build_feature_adaptor(
        None, ProjectionConfig(method="pca", target_dim=2), num_features=3
    ).fit([_train_features()])

    assert list(adaptor.parameters()) == []
    buffer_names = {name for name, _ in adaptor.named_buffers()}
    assert {"projection_matrix", "projection_mean"} <= buffer_names


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


def _synthetic_fold(
    tmp_path: Path, feature_dim: int = _TILE_DIM
) -> tuple[Dataset, Splits, FeatureStore]:
    """4 samples — 2 train, 1 tune, 1 test — whose feature scales differ wildly per
    split, so a leaked tune/test row would visibly move the fitted statistics."""
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        torch.save(torch.randn(6, feature_dim) * scale, feature_dir / f"{sample_id}.pt")

    dataset = Dataset(dataset_csv)
    return dataset, Splits(splits_csv, dataset), FeatureStore(feature_dir)


def _run_fold(
    tmp_path: Path,
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None = None,
    *,
    feature_dim: int = _TILE_DIM,
    aggregator: str = "mean_pool",
):
    dataset, splits, store = _synthetic_fold(tmp_path, feature_dim)
    fold_dir = tmp_path / "fold_0"
    result = train_one_fold(
        feature_store=store,
        dataset=dataset,
        fold_split=splits.folds[0],
        aggregator=AggregatorConfig(name=aggregator),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=2, learning_rate=1e-3, batch_size=1),
        fold_dir=fold_dir,
        normalization=normalization,
        projection=projection,
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


def test_projection_rewires_the_aggregator_to_the_target_width(tmp_path: Path):
    """The dim rewire, stated as the thing it exists to guarantee: with a projection
    active the downstream trainable parameter count is *independent of the encoder's
    native dim*, so a capacity difference can no longer masquerade as a ranking."""
    projection = ProjectionConfig(method="pca", target_dim=3)

    _, narrow_state, _ = _run_fold(
        tmp_path / "narrow", None, projection, feature_dim=4, aggregator="abmil"
    )
    _, wide_state, _ = _run_fold(
        tmp_path / "wide", None, projection, feature_dim=32, aggregator="abmil"
    )

    def _trainable(state_dict) -> int:
        return sum(
            value.numel()
            for key, value in state_dict.items()
            if not key.startswith("feature_adaptor")
        )

    assert _trainable(narrow_state) == _trainable(wide_state)
    # ...and the adaptor is what absorbs the difference: its frozen matrix is the only
    # thing whose size still tracks the native dim.
    assert narrow_state["feature_adaptor.projection_matrix"].shape == (3, 4)
    assert wide_state["feature_adaptor.projection_matrix"].shape == (3, 32)


def test_without_projection_the_aggregator_still_tracks_the_native_dim(tmp_path: Path):
    """The counterpart that makes the test above non-vacuous: this is the confound."""
    _, narrow_state, _ = _run_fold(
        tmp_path / "narrow", None, None, feature_dim=4, aggregator="abmil"
    )
    _, wide_state, _ = _run_fold(
        tmp_path / "wide", None, None, feature_dim=32, aggregator="abmil"
    )

    assert sum(v.numel() for v in narrow_state.values()) != sum(
        v.numel() for v in wide_state.values()
    )


def test_projection_state_rides_in_the_fold_checkpoint(tmp_path: Path):
    """The fitted map must be re-applied verbatim by the final-checkpoint test pass."""
    fold_dir, state_dict, store = _run_fold(
        tmp_path, None, ProjectionConfig(method="pca", target_dim=3)
    )

    support = torch.cat([store.load("s0"), store.load("s1")], dim=0)
    expected = build_feature_adaptor(
        None, ProjectionConfig(method="pca", target_dim=3), num_features=_TILE_DIM
    ).fit([store.load("s0"), store.load("s1")])

    assert torch.equal(
        state_dict["feature_adaptor.projection_matrix"], expected.projection_matrix
    )
    assert torch.equal(
        state_dict["feature_adaptor.projection_mean"], expected.projection_mean
    )
    # Fit on the Support split alone — the held-out rows are two orders of magnitude
    # wider, so a leak would move the mean visibly.
    assert not torch.allclose(
        state_dict["feature_adaptor.projection_mean"],
        torch.cat([store.load(f"s{i}") for i in range(4)], dim=0).mean(dim=0),
        atol=1e-2,
    )
    assert support.shape[0] == 12


def test_projection_without_normalization_needs_no_normalization_section(tmp_path: Path):
    """`normalization` and `projection` are independently configurable."""
    _, state_dict, _ = _run_fold(
        tmp_path, None, ProjectionConfig(method="random", target_dim=5)
    )

    assert "feature_adaptor.projection_matrix" in state_dict
    # The normalize stage is off, so its buffers stay at their identity values.
    assert torch.equal(
        state_dict["feature_adaptor.center"], torch.zeros(_TILE_DIM)
    )
    assert torch.equal(state_dict["feature_adaptor.scale"], torch.ones(_TILE_DIM))


def test_normalization_and_projection_compose_in_one_fold(tmp_path: Path):
    _, state_dict, store = _run_fold(
        tmp_path,
        NormalizationConfig(method="zscore"),
        ProjectionConfig(method="pca", target_dim=3),
    )

    support = torch.cat([store.load("s0"), store.load("s1")], dim=0).to(torch.float64)
    assert torch.allclose(
        state_dict["feature_adaptor.center"], support.mean(dim=0).float(), atol=1e-5
    )
    assert state_dict["feature_adaptor.projection_matrix"].shape == (3, _TILE_DIM)


def test_fold_sidecar_reports_the_projection(tmp_path: Path):
    fold_dir, _, _ = _run_fold(
        tmp_path, None, ProjectionConfig(method="pca", target_dim=3, seed=5)
    )

    sidecar = json.loads(
        (fold_dir / "feature_adapter.json").read_text(encoding="utf-8")
    )["projection"]

    assert sidecar["method"] == "pca"
    assert sidecar["target_dim"] == 3
    assert sidecar["seed"] == 5
    assert sidecar["n_fit_samples"] == 12  # 2 Support samples x 6 tiles
    assert sidecar["input_dim"] == _TILE_DIM
    assert sidecar["output_dim"] == 3
    ratios = sidecar["explained_variance_ratio"]
    assert len(ratios) == 3
    assert ratios == sorted(ratios, reverse=True)
    assert 0.0 < sum(ratios) <= 1.0 + 1e-6


def test_sidecar_omits_explained_variance_for_random_projection(tmp_path: Path):
    fold_dir, _, _ = _run_fold(
        tmp_path, None, ProjectionConfig(method="random", target_dim=3)
    )

    sidecar = json.loads(
        (fold_dir / "feature_adapter.json").read_text(encoding="utf-8")
    )["projection"]

    assert sidecar["method"] == "random"
    assert "explained_variance_ratio" not in sidecar


def test_projection_off_leaves_no_adaptor_in_the_run(tmp_path: Path):
    fold_dir, state_dict, _ = _run_fold(tmp_path, None, ProjectionConfig())

    assert not any(key.startswith("feature_adaptor") for key in state_dict)
    assert not (fold_dir / "feature_adapter.json").exists()


def test_pca_preflight_fires_inside_a_fold(tmp_path: Path):
    """The Support set here has 12 tile rows; asking for more components than that must
    fail with the shortfall named rather than produce a degenerate basis."""
    with pytest.raises(ValueError, match="only 12 feature row"):
        _run_fold(
            tmp_path,
            None,
            ProjectionConfig(method="pca", target_dim=13),
            feature_dim=32,  # wide enough that the target_dim <= D half passes
        )


def test_attention_reconstructs_the_projected_model_from_the_checkpoint(tmp_path: Path):
    """`save_attention` rebuilds MILModel from config + checkpoint and loads strictly.

    Under a projection that rebuild has to agree on *both* the extra buffers and the
    rewired aggregator width, or the strict load fails — the same class of breakage the
    normalize stage hit.
    """
    from soma.config import PipelineConfig, save_config
    from soma.heatmaps import save_attention

    dataset, splits, store = _synthetic_fold(tmp_path / "data")
    projection = ProjectionConfig(method="pca", target_dim=3)
    fold_dir = tmp_path / "run"
    train_one_fold(
        feature_store=store,
        dataset=dataset,
        fold_split=splits.folds[0],
        aggregator=AggregatorConfig(name="abmil"),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=1, learning_rate=1e-3, batch_size=1),
        fold_dir=fold_dir,
        projection=projection,
    )
    save_config(
        PipelineConfig(
            dataset_csv=tmp_path / "data" / "dataset.csv",
            splits_csv=tmp_path / "data" / "splits.csv",
            output_root=tmp_path / "out",
            dataset_type="slide",
            aggregator=AggregatorConfig(name="abmil"),
            task=TaskConfig(name="binary_classification", params={"num_classes": 2}),
            projection=projection,
        ),
        fold_dir / "config.yaml",
    )

    save_attention(fold_dir, dataset, store)

    assert list((fold_dir / "attention").rglob("*.npz"))


# ---------------------------------------------------------------------------
# Slide-encoder embedding path — end to end through train_one_fold (issue #285)
# ---------------------------------------------------------------------------

_EMBEDDING_DIM = 4


def _synthetic_embedding_fold(
    tmp_path: Path, feature_dim: int = _EMBEDDING_DIM
) -> tuple[Dataset, Splits, FeatureStore]:
    """The slide-encoder path: **one** feature vector per slide, not a bag of tiles.

    4 slides — 2 Support, 1 tune, 1 test — whose embedding scales differ wildly per
    split, so a leaked held-out row would visibly move the fitted statistics. The
    Support set is deliberately tiny: "K means K" is the regime this path runs in.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
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
        # (D,) — a single embedding per slide is what makes this the slide-encoder path.
        torch.save(torch.randn(feature_dim) * scale, feature_dir / f"{sample_id}.pt")

    dataset = Dataset(dataset_csv)
    return dataset, Splits(splits_csv, dataset), FeatureStore(feature_dir)


def _run_embedding_fold(
    tmp_path: Path,
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None = None,
    *,
    feature_dim: int = _EMBEDDING_DIM,
):
    dataset, splits, store = _synthetic_embedding_fold(tmp_path, feature_dim)
    fold_dir = tmp_path / "fold_0"
    result = train_one_fold(
        feature_store=store,
        dataset=dataset,
        fold_split=splits.folds[0],
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=2, learning_rate=1e-3, batch_size=1),
        fold_dir=fold_dir,
        normalization=normalization,
        projection=projection,
    )
    checkpoint = torch.load(
        result.train_result.checkpoint_path, weights_only=True, map_location="cpu"
    )
    return fold_dir, checkpoint["model_state_dict"], store


def test_embedding_path_fits_zscore_on_the_support_embeddings_only(tmp_path: Path):
    """The tracer for this path: the fit is over the K Support *embeddings* — one vector
    per slide — and nothing else.
    """
    _, state_dict, store = _run_embedding_fold(
        tmp_path, NormalizationConfig(method="zscore")
    )

    support = torch.stack([store.load("s0"), store.load("s1")]).to(torch.float64)
    assert torch.allclose(
        state_dict["feature_adaptor.center"], support.mean(dim=0).float(), atol=1e-5
    )
    assert torch.allclose(
        state_dict["feature_adaptor.scale"],
        support.std(dim=0, unbiased=False).float(),
        atol=1e-4,
    )
    # The held-out slides are ~2 orders of magnitude wider; had either leaked in, the
    # fitted scale could not match the Support-only estimate above.
    everything = torch.stack([store.load(f"s{i}") for i in range(4)])
    assert not torch.allclose(
        state_dict["feature_adaptor.scale"],
        everything.std(dim=0, unbiased=False),
        atol=1e-4,
    )


def test_embedding_path_evaluation_never_moves_the_fitted_buffers(tmp_path: Path):
    """train_one_fold evaluates tune and test after fitting; the buffers it saved must be
    exactly the ones it fit — the transform is frozen, not running."""
    dataset, splits, store = _synthetic_embedding_fold(tmp_path)
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=_EMBEDDING_DIM
    ).fit([store.load("s0"), store.load("s1")])
    expected_center = adaptor.center.clone()
    expected_scale = adaptor.scale.clone()

    _, state_dict, _ = _run_embedding_fold(
        tmp_path, NormalizationConfig(method="zscore")
    )

    assert torch.equal(state_dict["feature_adaptor.center"], expected_center)
    assert torch.equal(state_dict["feature_adaptor.scale"], expected_scale)


def test_embedding_path_pca_preflight_fires_when_k_is_smaller_than_target_dim(
    tmp_path: Path,
):
    """The preflight this path exists to protect: the fit sample count *is* K, so asking
    for a basis wider than the Support set must name the shortfall rather than hand back
    a degenerate basis. Two Support slides means two rows — nothing more is available."""
    with pytest.raises(ValueError, match="only 2 feature row"):
        _run_embedding_fold(
            tmp_path,
            None,
            ProjectionConfig(method="pca", target_dim=3),
            feature_dim=8,  # wide enough that the target_dim <= D half passes
        )


def test_embedding_path_projection_rewires_the_head_to_the_target_width(tmp_path: Path):
    """The dim rewire on this path: with a projection active the *head* — the only
    trainable thing here — is built against `target_dim`, so its parameter count is
    independent of the encoder's native dim."""
    projection = ProjectionConfig(method="random", target_dim=6)

    _, narrow_state, _ = _run_embedding_fold(
        tmp_path / "narrow", None, projection, feature_dim=4
    )
    _, wide_state, _ = _run_embedding_fold(
        tmp_path / "wide", None, projection, feature_dim=32
    )

    def _trainable(state_dict) -> int:
        return sum(
            value.numel()
            for key, value in state_dict.items()
            if not key.startswith("feature_adaptor")
        )

    assert _trainable(narrow_state) == _trainable(wide_state)
    # ...and the adaptor is what absorbs the difference: its frozen matrix is the only
    # thing whose size still tracks the native dim.
    assert narrow_state["feature_adaptor.projection_matrix"].shape == (6, 4)
    assert wide_state["feature_adaptor.projection_matrix"].shape == (6, 32)


def test_embedding_path_without_projection_the_head_tracks_the_native_dim(
    tmp_path: Path,
):
    """The counterpart that makes the test above non-vacuous: this is the confound."""
    _, narrow_state, _ = _run_embedding_fold(tmp_path / "narrow", None, None, feature_dim=4)
    _, wide_state, _ = _run_embedding_fold(tmp_path / "wide", None, None, feature_dim=32)

    assert sum(v.numel() for v in narrow_state.values()) != sum(
        v.numel() for v in wide_state.values()
    )


def test_embedding_path_checkpoint_reconstructs_and_scores(tmp_path: Path):
    """A fold's checkpoint must reload strictly into a model rebuilt from config alone,
    and then score. Under a projection that rebuild has to agree on *both* the extra
    adaptor buffers and the rewired head width — the width mismatch is invisible until
    something actually runs, so this reconstructs and runs."""
    normalization = NormalizationConfig(method="zscore")
    projection = ProjectionConfig(method="pca", target_dim=2)
    _, state_dict, store = _run_embedding_fold(tmp_path, normalization, projection)

    rebuilt_adaptor = build_feature_adaptor(
        normalization, projection, num_features=_EMBEDDING_DIM
    )
    restored = EmbeddingModel(
        task_head=BinaryClassificationHead(input_dim=2, num_classes=2),
        feature_adaptor=rebuilt_adaptor,
    )
    restored.load_state_dict(state_dict)  # strict: extra/missing keys would raise
    restored.eval()

    assert restored.feature_adaptor.is_fitted
    # The fitted state came back, not the identity values it was constructed with.
    expected = build_feature_adaptor(
        normalization, projection, num_features=_EMBEDDING_DIM
    ).fit([store.load("s0"), store.load("s1")])
    assert torch.equal(restored.feature_adaptor.center, expected.center)
    assert torch.equal(restored.feature_adaptor.projection_matrix, expected.projection_matrix)
    # ...and it scores: the head really is `target_dim` wide.
    logits = restored(store.load("s3").unsqueeze(0)).logits
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_embedding_path_adaptor_state_is_not_among_the_models_parameters(tmp_path: Path):
    """The optimizer builds from model.parameters(); the fitted statistics must not be
    there, or they would be trained rather than estimated."""
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=_EMBEDDING_DIM
    ).fit([torch.randn(8, _EMBEDDING_DIM)])
    head_kwargs = {"input_dim": _EMBEDDING_DIM, "num_classes": 2}
    model = EmbeddingModel(
        task_head=BinaryClassificationHead(**head_kwargs), feature_adaptor=adaptor
    )

    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any(name.startswith("feature_adaptor") for name in parameter_names)
    assert parameter_names == {
        name
        for name, _ in EmbeddingModel(
            task_head=BinaryClassificationHead(**head_kwargs)
        ).named_parameters()
    }


def test_embedding_model_without_adaptor_is_structurally_identical_to_today():
    """"Omitted section = absent module" on this path: same state_dict, same parameters,
    same logits — the byte-parity anchor for existing slide-encoder runs."""
    torch.manual_seed(0)
    legacy = EmbeddingModel(task_head=BinaryClassificationHead(input_dim=3, num_classes=2))
    torch.manual_seed(0)
    with_none = EmbeddingModel(
        task_head=BinaryClassificationHead(input_dim=3, num_classes=2),
        feature_adaptor=None,
    )

    assert list(with_none.state_dict()) == list(legacy.state_dict())
    assert [name for name, _ in with_none.named_parameters()] == [
        name for name, _ in legacy.named_parameters()
    ]
    X = torch.randn(2, 3)
    assert torch.equal(with_none(X).logits, legacy(X).logits)


def test_embedding_path_writes_a_feature_adapter_qc_sidecar(tmp_path: Path):
    fold_dir, _, _ = _run_embedding_fold(
        tmp_path,
        NormalizationConfig(method="zscore", eps=1e-4),
        ProjectionConfig(method="pca", target_dim=2, seed=7),
    )

    sidecar = json.loads((fold_dir / "feature_adapter.json").read_text(encoding="utf-8"))

    assert sidecar["normalization"]["method"] == "zscore"
    assert sidecar["normalization"]["eps"] == 1e-4
    assert sidecar["projection"]["method"] == "pca"
    assert sidecar["projection"]["seed"] == 7
    assert sidecar["projection"]["n_fit_samples"] == 2  # K Support embeddings
    assert sidecar["projection"]["input_dim"] == _EMBEDDING_DIM
    assert sidecar["projection"]["output_dim"] == 2


def test_embedding_path_with_both_blocks_off_leaves_no_adaptor_in_the_run(tmp_path: Path):
    fold_dir, state_dict, _ = _run_embedding_fold(tmp_path, None, ProjectionConfig())

    assert not any(key.startswith("feature_adaptor") for key in state_dict)
    assert not (fold_dir / "feature_adapter.json").exists()


def test_unsupported_path_refuses_projection_rather_than_ignoring_it(tmp_path: Path):
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
            projection=ProjectionConfig(method="pca", target_dim=2),
        )


def test_train_refuses_projection_on_the_dense_paths(tmp_path: Path):
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
            projection=ProjectionConfig(method="random", target_dim=2),
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
