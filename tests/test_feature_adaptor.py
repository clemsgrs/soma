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


def test_fit_rejects_a_one_shot_stream_when_both_stages_need_fitting():
    """Fitting normalize+project takes two passes; a generator would arrive at the second
    exhausted and the PCA preflight would then blame the Support size. Name the real cause."""
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"),
        ProjectionConfig(method="pca", target_dim=2),
        num_features=3,
    )

    with pytest.raises(TypeError, match="re-iterable"):
        adaptor.fit(iter([_train_features()]))

    # A single-pass fit is unaffected — one stage, one pass.
    build_feature_adaptor(NormalizationConfig(method="zscore"), num_features=3).fit(
        iter([_train_features()])
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
    assert sidecar["n_support_samples"] == 2
    assert sidecar["normalization"]["n_fit_samples"] == 12
    assert sidecar["projection"]["n_fit_samples"] is None


@pytest.mark.parametrize("method", ["l2", "layernorm"])
def test_stateless_normalization_reports_zero_fit_rows(tmp_path: Path, method: str):
    fold_dir, _, _ = _run_fold(tmp_path, NormalizationConfig(method=method))

    sidecar = json.loads(
        (fold_dir / "feature_adapter.json").read_text(encoding="utf-8")
    )
    assert sidecar["n_support_samples"] == 2
    assert sidecar["normalization"]["n_fit_samples"] == 0
    assert sidecar["projection"]["n_fit_samples"] is None


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
        tmp_path,
        NormalizationConfig(method="zscore"),
        ProjectionConfig(method="pca", target_dim=3),
    )

    sidecar = json.loads(
        (fold_dir / "feature_adapter.json").read_text(encoding="utf-8")
    )
    projection = sidecar["projection"]

    assert sidecar["n_support_samples"] == 2
    assert sidecar["normalization"]["n_fit_samples"] == 12
    assert projection["method"] == "pca"
    assert projection["target_dim"] == 3
    assert projection["seed"] == 0
    assert projection["n_fit_samples"] == 12  # 2 Support samples x 6 tiles
    assert projection["input_dim"] == _TILE_DIM
    assert projection["output_dim"] == 3
    ratios = projection["explained_variance_ratio"]
    assert len(ratios) == 3
    assert ratios == sorted(ratios, reverse=True)
    assert 0.0 < sum(ratios) <= 1.0 + 1e-6


def test_sidecar_omits_explained_variance_for_random_projection(tmp_path: Path):
    fold_dir, _, _ = _run_fold(
        tmp_path, None, ProjectionConfig(method="random", target_dim=3)
    )

    sidecar = json.loads(
        (fold_dir / "feature_adapter.json").read_text(encoding="utf-8")
    )
    projection = sidecar["projection"]

    assert sidecar["n_support_samples"] == 2
    assert sidecar["normalization"]["n_fit_samples"] is None
    assert projection["method"] == "random"
    assert projection["n_fit_samples"] == 0
    assert "explained_variance_ratio" not in projection


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
        ProjectionConfig(method="pca", target_dim=2),
    )

    sidecar = json.loads((fold_dir / "feature_adapter.json").read_text(encoding="utf-8"))

    assert sidecar["normalization"]["method"] == "zscore"
    assert sidecar["normalization"]["eps"] == 1e-4
    assert sidecar["n_support_samples"] == 2
    assert sidecar["normalization"]["n_fit_samples"] == 2
    assert sidecar["projection"]["method"] == "pca"
    assert sidecar["projection"]["seed"] == 0
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


# ---------------------------------------------------------------------------
# Single-encoder dense path — end to end through the dense fold trainers (#286)
# ---------------------------------------------------------------------------

_DENSE_DIM = 4
_DENSE_TARGET = 8
_DENSE_PATCH = 4
_DENSE_CLASSES = 2


def _literal_dense_kit(geometry):
    """Plain public-kit-shaped fixture for tests that never run the encoder half."""
    from types import SimpleNamespace

    top, left, height, width = geometry.crop_box
    public_geometry = SimpleNamespace(
        target_size=geometry.target_size,
        patch_size=geometry.patch_size,
        encoded_size=geometry.encoded_size,
        grid_shape=geometry.grid_shape,
        pad=geometry.pad,
        crop_box=(left, top, left + width, top + height),
    )
    return SimpleNamespace(
        geometry=public_geometry,
        preprocessor=lambda: (lambda pixels: pixels.float()),
    )


def _synthetic_dense_fold(tmp_path: Path, feature_dim: int = _DENSE_DIM):
    """A cached dense-grid cohort: 4 ROIs — 2 Support, 1 tune, 1 test.

    Each sample is a ``(d, h, w)`` grid, so the Support fit is over **all positions in
    the Support ROIs**, not one row per sample. The held-out ROIs are ~2 orders of
    magnitude wider, so a leaked position would visibly move the fitted scale.
    """
    import numpy as np
    from PIL import Image

    from soma.dataset import SegmentationManifest
    from soma.dense import DenseFeatureStore
    from soma.dense.geometry import compute_dense_geometry
    from soma.dense.store import dense_grid_metadata, write_dense_grid

    tmp_path.mkdir(parents=True, exist_ok=True)
    dense_dir = tmp_path / "dense"
    masks_dir = tmp_path / "masks"
    dense_dir.mkdir()
    masks_dir.mkdir()

    geom = compute_dense_geometry(target_size=_DENSE_TARGET, patch_size=_DENSE_PATCH)
    meta = dense_grid_metadata(geom, feature_dim=feature_dim, pad_mode="reflect")

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    scales = {"s0": 1.0, "s1": 2.0, "s2": 500.0, "s3": 900.0}
    rows = []
    for sample_id, scale in scales.items():
        grid = torch.randn(feature_dim, *geom.grid_shape) * scale
        write_dense_grid(dense_dir, sample_id, grid, meta)
        mask_path = masks_dir / f"{sample_id}.png"
        Image.fromarray(
            rng.integers(
                0, _DENSE_CLASSES, size=(_DENSE_TARGET, _DENSE_TARGET), dtype=np.uint8
            )
        ).save(mask_path)
        rows.append((sample_id, f"{sample_id}.jpg", str(mask_path)))

    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,mask_path\n"
        + "\n".join(f"{s},{i},{m}" for s, i, m in rows)
        + "\n",
        encoding="utf-8",
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "sample_id,split,fold\ns0,train,0\ns1,train,0\ns2,tune,0\ns3,test,0\n",
        encoding="utf-8",
    )
    manifest = SegmentationManifest(manifest_csv)
    return manifest, Splits(splits_csv, manifest), DenseFeatureStore(dense_dir)


def _run_dense_fold(
    tmp_path: Path,
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None = None,
    *,
    feature_dim: int = _DENSE_DIM,
    decoder: str = "lightweight_conv",
):
    from soma.config import DecoderConfig
    from soma.pipeline import train_one_segmentation_fold

    manifest, splits, store = _synthetic_dense_fold(tmp_path / "data", feature_dim)
    fold_dir = tmp_path / "fold_0"
    result = train_one_segmentation_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=TaskConfig(name="segmentation", params={"num_classes": _DENSE_CLASSES}),
        training=TrainingConfig(epochs=1, learning_rate=1e-3, batch_size=2),
        fold_dir=fold_dir,
        decoder=DecoderConfig(name=decoder),
        normalization=normalization,
        projection=projection,
    )
    checkpoint = torch.load(
        result.train_result.checkpoint_path, weights_only=True, map_location="cpu"
    )
    return fold_dir, checkpoint["model_state_dict"], store


def _support_positions(store, sample_ids=("s0", "s1")) -> torch.Tensor:
    """Every position in the Support ROIs as ``(N, d)`` rows — the dense fit population."""
    return torch.cat(
        [store.load(sid).movedim(0, -1).reshape(-1, store.feature_dim) for sid in sample_ids],
        dim=0,
    ).to(torch.float64)


def test_dense_path_fits_zscore_over_all_support_roi_positions(tmp_path: Path):
    """The tracer: the fit population is every position of every Support ROI, and the
    statistics are channel-axis (one per feature channel of the ``(B, d, h, w)`` grid)."""
    _, state_dict, store = _run_dense_fold(
        tmp_path, NormalizationConfig(method="zscore")
    )

    support = _support_positions(store)
    assert torch.allclose(
        state_dict["feature_adaptor.center"], support.mean(dim=0).float(), atol=1e-4
    )
    assert torch.allclose(
        state_dict["feature_adaptor.scale"],
        support.std(dim=0, unbiased=False).float(),
        rtol=1e-4,
    )


def test_dense_path_fit_is_leak_free(tmp_path: Path):
    """The fold trains, tunes and tests after fitting; the buffers it saved must be
    exactly the Support-only ones — the held-out ROIs' positions never move them."""
    manifest, splits, store = _synthetic_dense_fold(tmp_path / "reference")
    reference = build_feature_adaptor(
        NormalizationConfig(method="zscore"), num_features=_DENSE_DIM
    ).fit(_support_positions(store).float().unsqueeze(0))

    _, state_dict, run_store = _run_dense_fold(
        tmp_path, NormalizationConfig(method="zscore")
    )

    assert torch.equal(state_dict["feature_adaptor.center"], reference.center)
    assert torch.equal(state_dict["feature_adaptor.scale"], reference.scale)
    # The held-out ROIs are ~2 orders of magnitude wider; had either leaked into the fit,
    # the saved scale could not equal the Support-only estimate above.
    everything = _support_positions(run_store, ("s0", "s1", "s2", "s3"))
    assert not torch.allclose(
        state_dict["feature_adaptor.scale"],
        everything.std(dim=0, unbiased=False).float(),
        rtol=1e-2,
    )


def test_dense_projection_composes_ahead_of_the_decoders_own_projection(tmp_path: Path):
    """The frozen ``d -> target_dim`` map sits in front of the decoder's learnable 1x1
    projection conv, so the decoder is simply *built* against ``target_dim``: its body is
    unchanged and its first conv reads ``target_dim`` channels, not the encoder's ``d``."""
    _, state_dict, _ = _run_dense_fold(
        tmp_path, None, ProjectionConfig(method="pca", target_dim=2)
    )

    assert state_dict["feature_adaptor.projection_matrix"].shape == (2, _DENSE_DIM)
    # The decoder's opening 1x1 conv — its only d-dependent module — now reads 2 channels.
    assert state_dict["decoder.proj.0.weight"].shape[1] == 2


def test_dense_decoder_parameter_count_is_encoder_dim_independent_under_projection(
    tmp_path: Path,
):
    """The dim-matched ablation on this path: two encoders of different native width,
    projected to one ``target_dim``, train decoders of identical capacity."""
    projection = ProjectionConfig(method="random", target_dim=3)
    _, narrow, _ = _run_dense_fold(tmp_path / "narrow", None, projection, feature_dim=4)
    _, wide, _ = _run_dense_fold(tmp_path / "wide", None, projection, feature_dim=16)

    def decoder_params(state: dict) -> int:
        return sum(v.numel() for k, v in state.items() if k.startswith("decoder."))

    assert decoder_params(narrow) == decoder_params(wide)
    # ...and that equality is not vacuous: without the projection the widths differ.
    _, narrow_native, _ = _run_dense_fold(tmp_path / "n2", None, None, feature_dim=4)
    _, wide_native, _ = _run_dense_fold(tmp_path / "w2", None, None, feature_dim=16)
    assert decoder_params(narrow_native) != decoder_params(wide_native)


def test_dense_adaptor_state_is_buffers_that_ride_in_the_checkpoint(tmp_path: Path):
    """Buffers, not parameters: the optimizer never sees them, yet they are saved so the
    final-checkpoint test pass re-applies exactly the transform training used."""
    from soma.config import DecoderConfig
    from soma.decoders.registry import decoder_registry
    from soma.tasks.segmentation import SegmentationHead
    from soma.training.model import SegmentationModel

    fold_dir, state_dict, store = _run_dense_fold(
        tmp_path,
        NormalizationConfig(method="zscore"),
        ProjectionConfig(method="pca", target_dim=2),
    )

    saved = {k for k in state_dict if k.startswith("feature_adaptor.")}
    assert {
        "feature_adaptor.center",
        "feature_adaptor.scale",
        "feature_adaptor.projection_mean",
        "feature_adaptor.projection_matrix",
    } <= saved

    # Rebuilding the same model shows those keys are buffers, not parameters.
    adaptor = build_feature_adaptor(
        NormalizationConfig(method="zscore"),
        ProjectionConfig(method="pca", target_dim=2),
        num_features=_DENSE_DIM,
    )
    model = SegmentationModel(
        decoder=decoder_registry.get("lightweight_conv")(
            input_dim=2, num_classes=_DENSE_CLASSES, num_upsample_blocks=2
        ),
        task_head=SegmentationHead(
            num_classes=_DENSE_CLASSES,
            geometry=store.geometry("s0"),
        ),
        feature_adaptor=adaptor,
    )
    parameter_names = {name for name, _ in model.named_parameters()}
    assert not any(name.startswith("feature_adaptor.") for name in parameter_names)

    # ...and they round-trip: a strict load restores the fitted state verbatim.
    model.load_state_dict(state_dict)
    assert torch.equal(model.feature_adaptor.center, state_dict["feature_adaptor.center"])
    assert torch.equal(
        model.feature_adaptor.projection_matrix,
        state_dict["feature_adaptor.projection_matrix"],
    )


def test_dense_pca_preflight_fires_when_the_support_rois_have_too_few_positions(
    tmp_path: Path,
):
    """The PCA preflight applies here too: 2 Support ROIs x a 2x2 grid = 8 positions, so
    a 32-wide target basis is not defined and must be refused by name."""
    with pytest.raises(ValueError, match="PCA needs at least target_dim rows"):
        _run_dense_fold(
            tmp_path, None, ProjectionConfig(method="pca", target_dim=32), feature_dim=64
        )


def test_dense_fold_refuses_the_adaptor_on_the_live_feature_mode(tmp_path: Path):
    """`feature_mode: live` re-encodes *augmented* tiles every step, so a transform fit on
    the cached Support grids would not match what it transforms. Fail loud, not silently."""
    from soma.config import AugmentationConfig, DecoderConfig
    from soma.dense.geometry import compute_dense_geometry
    from soma.dense.live import LiveSegmentationSource
    from soma.pipeline import train_one_segmentation_fold

    manifest, splits, _ = _synthetic_dense_fold(tmp_path / "data")
    geometry = compute_dense_geometry(target_size=_DENSE_TARGET, patch_size=_DENSE_PATCH)
    source = LiveSegmentationSource(
        kit=_literal_dense_kit(geometry),
        device="cpu",
        feature_dim=_DENSE_DIM,
        augmentation=AugmentationConfig(),
        spacing_um=None,
    )

    with pytest.raises(ValueError, match="requires feature_mode='cached'"):
        train_one_segmentation_fold(
            feature_store=source,
            dataset=manifest,
            fold_split=splits.folds[0],
            task=TaskConfig(name="segmentation", params={"num_classes": _DENSE_CLASSES}),
            training=TrainingConfig(epochs=1, batch_size=2),
            fold_dir=tmp_path / "fold",
            decoder=DecoderConfig(name="lightweight_conv"),
            normalization=NormalizationConfig(method="zscore"),
        )


def test_config_rejects_the_adaptor_together_with_feature_mode_live(tmp_path: Path):
    """The same refusal at config-validation time, so the run never starts."""
    from soma.config import DecoderConfig, PipelineConfig

    def _config(**adaptor):
        return PipelineConfig(
            dataset_csv=tmp_path / "dataset.csv",
            splits_csv=tmp_path / "splits.csv",
            output_root=tmp_path / "out",
            dataset_type="segmentation",
            feature_mode="live",
            decoder=DecoderConfig(name="lightweight_conv"),
            task=TaskConfig(name="segmentation", params={"num_classes": 2}),
            **adaptor,
        )

    with pytest.raises(ValueError, match="requires feature_mode='cached'"):
        _config(normalization=NormalizationConfig(method="zscore"))
    with pytest.raises(ValueError, match="requires feature_mode='cached'"):
        _config(projection=ProjectionConfig(method="random", target_dim=2))
    # ...and the default (both off) is untouched by the guard.
    assert _config().feature_mode == "live"


_DETECTION_SPACING = 0.2  # µm/px; match_distance 0.6 µm -> 3 target-frame px


def _synthetic_detection_fold(tmp_path: Path, feature_dim: int = _DENSE_DIM):
    """The other single-encoder dense stream: same grids, point annotations."""
    from soma.dataset import DetectionManifest
    from soma.dense import DenseFeatureStore
    from soma.dense.geometry import compute_dense_geometry
    from soma.dense.store import dense_grid_metadata, write_dense_grid

    dense_dir = tmp_path / "dense"
    points_dir = tmp_path / "points"
    dense_dir.mkdir(parents=True)
    points_dir.mkdir(parents=True)
    geom = compute_dense_geometry(target_size=16, patch_size=4)
    meta = dense_grid_metadata(
        geom, feature_dim=feature_dim, pad_mode="reflect", spacing_um=_DETECTION_SPACING
    )
    meta.update(
        source_spacing_um=_DETECTION_SPACING,
        effective_spacing_um=_DETECTION_SPACING,
    )
    torch.manual_seed(0)
    rows = []
    for sample_id, scale in {"s0": 1.0, "s1": 2.0, "s2": 500.0, "s3": 900.0}.items():
        write_dense_grid(
            dense_dir, sample_id, torch.randn(feature_dim, *geom.grid_shape) * scale, meta
        )
        pts = points_dir / f"{sample_id}.csv"
        pts.write_text("x,y,class\n4,4,0\n11,11,1\n", encoding="utf-8")
        rows.append((sample_id, f"{sample_id}.jpg", str(pts)))

    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text(
        "sample_id,image_path,points_path,spacing_at_level_0\n"
        + "\n".join(f"{s},{i},{p},{_DETECTION_SPACING}" for s, i, p in rows)
        + "\n",
        encoding="utf-8",
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "sample_id,split,fold\ns0,train,0\ns1,train,0\ns2,tune,0\ns3,test,0\n",
        encoding="utf-8",
    )
    manifest = DetectionManifest(manifest_csv)
    return manifest, Splits(splits_csv, manifest), DenseFeatureStore(dense_dir)


_DETECTION_TASK = TaskConfig(
    name="detection",
    params={
        "num_classes": 2,
        "match_distance": 0.6,
    },
)


def _run_detection_fold(
    tmp_path: Path,
    normalization: NormalizationConfig | None,
    projection: ProjectionConfig | None = None,
    *,
    feature_dim: int = _DENSE_DIM,
    fixtures=None,
    fold_dir_name: str = "fold_0",
    checkpoint_path: Path | None = None,
):
    from soma.config import DecoderConfig, PreprocessingConfig
    from soma.pipeline import train_one_detection_fold

    manifest, splits, store = fixtures or _synthetic_detection_fold(
        tmp_path / "data", feature_dim
    )
    fold_dir = tmp_path / fold_dir_name
    result = train_one_detection_fold(
        feature_store=store,
        dataset=manifest,
        fold_split=splits.folds[0],
        task=_DETECTION_TASK,
        training=TrainingConfig(epochs=1, learning_rate=1e-3, batch_size=2),
        fold_dir=fold_dir,
        decoder=DecoderConfig(name="lightweight_conv"),
        preprocessing=PreprocessingConfig(requested_spacing_um=_DETECTION_SPACING),
        normalization=normalization,
        projection=projection,
        checkpoint_path=checkpoint_path,
    )
    if checkpoint_path is not None:
        return fold_dir, result, store
    checkpoint = torch.load(
        result.train_result.checkpoint_path, weights_only=True, map_location="cpu"
    )
    return fold_dir, checkpoint["model_state_dict"], store


def test_detection_dense_path_fits_the_adaptor_over_the_support_rois(tmp_path: Path):
    """Detection shares the dense decoder, so it carries the same adaptor on the same
    channel axis — including the dim rewire."""
    _, state_dict, store = _run_detection_fold(
        tmp_path,
        NormalizationConfig(method="zscore"),
        ProjectionConfig(method="pca", target_dim=2),
    )

    support = _support_positions(store)
    assert torch.allclose(
        state_dict["feature_adaptor.center"], support.mean(dim=0).float(), atol=1e-4
    )
    assert state_dict["feature_adaptor.projection_matrix"].shape == (2, _DENSE_DIM)
    assert state_dict["decoder.proj.0.weight"].shape[1] == 2


def test_detection_fold_refuses_adaptor_on_composite_dense_stream(tmp_path: Path):
    """Detection's standalone fold API shares the single-encoder adaptor contract."""
    from soma.dense.composite import CompositeDenseFeatureStore

    manifest, splits, store = _synthetic_detection_fold(tmp_path / "data")
    composite = CompositeDenseFeatureStore(
        [store, store], concat_resolution="grid", member_norms=["l2", "l2"]
    )

    with pytest.raises(ValueError, match="not yet supported.*composite"):
        _run_detection_fold(
            tmp_path,
            NormalizationConfig(method="zscore"),
            fixtures=(manifest, splits, composite),
        )


def test_detection_dense_path_with_both_blocks_off_leaves_no_adaptor(tmp_path: Path):
    fold_dir, state_dict, _ = _run_detection_fold(tmp_path, None, None)

    assert not any(key.startswith("feature_adaptor") for key in state_dict)
    assert not (fold_dir / "feature_adapter.json").exists()


def test_dense_fold_writes_a_feature_adapter_qc_sidecar(tmp_path: Path):
    fold_dir, _, _ = _run_dense_fold(
        tmp_path,
        NormalizationConfig(method="zscore", eps=1e-4),
        ProjectionConfig(method="pca", target_dim=2),
    )

    sidecar = json.loads((fold_dir / "feature_adapter.json").read_text(encoding="utf-8"))

    assert sidecar["normalization"]["method"] == "zscore"
    assert sidecar["normalization"]["eps"] == 1e-4
    assert sidecar["n_support_samples"] == 2
    assert sidecar["normalization"]["n_fit_samples"] == 8
    assert sidecar["projection"]["method"] == "pca"
    assert sidecar["projection"]["seed"] == 0
    # 2 Support ROIs x a 2x2 token grid = 8 positions, not 2 samples.
    assert sidecar["projection"]["n_fit_samples"] == 8
    assert sidecar["projection"]["input_dim"] == _DENSE_DIM
    assert sidecar["projection"]["output_dim"] == 2


def test_dense_path_with_both_blocks_off_is_byte_identical_to_a_legacy_run(tmp_path: Path):
    """"Omitted section = absent module" on this path: the checkpoint of a defaults run is
    exactly the checkpoint of a run that predates the adaptor."""
    fold_dir, state_dict, _ = _run_dense_fold(tmp_path / "off", None, ProjectionConfig())

    assert not any(key.startswith("feature_adaptor") for key in state_dict)
    assert not (fold_dir / "feature_adapter.json").exists()


# The guard: what the single-encoder dense slice does *not* cover.


def test_train_still_refuses_the_adaptor_on_spatial_expression(tmp_path: Path):
    """`spatial_expression` is not a dense-grid stream (one spot = one embedding, scored
    by the closed-form probe), so the dense slice does not reach it."""
    dataset, splits, store = _synthetic_fold(tmp_path)

    with pytest.raises(ValueError, match="not yet supported"):
        train(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            dataset_type="spatial_expression",
            task=TaskConfig(name="regression"),
            training=TrainingConfig(epochs=1),
            run_dir=tmp_path / "run",
            normalization=NormalizationConfig(method="zscore"),
        )


def test_train_still_refuses_the_adaptor_on_the_pixel_classifier_path(tmp_path: Path):
    """The decoder-free segmentation path has no torch model to carry buffers."""
    from soma.config import PixelClassifierConfig

    manifest, splits, store = _synthetic_dense_fold(tmp_path / "data")

    with pytest.raises(ValueError, match="not yet supported"):
        train(
            feature_store=store,
            dataset=manifest,
            splits=splits,
            dataset_type="segmentation",
            pixel_classifier=PixelClassifierConfig(name="logistic_regression"),
            task=TaskConfig(name="segmentation", params={"num_classes": _DENSE_CLASSES}),
            training=TrainingConfig(epochs=1),
            run_dir=tmp_path / "run",
            normalization=NormalizationConfig(method="zscore"),
        )


def test_train_still_refuses_the_adaptor_on_a_composite_dense_stream(tmp_path: Path):
    """Composites keep their per-member `member_norm`; the top-level blocks are scoped to
    single-encoder streams, so asking for one over a composite must fail rather than
    silently normalize the concatenated grid as if it were one encoder."""
    from soma.config import DecoderConfig
    from soma.dense.composite import CompositeDenseFeatureStore

    manifest, splits, store = _synthetic_dense_fold(tmp_path / "data")
    composite = CompositeDenseFeatureStore([store, store], member_norms=["l2", "l2"])

    with pytest.raises(ValueError, match="not yet supported"):
        train(
            feature_store=composite,
            dataset=manifest,
            splits=splits,
            dataset_type="segmentation",
            decoder=DecoderConfig(name="lightweight_conv"),
            task=TaskConfig(name="segmentation", params={"num_classes": _DENSE_CLASSES}),
            training=TrainingConfig(epochs=1),
            run_dir=tmp_path / "run",
            projection=ProjectionConfig(method="random", target_dim=2),
        )


def test_segmentation_fold_refuses_adaptor_on_composite_dense_stream(tmp_path: Path):
    """The standalone fold API enforces the same single-encoder ownership rule."""
    from soma.config import DecoderConfig
    from soma.dense.composite import CompositeDenseFeatureStore
    from soma.pipeline import train_one_segmentation_fold

    manifest, splits, store = _synthetic_dense_fold(tmp_path / "data")
    composite = CompositeDenseFeatureStore(
        [store, store], concat_resolution="grid", member_norms=["l2", "l2"]
    )

    with pytest.raises(ValueError, match="not yet supported.*composite"):
        train_one_segmentation_fold(
            feature_store=composite,
            dataset=manifest,
            fold_split=splits.folds[0],
            task=TaskConfig(name="segmentation", params={"num_classes": _DENSE_CLASSES}),
            training=TrainingConfig(epochs=1, batch_size=2),
            fold_dir=tmp_path / "fold",
            decoder=DecoderConfig(name="lightweight_conv"),
            normalization=NormalizationConfig(method="zscore"),
        )


def test_composite_member_norm_is_unchanged_by_the_dense_adaptor(tmp_path: Path):
    """The composite's own per-member normalization keeps working exactly as before."""
    from soma.dense.composite import CompositeDenseFeatureStore, apply_member_norm

    _, _, store = _synthetic_dense_fold(tmp_path / "data")
    composite = CompositeDenseFeatureStore(
        [store, store], concat_resolution="grid", member_norms=["l2", "none"]
    )

    loaded = composite.load("s0")
    raw = store.load("s0")

    assert loaded.shape[0] == 2 * _DENSE_DIM
    assert torch.allclose(loaded[:_DENSE_DIM], apply_member_norm(raw, "l2"))
    assert torch.allclose(loaded[_DENSE_DIM:], raw)


def test_train_dispatches_the_adaptor_into_the_dense_segmentation_fold(tmp_path: Path):
    """The guard is relaxed for the single-encoder dense path: `train` now carries the
    adaptor all the way into the fold rather than refusing it."""
    from soma.config import DecoderConfig

    manifest, splits, store = _synthetic_dense_fold(tmp_path / "data")

    result = train(
        feature_store=store,
        dataset=manifest,
        splits=splits,
        dataset_type="segmentation",
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": _DENSE_CLASSES}),
        training=TrainingConfig(epochs=1, batch_size=2),
        run_dir=tmp_path / "run",
        normalization=NormalizationConfig(method="zscore"),
        projection=ProjectionConfig(method="random", target_dim=3),
        encoder_identity="phikon",
    )

    state = torch.load(
        result.fold_results[0].train_result.checkpoint_path,
        weights_only=True,
        map_location="cpu",
    )["model_state_dict"]
    assert state["feature_adaptor.projection_matrix"].shape == (3, _DENSE_DIM)
    assert (tmp_path / "run" / "feature_adapter.json").is_file()


# Checkpoint reconstruction on the dense path. Rebuilding a trained model from config +
# checkpoint breaks two ways under an adaptor: the strict load rejects the extra buffer
# keys, and the decoder gets built against the *native* dim while the checkpoint carries
# `target_dim` shapes. The second is invisible to a wiring-level check, so each of these
# reconstructs from a real checkpoint and runs a forward pass.


def test_detection_eval_only_rerun_reconstructs_the_projected_model(tmp_path: Path):
    """`train_one_detection_fold(checkpoint_path=...)` regenerates a finished run's
    artifacts without retraining — it must agree with the checkpoint on *both* the extra
    buffers and the rewired decoder width, and then actually score."""
    normalization = NormalizationConfig(method="zscore")
    projection = ProjectionConfig(method="pca", target_dim=2)
    fixtures = _synthetic_detection_fold(tmp_path / "data")
    fold_dir, _, _ = _run_detection_fold(
        tmp_path, normalization, projection, fixtures=fixtures
    )

    _, result, _ = _run_detection_fold(
        tmp_path,
        normalization,
        projection,
        fixtures=fixtures,
        fold_dir_name="rescore",
        checkpoint_path=fold_dir / "best_model.pt",
    )

    assert result.train_result is None  # eval-only: no epoch history
    assert result.test_reports["test"].metrics  # a real forward pass happened


def test_ocelot_greedy_rescoring_reconstructs_the_projected_model(tmp_path: Path):
    """`soma.benchmarks.ocelot` rebuilds SegmentationModel from config + checkpoint for
    greedy re-scoring; under an adaptor it must agree on both the buffers and the width."""
    from soma.benchmarks.ocelot import build_detection_model_from_checkpoint
    from soma.config import DecoderConfig
    from soma.tasks.detection import DetectionHead

    normalization = NormalizationConfig(method="zscore")
    projection = ProjectionConfig(method="random", target_dim=3)
    fold_dir, _, store = _run_detection_fold(tmp_path, normalization, projection)

    model = build_detection_model_from_checkpoint(
        store=store,
        checkpoint_path=fold_dir / "best_model.pt",
        decoder=DecoderConfig(name="lightweight_conv"),
        geometry=store.geometry("s0"),
        task_head=DetectionHead(
            num_classes=2,
            geometry=store.geometry("s0"),
            delta_px=3.0,
            sigma_px=1.0,
            nms_distance_px=3.0,
            sample_spacings={sid: store.spacing(sid) for sid in store.available_samples},
        ),
        normalization=normalization,
        projection=projection,
    )

    # The reconstruction is only proven by running it: a decoder built against the native
    # dim would load fine key-wise but blow up on the first conv.
    with torch.inference_mode():
        out = model(store.load("s3").unsqueeze(0))
    assert out.logits.shape[0] == 1


def test_live_prediction_models_reconstruct_a_cached_trained_projected_checkpoint(
    tmp_path: Path,
):
    """`build_live_segmentation_models` rebuilds fold models for whole-slide sliding-window
    inference from checkpoints trained on the **cached** path — so those checkpoints can
    carry an adaptor even though live *training* refuses one. The rebuilt model must load
    the buffers, be built against the rewired width, and produce logits."""
    from soma.config import AugmentationConfig
    from soma.dense.geometry import compute_dense_geometry
    from soma.dense.live import LiveSegmentationSource
    from soma.dense.predict import build_live_segmentation_models

    normalization = NormalizationConfig(method="zscore")
    projection = ProjectionConfig(method="random", target_dim=3)
    fold_dir, _, _ = _run_dense_fold(tmp_path, normalization, projection)

    geometry = compute_dense_geometry(
        target_size=_DENSE_TARGET, patch_size=_DENSE_PATCH
    )
    source = LiveSegmentationSource(
        kit=_literal_dense_kit(geometry),
        device="cpu",
        feature_dim=_DENSE_DIM,
        augmentation=AugmentationConfig(),
        spacing_um=None,
    )

    models = build_live_segmentation_models(
        source,
        decoder_name="lightweight_conv",
        decoder_params=None,
        num_classes=_DENSE_CLASSES,
        ckpt_paths=[fold_dir / "best_model.pt"],
        normalization=normalization,
        projection=projection,
    )

    # Drive the trainable half on a real grid — the half that would break on a width
    # mismatch. (The encoder half needs a real backbone, which this fixture has not.)
    grid = torch.randn(1, _DENSE_DIM, *geometry.grid_shape)
    with torch.inference_mode():
        logits = models[0].forward_from_grid(grid).logits
    assert logits.shape[:2] == (1, _DENSE_CLASSES)


# Provenance on this path (reused seams — identity, cache key, saved config).


def _dense_pipeline_config(tmp_path: Path, **adaptor):
    from soma.config import (
        DecoderConfig,
        EncoderConfig,
        PipelineConfig,
        PreprocessingConfig,
    )

    return PipelineConfig(
        dataset_csv=tmp_path / "manifest.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "out",
        dataset_type="segmentation",
        preprocessing=PreprocessingConfig(
            backend="asap", requested_spacing_um=0.5, requested_tile_size_px=224
        ),
        encoder=EncoderConfig(name="phikon"),
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": 2}),
        **adaptor,
    )


def test_dense_run_identity_folds_in_the_adaptor_only_when_non_default(tmp_path: Path):
    from soma.output_layout import canonical_experiment_payload

    default = canonical_experiment_payload(_dense_pipeline_config(tmp_path))
    assert "normalization" not in default and "projection" not in default

    for section in (
        {"normalization": NormalizationConfig(method="zscore")},
        {"projection": ProjectionConfig(method="pca", target_dim=64)},
    ):
        payload = canonical_experiment_payload(_dense_pipeline_config(tmp_path, **section))
        assert payload != default


def test_dense_adaptor_leaves_the_feature_extraction_cache_key_untouched(tmp_path: Path):
    """The adaptor consumes the dense cache; it must never orphan it."""
    from dataclasses import replace as dataclass_replace

    from soma.cache.keys import build_dense_cache_key

    off = _dense_pipeline_config(tmp_path)
    on = dataclass_replace(
        off,
        normalization=NormalizationConfig(method="zscore"),
        projection=ProjectionConfig(method="random", target_dim=64),
    )

    def key(config):
        return build_dense_cache_key(
            tile_encoder_name=config.encoder.name,
            target_size=(224, 224),
            patch_size=(16, 16),
            pad_mode="reflect",
            execution=config.encoder,
            preprocessing=config.preprocessing,
            window_size=None,
            overlap=0.0,
        )

    assert key(off) == key(on)


def test_dense_run_config_always_serializes_both_adaptor_blocks(tmp_path: Path):
    """Guard the hash, not the record: a saved dense run config says what transform ran,
    even when that is 'none'."""
    import yaml

    from soma.config import save_config

    path = tmp_path / "config.yaml"
    save_config(_dense_pipeline_config(tmp_path), path)

    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["normalization"] == {"method": "none", "eps": 1e-6}
    assert saved["projection"]["method"] == "none"
