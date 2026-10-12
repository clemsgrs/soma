"""``soma.save`` / ``soma.load``: one persisted model format, the bundle (ADR 0016).

Every test goes through the public verbs; models are built with the pipeline's
builders, which record how ``soma.load`` rebuilds them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

import soma
from soma.data import Cohort, GridGeometry, from_arrays
from soma.learn import loaders
from soma.pipeline.build import build_grid_model, build_set_model
from soma.testing import FakeEncoder
from soma.training.model import DenseModel, MILModel

D = 4
BUNDLE_FILES = ["model.json", "preprocessing.json", "provenance.json", "weights.safetensors"]


def _cohort_and_bags():
    ids = [f"s{i}" for i in range(8)]
    labels = [0, 1] * 4
    frame = pd.DataFrame({"sample_id": ids, "label": labels})
    splits = pd.DataFrame({"sample_id": ids, "split": ["train"] * 4 + ["tune"] * 2 + ["test"] * 2})
    cohort = Cohort.from_frames(frame, splits, targets=["label"])
    generator = torch.Generator().manual_seed(0)
    bags = {}
    for sid, label in zip(ids, labels):
        bag = torch.randn(5, D, generator=generator)
        bag[:, 0] += 2.0 if label else -2.0
        bags[sid] = bag
    return cohort, from_arrays(bags)


def _set_model(adaptor=None):
    torch.manual_seed(0)
    model = build_set_model(
        kind="mil",
        feature_dim=D,
        aggregator={"name": "abmil", "params": {"hidden_dim": 8}},
        head={
            "name": "binary_classification",
            "params": {"num_classes": 2, "label_map": {0: 0, 1: 1}, "metrics": ["auroc"]},
        },
        normalization={"method": "zscore"},
        adaptor=adaptor,
    )
    return model


def _fitted(tmp_path: Path):
    cohort, source = _cohort_and_bags()
    from soma.training.feature_adaptor import FeatureAdaptor

    adaptor = FeatureAdaptor(method="zscore", num_features=D)
    adaptor.fit([source.load(sid) for sid in cohort.folds[0].train])
    model = _set_model(adaptor=adaptor)
    fold = loaders(source, cohort, head=model.task_head, batch_size=2)
    result = soma.fit(model, fold.train, fold.tune, max_epochs=2, device="cpu", progress=False)
    return result, fold


def test_a_fit_round_trips_through_its_bundle(tmp_path: Path):
    result, fold = _fitted(tmp_path)
    path = soma.save(
        result,
        tmp_path / "bundle",
        preprocessing={"classes": [0, 1], "metrics": ["auroc"]},
        provenance={"fold": 0, "run_id": "r1"},
    )

    assert path == tmp_path / "bundle"
    assert sorted(p.name for p in path.iterdir()) == BUNDLE_FILES
    bundle = soma.load(path)
    assert isinstance(bundle, soma.ModelBundle) and isinstance(bundle.model, MILModel)
    assert not bundle.model.training
    assert bundle.preprocessing == {"classes": [0, 1], "metrics": ["auroc"]}
    assert bundle.provenance["fold"] == 0 and bundle.provenance["run_id"] == "r1"
    assert bundle.provenance["selected_epoch"] == result.selected_epoch
    assert bundle.provenance["selection_monitor"] == [result.selection_monitor[0], result.selection_monitor[1]]
    assert bundle.provenance["soma_version"] == soma.__version__
    # The fitted adaptor rides in the weights as buffers, so the loaded model scores the
    # same features the fitted one did.
    original = result.learner.model.eval()
    assert torch.equal(bundle.model.feature_adaptor.center, original.feature_adaptor.center)
    loaded = soma.predict(bundle.model, fold.tests["test"])
    assert loaded.frame.equals(soma.predict(original, fold.tests["test"]).frame)


def test_the_bundle_holds_the_full_state_dict_and_a_rebuild_recipe(tmp_path: Path):
    from safetensors.torch import load_file

    result, _ = _fitted(tmp_path)
    path = soma.save(result, tmp_path / "bundle")
    weights = load_file(path / "weights.safetensors")
    assert sorted(weights) == sorted(result.learner.model.state_dict())
    assert "feature_adaptor.center" in weights and "feature_adaptor.fitted" in weights
    recipe = json.loads((path / "model.json").read_text())
    assert recipe["shape"] == "set"
    assert recipe["builder"] == "soma.pipeline.build.build_set_model"
    assert json.loads((path / "preprocessing.json").read_text()) == {}


def test_saving_over_a_bundle_replaces_it_and_leaves_no_staging_behind(tmp_path: Path):
    result, _ = _fitted(tmp_path)
    soma.save(result, tmp_path / "bundle", preprocessing={"version": 1})
    soma.save(result, tmp_path / "bundle", preprocessing={"version": 2})
    assert soma.load(tmp_path / "bundle").preprocessing == {"version": 2}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle"]


def _rewrite_version(path: Path, version: str) -> None:
    provenance = json.loads((path / "provenance.json").read_text())
    provenance["soma_version"] = version
    (path / "provenance.json").write_text(json.dumps(provenance))


def test_a_bundle_from_another_soma_major_version_is_refused(tmp_path: Path):
    result, _ = _fitted(tmp_path)
    path = soma.save(result, tmp_path / "bundle")
    major = int(str(soma.__version__).split(".")[0])
    _rewrite_version(path, f"{major + 1}.0.0")
    with pytest.raises(ValueError, match="major"):
        soma.load(path)


def test_a_bundle_from_another_minor_version_loads_with_a_warning(tmp_path: Path):
    result, _ = _fitted(tmp_path)
    path = soma.save(result, tmp_path / "bundle")
    major = int(str(soma.__version__).split(".")[0])
    _rewrite_version(path, f"{major}.999.0")
    with pytest.warns(UserWarning, match="999"):
        bundle = soma.load(path)
    assert isinstance(bundle.model, MILModel)


def test_a_hand_built_model_saves_but_cannot_be_rebuilt_from_its_bundle(tmp_path: Path):
    from soma.aggregators import ABMIL
    from soma.tasks.classification import BinaryClassificationHead

    cohort, source = _cohort_and_bags()
    aggregator = ABMIL(input_dim=D, hidden_dim=8)
    model = MILModel(aggregator, BinaryClassificationHead(input_dim=aggregator.output_dim, num_classes=2))
    fold = loaders(source, cohort, head=model.task_head, batch_size=2)
    result = soma.fit(model, fold.train, max_epochs=1, device="cpu", progress=False)

    path = soma.save(result, tmp_path / "bundle")
    assert (path / "weights.safetensors").is_file()
    with pytest.raises(ValueError, match="builder"):
        soma.load(path)


def _grid_model(encoder_spec):
    geometry = GridGeometry.from_sizes(target_size=32, patch_size=8)
    torch.manual_seed(0)
    return build_grid_model(
        feature_dim=3,
        decoder={"name": "lightweight_conv", "params": {}},
        head={"name": "segmentation", "params": {"num_classes": 2, "geometry": geometry}},
        encoder=encoder_spec,
        stage=FakeEncoder(target_size=32, patch_size=8),
    )


def test_loading_a_live_bundle_builds_its_frozen_encoder_eagerly_and_shares_it(tmp_path: Path, monkeypatch):
    import soma.extract
    from soma.learn import FitResult

    built: list[dict] = []

    def live_encoder(encoder, **kwargs):
        built.append({"encoder": encoder, **kwargs})
        return FakeEncoder(target_size=kwargs["target_size"], patch_size=8)

    monkeypatch.setattr(soma.extract, "live_encoder", live_encoder)
    spec = {"encoder": "fake", "target_size": 32, "spacing_um": 0.5}
    model = _grid_model(spec)
    learner = soma.learn.DenseLearner(model)
    fit = FitResult(learner, pd.DataFrame(), 0, ("loss", 0.0), None, 0)
    path = soma.save(fit, tmp_path / "bundle")
    assert built == []  # the stage the pipeline passed is not rebuilt at save time

    first, second = soma.load(path), soma.load(path)
    assert built == [spec]  # built at load time, once, before any predict
    assert isinstance(first.model, DenseModel) and first.model.encoder is second.model.encoder
    pixels = torch.randint(0, 255, (1, 3, 32, 32), dtype=torch.uint8)
    with torch.inference_mode():
        assert torch.equal(first.model(pixels=pixels).logits, model.eval()(pixels=pixels).logits)


def test_a_live_bundle_fails_at_load_time_when_its_encoder_cannot_be_built(tmp_path: Path, monkeypatch):
    import soma.extract
    from soma.learn import FitResult

    def gated(encoder, **kwargs):
        raise PermissionError("401: gated weights")

    model = _grid_model({"encoder": "gated", "target_size": 32, "spacing_um": 0.5})
    fit = FitResult(soma.learn.DenseLearner(model), pd.DataFrame(), 0, ("loss", 0.0), None, 0)
    path = soma.save(fit, tmp_path / "bundle")
    monkeypatch.setattr(soma.extract, "live_encoder", gated)
    with pytest.raises(PermissionError, match="gated"):
        soma.load(path)
