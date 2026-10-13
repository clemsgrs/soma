"""``soma.components``: the one discovery surface over every registry (ADR 0018)."""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import pytest

import soma
from soma import discovery


def _entry(kind: str, name: str) -> dict:
    (entry,) = [entry for entry in soma.components(kind) if entry["name"] == name]
    return entry


DOMAINS = [
    "aggregators",
    "augmentations",
    "benchmarks",
    "callbacks",
    "decoders",
    "encoders",
    "heads",
    "learners",
    "optimizers",
    "samplers",
    "schedulers",
]


def test_without_a_kind_it_names_the_domains():
    assert soma.components() == DOMAINS


@pytest.mark.parametrize("kind", DOMAINS)
def test_every_domain_lists_named_entries(kind):
    entries = soma.components(kind)

    assert entries
    assert [entry["name"] for entry in entries] == sorted(entry["name"] for entry in entries)
    assert all({"name", "class", "params"} <= set(entry) for entry in entries)


def test_an_unknown_kind_names_the_domains():
    with pytest.raises(ValueError, match="aggregators, augmentations"):
        soma.components("widgets")


def _soma_entries():
    for kind in DOMAINS:
        if kind == "encoders":  # slide2vec's, not soma's
            continue
        for entry in soma.components(kind):
            yield pytest.param(kind, entry["name"], id=f"{kind}:{entry['name']}")


@pytest.mark.parametrize(("kind", "name"), list(_soma_entries()))
def test_every_soma_entry_declares_params_mirroring_its_constructor(kind, name):
    registry = discovery.registry(kind)
    target = registry.get(name)
    signature = inspect.signature(target.build_config if kind == "benchmarks" else target)
    defaults = registry.info(name)["params"]

    assert defaults is not None, f"{kind} '{name}' is registered without params"
    for field, default in defaults.items():
        assert field in signature.parameters, f"{kind} '{name}': {field!r} is not a constructor keyword"
        declared = signature.parameters[field].default
        if declared is not inspect.Parameter.empty:
            assert default == declared, f"{kind} '{name}': {field} defaults to {default!r}, not {declared!r}"
    assert registry.validate(name, {}) == {}  # the defaults validate


def test_a_domain_lists_its_entries_with_params_and_defaults():
    abmil = _entry("aggregators", "abmil")

    assert abmil["class"] == "soma.aggregators.mil.abmil.ABMIL"
    assert abmil["params"] == {"hidden_dim": 128, "activation": "tanh", "gated": True, "dropout": 0.25}


def test_encoders_are_slide2vecs_listed_with_their_level_and_params():
    """A slide2vec encoder's params are ``Model.from_preset``'s keywords (ADR 0019)."""
    levels = {entry["name"]: entry["level"] for entry in soma.components("encoders")}

    assert levels["uni2"] == "tile"
    assert levels["prism"] == "slide"
    assert _entry("encoders", "uni2")["params"] == {
        "output_variant": None,
        "allow_non_recommended_settings": False,
    }
    with pytest.raises(TypeError, match="slide2vec"):
        soma.register("encoders", "mine", object)


def test_an_encoder_name_is_its_own_registry_key():
    """``key_of`` holds for the encoders registry, whose entries are slide2vec's."""
    registry = discovery.registry("encoders")

    assert registry.key_of("uni2") == "uni2"
    assert registry.key_of("composite") == "composite"
    assert registry.key_of("not_an_encoder") == "not_an_encoder"
    assert registry.key_of("no_such_module.Encoder") == "no_such_module.Encoder"


def test_the_composite_encoder_is_an_entry_whose_params_list_the_members():
    composite = _entry("encoders", "composite")

    assert composite["class"] == "soma.dense.composite.CompositeDenseFeatureStore"
    assert composite["params"] == {"members": (), "concat_resolution": "grid", "concat_grid_size": None}
    registry = discovery.registry("encoders")
    params = {"members": [{"name": "uni", "dense": {"window_size": 224}}, {"name": "phikon"}]}
    assert registry.validate("composite", params) == params
    with pytest.raises(ValueError, match="at least one member"):
        registry.validate("composite", {"members": []})


# -- the registration ladder: dotted path (zero ceremony) vs soma.register ------------


@dataclass
class FancyMILParams:
    width: int = 64

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"width must be a positive integer, got {self.width}")


class FancyMIL:
    def __init__(self, input_dim: int, *, width: int = 64) -> None:
        self.input_dim, self.width = input_dim, width


@pytest.fixture
def aggregators(monkeypatch):
    """The aggregator registry, restored after the test."""
    registry = discovery.registry("aggregators")
    monkeypatch.setattr(registry, "_entries", dict(registry._entries))
    monkeypatch.setattr(registry, "_warned", set())
    return registry


def test_a_registered_custom_class_is_listed_and_validates(aggregators):
    soma.register("aggregators", "fancy_mil", FancyMIL, params=FancyMILParams)

    assert _entry("aggregators", "fancy_mil") == {
        "name": "fancy_mil",
        "class": "tests.test_components.FancyMIL",
        "params": {"width": 64},
    }
    assert aggregators.validate("fancy_mil", {"width": 8}) == {"width": 8}
    with pytest.raises(ValueError, match="width must be a positive integer"):
        aggregators.validate("fancy_mil", {"width": 0})


def test_a_dotted_path_class_resolves_with_a_warning_unvalidated_and_unlisted(aggregators):
    dotted = "tests.test_components.FancyMIL"

    with pytest.warns(UserWarning, match=r"aggregators 'tests\.test_components\.FancyMIL' is not registered"):
        assert aggregators.resolve(dotted) is FancyMIL
    assert aggregators.validate(dotted, {"width": 0}) == {"width": 0}
    assert dotted not in [entry["name"] for entry in soma.components("aggregators")]


def _config(dataset_type: str = "slide", **components):
    from soma.config import PipelineConfig

    return PipelineConfig(
        dataset_csv="data.csv",
        splits_csv="splits.csv",
        output_root="out",
        dataset_type=dataset_type,
        **components,
    )


def test_config_names_a_registered_class_by_its_key_so_clam_mb_gets_the_branch_aware_head():
    from soma.config import AggregatorConfig, TaskConfig
    from soma.pipeline.build import _bag_head_class
    from soma.tasks import BranchAwareClassificationHead, MulticlassClassificationHead

    config = _config(
        aggregator=AggregatorConfig(name="soma.aggregators.mil.clam.CLAM_MB"),
        task=TaskConfig(name="multiclass_classification"),
    )

    assert config.aggregator.name == "clam_mb"
    head_cls = _bag_head_class(
        MulticlassClassificationHead,
        task_name=config.task.name,
        aggregator_name=config.aggregator.name,
    )
    assert head_cls is BranchAwareClassificationHead


def test_a_dotted_survival_head_honours_the_loss_selector():
    from soma.config import TaskConfig, TrainingConfig
    from soma.tasks.survival import CoxSurvivalHead, resolve_survival_task

    config = _config(
        task=TaskConfig(name="soma.tasks.survival.SurvivalHead", params={"loss": "cox"}),
        training=TrainingConfig(batch_size=4),
    )

    assert config.task.name == "survival"
    assert resolve_survival_task(config.task.name, config.task.params) == (CoxSurvivalHead, "cox")


def test_config_names_a_registered_decoder_by_its_key():
    from soma.config import DecoderConfig, TaskConfig

    config = _config(
        "segmentation",
        decoder=DecoderConfig(name="soma.decoders.lightweight_conv.LightweightConvDecoder"),
        task=TaskConfig(name="segmentation"),
    )

    assert config.decoder.name == "lightweight_conv"


def test_config_keeps_an_unregistered_dotted_path_as_written(aggregators):
    from soma.config import AggregatorConfig, TaskConfig

    dotted = "tests.test_components.FancyMIL"
    with pytest.warns(UserWarning, match="is not registered"):
        config = _config(aggregator=AggregatorConfig(name=dotted), task=TaskConfig(name="binary_classification"))

    assert config.aggregator.name == dotted


def test_components_is_the_only_discovery_surface():
    import soma.aggregators
    import soma.benchmarks
    import soma.decoders
    import soma.encoders
    import soma.tasks
    from soma.registry import Registry

    for module, name in [
        (soma, "list_models"),
        (soma, "list_aggregators"),
        (soma, "list_decoders"),
        (soma, "list_task_heads"),
        (soma.encoders, "list_models"),
        (soma.aggregators, "list_aggregators"),
        (soma.decoders, "list_decoders"),
        (soma.tasks, "list_task_heads"),
        (soma.benchmarks, "list_benchmarks"),
    ]:
        assert not hasattr(module, name), f"{module.__name__}.{name} is still exported"
    assert not hasattr(Registry, "list_with_metadata")


def test_building_a_registered_aggregator_validates_its_params(aggregators):
    from soma.pipeline.build import build_set_model

    soma.register("aggregators", "fancy_mil", FancyMIL, params=FancyMILParams)

    with pytest.raises(ValueError, match="width must be a positive integer"):
        build_set_model(
            kind="mil",
            feature_dim=8,
            head={"name": "binary_classification", "params": {"num_classes": 2}},
            aggregator={"name": "fancy_mil", "params": {"width": 0}},
            adaptor=None,
        )


def test_a_misspelt_optimizer_param_is_named_before_training():
    from soma.learn import optimizer_registry, resolve_leaf

    with pytest.raises(ValueError, match=r"optimizers 'adamw' has no param\(s\) 'learning_rate'"):
        resolve_leaf({"name": "adamw", "params": {"learning_rate": 1e-3}}, optimizer_registry)
