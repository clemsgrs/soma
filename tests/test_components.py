"""``soma.components``: the one discovery surface over every registry (ADR 0018)."""

from __future__ import annotations

import dataclasses
import gc
import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import soma
import soma.benchmark  # noqa: F401  (registers the benchmarks kind: option C)
from soma import discovery
from soma._internal.params import is_run_decided
from soma.registry import Registry


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
    "presets",
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


def _fresh_python(script: str) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter importing this checkout's soma first."""
    checkout = str(Path(soma.__file__).resolve().parents[1])
    path = os.pathsep.join(filter(None, [checkout, os.environ.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": path}
    )


def test_a_registry_announces_its_kind_while_it_lives():
    widgets = Registry("widgets")

    assert "widgets" in soma.components()
    assert discovery.registry("widgets") is widgets

    del widgets
    gc.collect()
    assert "widgets" not in soma.components()


def test_a_kind_has_one_registry():
    with pytest.raises(ValueError, match="already"):
        Registry("aggregators")


def test_registries_have_no_lazy_bundled_module():
    assert "bundled_module" not in inspect.signature(Registry).parameters


def test_import_soma_lists_the_core_kinds_and_not_the_benchmarks():
    result = _fresh_python(
        "import sys, soma\n"
        "assert not [m for m in sys.modules if m.startswith('soma.benchmark')], 'core imported soma.benchmark'\n"
        "print(' '.join(soma.components()))\n"
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [kind for kind in DOMAINS if kind != "benchmarks"]


def test_the_benchmarks_kind_needs_import_soma_benchmark():
    result = _fresh_python(
        "import soma\n"
        "try:\n"
        "    soma.components('benchmarks')\n"
        "except ValueError as error:\n"
        "    print(error)\n"
        "import soma.benchmark\n"
        "print('eva/bach' in [entry['name'] for entry in soma.components('benchmarks')])\n"
        "print('benchmarks' in soma.components())\n"
    )

    assert result.returncode == 0, result.stderr
    hint, listed, kind = result.stdout.strip().splitlines()
    assert "unknown component kind 'benchmarks'" in hint
    assert "import soma.benchmark" in hint
    assert (listed, kind) == ("True", "True")


def test_a_benchmark_component_name_needs_the_import_and_a_dotted_path_does_not():
    result = _fresh_python(
        "import sys, soma\n"
        "from soma.models.aggregators import aggregator_registry\n"
        "try:\n"
        "    aggregator_registry.get('eva_abmil')\n"
        "except KeyError as error:\n"
        "    print(error.args[0])\n"
        "cls = aggregator_registry.resolve('soma.benchmark.eva.slide.EvaABMIL')\n"
        "print(cls.__name__, 'eva_abmil' in aggregator_registry)\n"
    )

    assert result.returncode == 0, result.stderr
    hint, resolved = result.stdout.strip().splitlines()
    assert "'eva_abmil' not found in aggregators registry" in hint
    assert "import soma.benchmark" in hint
    assert resolved == "EvaABMIL True"


def test_the_cli_imports_soma_benchmark_at_startup():
    result = _fresh_python("import sys, soma.cli\nprint('soma.benchmark' in sys.modules)\n")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True"


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
    # A field the run decides (a class scheme, a derived size) is the documented exception.
    run_decided = {f.name for f in dataclasses.fields(registry._entries[name].params) if is_run_decided(f)}
    for field, default in defaults.items():
        if field in run_decided:
            continue
        assert field in signature.parameters, f"{kind} '{name}': {field!r} is not a constructor keyword"
        declared = signature.parameters[field].default
        if declared is not inspect.Parameter.empty:
            assert default == declared, f"{kind} '{name}': {field} defaults to {default!r}, not {declared!r}"
    assert registry.validate(name, {}) == {}  # the defaults validate


def test_a_domain_lists_its_entries_with_params_and_defaults():
    abmil = _entry("aggregators", "abmil")

    assert abmil["class"] == "soma.models.aggregators.mil.abmil.ABMIL"
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

    assert composite["class"] == "soma.extract.dense.composite.CompositeDenseFeatureStore"
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


def _plan(tmp_path, **kwargs):
    """``kwargs``' experiment resolved against a two-class cohort of six samples."""
    from soma import resolve
    from soma.plan import load_cohort
    from tests.experiment_configs import experiment

    records, splits = tmp_path / "dataset.csv", tmp_path / "splits.csv"
    ids = [f"s{i}" for i in range(6)]
    records.write_text("sample_id,image_path,label,time,event,bin\n" + "".join(
        f"{sid},/{sid}.tif,{i % 2},{i + 1.0},{i % 2},{i % 2}\n" for i, sid in enumerate(ids)
    ))
    splits.write_text("sample_id,split\n" + "".join(
        f"{sid},{split}\n" for sid, split in zip(ids, ["train"] * 3 + ["tune", "test", "test"])
    ))
    config = experiment(records, splits, **kwargs)
    return resolve(config, load_cohort(config))


def test_a_dotted_path_naming_a_registered_class_is_that_entry(tmp_path):
    """The plan keeps the name as written, so it reloads in a fresh process (importing the
    path registers the class); its params take the entry's defaults, and name-keyed
    routing (``clam_mb``'s branch-aware head) sees the entry's key."""
    from soma.models.aggregators.registry import aggregator_registry
    from soma.pipeline.build import _bag_head_class
    from soma.models.heads import BranchAwareClassificationHead, MulticlassClassificationHead

    dotted = "soma.models.aggregators.mil.clam.CLAM_MB"
    plan = _plan(tmp_path, aggregator=dotted, head="multiclass_classification")

    aggregator = plan.learner.params["aggregator"]
    assert aggregator.name == dotted
    assert aggregator.params == aggregator_registry.canonical_params("clam_mb")
    key = aggregator_registry.key_of(aggregator.name)
    assert key == "clam_mb"
    head_cls = _bag_head_class(MulticlassClassificationHead, task_name="multiclass_classification", aggregator_key=key)
    assert head_cls is BranchAwareClassificationHead


def test_a_dotted_survival_head_selects_its_objective_by_class(tmp_path):
    from soma.models.heads.survival import CoxSurvivalHead, resolve_survival_task

    dotted = "soma.models.heads.survival.CoxSurvivalHead"
    plan = _plan(tmp_path, head=dotted, aggregator="abmil", loaders={"batch_size": 4})

    assert plan.learner.params["head"].name == dotted
    assert resolve_survival_task(dotted) == (CoxSurvivalHead, "cox")


def test_a_dotted_registered_decoder_takes_its_entry_s_defaults():
    from soma.config import Component
    from soma.models.decoders.registry import decoder_registry
    from soma.plan import _expand

    dotted = "soma.models.decoders.lightweight_conv.LightweightConvDecoder"
    leaf = _expand(decoder_registry, Component(dotted))
    assert leaf.name == dotted
    assert leaf.params == decoder_registry.canonical_params("lightweight_conv")
    assert decoder_registry.key_of(dotted) == "lightweight_conv"


def test_an_unregistered_dotted_path_keeps_its_name_and_params_verbatim(aggregators, tmp_path):
    dotted = "tests.test_components.FancyMIL"
    with pytest.warns(UserWarning, match="is not registered"):
        plan = _plan(tmp_path, aggregator={"name": dotted, "params": {"width": 0}})

    assert plan.learner.params["aggregator"].name == dotted
    assert plan.learner.params["aggregator"].params == {"width": 0}


def test_components_is_the_only_discovery_surface():
    import soma.models.aggregators
    import soma.benchmark
    import soma.models.decoders
    import soma.models.encoders
    import soma.models.heads
    from soma.registry import Registry

    for module, name in [
        (soma, "list_models"),
        (soma, "list_aggregators"),
        (soma, "list_decoders"),
        (soma, "list_task_heads"),
        (soma.models.encoders, "list_models"),
        (soma.models.aggregators, "list_aggregators"),
        (soma.models.decoders, "list_decoders"),
        (soma.models.heads, "list_task_heads"),
        (soma.benchmark, "list_benchmarks"),
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
