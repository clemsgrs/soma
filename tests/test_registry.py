"""Tests for soma.registry — generic component registry."""

import sys
import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from soma.registry import Registry


# -- params at registration (ADR 0018) ------------------------------------------------


@dataclass
class ScaleParams:
    factor: float = 1.0
    clip: bool = False

    def __post_init__(self) -> None:
        if self.factor <= 0:
            raise ValueError(f"factor must be positive, got {self.factor}")


class Scale:
    def __init__(self, *, factor: float = 1.0, clip: bool = False) -> None:
        self.factor, self.clip = factor, clip


def test_a_registered_class_validates_its_params():
    reg = Registry("widgets")
    reg.register("scale", Scale, params=ScaleParams)

    assert reg.validate("scale", {"factor": 2.0}) == {"factor": 2.0}
    with pytest.raises(ValueError, match="factor must be positive"):
        reg.validate("scale", {"factor": -1.0})
    with pytest.raises(ValueError, match=r"widgets 'scale'.*'gain'.*clip, factor"):
        reg.validate("scale", {"gain": 2.0})


class Unregistered:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


def test_a_dotted_path_resolves_with_one_warning_and_no_validation():
    reg = Registry("widgets")
    reg.register("scale", Scale, params=ScaleParams)
    dotted = "tests.test_registry.Unregistered"

    with pytest.warns(UserWarning, match=r"widgets 'tests\.test_registry\.Unregistered'.*unvalidated") as record:
        assert reg.resolve(dotted) is Unregistered
        assert reg.validate(dotted, {"anything": -1}) == {"anything": -1}
        assert reg.resolve(dotted) is Unregistered
    assert len(record) == 1
    assert dotted not in reg.list()


def test_a_dotted_path_naming_a_registered_class_is_that_entry():
    reg = Registry("widgets")
    reg.register("scale", Scale, params=ScaleParams)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert reg.resolve("tests.test_registry.Scale") is Scale
        with pytest.raises(ValueError, match="factor must be positive"):
            reg.validate("tests.test_registry.Scale", {"factor": -1.0})


def test_infer_returns_the_fields_the_cohort_decides():
    seen = []

    def infer_factor(cohort, params):
        seen.append(dict(params))
        return {"factor": float(len(cohort.records))}

    reg = Registry("widgets")
    reg.register("scale", Scale, params=ScaleParams, infer=infer_factor)
    reg.register("plain", Unregistered, params=ScaleParams)
    cohort = SimpleNamespace(records=["a", "b", "c"])

    assert reg.infer("scale", cohort, {"clip": True}) == {"factor": 3.0}
    assert seen == [{"clip": True}]
    assert reg.infer("plain", cohort, {}) == {}


def test_params_must_be_a_dataclass():
    reg = Registry("widgets")
    with pytest.raises(TypeError, match="dataclass"):
        reg.register("scale", Scale, params=dict)


def test_register_and_retrieve():
    reg = Registry("test")
    reg.register("foo", dict, metadata={"version": 1})

    assert reg.get("foo") is dict
    assert "foo" in reg


def test_register_duplicate_raises():
    reg = Registry("test")
    reg.register("foo", int)

    with pytest.raises(ValueError, match="already registered"):
        reg.register("foo", str)


def test_get_unknown_raises():
    reg = Registry("test")

    with pytest.raises(KeyError, match="not found"):
        reg.get("nonexistent")


def test_list_returns_all_names():
    reg = Registry("test")
    reg.register("alpha", int)
    reg.register("beta", str)

    assert sorted(reg.list()) == ["alpha", "beta"]


def test_list_empty_registry():
    reg = Registry("test")
    assert reg.list() == []


def test_metadata_retrieval():
    reg = Registry("test")
    reg.register("foo", int, metadata={"dim": 512, "paper": "arxiv:1234"})

    info = reg.info("foo")
    assert info["name"] == "foo"
    assert info["class"] == "builtins.int"
    assert info["params"] is None  # registered without params
    assert info["dim"] == 512
    assert info["paper"] == "arxiv:1234"


def test_info_reports_the_params_defaults():
    reg = Registry("test")
    reg.register("scale", Scale, params=ScaleParams)

    assert reg.info("scale")["params"] == {"factor": 1.0, "clip": False}


def test_info_unknown_raises():
    reg = Registry("test")

    with pytest.raises(KeyError):
        reg.info("nonexistent")


#: Handed to the bundled module written by ``test_bundled_module_loads_on_first_miss``.
PENDING: list[Registry] = []


def test_bundled_module_loads_on_first_miss(tmp_path, monkeypatch):
    module = tmp_path / "soma_test_bundled_components.py"
    module.write_text(
        "from tests.test_registry import PENDING\n"
        "PENDING[0].register('late', float)\n"
        "assert 'late' in PENDING[0]  # no re-entrant import while registering\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "soma_test_bundled_components", raising=False)
    reg = Registry("test", bundled_module="soma_test_bundled_components")
    PENDING[:] = [reg]
    reg.register("early", int)
    assert reg.get("early") is int
    assert "soma_test_bundled_components" not in sys.modules  # nothing imported on a hit
    assert "late" in reg
    assert reg.get("late") is float
    assert sorted(reg.list()) == ["early", "late"]
    with pytest.raises(KeyError, match="not found"):
        reg.get("missing")  # the bundled module is imported once only


def test_listing_includes_the_bundled_module(tmp_path, monkeypatch):
    module = tmp_path / "soma_test_bundled_listing.py"
    module.write_text("from tests.test_registry import PENDING\nPENDING[0].register('late', float)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "soma_test_bundled_listing", raising=False)
    reg = Registry("test", bundled_module="soma_test_bundled_listing")
    PENDING[:] = [reg]
    reg.register("early", int)

    assert sorted(reg.list()) == ["early", "late"]
