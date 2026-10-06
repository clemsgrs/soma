"""Tests for soma.registry — generic component registry."""

import pytest

from soma.registry import Registry


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
    assert info["dim"] == 512
    assert info["paper"] == "arxiv:1234"


def test_info_unknown_raises():
    reg = Registry("test")

    with pytest.raises(KeyError):
        reg.info("nonexistent")


def test_list_with_metadata():
    reg = Registry("test")
    reg.register("a", int, metadata={"task": "vision"})
    reg.register("b", str, metadata={"task": "text"})
    reg.register("c", float, metadata={"task": "vision"})

    all_info = reg.list_with_metadata()
    assert len(all_info) == 3
    assert all_info[0]["name"] == "a"


def test_decorator_style():
    reg = Registry("test")

    @reg.register_decorator("my_component", metadata={"version": 2})
    class MyComponent:
        pass

    assert reg.get("my_component") is MyComponent
    assert reg.info("my_component")["version"] == 2


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
    monkeypatch.delitem(__import__("sys").modules, "soma_test_bundled_components", raising=False)
    reg = Registry("test", bundled_module="soma_test_bundled_components")
    PENDING[:] = [reg]
    reg.register("early", int)
    assert reg.list() == ["early"]  # nothing imported until a lookup misses
    assert "late" in reg
    assert reg.get("late") is float
    assert sorted(reg.list()) == ["early", "late"]
    with pytest.raises(KeyError, match="not found"):
        reg.get("missing")  # the bundled module is imported once only
