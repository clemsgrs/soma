"""The import rule (design §2, ADR 0022), as the import-linter contracts in pyproject.toml.

CI runs ``lint-imports`` before the suite. These tests keep the contracts' module lists in
step with the package: a new protocol, private module or core namespace must be named.
"""

from __future__ import annotations

import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

import soma

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path(soma.__file__).resolve().parent


def _contracts() -> dict[str, dict]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["importlinter"]
    assert config["root_package"] == "soma"
    return {contract["id"]: contract for contract in config["contracts"]}


def _module(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def test_there_are_four_contracts():
    assert sorted(_contracts()) == ["benchmark-public-api", "layers", "protocols-independent", "verbs-config-free"]


def test_the_layers_put_the_cli_over_the_benchmark_over_every_other_module():
    contract = _contracts()["layers"]
    top_level = {path.stem if path.suffix else path.name for path in PACKAGE.iterdir() if path.suffix == ".py" or (path / "__init__.py").is_file()}

    assert contract["type"] == "layers"
    assert contract["containers"] == ["soma"]
    assert contract["exhaustive"] is True
    cli, benchmark, rest = contract["layers"]
    assert (cli, benchmark) == ("cli", "benchmark")
    assert set(rest.split(" : ")) == top_level - {"cli", "benchmark", "__init__", "__main__"}


def test_the_protocol_packages_are_independent():
    contract = _contracts()["protocols-independent"]
    protocols = {_module(path) for path in (PACKAGE / "benchmark").glob("*/__init__.py")}

    assert contract["type"] == "independence"
    assert set(contract["modules"]) == protocols
    assert len(protocols) == 6


def test_the_benchmark_package_imports_no_private_module():
    contract = _contracts()["benchmark-public-api"]
    private = {
        _module(path)
        for path in PACKAGE.rglob("*")
        if path.name.startswith("_") and not path.name.startswith("__") and (path.suffix == ".py" or path.is_dir())
    }

    assert contract["type"] == "forbidden"
    assert contract["source_modules"] == ["soma.benchmark"]
    assert set(contract["forbidden_modules"]) == private
    assert "soma._internal" in private
    assert contract["allow_indirect_imports"] is True


def test_no_core_namespace_but_the_pipeline_and_the_presets_imports_config_or_plan():
    contract = _contracts()["verbs-config-free"]
    namespaces = {f"soma.{path.name}" for path in PACKAGE.iterdir() if (path / "__init__.py").is_file()}

    assert contract["type"] == "forbidden"
    # The pipeline consumes a config; the presets build one; the benchmark sits above core.
    assert set(contract["source_modules"]) == (namespaces | {"soma.registry", "soma.discovery"}) - {
        "soma.pipeline",
        "soma.presets",
        "soma.benchmark",
    }
    assert set(contract["forbidden_modules"]) == {"soma.config", "soma.plan"}
    assert contract["allow_indirect_imports"] is True


@pytest.mark.skipif(shutil.which("lint-imports") is None, reason="import-linter is a dev dependency")
def test_the_contracts_hold():
    result = subprocess.run(["lint-imports", "--no-cache"], cwd=ROOT, capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
