"""The import rule (design §2, ADR 0022), as the import-linter contracts in pyproject.toml.

CI runs ``lint-imports`` before the suite. These tests keep the contracts' module lists in
step with the package: a new protocol, private module or core namespace must be named.
The tests are a second root package: like the benchmark package, they import no private
module (contract 5). Neither imports a private core name either, which import-linter cannot
see (ADR 0022, ADR 0023).
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

import soma

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(soma.__file__).resolve().parent
TESTS = ROOT / "tests"


def _contracts() -> dict[str, dict]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["importlinter"]
    assert config["root_packages"] == ["soma", "tests"]
    return {contract["id"]: contract for contract in config["contracts"]}


def _module(path: Path) -> str:
    parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _private_modules() -> set[str]:
    return {
        _module(path)
        for path in PACKAGE.rglob("*")
        if path.name.startswith("_") and not path.name.startswith("__") and (path.suffix == ".py" or path.is_dir())
    }


def test_there_are_five_contracts():
    assert sorted(_contracts()) == [
        "benchmark-public-api",
        "layers",
        "protocols-independent",
        "tests-public-api",
        "verbs-config-free",
    ]


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
    private = _private_modules()

    assert contract["type"] == "forbidden"
    assert contract["source_modules"] == ["soma.benchmark"]
    assert set(contract["forbidden_modules"]) == private
    assert "soma._internal" in private
    assert contract["allow_indirect_imports"] is True


def test_the_tests_import_no_private_module():
    contract = _contracts()["tests-public-api"]

    assert contract["type"] == "forbidden"
    assert contract["source_modules"] == ["tests"]
    assert set(contract["forbidden_modules"]) == _private_modules()
    assert contract["allow_indirect_imports"] is True


def _is_private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _private_imports(path: Path) -> list[str]:
    """Each import in ``path`` of a ``_``-prefixed module or name of soma or the tests.

    A relative import counts as one inside ``tests``: every test directory is a package.
    """
    roots = {"soma", "tests"}
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names if alias.name.split(".")[0] in roots]
            found += [name for name in names if any(_is_private(part) for part in name.split("."))]
        elif isinstance(node, ast.ImportFrom) and (node.level > 0 or (node.module or "").split(".")[0] in roots):
            module = node.module or ""
            private_module = any(_is_private(part) for part in module.split("."))
            prefix = "." * node.level + (f"{module}." if module else "")
            found += [f"{prefix}{alias.name}" for alias in node.names if private_module or _is_private(alias.name)]
    return found


def test_a_relative_import_of_a_private_name_is_caught(tmp_path):
    # The test directories are packages, so ``from .x import _y`` reaches another test file.
    probe = tmp_path / "probe.py"
    probe.write_text(
        "from .helper import _x\nfrom . import _y\nfrom ._helper import public\nfrom .helper import public\n",
        encoding="utf-8",
    )

    assert _private_imports(probe) == [".helper._x", "._y", "._helper.public"]


def test_no_test_imports_a_private_name():
    # Contract 5 sees modules only; ``from soma.x import _y`` goes through a public module.
    # A new test goes through a public verb or a conformance suite (ADR 0023).
    offenders = {
        str(path.relative_to(ROOT)): names
        for path in sorted(TESTS.rglob("*.py"))
        if "fixtures" not in path.relative_to(TESTS).parts and (names := _private_imports(path))
    }

    assert offenders == {}


def _private_core_names(source: str, *, package: str) -> list[str]:
    """Each ``_``-prefixed name ``source`` imports from a core module: a module of
    ``soma`` outside ``soma.benchmark``. Relative imports resolve against ``package``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = package.split(".")[: len(package.split(".")) - node.level + 1]
            module = ".".join([*base, *([node.module] if node.module else [])])
        else:
            module = node.module or ""
        parts = module.split(".")
        if parts[0] == "soma" and parts[:2] != ["soma", "benchmark"]:
            found += [f"{module}.{alias.name}" for alias in node.names if _is_private(alias.name)]
    return found


def test_a_private_core_name_imported_by_the_benchmark_package_is_caught():
    source = (
        "from soma.evaluation.metrics import _x, public\n"
        "from ...evaluation import _y\n"
        "from soma.benchmark.spec import _z\n"
        "from .froc import _w\n"
        "from soma import __version__\n"
    )

    assert _private_core_names(source, package="soma.benchmark.monkey") == [
        "soma.evaluation.metrics._x",
        "soma.evaluation._y",
    ]


def test_the_benchmark_package_imports_no_private_core_name():
    # Contract 3 sees modules only; ``from soma.x import _y`` goes through a public
    # module. A core gap the benchmark needs gets a public spelling (ADR 0022); the
    # benchmark package's own private names are its own.
    offenders = {}
    for path in sorted((PACKAGE / "benchmark").rglob("*.py")):
        package = _module(path) if path.name == "__init__.py" else _module(path).rpartition(".")[0]
        if names := _private_core_names(path.read_text(encoding="utf-8"), package=package):
            offenders[str(path.relative_to(ROOT))] = names

    assert offenders == {}


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
