"""The package layout and the root surface (design §7), and the tests tree that mirrors it (§10).

The root exports the verbs, the nouns they take or return, and the namespaces;
everything else is reached through its namespace.
"""

from __future__ import annotations

import importlib
import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

import soma

VERBS = {
    "extract",
    "fit",
    "predict",
    "evaluate",
    "save",
    "load",
    "seed_everything",
    "Pipeline",
    "resolve",
    "components",
    "register",
}
NOUNS = {
    "Cohort",
    "SampleRecord",
    "FoldSplit",
    "ImageManifest",
    "AnnotationManifest",
    "SetSource",
    "GridSource",
    "TargetSource",
    "ImageSource",
    "FitResult",
    "ModelBundle",
    "Predictions",
    "PointPredictions",
    "LabelMapPredictions",
    "Evaluation",
    "ExperimentConfig",
    "Plan",
    "PipelineResult",
}
NAMESPACES = {"data", "extract", "models", "learn", "evaluation", "pipeline", "presets", "testing"}

ROOT_MODULES = {"__init__.py", "__main__.py", "cli.py", "config.py", "discovery.py", "plan.py", "registry.py"}
#: The benchmark side: one package, imported explicitly (design §2, ADR 0022).
BENCHMARK_PACKAGE = "benchmark"
#: One package per benchmark protocol, each with its own reference rows.
PROTOCOLS = ("eva", "hest", "ocelot", "midog", "monkey", "croma")


def _holds_source(path: Path) -> bool:
    return path.suffix == ".py" or (path.is_dir() and any(path.rglob("*.py")))


def test_the_package_top_level_holds_only_the_namespaces_and_the_root_modules():
    package = Path(soma.__file__).parent

    entries = {path.name for path in package.iterdir() if _holds_source(path)}

    assert entries == ROOT_MODULES | NAMESPACES | {"_internal", BENCHMARK_PACKAGE}


@pytest.mark.parametrize("module", ["soma.benchmarks", "soma.curation", "soma.leaderboard", "soma.robustness"])
def test_the_old_benchmark_modules_are_gone(module):
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize(
    "module",
    [
        # the generic mechanism
        "soma.benchmark.spec",
        "soma.benchmark.registry",
        "soma.benchmark.run",
        "soma.benchmark.reproduction",
        "soma.benchmark.leaderboard",
        "soma.benchmark.curation",
        "soma.benchmark.coverage",
        "soma.benchmark.tiles",
        "soma.benchmark.detection",
        "soma.benchmark.attention",
        # one package per benchmark: curate, recipes, scorer
        "soma.benchmark.eva",
        "soma.benchmark.eva.segmentation",
        "soma.benchmark.eva.slide",
        "soma.benchmark.eva.curate",
        "soma.benchmark.eva.curate_segmentation",
        "soma.benchmark.eva.curate_slide",
        "soma.benchmark.hest",
        "soma.benchmark.hest.probe",
        "soma.benchmark.hest.curate",
        "soma.benchmark.ocelot",
        "soma.benchmark.ocelot.curate",
        "soma.benchmark.midog",
        "soma.benchmark.midog.curate",
        "soma.benchmark.midog.f1",
        "soma.benchmark.monkey",
        "soma.benchmark.monkey.curate",
        "soma.benchmark.monkey.froc",
        "soma.benchmark.croma",
        "soma.benchmark.croma.curate",
        "soma.benchmark.croma.robustness",
    ],
)
def test_each_benchmark_module_lives_in_the_benchmark_package(module):
    assert importlib.util.find_spec(module) is not None


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_each_protocol_ships_its_reference_rows_beside_its_code(protocol):
    package = Path(soma.__file__).parent / BENCHMARK_PACKAGE / protocol

    assert (package / "__init__.py").is_file()
    assert (package / "reference.csv").is_file()


def test_the_benchmark_package_holds_only_the_generic_mechanism_and_the_protocols():
    package = Path(soma.__file__).parent / BENCHMARK_PACKAGE

    entries = {path.name for path in package.iterdir() if _holds_source(path)}

    generic = {"spec", "registry", "run", "reproduction", "leaderboard", "curation", "coverage", "tiles"}
    assert entries == {"__init__.py", "detection.py", "attention.py"} | {f"{name}.py" for name in generic} | set(PROTOCOLS)


@pytest.mark.parametrize(
    "module",
    [
        # extract/: extraction, the caches, the dense stores and readers, preprocessing
        "soma.extract.extractor",
        "soma.extract.facade",
        "soma.extract.rois",
        "soma.extract.contracts",
        "soma.extract.cache.keys",
        "soma.extract.dense.store",
        "soma.extract.dense.reader",
        "soma.extract.dense.sliding",
        "soma.extract.dense.composite",
        "soma.extract.dense_extraction",
        "soma.extract.dense_slide_extraction",
        "soma.extract.tile_extraction",
        "soma.extract.slide2vec_adapter",
        "soma.extract.spacing",
        "soma.extract.preprocessing.supplied_coordinates",
        # models/: aggregators, heads, decoders, encoders, the composed models
        "soma.models.aggregators.mil.abmil",
        "soma.models.heads.classification",
        "soma.models.decoders.lightweight_conv",
        "soma.models.encoders.validation",
        "soma.models.model",
        "soma.models.detection_targets",
        # learn/: the datasets, collate, feature adaptor, seed and survival sampler
        "soma.learn.bag_dataset",
        "soma.learn.sample_dataset",
        "soma.learn.patient_dataset",
        "soma.learn.collate",
        "soma.learn.feature_adaptor",
        "soma.learn.seed",
        "soma.learn.survival_sampler",
        # evaluation/: detection scoring, grid prediction, heatmaps
        "soma.evaluation.detection.peaks",
        "soma.evaluation.detection.matching",
        "soma.evaluation.detection.io",
        "soma.evaluation.sliding_window",
        "soma.evaluation.heatmaps",
        # pipeline/: fold planning
        "soma.pipeline.fold_planning",
        # _internal/: the only private home
        "soma._internal.atomic_io",
        "soma._internal.logging",
        "soma._internal.class_scheme",
        "soma._internal.params",
    ],
)
def test_each_moved_module_lives_in_its_namespace(module):
    assert importlib.util.find_spec(module) is not None


def test_the_root_exports_exactly_the_verbs_nouns_and_namespaces():
    assert len(soma.__all__) == len(set(soma.__all__))
    assert set(soma.__all__) == VERBS | NOUNS | NAMESPACES


def test_the_root_holds_no_other_public_name_than_its_exports_and_modules():
    public = {name for name in vars(soma) if not name.startswith("_")}

    stray = {name for name in public - set(soma.__all__) if not isinstance(getattr(soma, name), ModuleType)}

    assert stray == set()


@pytest.mark.parametrize(
    ("name", "home"),
    [
        ("ExperimentConfig", "soma.config"),
        ("Plan", "soma.plan"),
        ("resolve", "soma.plan"),
        ("components", "soma.discovery"),
        ("register", "soma.discovery"),
        ("Cohort", "soma.data"),
        ("SampleRecord", "soma.data"),
        ("FoldSplit", "soma.data"),
        ("ImageManifest", "soma.data"),
        ("AnnotationManifest", "soma.data"),
        ("SetSource", "soma.data"),
        ("GridSource", "soma.data"),
        ("TargetSource", "soma.data"),
        ("ImageSource", "soma.data"),
        ("fit", "soma.learn"),
        ("save", "soma.learn"),
        ("load", "soma.learn"),
        ("FitResult", "soma.learn"),
        ("ModelBundle", "soma.learn"),
        ("seed_everything", "soma.learn.seed"),
        ("predict", "soma.evaluation"),
        ("evaluate", "soma.evaluation"),
        ("Predictions", "soma.evaluation"),
        ("PointPredictions", "soma.evaluation"),
        ("LabelMapPredictions", "soma.evaluation"),
        ("Evaluation", "soma.evaluation"),
        ("Pipeline", "soma.pipeline"),
        ("PipelineResult", "soma.pipeline"),
    ],
)
def test_a_root_name_is_the_object_its_namespace_holds(name, home):
    assert getattr(soma, name) is getattr(importlib.import_module(home), name)


def test_the_extract_namespace_is_the_verb():
    assert callable(soma.extract)
    assert soma.extract.extract is importlib.import_module("soma.extract.facade").extract


def test_a_star_import_brings_every_export():
    namespace: dict[str, object] = {}
    exec("from soma import *", namespace)

    assert set(namespace) - {"__builtins__"} == VERBS | NOUNS | NAMESPACES
    for name in NAMESPACES:
        assert isinstance(namespace[name], ModuleType)
    assert callable(namespace["extract"])


@pytest.mark.parametrize(
    ("name", "home"),
    [
        ("CacheOptions", "soma.config"),
        ("Component", "soma.config"),
        ("EncoderLeaf", "soma.config"),
        ("CachedSetSource", "soma.data"),
        ("CachedGridSource", "soma.data"),
        ("TileImageSource", "soma.data"),
        ("LabelMapSource", "soma.data"),
        ("PointSource", "soma.data"),
        ("from_arrays", "soma.data"),
        ("FeatureSource", "soma.extract"),
        ("PooledFeatureSource", "soma.extract"),
        ("DenseSourceProvenance", "soma.extract.dense"),
        ("overlay_mask_on_slide", "soma.extract.preprocessing"),
        ("EmbeddingModel", "soma.models"),
        ("MILModel", "soma.models"),
        ("DenseModel", "soma.models"),
        ("ABMIL", "soma.models"),
        ("Trainer", "soma.learn"),
        ("MILLearner", "soma.learn"),
        ("SampleDataset", "soma.learn.sample_dataset"),
        ("Baseline", "soma.evaluation"),
        ("FoldResult", "soma.pipeline"),
        ("evaluate_representation", "soma.pipeline.representation"),
    ],
)
def test_a_name_off_the_root_is_reached_through_its_namespace(name, home):
    assert not hasattr(soma, name)
    assert hasattr(importlib.import_module(home), name)


# The tests tree (design §10, ADR 0023): tests/ mirrors soma/'s namespaces, plus e2e/ for
# the end-to-end scenarios and repo/ for packaging, workflow, docs and release checks.

REPO = Path(__file__).resolve().parents[2]
TESTS = REPO / "tests"
#: Test directories that mirror no namespace.
TEST_ONLY_DIRECTORIES = {"e2e", "repo"}


def _packages(root: Path) -> set[str]:
    return {path.name for path in root.iterdir() if (path / "__init__.py").is_file()}


def test_the_tests_tree_mirrors_the_package_namespaces():
    namespaces = {name for name in _packages(Path(soma.__file__).parent) if not name.startswith("_")}

    assert _packages(TESTS) == namespaces | TEST_ONLY_DIRECTORIES


def test_the_benchmark_tests_hold_one_directory_per_protocol():
    assert _packages(TESTS / "benchmark") == set(PROTOCOLS)


def test_every_test_directory_is_a_package():
    # Packages give each test module a unique dotted name and let import-linter see it.
    directories = {path.parent for path in TESTS.rglob("test_*.py") if "fixtures" not in path.relative_to(TESTS).parts}

    assert [str(d.relative_to(REPO)) for d in sorted(directories) if not (d / "__init__.py").is_file()] == []


def test_project_protocol_tests_run_beside_their_protocol_outside_the_suite():
    # BEETLE tests a project protocol (ADR 0010): ``python -m pytest examples/beetle/tests``.
    testpaths = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["pytest"]["ini_options"][
        "testpaths"
    ]

    assert testpaths == ["tests"]
    assert list(TESTS.rglob("test_beetle*.py")) == []
    assert sorted((REPO / "examples" / "beetle" / "tests").glob("test_*.py"))
