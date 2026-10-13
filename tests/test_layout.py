"""The package layout and the root surface (design §7).

The root exports the verbs, the nouns they take or return, and the namespaces;
everything else is reached through its namespace.
"""

from __future__ import annotations

import importlib
import importlib.util
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
#: The benchmark side, gathered into ``soma.benchmark`` by the next slice.
AWAITING_BENCHMARK_PACKAGE = {"benchmarks", "curation", "leaderboard.py", "robustness.py"}


def _holds_source(path: Path) -> bool:
    return path.suffix == ".py" or (path.is_dir() and any(path.rglob("*.py")))


def test_the_package_top_level_holds_only_the_namespaces_and_the_root_modules():
    package = Path(soma.__file__).parent

    entries = {path.name for path in package.iterdir() if _holds_source(path)}

    assert entries == ROOT_MODULES | NAMESPACES | {"_internal"} | AWAITING_BENCHMARK_PACKAGE


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
