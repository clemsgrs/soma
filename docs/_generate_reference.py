"""Generate the compact Sphinx reference page from public soma metadata."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from inspect import signature
from pathlib import Path
from textwrap import dedent
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for sibling in (ROOT.parent / "slide2vec", ROOT.parent / "hs2p"):
    if sibling.exists():
        sys.path.insert(0, str(sibling))

from soma.aggregators import aggregator_registry
from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    HeatmapConfig,
    PipelineConfig,
    PreprocessingConfig,
    PreviewConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.dataset import Dataset, Splits
from soma.extraction import FeatureExtractor
from soma.features import FeatureStore
from soma.pipeline import Pipeline, train
from soma.tasks import task_registry
from soma.tile_extraction import TileFeatureExtractor


def _field_names(cls: type) -> str:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    return ", ".join(f"``{field.name}``" for field in fields(cls))


def _constructor_knobs(cls: type) -> str:
    params = [
        f"``{param.name}``"
        for param in signature(cls.__init__).parameters.values()
        if param.name != "self"
    ]
    return ", ".join(params)


def _list_table(rows: list[tuple[str, str, str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Name",
            "     - Class",
            "     - Constructor knobs",
            "     - Notes",
        ]
    )
    for name, cls_name, knobs, notes in rows:
        lines.extend(
            [
                f"   * - ``{name}``",
                f"     - ``{cls_name}``",
                f"     - {knobs}",
                f"     - {notes}",
            ]
        )
    return "\n".join(lines)


def _api_table(rows: list[tuple[str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Symbol",
            "     - Description",
        ]
    )
    for symbol, desc in rows:
        lines.extend(
            [
                f"   * - ``{symbol}``",
                f"     - {desc}",
            ]
        )
    return "\n".join(lines)


def _config_table(rows: list[tuple[str, str, str]]) -> str:
    lines = [".. list-table::", "   :header-rows: 1", ""]
    lines.extend(
        [
            "   * - Config",
            "     - Main fields",
            "     - Purpose",
        ]
    )
    for name, fields_text, purpose in rows:
        lines.extend(
            [
                f"   * - ``{name}``",
                f"     - {fields_text}",
                f"     - {purpose}",
            ]
        )
    return "\n".join(lines)


def build_reference_rst() -> str:
    """Return the full compact reference page as reStructuredText."""

    config_rows = [
        ("PreprocessingConfig", _field_names(PreprocessingConfig), "Whole-slide segmentation and tiling geometry"),
        ("ExecutionConfig", _field_names(ExecutionConfig), "Runtime execution settings"),
        ("PreviewConfig", _field_names(PreviewConfig), "Preview rendering settings"),
        ("EncoderConfig", _field_names(EncoderConfig), "Foundation-model encoder selection and runtime behavior"),
        ("CacheConfig", _field_names(CacheConfig), "Shared cache policy"),
        ("AggregatorConfig", _field_names(AggregatorConfig), "MIL bag aggregation"),
        ("TaskConfig", _field_names(TaskConfig), "Task head selection"),
        ("EvalConfig", _field_names(EvalConfig), "Metric and subgroup reporting"),
        ("TrainingConfig", _field_names(TrainingConfig), "Training hyperparameters"),
        ("HeatmapConfig", _field_names(HeatmapConfig), "Attention heatmap rendering"),
        ("PipelineConfig", _field_names(PipelineConfig), "Complete experiment specification"),
    ]

    aggregator_notes = {
        "mean_pool": "Baseline pooling; ``input_dim`` is injected from the feature store",
        "max_pool": "Baseline pooling; ``input_dim`` is injected from the feature store",
        "abmil": "Attention MIL; ``input_dim`` is injected from the feature store",
        "clam_sb": "General-purpose CLAM; ``input_dim`` is injected from the feature store",
        "clam_mb": "Multi-branch classification only; ``input_dim`` is injected from the feature store",
        "dsmil": "Dual-stream MIL; ``input_dim`` is injected from the feature store",
        "dtfdmil": "Two-stage distillation MIL; ``input_dim`` is injected from the feature store",
        "hipt": "Hierarchical aggregation; ``input_dim`` is injected from the feature store",
        "transmil": "Transformer MIL; ``input_dim`` is injected from the feature store",
    }

    task_notes = {
        "binary_classification": "Requires exactly two classes",
        "multiclass_classification": "Two or more classes; use binary_classification when the problem is strictly binary",
        "branch_aware_classification": "CLAM-MB compatible branch head",
        "ordinal_classification": "Ordered integer labels",
        "regression": "Continuous targets",
    }

    aggregator_rows = []
    for name in aggregator_registry.list():
        cls = aggregator_registry.get(name)
        aggregator_rows.append(
            (
                name,
                cls.__name__,
                _constructor_knobs(cls),
                aggregator_notes.get(name, cls.__doc__.splitlines()[0] if cls.__doc__ else ""),
            )
        )

    task_rows = []
    for name in task_registry.list():
        cls = task_registry.get(name)
        task_rows.append(
            (
                name,
                cls.__name__,
                _constructor_knobs(cls),
                task_notes.get(name, cls.__doc__.splitlines()[0] if cls.__doc__ else ""),
            )
        )

    api_rows = [
        ("Pipeline", "Orchestrates the full pipeline: extract → train all folds → summarize"),
        ("train", "Train and evaluate all folds, then summarize"),
        ("list_models", "List available encoder presets, optionally filtered by level"),
        ("list_aggregators", "List registered aggregator presets"),
        ("list_task_heads", "List registered task-head presets"),
        ("Dataset", "Load and validate dataset.csv"),
        ("Splits", "Load and validate splits.csv"),
        ("FeatureExtractor", "Delegates tile/slide feature extraction to slide2vec"),
        ("FeatureStore", "Index and load precomputed tile embeddings from disk"),
        ("TileFeatureExtractor", "Encode individual tile images into 1D feature vectors"),
    ]

    body = dedent(
        """\
        Compact Parameter Reference
        ===========================

        This page is generated from the public config dataclasses and component
        registries. It provides a compact index of the main public surfaces.

        Public API
        ----------

        """
    )
    body += _api_table(api_rows)
    body += "\n\nConfiguration dataclasses\n-------------------------\n\n"
    body += _config_table(config_rows)
    body += "\n\nAggregator registry\n-------------------\n\n"
    body += _list_table(aggregator_rows)
    body += "\n\nTask registry\n-------------\n\n"
    body += _list_table(task_rows)
    body += "\n\nUse this page as a concise index. Use the guide pages for workflow and the\ndocstrings for the exact API contract.\n"
    return body


def write_reference_rst(path: str | Path | None = None) -> Path:
    """Write the generated reference page to disk."""

    target = Path(path) if path is not None else Path(__file__).with_name("reference.rst")
    target.write_text(build_reference_rst(), encoding="utf-8")
    return target


def main() -> None:
    write_reference_rst()


if __name__ == "__main__":
    main()
