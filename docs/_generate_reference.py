"""Generate the CLI guide page from public soma metadata."""

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


def _field_names(cls: type, *, exclude: set[str] | None = None) -> str:
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    excluded = exclude or set()
    return ", ".join(f"``{field.name}``" for field in fields(cls) if field.name not in excluded)


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


def build_cli_rst() -> str:
    body = dedent(
        """\
        CLI Guide
        =========

        ``soma`` exposes a compact command-line interface for running
        experiments from YAML config files and for listing the available model
        presets.

        Basic usage
        -----------

        The main entrypoint takes a config path directly::

            soma /path/to/config.yaml

        You can also invoke it through Python if you prefer::

            python -m soma /path/to/config.yaml

        Available commands
        ------------------

        ``soma CONFIG``
           Run a full pipeline from the given YAML config file.

        ``soma list encoders [--level {tile,slide,patient}]``
           List all registered encoder presets. ``--level`` narrows results to
           ``tile``, ``slide``, or ``patient`` encoders.

        ``soma list aggregators``
           List all registered MIL aggregator presets.

        ``soma list decoders``
           List all registered dense decoder presets.

        ``soma list pixel-classifiers``
           List all registered per-pixel classifier presets.

        ``soma list tasks``
           List all registered task-head presets.

        What the CLI expects
        --------------------

        The config file follows the canonical nested schema below. Every key is
        optional except those marked *required*. Omit a section entirely to
        accept all its defaults.

        Full config reference
        ---------------------

        .. code-block:: yaml

           # ── Run ──────────────────────────────────────────────────────────
           run:
             output_root: runs          # required – directory for run artifacts
             seed: 0
             tags:
               - baseline               # free-form labels stored in metadata

           # ── Data ─────────────────────────────────────────────────────────
           data:
             dataset_csv: data/dataset.csv   # required – slide list and labels
             splits_csv: data/splits.csv     # required – train/tune/test folds
             dataset_type: slide             # required – slide | tile | patient

           # ── Preprocessing ────────────────────────────────────────────────
           preprocessing:
             backend: auto                   # auto | hs2p | sam2
             requested_spacing_um: null      # primary scale knob (µm/px)
             requested_tile_size_px: null    # tile edge length at the target spacing
             requested_region_size_px: null  # HIPT region size (hierarchical only)
             region_tile_multiple: null      # tiles-per-region (hierarchical only)
             # Tissue segmentation method. Options: sam2 | hsv | otsu | threshold.
             # Leave empty/unused when pre-computed tissue masks are provided.
             tissue_method: hsv
             min_coverage:                   # tissue coverage threshold (min tissue fraction per tile)
               tissue: 0.1
             overlap: 0.0
             seg_downsample: 64
             sam2_device: cpu
             sam2_num_workers: null
             tolerance: 0.05
             ref_tile_size_px: null
             a_t: 4
             tissue_mask_tissue_value: 1
             preview:
               save_mask_preview: true
               save_tiling_preview: true
               downsample: 32
               tissue_contour_color: [37, 94, 59]

           # ── Cache ────────────────────────────────────────────────────────
           cache:
             enabled: true
             root_dir: null              # null → inside output_root
             reuse_policy: strict        # strict | relaxed
             fingerprint_files: false    # hash slide/mask contents for cache identity
             validate_payloads: false    # load cached tensors to verify shape/dim

           # ── Encoder ──────────────────────────────────────────────────────
           encoder:
             name: uni2                  # required – see `soma list encoders`
             batch_size: 32
             adaptive_batching: false
             output_variant: null        # preset-specific feature variant
             allow_non_recommended_settings: false
             save_tile_features: false

           # ── Aggregation (slide dataset_type only) ────────────────────────
           aggregation:
             name: abmil                 # see `soma list aggregators`
             params:
               hidden_dim: 256
               dropout: 0.25

           # ── Task ─────────────────────────────────────────────────────────
           task:
             name: binary_classification  # required – see `soma list tasks`
             params: {}

           # ── Evaluation ───────────────────────────────────────────────────
           evaluation:
             metrics:
               - auroc
               - balanced_accuracy
             subgroups:
               columns: []              # dataset.csv columns for metric breakdowns

           # ── Training ─────────────────────────────────────────────────────
           training:
             epochs: 50
             learning_rate: 1.0e-4
             weight_decay: 1.0e-5
             optimizer: adam            # adam | sgd | adamw
             scheduler: cosine          # cosine | step | none
             patience: 10               # early-stopping patience (epochs)
             monitor: tune_loss         # tune_loss or a tune metric name
             monitor_mode: min          # min | max
             batch_size: 1
             gradient_accumulation: 1
             tune_is_test: false
             allow_missing_tune: false

           # ── Reports ──────────────────────────────────────────────────────
           reports:
             heatmaps:
               enabled: false
               cmap: coolwarm
               alpha: 0.5
               blur_sigma: 0.0

        See also
        --------

        * :doc:`pipeline` – Python API equivalent of each config section
        * :doc:`getting-started` – end-to-end walkthrough

        """
    )
    return body


def write_cli_rst(path: str | Path | None = None) -> Path:
    """Write the generated CLI guide page to disk."""

    target = Path(path) if path is not None else Path(__file__).with_name("cli.rst")
    target.write_text(build_cli_rst(), encoding="utf-8")
    return target


def main() -> None:
    write_cli_rst()


if __name__ == "__main__":
    main()
