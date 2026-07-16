"""Generate the CLI guide page from public soma metadata."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from inspect import signature
from importlib import resources
from pathlib import Path
from textwrap import dedent, indent
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for sibling in (ROOT.parent / "slide2vec", ROOT.parent / "hs2p"):
    if sibling.exists():
        sys.path.insert(0, str(sibling))

from soma.aggregators import aggregator_registry
from soma.benchmarks import (
    expected_rows,
    get_benchmark,
    list_benchmarks,
    load_results,
    reproduction_report,
)
from soma.benchmarks import eva as eva_bench
from soma.benchmarks import ocelot as ocelot_bench
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
from soma.training.probe import DEFAULT_PCA_COMPONENTS, ridge_alpha


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


def _default_config_yaml_block() -> str:
    """Return the bundled default YAML as an indented RST code block body."""
    text = (
        resources.files("soma.configs")
        .joinpath("default.yaml")
        .read_text(encoding="utf-8")
        .strip()
    )
    return indent(text, "   ")


def build_cli_rst() -> str:
    config_reference = _default_config_yaml_block()
    body = (
        dedent(
        """\
        CLI
        ===

        ``soma`` exposes a compact command-line interface for running
        experiments from YAML config files and for listing the available model
        presets.

        .. figure:: /_static/figures/run-flow.svg
           :figclass: soma-figure
           :alt: Three input files flow into one soma command that schedules tiling, feature extraction, training, and metrics.

           You provide three files — a dataset, splits, and a config. ``soma`` then
           schedules every step: tiling, feature extraction, training, and metrics.

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

        ``soma list benchmarks``
           List all registered foundation-model benchmarks — the names
           ``soma reproduce`` and ``soma leaderboard`` accept.

        Benchmarking commands
        ---------------------

        These drive the registered benchmarks (see :doc:`benchmarking` for the
        end-to-end curate → configure → run → leaderboard → reproduce story).

        ``soma reproduce NAME [--raw-root DIR | --curated-dir DIR | --from-run-dir DIR] [--seeds N]``
           Curate → run → score a registered benchmark. When a matching packaged
           reference exists, report its delta and highlight potential drift;
           otherwise, explicitly skip the comparison. Reference comparisons are
           informational and never determine command success. ``NAME`` is a
           registered benchmark (``ocelot``, ``eva/bach``) or a family prefix
           (``eva``) that fans out over every ``eva/<dataset>``. Three manifest
           sources: ``--raw-root`` curates from raw data; ``--curated-dir`` reuses an
           already-curated manifest dir (``dataset.csv`` + ``splits.csv``), skipping
           curation; ``--from-run-dir`` re-scores an existing run without retraining.
           ``--seeds 1`` is the quickest smoke.

        ``soma leaderboard [NAME] --root OUTPUT_ROOT [--vary AXIS] [--fix AXIS=VALUE] [--like DIR]``
           Render a faceted leaderboard over the completed run dirs under an
           output root. A benchmark ``NAME`` supplies the canonical facet and
           reference band; ``--vary`` / ``--fix`` / ``--like`` shape the facet on
           top of it.

        What the CLI expects
        --------------------

        The config file follows the canonical nested schema below. This block
        is generated from ``soma/configs/default.yaml``, the bundled defaults
        merged by :func:`soma.config.load_config`. Copy it when you want the
        baseline public YAML shape, then replace neutral defaults such as
        ``encoder: null`` and ``aggregation: null`` for your run.

        Full config reference
        ---------------------

        .. code-block:: yaml

        """
        )
        + config_reference
        + "\n\n"
        + dedent(
            """\
        See also
        --------

        * :doc:`getting-started` – Python API equivalent of each config section
        * :doc:`getting-started` – end-to-end walkthrough

        """
        )
    )
    return body.rstrip() + "\n"


def write_cli_rst(path: str | Path | None = None) -> Path:
    """Write the generated CLI guide page to disk."""

    target = Path(path) if path is not None else Path(__file__).with_name("cli.rst")
    target.write_text(build_cli_rst(), encoding="utf-8")
    return target


# --- benchmark pages (generated from the registered Benchmark definitions) -----------
#
# Each per-benchmark page is generated from its registered ``Benchmark`` object: the
# protocol summary comes from ``facet`` / ``primary_metric`` / ``canonical_seeds`` /
# ``reference_environment``, the reproduce command from ``benchmark.name``, and the
# reference numbers are read from the benchmark's ``expected()`` reference rows (packaged
# ``soma/benchmarks/reference/<name>.csv``) and rendered as a readable ``list-table`` — the
# gate band beside the measured cells, a linked source instead of dumping the CSV's verbose
# ``source`` column (no hand-typed numbers, no drift, no ``TBD``). A generator + a
# checked-in file kept in sync by ``tests/test_docs.py`` mirrors the ``cli.rst`` mechanism.

_GENERATED_PAGE_NOTE = (
    ".. note::\n\n"
    "   This page is generated from the registered benchmark definition — the protocol\n"
    "   summary and reference numbers from the ``Benchmark`` object's ``expected()`` rows\n"
    "   (packaged ``{csv}``), and the command from the benchmark name. Edit the registry\n"
    "   (``{module}``) and the CSV, not this page; ``python docs/_generate_reference.py``\n"
    "   re-emits it and ``tests/test_docs.py`` guards the two from drifting."
)

# Upstream provenance for each benchmark family's published reference band — the leaderboard
# the numbers were captured from, rendered as one clickable link next to the reference
# (replaces dumping the CSV's verbose ``source`` column on the page). A family absent here
# has a self-referential reference (e.g. a soma seed-0 regression gate), so there is no
# external source to link.
_REFERENCE_SOURCE = {
    "eva": (
        "kaiko-ai/eva pathology leaderboard",
        "https://github.com/kaiko-ai/eva/blob/main/tools/data/leaderboards/pathology.csv",
    ),
    "hest": (
        "HEST-Benchmark leaderboard (mahmoodlab/HEST)",
        "https://github.com/mahmoodlab/HEST#hest-benchmark",
    ),
}


def _kv_table(col_a: str, col_b: str, rows: list[tuple[str, str]], *, widths: str = "30 70") -> str:
    """A two-column ``list-table`` (header + ``rows``)."""
    lines = [".. list-table::", "   :header-rows: 1", f"   :widths: {widths}", ""]
    lines.extend([f"   * - {col_a}", f"     - {col_b}"])
    for a, b in rows:
        lines.extend([f"   * - {a}", f"     - {b}"])
    return "\n".join(lines)


def _reference_source_link(family: str) -> str:
    """A single clickable ``label <url>`` for where a benchmark's reference band came from."""
    label, url = _REFERENCE_SOURCE[family]
    return f"`{label} <{url}>`__"


def _eva_results_section() -> str:
    """Render the public reproduced-versus-reference EVA comparison."""
    rows = load_results("eva")
    if not rows:
        return (
            "No reproduced cells have been recorded yet. Run, for example::\n\n"
            "    soma reproduce eva/bach --encoder virchow2 "
            "--raw-root /path/to/eva/bach --record\n\n"
            "to record a soma score next to the published EVA reference."
        )
    lines = [
        "We benchmarked two encoders: soma closely reproduces\n"
        "EVA's published balanced accuracy scores.\n",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Dataset",
        "     - Encoder",
        "     - soma (mean ± std)",
        "     - EVA reference",
    ]
    relative_differences = []
    for row in rows:
        measured = f"{row.measured:.3f}" + (
            f" ± {row.std:.3f}" if row.std is not None else ""
        )
        gates = [
            g
            for g in expected_rows("eva", metric=row.metric, **row.key)
            if not g.is_external
        ]
        reference = f"{gates[0].expected:.3f}" if gates else "—"
        if gates and gates[0].expected:
            relative_differences.append(
                abs(100 * (row.measured - gates[0].expected) / gates[0].expected)
            )
        lines.extend(
            [
                f"   * - {row.key.get('dataset', '')}",
                f"     - {row.key.get('encoder', '')}",
                f"     - {measured}",
                f"     - {reference}",
            ]
        )

    if not relative_differences:
        return "\n".join(lines)
    relative_differences.sort()
    mid = len(relative_differences) // 2
    median = (
        relative_differences[mid]
        if len(relative_differences) % 2
        else (relative_differences[mid - 1] + relative_differences[mid]) / 2
    )
    return (
        "\n".join(lines)
        + f"\n\nAcross these {len(rows)} recorded dataset–encoder comparisons, the median "
        f"relative difference is **{median:.2f}%**."
    )


def _ocelot_guidance_section(bench) -> str:
    """The non-gating external/guidance anchors as a clickable, clearly-labelled section.

    Generated from the registered benchmark's ``external`` reference rows (issue #226): each
    renders as an anonymous RST hyperlink (``label <url>``__) so the docs page links the
    snapshotted source, framed as context — never a target ``soma reproduce`` gates on.
    """
    external = [r for r in bench.expected() if r.is_external]
    bullets = []
    for row in external:
        note = f" — {row.source}" if row.source else ""
        bullets.append(
            f"* `{row.label} <{row.url}>`__ — ``{row.metric}`` ≈ {row.expected:.2f}{note}"
        )
    return (
        "Guidance anchors (non-gating)\n-----------------------------\n\n"
        "External reference points shown for **context only** — the official challenge\n"
        "baseline and best-reported numbers, snapshotted (not live-scraped) from\n"
        "`histoboard <https://wearewaiv.github.io/histoboard/>`__. They measure a\n"
        "*different* protocol than soma's frozen probe (fully-supervised, end-to-end, not\n"
        "tied to any encoder), so ``soma reproduce`` **never gates** on them; they only show\n"
        "how far the frozen-probe result stands from the best reported result:\n\n"
        + "\n".join(bullets)
    )


def build_ocelot_benchmark_rst() -> str:
    """Generate the OCELOT benchmark page from the registered ``ocelot`` benchmark."""
    bench = get_benchmark("ocelot")
    fixed = bench.facet.fixed
    varied = ", ".join(f"``{v}``" for v in bench.facet.varied)
    seeds = ", ".join(str(s) for s in bench.canonical_seeds)
    anchor = f"``{ocelot_bench.ANCHOR_ENCODER}`` @ {ocelot_bench.ANCHOR_SPACING:g} µm/px"

    protocol_rows = [(f"``{key}``", f"``{value}``") for key, value in fixed.items()]
    protocol_rows += [
        ("varied axes", varied),
        ("primary metric", f"``{bench.primary_metric}``"),
        ("canonical seeds", f"``{seeds}``"),
        ("anchor", f"{anchor} (seed 0)"),
    ]
    axes_rows = [
        (f"``{enc}``", f"{spacing:g}")
        for enc, spacing in sorted(ocelot_bench._CONFIG_FILES)
    ]
    env_rows = [(f"``{key}``", f"``{value}``") for key, value in bench.reference_environment.items()]
    gate_rows = [
        (f"``{row.metric}``", f"{row.expected:.4f} ± {row.tolerance:.3f}")
        for row in bench.expected()
        if not row.is_external
    ]

    sections = [
        "OCELOT\n======",
        "*Maps to task:* :doc:`detection` — soma's :doc:`detection path <detection>`\n"
        "reproduced on the `OCELOT 2023 <https://ocelot2023.grand-challenge.org/>`_\n"
        "cell-detection challenge.",
        _GENERATED_PAGE_NOTE.format(
            csv="soma/benchmarks/reference/ocelot.csv", module="soma/benchmarks/ocelot.py"
        ),
        "OCELOT 2023 provides paired cell + tissue patches from TCGA. This benchmark is\n"
        "**cell-only**: a **frozen** foundation-model encoder produces a dense token grid,\n"
        "a ``lightweight_conv`` decoder regresses a per-class peak heatmap, and the\n"
        ":class:`~soma.tasks.detection.DetectionHead` scores it with OCELOT's class-aware\n"
        "**mean F1 @ δ = 3 µm**, greedy-matched — the leaderboard-comparable operating\n"
        "point (per-class score thresholds swept on ``tune``, frozen, applied once to\n"
        "``test``). See :doc:`detection` for the canonical matcher and px↔µm definitions.",
        "Protocol\n--------\n\n"
        "The recipe backbone is held fixed; ``soma reproduce`` varies only the ``encoder``\n"
        "and fixes image spacing at the anchor.\n\n"
        + _kv_table("Axis / setting", "Value", protocol_rows),
        "Packaged spacing protocols\n--------------------------\n\n"
        "Reproduce fixes spacing at the anchor, but ``build_config`` still resolves a\n"
        "committed protocol per ``(encoder, spacing)`` — the 2×2 magnification-alignment\n"
        "ablation plus the native anchor. Use these for a custom spacing sweep compared on a\n"
        ":doc:`leaderboard <benchmarking>`, like any other non-encoder axis:\n\n"
        + _kv_table("Encoder", "Spacing (µm/px)", axes_rows, widths="50 50"),
        "Reference band\n--------------\n\n"
        "The tolerance band ``soma reproduce`` uses to highlight potential drift — a\n"
        "**config-agnostic** banner\n"
        "(soma's own frozen-probe Virchow2 @ 0.2 µm/px seed-0 headline, used as a regression\n"
        "anchor, not an external leaderboard number). The non-gating external anchors —\n"
        "fully-supervised end-to-end baselines from a *different* protocol — are surfaced\n"
        "with clickable links under *Guidance anchors* below:\n\n"
        + _kv_table("Metric", "Reference band (expected ± tolerance)", gate_rows, widths="40 60"),
        _ocelot_guidance_section(bench),
        "Reference environment\n---------------------\n\n"
        "The recorded anchor environment the reference number was produced in:\n\n"
        + _kv_table("Component", "Version", env_rows, widths="40 60"),
        "Reproduce\n---------\n\n"
        "One command curates the raw data, trains the anchor for the canonical seed,\n"
        "greedy-scores it, and reports ``mean_f1`` beside the band above::\n\n"
        "    soma reproduce ocelot --raw-root /path/to/ocelot\n\n"
        "Fast paths: ``--from-run-dir <dir>`` re-scores an existing run with the greedy\n"
        "matcher (no training); ``--seeds 1`` is the quickest smoke. Compare encoders with\n"
        "``--encoder`` (e.g. ``soma reproduce ocelot --encoder uni2 --raw-root ...``); to\n"
        "compare spacings, run per-spacing configs and a :doc:`leaderboard <benchmarking>`.",
        ".. seealso::\n\n"
        "   * :doc:`detection` — the detection modeling substrate (head, target encoding,\n"
        "     loss, F1@δ evaluator).\n"
        "   * :doc:`benchmarking` — the shared curate → run → leaderboard → reproduce guide.\n"
        "   * :doc:`curation` — the OCELOT curator and split policy.",
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def build_eva_benchmark_rst() -> str:
    """Generate the EVA benchmark page from the registered ``eva/<dataset>`` family."""
    family = [get_benchmark(n) for n in list_benchmarks() if n.startswith("eva/")]
    head = family[0]  # protocol constants are shared across the family
    seeds = ", ".join(str(s) for s in head.canonical_seeds)
    protocol_rows = [
        ("head", "linear probe (``aggregation: null`` — each patch is its own bag)"),
        (
            "optimizer",
            f"AdamW, lr ``{eva_bench.LEARNING_RATE:g}``, "
            f"weight_decay ``{eva_bench.WEIGHT_DECAY:g}``",
        ),
        ("batch size", f"``{eva_bench.HEAD_BATCH_SIZE}``"),
        ("budget", f"eva's ``max_steps={eva_bench.MAX_STEPS}`` mapped to soma epochs"),
        ("metric", "``balanced_accuracy``"),
        ("varied axis", "``encoder``"),
        ("primary metric", f"``{head.primary_metric}`` (from ``summary.json``)"),
        ("canonical seeds", f"``{seeds}`` (averaged)"),
    ]

    dataset_names = [str(benchmark.facet.fixed["dataset"]) for benchmark in family]
    dataset_list = ", ".join(dataset_names[:-1]) + f", and {dataset_names[-1]}"
    raw_layout_rows = [
        (
            "`BACH <https://zenodo.org/records/3632035>`__ (``bach``)",
            "``ICIAR2018_BACH_Challenge/Photos/<class>/*.tif``",
        ),
        (
            "`BreaKHis <https://web.inf.ufpr.br/vri/databases/"
            "breast-cancer-histopathological-database-breakhis/>`__ (``breakhis``)",
            "``BreaKHis_v1/histology_slides/…/40X/*.png``; soma selects EVA classes",
        ),
        (
            "`CRC <https://zenodo.org/records/1214456>`__ (``crc``)",
            "``NCT-CRC-HE-100K/`` and ``CRC-VAL-HE-7K/``",
        ),
        (
            "`Gleason Arvaniti <https://dataverse.harvard.edu/dataset.xhtml?"
            "persistentId=doi:10.7910/DVN/OCYCMP>`__ (``gleason_arvaniti``)",
            "the ``ZT{76_39,111_4,199_1,204_6}*.tar.gz`` TMA archives and "
            "``Gleason_masks_train.tar.gz``",
        ),
        (
            "`MHIST <https://bmirds.github.io/MHIST/#accessing-dataset>`__ (``mhist``)",
            "``images/*.png`` and ``annotations.csv``",
        ),
        (
            "`PatchCamelyon <https://zenodo.org/records/2546921>`__ "
            "(``patch_camelyon``)",
            "the six ``camelyonpatch_level_2_split_{train,valid,test}_{x,y}.h5`` files",
        ),
    ]

    sections = [
        "EVA\n===",
        "Reproduce the `kaiko-ai/eva <https://github.com/kaiko-ai/eva>`_\n"
        "patch-classification leaderboard with frozen tile encoders and linear\n"
        ":doc:`classification` heads.\n\n"
        "EVA provides 6 registered datasets: "
        + dataset_list
        + ". All share the same linear-probe protocol.\n\n"
        "**Pipeline:** labelled patches → frozen encoder → linear head → balanced accuracy",
        "Prepare the data\n----------------\n\n"
        "soma does not download benchmark data. Download one dataset from its official\n"
        "source and unpack it in the directory you will pass as ``--raw-root``:\n\n"
        + _kv_table("Dataset and source", "Raw-root contents", raw_layout_rows, widths="38 62")
        + "\n\nFor example, prepare BACH from its public archive::\n\n"
        "    mkdir -p /path/to/eva/bach\n"
        "    curl -L 'https://zenodo.org/records/3632035/files/"
        "ICIAR2018_BACH_Challenge.zip?download=1' -o /tmp/bach.zip\n"
        "    unzip /tmp/bach.zip -d /path/to/eva/bach",
        "Run the benchmark\n-----------------\n\n"
        "Pick any tile-level :doc:`encoder <encoders>` supported by soma and pass the\n"
        "downloaded dataset directory as ``--raw-root``. ``soma reproduce`` runs the\n"
        "built-in EVA curator automatically, writes the manifests under\n"
        "``<raw-root>/curated``, extracts features, trains the linear probe, and reports\n"
        "balanced accuracy. No separate curation command is required. For example::\n\n"
        "    soma reproduce eva/bach --encoder virchow2 --raw-root /path/to/eva/bach\n\n"
        "Or run EVA's 6 datasets in one go::\n\n"
        "    soma reproduce eva --encoder virchow2 --raw-root /path/to/eva",
        "Results\n-------\n\n"
        + _eva_results_section()
        + "\n\nSee the "
        + _reference_source_link("eva")
        + " for the official reference leaderboard.",
        "Protocol details\n----------------\n\n"
        + _kv_table("Setting", "Value", protocol_rows),
        "See :doc:`benchmarking` for the shared benchmark workflow and :doc:`classification`\n"
        "for task-head details.",
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def _hest_results_section() -> str:
    """Render the public reproduced-versus-reference HEST comparison."""
    report = reproduction_report("hest")
    if not report.cells:
        return (
            "No reproduced cells have been recorded yet. Run, for example::\n\n"
            "    soma reproduce hest/IDC --encoder uni2 "
            "--raw-root /path/to/hest-bench --record\n\n"
            "to record a soma score next to the published HEST reference."
        )

    lines = [
        "We benchmarked three encoders: soma closely reproduces\n"
        "HEST's published Pearson scores.\n",
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 24 28 24 24",
        "",
        "   * - Task",
        "     - Encoder",
        "     - soma",
        "     - HEST reference",
    ]
    for cell in report.cells:
        lines.extend(
            [
                f"   * - {cell.dataset}",
                f"     - ``{cell.encoder}``",
                f"     - {cell.measured:.4f}",
                f"     - {cell.reference:.4f}",
            ]
        )

    rels = sorted(
        abs(100 * cell.delta / cell.reference)
        for cell in report.cells
        if cell.reference
    )
    if not rels:
        return "\n".join(lines)

    mid = len(rels) // 2
    median_rel = rels[mid] if len(rels) % 2 else (rels[mid - 1] + rels[mid]) / 2
    summary = (
        f"Across these {len(report.cells)} recorded task–encoder comparisons, the median "
        f"relative difference is **{median_rel:.2f}%**."
    )
    return "\n".join(lines) + "\n\n" + summary


def build_hest_benchmark_rst() -> str:
    """Generate the HEST benchmark page from the registered ``hest/<task>`` family.

    Family-aware (like EVA): renders every registered ``hest/<task>`` sub-benchmark, so a
    registered task appears automatically. The page documents the shared protocol, scoped
    data download, reproduction commands, and recorded results.
    """
    family = [get_benchmark(n) for n in list_benchmarks() if n.startswith("hest/")]
    head = family[0]  # protocol constants are shared across the family
    seeds = ", ".join(str(s) for s in head.canonical_seeds)
    alpha = ridge_alpha(50)  # 50 highly-variable genes per HEST task

    def _section(title: str, body: str) -> str:
        return f"{title}\n{'-' * len(title)}\n\n{body}"

    protocol_rows = [
        ("head", "closed-form Ridge probe — no trained head, no gradient loop"),
        (
            "features",
            "``StandardScaler`` → ``PCA(n_components="
            + f"{DEFAULT_PCA_COMPONENTS})`` fit on the fold's train spots (X only)",
        ),
        (
            "estimator",
            "``Ridge(solver='lsqr', fit_intercept=False)``, penalty "
            + f"``alpha = {alpha:g}`` = 100 / ({DEFAULT_PCA_COMPONENTS}·50)",
        ),
        ("targets", "50-gene ``log1p(counts)`` vector per 112 µm spot (baked by the curator)"),
        (
            "metric",
            "``pearson`` — per gene, pooled over test spots → mean over 50 genes → mean over folds",
        ),
        ("task family", f"``{head.facet.fixed['task']}``"),
        ("varied axis", "``encoder``"),
        ("primary metric", f"``{head.primary_metric}`` (from ``summary.json``)"),
        ("canonical seeds", f"``{seeds}`` (the probe is closed-form — one seed suffices)"),
    ]

    task_names = [str(benchmark.facet.fixed["dataset"]) for benchmark in family]
    task_list = ", ".join(task_names[:-1]) + f", and {task_names[-1]}"

    download_cmd = (
        "    hf download MahmoodLab/hest-bench --include 'IDC/*' --exclude 'fm_v1/*' \\\n"
        "        --repo-type dataset --local-dir /path/to/hest-bench"
    )

    sections = [
        "HEST\n====",
        "Predict a 50-gene expression vector from each 112 µm tile with a frozen encoder,\n"
            "reproducing `HEST-Benchmark <https://github.com/mahmoodlab/HEST>`_ (Jaume et al.,\n"
            "NeurIPS 2024).\n\n"
            "HEST provides 9 registered datasets: "
            + task_list
            + ".\nAll share the same closed-form "
            ":doc:`spatial-expression probe <regression>` protocol.\n\n"
            "**Pipeline:** spot tiles → frozen encoder → Ridge+PCA probe → mean Pearson",
        _section(
            "Prepare the data",
            "Install soma with the optional HEST readers::\n\n"
            "    pip install 'soma-pathology[hest]'\n\n"
            "Use the Hugging Face CLI to download one task while excluding HEST's\n"
            "precomputed ``fm_v1`` features; soma re-extracts them locally::\n\n"
            + download_cmd
            + "\n\nThe ``hf`` CLI downloads the data. Omit ``--include`` to download\n"
            "every registered task under the same local root.",
        ),
        _section(
            "Run the benchmark",
            "Pick any tile-level :doc:`encoder <encoders>` supported by soma and pass the\n"
            "downloaded task directory as ``--raw-root``. ``soma reproduce`` runs the\n"
            "built-in HEST curator automatically, writes the manifests under\n"
            "``<raw-root>/curated``, preserves HEST's fold assignments, extracts features,\n"
            "runs the Ridge probe, and reports the mean Pearson score. No separate curation\n"
            "command is required. Some model weights require ``hf auth login``. For example::\n\n"
            "    soma reproduce hest/IDC --encoder virchow2 "
            "--raw-root /path/to/hest-bench/IDC\n\n"
            "Or run HEST's 9 datasets in one go::\n\n"
            "    soma reproduce hest --encoder virchow2 --raw-root /path/to/hest-bench",
        ),
        _section(
            "Results",
            _hest_results_section()
            + "\n\nSee the "
            + _reference_source_link("hest")
            + " for the official reference leaderboard.",
        ),
        _section("Protocol details", _kv_table("Setting", "Value", protocol_rows)),
        "See :doc:`benchmarking` for the shared benchmark workflow and :doc:`regression`\n"
        "for the probe and metric.",
    ]
    return "\n\n".join(sections).rstrip() + "\n"


def write_benchmark_rst(directory: str | Path | None = None) -> list[Path]:
    """Write the generated per-benchmark pages to disk."""
    base = Path(directory) if directory is not None else Path(__file__).parent
    written = []
    for filename, builder in (
        ("ocelot-detection-benchmark.rst", build_ocelot_benchmark_rst),
        ("eva-patch-classification-benchmark.rst", build_eva_benchmark_rst),
        ("hest-gene-expression-benchmark.rst", build_hest_benchmark_rst),
    ):
        target = base / filename
        target.write_text(builder(), encoding="utf-8")
        written.append(target)
    return written


def main() -> None:
    write_cli_rst()
    write_benchmark_rst()


if __name__ == "__main__":
    main()
