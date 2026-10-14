"""Command-line interface for soma."""

from __future__ import annotations

import argparse
import inspect
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

# The CLI imports the benchmark package at startup (ADR 0022): it registers the benchmarks
# and their components, which `soma list`, `soma reproduce`, `soma leaderboard` and a
# saved benchmark config need. `import soma` alone does not.
import soma.benchmark  # noqa: F401
from soma._internal.logging import configure_cli_logging
from soma.config import ExperimentConfig
from soma.discovery import components
from soma.pipeline import Pipeline

# The reproduce orchestration lives in soma.benchmark.run (issue #370): the CLI is a
# thin caller of the same importable code (`soma.benchmark.run_benchmark`). The aliases
# below remain the CLI's patchable module-level seams — `_reproduce_one` threads them back
# through the shared implementation so replacing e.g. ``soma.cli._provenance`` or
# ``soma.cli.Pipeline`` still takes effect.
from soma.benchmark import run as _reproduce_impl
from soma.benchmark.run import (
    ReportedScoreError,  # noqa: F401  (kept importable from soma.cli)
    _MissingReproduceSourceError,
    _PanelCellRuntimeFailure,
    _git_commit,  # noqa: F401  (kept importable from soma.cli)
    _provenance,
    _record_axes,  # noqa: F401  (kept importable from soma.cli)
    _record_reference_row,  # noqa: F401  (kept importable from soma.cli)
    _record_result,  # noqa: F401  (kept importable from soma.cli)
    _reproduce_manifest,
    _reproduce_output_root,
    _runtime_croma_version,
)


def _cached_source(config: ExperimentConfig, feature_dir: Path):
    """The cached source a ``--feature-dir`` holds: grids for the grid shape, else features."""
    from soma.data import CachedSetSource
    from soma.extract.dense import CachedGridSource, DenseSourceProvenance
    from soma.plan import learner_shape

    if learner_shape(config) == "grid":
        return CachedGridSource(
            feature_dir,
            provenance=DenseSourceProvenance(
                kind="dense_cache", feature_dir=feature_dir, dataset_csv=config.cohort.records
            ),
        )
    return CachedSetSource(feature_dir)


def _load_config(config_path: Path) -> ExperimentConfig:
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    try:
        return ExperimentConfig.from_yaml(config_path)
    except Exception as exc:
        print(f"Error: failed to load config from {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)


def _preset_parameters(name: str) -> list[inspect.Parameter]:
    """The preset ``name``'s arguments, from its signature: soma.register supplies no metadata."""
    from soma.presets import preset_registry

    parameters = inspect.signature(preset_registry.get(name)).parameters.values()
    return [p for p in parameters if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def _selected_preset(argv: Sequence[str], *, positional: bool) -> str | None:
    """The registered preset ``argv`` selects, read before the parser that takes its
    arguments exists: ``soma init``'s positional (its flags may come first, so every
    preset's flags are declared and a flag's value is not taken for the name),
    ``soma run``'s ``--preset NAME``."""
    from soma.presets import preset_registry

    selector = argparse.ArgumentParser(add_help=False, exit_on_error=False)
    if positional:
        selector.add_argument("preset", nargs="?")
        flags = {_flag(p.name): p.name for entry in components("presets") for p in _preset_parameters(entry["name"])}
        for flag, argument in flags.items():
            selector.add_argument(flag, dest=f"preset_{argument}")
    else:
        selector.add_argument("--preset")
    try:
        name = selector.parse_known_args(list(argv))[0].preset
    except argparse.ArgumentError:  # the full parser reports it
        return None
    return name if name is not None and name in preset_registry else None


def _flag(argument: str) -> str:
    return f"--{argument.replace('_', '-')}"


def _add_preset_arguments(parser: argparse.ArgumentParser, name: str | None) -> None:
    """The selected preset's own arguments as flags, an ``int``/``float`` annotation as the
    flag's type; one named like the command's own flag is a usage error."""
    if name is None:
        return
    group = parser.add_argument_group(f"{name} arguments", "The preset's own arguments (soma list presets).")
    for p in _preset_parameters(name):
        kind = next((kind for kind in (int, float) if p.annotation in (kind, kind.__name__)), None)
        try:
            group.add_argument(
                _flag(p.name), dest=f"preset_{p.name}", metavar=p.name.upper(), type=kind, default=None,
                help="Required." if p.default is p.empty else f"Default: {p.default!r}.".replace("%", "%%"),
            )
        except argparse.ArgumentError:
            parser.error(f"preset {name}: argument {p.name} clashes with a {parser.prog} flag")


def _given_preset_arguments(args: argparse.Namespace) -> dict[str, Any]:
    """The preset-argument flags given on the command line, by argument name."""
    return {
        key.removeprefix("preset_"): value for key, value in vars(args).items()
        if key.startswith("preset_") and value is not None
    }


def _build_preset(parser: argparse.ArgumentParser, name: str, args: argparse.Namespace) -> ExperimentConfig:
    """The preset ``name`` called with the flags ``args`` gives, every required one present
    (the parser took only its own arguments)."""
    from soma.presets import preset_registry

    given = _given_preset_arguments(args)
    missing = [p.name for p in _preset_parameters(name) if p.default is p.empty and p.name not in given]
    if missing:
        parser.error(f"preset {name} needs {', '.join(map(_flag, missing))}")
    try:
        return preset_registry.get(name)(**given)
    except (TypeError, ValueError) as exc:
        parser.error(f"preset {name}: {exc}")


def _cmd_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    preset_only = {"--cohort": args.cohort, "--splits": args.splits, "--out": args.out, "--seed": args.seed}
    if args.preset is None:
        if args.config is None:
            parser.error("pass a config path or --preset NAME")
        given = [flag for flag, value in preset_only.items() if value is not None]
        if given:
            parser.error(f"{', '.join(given)} only go with --preset: edit the config instead")
        config = _load_config(Path(args.config))
    else:
        if args.config is not None:
            parser.error("pass a config path or --preset NAME, not both")
        missing = [flag for flag in ("--cohort", "--splits", "--out") if preset_only[flag] is None]
        if missing:
            parser.error(f"--preset needs {', '.join(missing)}")
        config = _build_preset(parser, args.preset, args)
        config.cohort.records = args.cohort
        config.cohort.splits = args.splits
        config.run.output_dir = args.out
        if args.seed is not None:
            config.run.seed = args.seed
    feature_dir = args.feature_dir
    if feature_dir is not None and not feature_dir.is_dir():
        print(f"Error: --feature-dir is not a directory: {feature_dir}", file=sys.stderr)
        sys.exit(1)
    source = _cached_source(config, feature_dir) if feature_dir is not None else None
    Pipeline(config, source=source).run()


def _build_run_parser(argv: Sequence[str] = ()) -> argparse.ArgumentParser:
    """The ``soma run`` parser for ``argv``: with the flags of the preset it selects, if any."""
    parser = argparse.ArgumentParser(
        prog="soma run",
        description=(
            "Run an experiment: a YAML config (soma init writes one; a run's config.yaml or "
            "plan.yaml replays it), or a preset on a cohort. The run directory keeps the "
            "config as given (config.yaml) and its resolved plan (plan.yaml). "
            "soma run --preset NAME --help lists the preset's own arguments."
        ),
    )
    parser.add_argument("config", nargs="?", default=None, help="Experiment YAML.")
    parser.add_argument(
        "--preset", choices=[entry["name"] for entry in components("presets")], default=None,
        help="Build the experiment from this preset instead of a YAML file.",
    )
    parser.add_argument("--cohort", default=None, help="dataset.csv of the samples and their targets (with --preset).")
    parser.add_argument("--splits", default=None, help="splits.csv (with --preset).")
    parser.add_argument("--out", default=None, help="Output directory (with --preset).")
    parser.add_argument("--seed", type=int, default=None, help="Seed (with --preset; default 0).")
    parser.add_argument(
        "--feature-dir", type=Path, default=None,
        help="Train on features already extracted: a pooled feature cache or dense grids directory.",
    )
    _add_preset_arguments(parser, _selected_preset(argv, positional=False))
    parser.set_defaults(func=_cmd_run)
    return parser


def _init_yaml(name: str, args: argparse.Namespace, config: ExperimentConfig) -> str:
    """``config`` with every default spelled out, as an editable YAML document."""
    from soma.plan import expand

    data = expand(config).to_dict()
    flags = " ".join(f"{_flag(argument)} {value}" for argument, value in _given_preset_arguments(args).items())
    header = (
        f"# soma init {name} {flags}".rstrip() + "\n"
        "# Every field at the preset's value or its default. A null field is unset: the run\n"
        "# fills in what the cohort or the machine decides (a segmentation or detection head's\n"
        "# classes and match distance are yours to state). Point cohort.records and\n"
        "# cohort.splits at your files, edit what you need, then: soma run <this file>\n"
    )
    return header + yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def _cmd_init(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    config = _build_preset(parser, args.preset, args)
    try:
        text = _init_yaml(args.preset, args, config)
    except (TypeError, ValueError) as exc:  # the arguments make a config check refuses
        parser.error(f"preset {args.preset}: {exc}")
    sys.stdout.write(text)


def _build_init_parser(argv: Sequence[str] = ()) -> argparse.ArgumentParser:
    """The ``soma init`` parser for ``argv``: with the flags of the preset it names, if any."""
    parser = argparse.ArgumentParser(
        prog="soma init",
        description=(
            "Print a preset as an editable experiment YAML with every default filled in: "
            "soma init NAME [arguments] > exp.yaml, edit, then soma run exp.yaml. "
            "soma init NAME --help lists the preset's own arguments."
        ),
    )
    parser.add_argument("preset", choices=[entry["name"] for entry in components("presets")], help="Preset name.")
    _add_preset_arguments(parser, _selected_preset(argv, positional=True))
    parser.set_defaults(func=_cmd_init)
    return parser


def _print_table(title: str, columns: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=title, show_header=True, header_style="bold", min_width=max(len(title) + 4, 12))
    for index, column in enumerate(columns):
        table.add_column(column, no_wrap=index == 0)
    for row in rows or [("(none)",) + ("",) * (len(columns) - 1)]:
        table.add_row(*row)
    Console().print(table)


def _format_params(params: dict[str, Any] | None) -> str:
    if params is None:
        return ""
    return ", ".join(f"{name}={value!r}" for name, value in params.items())


def _cmd_list(args: argparse.Namespace) -> None:
    """A thin print over :func:`soma.components`."""
    if args.kind is None:
        _print_table("Components", ("Kind",), [(kind,) for kind in components()])
        return
    if args.level is not None and args.kind != "encoders":
        print("Error: --level applies to encoders only", file=sys.stderr)
        sys.exit(2)
    entries = components(args.kind)
    if args.level is not None:
        entries = [entry for entry in entries if entry.get("level") == args.level]
    title = args.kind.capitalize() if args.level is None else f"Encoders ({args.level})"
    if args.kind == "encoders":
        rows = [(entry["name"], str(entry.get("level", ""))) for entry in entries]
        _print_table(title, ("Name", "Level"), rows)
    else:
        rows = [(entry["name"], _format_params(entry["params"])) for entry in entries]
        _print_table(title, ("Name", "Params"), rows)


def _build_list_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma list",
        description="List the component kinds, or one kind's registered components and their params.",
    )
    parser.add_argument(
        "kind",
        nargs="?",
        choices=components(),
        default=None,
        help="Component kind to list; omit it to list the kinds.",
    )
    parser.add_argument(
        "--level",
        choices=("tile", "slide", "patient"),
        default=None,
        help="Restrict encoder listing to one level.",
    )
    parser.set_defaults(func=_cmd_list)
    return parser


def _resolve_reproduce_targets(name: str) -> list[Any]:
    """Benchmarks a ``reproduce NAME`` drives: the single benchmark, or a whole family.

    ``NAME`` may be a directly registered benchmark (``ocelot``, ``eva/bach``) or a family
    prefix (``eva``) that fans out over every registered ``NAME/<member>``. Returns an empty
    list when nothing matches (fail-fast handled by the caller).
    """
    from soma.benchmark import benchmark_registry, get_benchmark

    try:
        return [get_benchmark(name)]
    except KeyError:
        return [get_benchmark(n) for n in sorted(benchmark_registry.list()) if n.startswith(f"{name}/")]


def _recorded_croma_version(
    run_dir: str | Path | None, *, historical_run: bool
) -> str:
    """Croma producer version, threading the CLI's patchable ``_runtime_croma_version``."""
    return _reproduce_impl._recorded_croma_version(
        run_dir, historical_run=historical_run, runtime_version=_runtime_croma_version
    )


def _reproduce_one(
    benchmark,
    args: argparse.Namespace,
    *,
    family_root: str | None = None,
    manifest=None,
    isolate_runtime_failures: bool = False,
) -> int:
    """Curate → run → score one benchmark (thin caller of the importable orchestration).

    Delegates to :func:`soma.benchmark.run._reproduce_one` — the same code behind the
    public :func:`soma.benchmark.run_benchmark` (issue #370) — threading this module's
    collaborators through the seams so the CLI's behavior (and its monkeypatchable
    ``Pipeline`` / ``_provenance`` / ``_reproduce_manifest`` / ``_runtime_croma_version``
    globals) stays byte-identical.
    """
    return _reproduce_impl._reproduce_one(
        benchmark,
        args,
        family_root=family_root,
        manifest=manifest,
        isolate_runtime_failures=isolate_runtime_failures,
        pipeline_cls=Pipeline,
        provenance=_provenance,
        resolve_manifest=_reproduce_manifest,
        recorded_croma_version=_recorded_croma_version,
    )


def _preflight_config_compatibility(
    benchmark, encoder: str, config, capabilities, metadata: dict[str, Any]
) -> str | None:
    """Return one actionable incompatibility derived from a Benchmark config."""
    from slide2vec.encoders import (
        resolve_encoder_output,
    )

    encoder_leaf = config.features.encoder
    if encoder_leaf is None or encoder_leaf.name != capabilities.name:
        resolved = None if encoder_leaf is None else encoder_leaf.name
        return (
            f"Benchmark builder resolved encoder {resolved!r}, expected "
            f"{capabilities.name!r}."
        )

    from soma.plan import learner_shape, sample_unit
    from soma.pipeline.extraction import resolve_pipeline_dense, resolve_pipeline_preprocessing

    dense = resolve_pipeline_dense(config) if learner_shape(config) == "grid" else None
    feature_kind = None if dense is None else dense.feature_kind
    if feature_kind == "patch_features" and not capabilities.dense:
        return (
            f"missing dense capability: Benchmark {benchmark.name!r} requires dense "
            "patch features, but slide2vec reports dense=False. Select a compatible "
            "Benchmark or change the Encoder plugin implementation."
        )
    if feature_kind == "patch_features_prenorm" and not capabilities.dense_prenorm:
        return (
            f"missing pre-norm dense capability: Benchmark {benchmark.name!r} requires the "
            "pre-norm patch grid (feature_kind='patch_features_prenorm'), but slide2vec "
            "reports dense_prenorm=False (timm ViT encoders only). Select a compatible "
            "Benchmark or change the Encoder plugin implementation."
        )
    if feature_kind == "cls_attention" and not capabilities.attention:
        return (
            f"missing attention capability: Benchmark {benchmark.name!r} requires CLS "
            "attention grids, but slide2vec reports attention=False. Select a compatible "
            "Benchmark or change the Encoder plugin implementation."
        )
    if feature_kind is None and not capabilities.pooled:
        return (
            f"missing pooled capability: Benchmark {benchmark.name!r} requires pooled "
            "features, but slide2vec reports pooled=False. Select a compatible Benchmark "
            "or change the Encoder plugin implementation."
        )

    declared_geometry = feature_kind is not None or sample_unit(config) in {"slide", "patient"}
    resolve_encoder_output(
        encoder,
        requested_output_variant=encoder_leaf.output_variant,
        metadata=metadata,
    )
    if not declared_geometry:
        return None

    from slide2vec.api import EncoderInputContract

    try:
        if dense is not None:
            EncoderInputContract.declared_dense(
                encoder,
                target_size_px=int(dense.target_size),
                window_size=dense.window_size,
            )
        else:
            preprocessing = resolve_pipeline_preprocessing(config)
            EncoderInputContract.declared_pooled(
                encoder,
                requested_tile_size_px=int(preprocessing.tiling.requested_tile_size_px),
                allow_non_recommended_settings=encoder_leaf.allow_non_recommended_settings,
            )
    except ValueError as exc:
        return (
            f"fixed encoder geometry is incompatible: {exc} "
            "Select a compatible Benchmark or change the Encoder plugin implementation."
        )
    return None


def _preflight_reproduce_panel(benchmark, encoders: list[str]) -> list[str]:
    """Collect one concrete Benchmark's Encoder incompatibilities before curation."""
    import tempfile

    from slide2vec.encoders import (
        encoder_registry,
        list_encoder_provider_diagnostics,
        resolve_encoder_capabilities,
    )

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="soma-reproduce-preflight-") as temp_dir:
        root = Path(temp_dir)
        dataset_csv = root / "dataset.csv"
        splits_csv = root / "splits.csv"
        dataset_csv.write_text(
            "sample_id,image_path,label\npreflight,/preflight.png,0\n",
            encoding="utf-8",
        )
        splits_csv.write_text(
            "sample_id,split,fold\npreflight,train,0\n",
            encoding="utf-8",
        )
        overrides = {
            "cache": {"enabled": True, "root_dir": str(root / "feature_cache")}
        }
        for encoder in encoders:
            try:
                capabilities = resolve_encoder_capabilities(encoder)
                metadata = encoder_registry.info(encoder)
            except KeyError:
                failure = (
                    f"{encoder!r}: preset is unavailable from slide2vec's Encoder "
                    "registry. Check the preset name and that its provider package is "
                    "installed."
                )
                diagnostics = list_encoder_provider_diagnostics()
                if diagnostics:
                    failure += "\n" + "\n".join(
                        "     Skipped slide2vec Encoder provider "
                        f"{item.provider_key!r} ({item.provider}): "
                        f"{item.exception_type}: {item.message}"
                        for item in diagnostics
                    )
                failures.append(failure)
                continue
            except ValueError as exc:
                failures.append(
                    f"{encoder!r}: invalid slide2vec capability contract: {exc}"
                )
                continue

            try:
                config = benchmark.build_config(
                    encoder=encoder,
                    dataset_csv=dataset_csv,
                    splits_csv=splits_csv,
                    output_root=root / encoder,
                    seed=benchmark.canonical_seeds[0],
                    overrides=overrides,
                )
                incompatibility = _preflight_config_compatibility(
                    benchmark, encoder, config, capabilities, metadata
                )
            except (KeyError, ValueError, TypeError) as exc:
                failures.append(
                    f"{encoder!r}: Benchmark config validation failed: {exc}"
                )
                continue
            if incompatibility is not None:
                failures.append(f"{encoder!r}: {incompatibility}")

    return failures


def _plural_leaderboard_args(benchmark, output_root: Path) -> argparse.Namespace:
    """Request the ordinary canonical Leaderboard with only Encoder varied."""
    return argparse.Namespace(
        name=benchmark.name,
        root=output_root,
        vary=["encoder"],
        fix=None,
        like=None,
        metric=None,
        split=None,
    )


def _panel_runtime_failure_context(exc: BaseException) -> str:
    """Render one runtime failure as a deterministic, single-line diagnostic."""
    detail = " ".join(str(exc).split()) or "(no error message)"
    return f"{type(exc).__name__}: {detail}"


def _completed_run_dirs(output_root: Path) -> set[Path]:
    """Return completed Run directories currently visible to a Leaderboard scan."""
    from soma.benchmark.leaderboard import discover_triples

    return {
        run_dir
        for run_dirs in discover_triples(output_root).values()
        for run_dir in run_dirs
    }


def _run_reproduce_panel(
    benchmark,
    args: argparse.Namespace,
    encoders: list[str],
    *,
    family_root: str | None = None,
) -> tuple[list[tuple[str, str]], int]:
    """Run and project one preflighted concrete Benchmark's Encoder panel."""
    try:
        manifest = _reproduce_manifest(benchmark, args, family_root=family_root)
    except _MissingReproduceSourceError as exc:
        print(f"Error: {benchmark.name}: {exc}", file=sys.stderr)
        return [], 2

    output_root = _reproduce_output_root(benchmark, args, family_root=family_root)
    completed_before = _completed_run_dirs(output_root)
    failures: list[tuple[str, str]] = []
    fully_completed_encoders = 0
    for encoder in encoders:
        cell_args = argparse.Namespace(**vars(args))
        cell_args.encoder = encoder
        cell_args.encoders = None
        try:
            code = _reproduce_one(
                benchmark,
                cell_args,
                family_root=family_root,
                manifest=manifest,
                isolate_runtime_failures=True,
            )
        except _PanelCellRuntimeFailure as exc:
            failures.append((encoder, _panel_runtime_failure_context(exc.cause)))
            continue
        if code:
            return failures, code
        fully_completed_encoders += 1

    leaderboard_code = 0
    completed_during_panel = _completed_run_dirs(output_root) - completed_before
    benchmark_label = (
        f" for Benchmark {benchmark.name!r}" if family_root is not None else ""
    )
    if fully_completed_encoders or completed_during_panel:
        if failures:
            completed_run_count = len(completed_during_panel)
            run_label = "Run" if completed_run_count == 1 else "Runs"
            print(
                f"PARTIAL Encoder panel{benchmark_label}: "
                f"{fully_completed_encoders}/{len(encoders)} encoders completed; "
                f"rendering the canonical Leaderboard from {completed_run_count} "
                f"completed {run_label}. Completed Runs remain valid.",
                flush=True,
            )
        leaderboard_code = _cmd_leaderboard(
            _plural_leaderboard_args(benchmark, output_root)
        )
    elif failures:
        print(
            f"Encoder panel{benchmark_label}: 0/{len(encoders)} encoders completed; "
            "no Leaderboard was written.",
            flush=True,
        )
    return failures, leaderboard_code


def _cmd_reproduce(args: argparse.Namespace) -> int:
    from soma.benchmark import get_benchmark

    encoders = getattr(args, "encoders", None)
    if encoders is not None:
        duplicates = sorted({name for name in encoders if encoders.count(name) > 1})
        if duplicates:
            print(
                f"Error: duplicate --encoders names: {', '.join(duplicates)}.",
                file=sys.stderr,
            )
            return 2
        if args.from_run_dir is not None:
            print(
                "Error: --encoders cannot be used with --from-run-dir; "
                "re-score one existing Run with --encoder instead.",
                file=sys.stderr,
            )
            return 2

    targets = _resolve_reproduce_targets(args.name)
    if not targets:
        try:
            get_benchmark(args.name)  # re-raise for the canonical "Unknown benchmark …" message
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
        return 2

    is_family = any(b.name != args.name for b in targets)
    if is_family and args.from_run_dir is not None:
        print(
            f"Error: --from-run-dir re-scores one run, so it needs a single sub-benchmark "
            f"(e.g. '{targets[0].name}'), not the '{args.name}' family.",
            file=sys.stderr,
        )
        return 2

    if encoders is not None:
        preflight_failures = [
            (benchmark, _preflight_reproduce_panel(benchmark, encoders))
            for benchmark in targets
        ]
        preflight_failures = [
            (benchmark, failures)
            for benchmark, failures in preflight_failures
            if failures
        ]
        if preflight_failures:
            if is_family:
                lines = ["Error: Encoder family panel preflight failed:"]
                index = 1
                for benchmark, failures in preflight_failures:
                    for failure in failures:
                        lines.append(
                            f"  {index}. Benchmark {benchmark.name!r}, Encoder {failure}"
                        )
                        index += 1
                message = "\n".join(lines)
            else:
                benchmark, failures = preflight_failures[0]
                message = (
                    f"Error: Encoder panel preflight failed for Benchmark "
                    f"{benchmark.name!r}:\n"
                    + "\n".join(
                        f"  {index}. {failure}"
                        for index, failure in enumerate(failures, 1)
                    )
                )
            print(
                message
                + "\nNo curation, Pipeline, extraction, training, or Run writes started.",
                file=sys.stderr,
            )
            return 2
        runtime_failures: list[tuple[str, str, str]] = []
        codes: list[int] = []
        for benchmark in targets:
            failures, code = _run_reproduce_panel(
                benchmark,
                args,
                encoders,
                family_root=args.name if is_family else None,
            )
            runtime_failures.extend(
                (benchmark.name, encoder, context) for encoder, context in failures
            )
            codes.append(code)
            if code == 2:
                break

        if runtime_failures:
            label = "Encoder family panel" if is_family else "Encoder panel"
            print(
                f"{label} runtime failures ({len(runtime_failures)}):",
                file=sys.stderr,
            )
            for benchmark_name, encoder, context in runtime_failures:
                cell = f"{benchmark_name}, {encoder}" if is_family else encoder
                print(f"  - {cell}: {context}", file=sys.stderr)
            return max(1, *codes)
        return max(codes) if codes else 2

    codes = [
        _reproduce_one(bench, args, family_root=args.name if is_family else None)
        for bench in targets
    ]
    return max(codes) if codes else 2


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    """Render a faceted leaderboard over the run dirs under ``--root`` (ADR 0003).

    Two entry points share one flat projection: a positional benchmark name supplies the
    canonical facet + primary metric + reference rows, while a bare ``--root`` discovers
    the ``(dataset, splits, task)`` triples under the root and requires disambiguation when
    several exist. ``--vary``/``--fix``/``--like`` shape the facet on top of either.
    """
    from soma.benchmark.leaderboard import (
        LeaderboardFacet,
        _AXIS_ALIASES,
        _MISSING,
        axis_value,
        discover_triples,
        format_table,
        load_run_record,
        project_leaderboard,
        write_leaderboard,
    )

    benchmark = None
    if args.name is not None:
        from soma.benchmark import get_benchmark

        try:
            benchmark = get_benchmark(args.name)
        except KeyError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    if args.root is None:
        print("Error: leaderboard needs --root <output_root>.", file=sys.stderr)
        return 2

    include_incomplete = False
    if benchmark is not None and args.metric is None:
        from soma.benchmark import get_reported_metrics

        include_incomplete = len(get_reported_metrics(benchmark)) > 1
    triples = discover_triples(args.root, include_incomplete=include_incomplete)
    if not triples:
        print(f"Error: no completed runs found under {args.root}.", file=sys.stderr)
        return 2

    # Facet: benchmark canonical facet as the base, then CLI overrides.
    vary: tuple[str, ...] = tuple(args.vary) if args.vary else (
        tuple(benchmark.facet.varied) if benchmark is not None else ()
    )
    fixed: dict[str, object] = dict(benchmark.facet.fixed) if benchmark is not None else {}
    for pair in args.fix or []:
        if "=" not in pair:
            print(f"Error: --fix expects axis=value, got {pair!r}", file=sys.stderr)
            return 2
        key, _, value = pair.partition("=")
        fixed[key.strip()] = value.strip()

    like_record = None
    if args.like is not None:
        like_record = load_run_record(args.like)
        if like_record is None:
            print(f"Error: --like run dir is not a completed run: {args.like}", file=sys.stderr)
            return 2
        # Fix every recognised axis except the varied one(s), by the example's value.
        for axis in _AXIS_ALIASES:
            if axis in vary:
                continue
            value = axis_value(like_record.canonical_spec, axis)
            if value is not _MISSING and value is not None:
                fixed.setdefault(axis, value)

    # Resolve the triple to render.
    if len(triples) == 1:
        triple = next(iter(triples))
    elif like_record is not None:
        triple = like_record.triple
    else:
        candidates = triples
        if "task" in fixed:
            candidates = {t: d for t, d in triples.items() if t[2] == str(fixed["task"])}
        if len(candidates) == 1:
            triple = next(iter(candidates))
        else:
            print(
                f"Error: {len(triples)} (dataset, splits, task) triples under {args.root}; "
                "disambiguate with --like <run_dir> or narrower filters:",
                file=sys.stderr,
            )
            for (dataset_ck, splits_ck, task), dirs in sorted(triples.items()):
                print(
                    f"  task={task} dataset={dataset_ck[:8]} splits={splits_ck[:8]} "
                    f"({len(dirs)} runs)",
                    file=sys.stderr,
                )
            return 2

    facet = LeaderboardFacet(vary=vary, fixed=fixed)
    try:
        table = project_leaderboard(
            triples[triple], facet, metric=args.metric, benchmark=benchmark, split=args.split
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    paths = write_leaderboard(table, args.root, name=args.name)
    print(format_table(table))
    print(f"\nWrote: {paths['csv']}  {paths['json']}  {paths['html']}")
    return 0


def _build_leaderboard_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma leaderboard",
        description="Render a faceted leaderboard over run dirs under an output root.",
    )
    parser.add_argument("name", nargs="?", default=None, help="Registered benchmark name (canonical facet).")
    parser.add_argument("--root", type=Path, default=None, help="Output root whose run dirs to project.")
    parser.add_argument("--vary", action="append", default=None, help="Axis to surface/rank across (repeatable).")
    parser.add_argument("--fix", action="append", default=None, help="Hold an axis fixed: axis=value (repeatable).")
    parser.add_argument("--like", type=Path, default=None, help="Fix all axes but --vary by this run dir's example.")
    parser.add_argument("--metric", type=str, default=None, help="Override the ranking metric.")
    parser.add_argument("--split", type=str, default=None, help="Override the split ranked on (default: test).")
    parser.set_defaults(func=_cmd_leaderboard)
    return parser


def _build_reproduce_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma reproduce",
        description="Curate → run → score a registered benchmark and check its tolerance band.",
    )
    parser.add_argument("name", help="Registered benchmark name (see `soma list benchmarks`).")
    parser.add_argument(
        "--from-run-dir",
        type=Path,
        default=None,
        help="Re-score an existing run directory (no curation, no training).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Run seeds 0..N-1 instead of the canonical set (--seeds 1 = fastest smoke).",
    )
    parser.add_argument("--raw-root", type=Path, default=None, help="Raw dataset root (full mode).")
    parser.add_argument(
        "--curated-dir",
        type=Path,
        default=None,
        help="Reuse an already-curated manifest dir (dataset.csv + splits.csv); "
        "skips curation. Alternative to --raw-root.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Curated manifest dir (default <raw-root>/curated).")
    parser.add_argument("--output-root", type=Path, default=None, help="Where runs are written.")
    parser.add_argument("--cache-root", type=Path, default=None, help="Shared feature-cache root (reused across seeds).")
    encoder_group = parser.add_mutually_exclusive_group()
    encoder_group.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Run one Encoder preset (Benchmark default if omitted).",
    )
    encoder_group.add_argument(
        "--encoders",
        type=str,
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Run an ordered Encoder panel, checking the complete panel before any work "
            "starts, then write one cross-encoder Leaderboard per concrete Benchmark. "
            "If a capability is missing, select a compatible Benchmark or fix the "
            "Encoder plugin."
        ),
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Append the measured number + provenance to the results ledger "
        "(soma/benchmark/<name>/results.csv) so 'reproduced' becomes a committed fact.",
    )
    parser.set_defaults(func=_cmd_reproduce)
    return parser


def _cmd_prepare_croma(args: argparse.Namespace) -> None:
    from soma.benchmark.croma.robustness import prepare_croma

    prepared = prepare_croma(args.raw_root, rebuild=args.rebuild)
    counts = ", ".join(f"{cohort.name}: {cohort.rows}" for cohort in prepared)
    print(f"Prepared PathoROB data under {args.raw_root} ({counts}).")


def _build_prepare_croma_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma prepare-croma",
        description="Acquire and decode the pinned PathoROB tile sources.",
    )
    parser.add_argument("raw_root", type=Path, help="Destination prepared-data root.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Deliberately replace a partial or revision-mismatched destination.",
    )
    parser.set_defaults(func=_cmd_prepare_croma)
    return parser


def _evaluation_targets(args: argparse.Namespace, predictions):
    """The cohort targets ``soma evaluate`` scores against: one split, or every row.

    A multi-output regression's ``value_<k>`` predictions read the same-named cohort
    columns; any other noun reads its family's target keys.
    """
    import pandas as pd

    from soma.data import Cohort
    from soma.evaluation.predictions import family_target_keys
    from soma.evaluation.scoring import _output_columns

    keys = _output_columns(predictions) or list(family_target_keys(predictions.task_family))
    if args.splits is not None:
        cohort = Cohort.from_csv(args.cohort, args.splits, targets=keys, allow_missing_test=True)
        return cohort.targets(args.split or "test", fold=args.fold or 0)
    if args.split is not None or args.fold is not None:
        raise ValueError("--split and --fold select from --splits; pass a splits CSV")
    records = pd.read_csv(args.cohort)
    unsplit = pd.DataFrame({"sample_id": records["sample_id"], "split": "test"})
    cohort = Cohort.from_frames(
        records, unsplit, targets=keys, sidecar_dir=Path(args.cohort).parent
    )
    return cohort.targets("test")


def _read_predictions(path: str):
    """A label-map directory, a points parquet, or a set-shape predictions parquet."""
    import json as _json

    import pyarrow.parquet as pq

    from soma.evaluation import LabelMapPredictions, PointPredictions, Predictions

    location = Path(path)
    if location.is_dir():
        return LabelMapPredictions.load(location)
    raw = (pq.read_schema(location).metadata or {}).get(b"soma")
    if raw is not None and _json.loads(raw).get("kind") == "points":
        return PointPredictions.from_parquet(location)
    return Predictions.from_parquet(location)


def _evaluation_grid_targets(args: argparse.Namespace, predictions):
    """The annotation source a grid noun scores against: the cohort CSV's
    ``label_mask_path`` / ``points_path`` columns, restricted to one split (or every row)."""
    import pandas as pd

    from soma.data import AnnotationManifest, Cohort, LabelMapSource, PointSource
    from soma.evaluation import LabelMapPredictions

    annotations = AnnotationManifest.from_csv(args.cohort)
    if isinstance(predictions, LabelMapPredictions):
        first = predictions[predictions.sample_ids[0]] if predictions.sample_ids else None
        size = (0, 0) if first is None else tuple(int(v) for v in first.shape)
        source = LabelMapSource.from_manifests(annotations, size=size)
    else:
        source = PointSource.from_manifest(annotations)
    if args.splits is not None:
        cohort = Cohort.from_csv(args.cohort, args.splits, targets=[], allow_missing_test=True)
        split = cohort.folds[args.fold or 0]
        name = args.split or "test"
        ids = split.tune if name == "tune" else split.train if name == "train" else split.tests[name]
        return source.subset(list(ids))
    if args.split is not None or args.fold is not None:
        raise ValueError("--split and --fold select from --splits; pass a splits CSV")
    return source.subset(list(pd.read_csv(args.cohort)["sample_id"].astype(str)))


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from rich.console import Console
    from rich.table import Table

    from soma.evaluation import LabelMapPredictions, PointPredictions, evaluate

    try:
        predictions = _read_predictions(args.predictions)
        if isinstance(predictions, (LabelMapPredictions, PointPredictions)):
            targets = _evaluation_grid_targets(args, predictions)
        else:
            targets = _evaluation_targets(args, predictions)
        evaluation = evaluate(
            predictions,
            targets,
            metrics=args.metrics or "default",
            subgroups=args.subgroups or (),
        )
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2
    table = Table(
        title=f"{predictions.task_family} — {evaluation.num_samples} samples",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for name, value in evaluation.metrics.items():
        table.add_row(name, f"{value:.4f}" if isinstance(value, float) else str(value))
    Console().print(table)
    if args.out is not None:
        evaluation.to_json(args.out)
        print(f"Evaluation written to {args.out}")
    return 0


def _build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma evaluate",
        description=(
            "Score predictions against a cohort's targets, joined by sample_id. Every "
            "predicted sample needs a target and every target a prediction. Set-shape "
            "predictions score against dataset.csv's target columns; label-map directories "
            "against its label_mask_path rasters (class indices) and point predictions "
            "against its points_path files."
        ),
    )
    parser.add_argument(
        "predictions",
        help=(
            "Predictions parquet (sample_id + the head's columns), a points parquet, or a "
            "label-map directory (PNGs + index.parquet)."
        ),
    )
    parser.add_argument("--cohort", required=True, help="dataset.csv holding the targets.")
    parser.add_argument(
        "--splits",
        default=None,
        help="splits.csv; without it every row of --cohort is scored as one unsplit frame.",
    )
    parser.add_argument("--split", default=None, help="Split to score (default: test).")
    parser.add_argument("--fold", type=int, default=None, help="Fold to score (default: 0).")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metric registry keys or dotted paths (default: the family's defaults).",
    )
    parser.add_argument(
        "--subgroups", nargs="+", default=None, help="Metadata columns to break metrics down by."
    )
    parser.add_argument("--out", default=None, help="Write the evaluation as JSON to this path.")
    parser.set_defaults(func=_cmd_evaluate)
    return parser


def _cmd_compact_index(args: argparse.Namespace) -> int:
    from soma.pipeline.layout import compact_run_index

    path = Path(args.path)
    if path.is_dir():
        path = path / "indexes" / "runs.csv"
    if not path.is_file():
        print(f"Error: no run index at {path}", file=sys.stderr)
        return 2
    kept = compact_run_index(path)
    print(f"Compacted {path}: {kept} run(s) kept.")
    return 0


def _build_compact_index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma compact-index",
        description=(
            "Rewrite an output root's append-only runs.csv with one row per run_id "
            "(the latest status). Readers already dedupe; this only shrinks the file."
        ),
    )
    parser.add_argument("path", help="Output root (containing indexes/runs.csv) or the runs.csv itself.")
    parser.set_defaults(func=_cmd_compact_index)
    return parser


def _print_top_level_help() -> None:
    print(
        "usage: soma run CONFIG [--feature-dir DIR]\n"
        "       soma run --preset NAME --cohort CSV --splits CSV --out DIR [--seed N] [PRESET ARGUMENTS]\n"
        "       soma init NAME [PRESET ARGUMENTS]\n"
        "       soma list [KIND] [--level {tile,slide,patient}]\n"
        "       soma prepare-croma RAW_ROOT [--rebuild]\n"
        "       soma reproduce NAME [--from-run-dir DIR] [--seeds N] [--raw-root DIR]\n"
        "       soma leaderboard [NAME] --root OUTPUT_ROOT [--vary AXIS] [--fix AXIS=VALUE] [--like DIR]\n"
        "       soma evaluate PREDICTIONS --cohort CSV [--splits CSV --split NAME --fold N] [--metrics M ...] [--subgroups COL ...] [--out JSON]\n"
        "       soma compact-index OUTPUT_ROOT\n"
        "\n"
        "commands:\n"
        "  run          run an experiment from a YAML config (--feature-dir: on cached features),\n"
        "               or from a preset (--preset) on a cohort\n"
        "  init         print a preset as an editable YAML config with every default filled in\n"
        "  list         list the component kinds, or one kind's components and their params\n"
        "  prepare-croma  acquire and decode the pinned PathoROB tile sources\n"
        "  reproduce    curate → run → score a registered benchmark, check its tolerance band\n"
        "  leaderboard  render a faceted view over the run dirs under an output root\n"
        "  evaluate     score a predictions parquet against a cohort's targets\n"
        "  compact-index  rewrite the append-only runs.csv with one row per run\n"
        "\n"
        "examples:\n"
        "  soma run /path/to/config.yaml\n"
        "  python -m soma run /path/to/config.yaml\n"
        "  soma init mil_classification --encoder uni2 > exp.yaml\n"
        "  soma run --preset mil_classification --encoder uni2 --cohort dataset.csv --splits splits.csv --out runs\n"
        "  soma list presets\n"
        "  soma list benchmarks\n"
        "  soma prepare-croma /data/croma\n"
        "  soma reproduce ocelot --from-run-dir /runs/ocelot\n"
        "  soma reproduce eva/bach --encoder uni2 --raw-root /data/eva/bach\n"
        "  soma reproduce eva --raw-root /data/eva   # fan out over the eva/<dataset> family\n"
        "  soma leaderboard --root /runs/sweep --vary encoder"
    )


def _print_list_help() -> None:
    print(
        "usage: soma list [KIND] [--level {tile,slide,patient}]\n"
        "\n"
        "commands:\n"
        "  (no KIND)    list the component kinds\n"
        "  KIND         list one kind's registered components with their params and defaults\n"
        f"               ({', '.join(components())})\n"
        "\n"
        "options:\n"
        "  --level      restrict the encoder listing to one level\n"
        "\n"
        "examples:\n"
        "  soma list\n"
        "  soma list aggregators\n"
        "  soma list encoders --level tile"
    )


def main(argv: list[str] | None = None) -> None:
    configure_cli_logging()
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        _print_top_level_help()
        raise SystemExit(2)

    if args[0] in {"-h", "--help"}:
        _print_top_level_help()
        return

    if args[0] == "list":
        if len(args) > 1 and args[1] in {"-h", "--help"}:
            _print_list_help()
            return
        parser = _build_list_parser()
        parsed = parser.parse_args(args[1:])
        parsed.func(parsed)
        return

    if args[0] == "reproduce":
        parser = _build_reproduce_parser()
        parsed = parser.parse_args(args[1:])
        raise SystemExit(parsed.func(parsed))

    if args[0] == "leaderboard":
        parser = _build_leaderboard_parser()
        parsed = parser.parse_args(args[1:])
        raise SystemExit(parsed.func(parsed))

    if args[0] == "evaluate":
        parser = _build_evaluate_parser()
        parsed = parser.parse_args(args[1:])
        code = parsed.func(parsed)
        if code:
            raise SystemExit(code)
        return

    if args[0] == "compact-index":
        parser = _build_compact_index_parser()
        parsed = parser.parse_args(args[1:])
        raise SystemExit(parsed.func(parsed))

    if args[0] == "prepare-croma":
        parser = _build_prepare_croma_parser()
        parsed = parser.parse_args(args[1:])
        parsed.func(parsed)
        return

    if args[0] in {"run", "init"}:
        parser = (_build_run_parser if args[0] == "run" else _build_init_parser)(args[1:])
        parsed = parser.parse_args(args[1:])
        parsed.func(parser, parsed)
        return

    print(
        f"Error: unknown command {args[0]!r}; run a config with 'soma run CONFIG' (soma --help lists the commands).",
        file=sys.stderr,
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
