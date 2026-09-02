"""Command-line interface for soma."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from soma.aggregators import list_aggregators
from soma.config import load_config
from soma.decoders import list_decoders
from soma.encoders import list_models
from soma.pipeline import Pipeline
from soma.pixel_classifiers import list_pixel_classifiers
from soma.tasks import list_task_heads

# The reproduce orchestration lives in soma.benchmarks.run (issue #370): the CLI is a
# thin caller of the same importable code (`soma.benchmarks.run_benchmark`). The aliases
# below remain the CLI's patchable module-level seams — `_reproduce_one` threads them back
# through the shared implementation so replacing e.g. ``soma.cli._provenance`` or
# ``soma.cli.Pipeline`` still takes effect.
from soma.benchmarks import run as _reproduce_impl
from soma.benchmarks.run import (
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


def _parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Turn ``--set a.b.c=value`` strings into a nested override dict.

    Keys are dotted paths into the config layout (``data.dataset_csv``,
    ``run.output_root``, ``training.epochs`` …). Values are parsed as YAML scalars so
    types come through naturally (``epochs=2`` → int, ``pin_memory=false`` → bool, paths
    stay strings). Lets a committed config be repointed at a new machine without editing
    it on disk.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            print(f"Error: --set expects key=value, got {pair!r}", file=sys.stderr)
            sys.exit(2)
        key, _, raw_value = pair.partition("=")
        key = key.strip()
        if not key:
            print(f"Error: --set has an empty key in {pair!r}", file=sys.stderr)
            sys.exit(2)
        value = yaml.safe_load(raw_value)
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            existing = cursor.get(part)
            if not isinstance(existing, dict):
                existing = {}
                cursor[part] = existing
            cursor = existing
        cursor[parts[-1]] = value
    return overrides


def _run_config_path(config_path: Path, overrides: dict[str, Any] | None = None) -> None:
    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(config_path, overrides=overrides)
    except Exception as exc:
        print(f"Error: failed to load config from {config_path}: {exc}", file=sys.stderr)
        sys.exit(1)

    Pipeline(config).run()


def _print_table(title: str, values: list[str]) -> None:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(
        title=title,
        show_header=True,
        header_style="bold",
        min_width=max(len(title) + 4, 12),
    )
    table.add_column("Name")
    if not values:
        table.add_row("(none)")
    else:
        for value in values:
            table.add_row(value)
    console.print(table)


def _cmd_list(args: argparse.Namespace) -> None:
    kind = args.kind
    if kind == "encoders":
        values = list_models(level=args.level)
        title = "Encoders" if args.level is None else f"Encoders ({args.level})"
    elif kind == "aggregators":
        values = list_aggregators()
        title = "Aggregators"
    elif kind == "decoders":
        values = list_decoders()
        title = "Decoders"
    elif kind == "pixel-classifiers":
        values = list_pixel_classifiers()
        title = "Pixel Classifiers"
    elif kind == "benchmarks":
        from soma.benchmarks import list_benchmarks

        values = list_benchmarks()
        title = "Benchmarks"
    else:
        values = list_task_heads()
        title = "Task Heads"
    _print_table(title, values)


def _build_list_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma list",
        description="List encoders, aggregators, dense registries, or task heads.",
    )
    parser.add_argument(
        "kind",
        choices=(
            "encoders",
            "aggregators",
            "decoders",
            "pixel-classifiers",
            "tasks",
            "benchmarks",
        ),
        help="Component family to list.",
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
    from soma.benchmarks import get_benchmark, list_benchmarks

    try:
        return [get_benchmark(name)]
    except KeyError:
        return [get_benchmark(n) for n in list_benchmarks() if n.startswith(f"{name}/")]


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

    Delegates to :func:`soma.benchmarks.run._reproduce_one` — the same code behind the
    public :func:`soma.benchmarks.run_benchmark` (issue #370) — threading this module's
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

    if config.encoder is None or config.encoder.name != capabilities.name:
        resolved = None if config.encoder is None else config.encoder.name
        return (
            f"Benchmark builder resolved encoder {resolved!r}, expected "
            f"{capabilities.name!r}."
        )

    feature_kind = config.preprocessing.feature_kind
    if feature_kind == "patch_features" and not capabilities.dense:
        return (
            f"missing dense capability: Benchmark {benchmark.name!r} requires dense "
            "patch features, but slide2vec reports dense=False. Select a compatible "
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

    declared_geometry = feature_kind is not None or config.dataset_type in {
        "slide",
        "patient",
    }
    resolve_encoder_output(
        encoder,
        requested_output_variant=config.encoder.output_variant,
        metadata=metadata,
    )
    if not declared_geometry:
        return None

    from soma.encoders.validation import resolve_preprocessing_config

    preprocessing = resolve_preprocessing_config(config.encoder, config.preprocessing)
    from slide2vec.api import EncoderInputContract

    try:
        if feature_kind is not None:
            EncoderInputContract.declared_dense(
                encoder,
                target_size_px=int(preprocessing.requested_tile_size_px),
                window_size=preprocessing.dense_window_size,
            )
        else:
            EncoderInputContract.declared_pooled(
                encoder,
                requested_tile_size_px=int(preprocessing.requested_tile_size_px),
                allow_non_recommended_settings=(
                    config.encoder.allow_non_recommended_settings
                ),
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
    from soma.leaderboard import discover_triples

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
    from soma.benchmarks import get_benchmark

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
    from soma.leaderboard import (
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
        from soma.benchmarks import get_benchmark

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
        from soma.benchmarks import get_reported_metrics

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
        "(soma/benchmarks/results/<name>.csv) so 'reproduced' becomes a committed fact.",
    )
    parser.set_defaults(func=_cmd_reproduce)
    return parser


def _cmd_prepare_croma(args: argparse.Namespace) -> None:
    from soma.robustness import prepare_croma

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


def _cmd_compact_index(args: argparse.Namespace) -> int:
    from soma.output_layout import compact_run_index

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
        "usage: soma CONFIG\n"
        "       soma list {encoders,aggregators,decoders,pixel-classifiers,tasks,benchmarks} [--level {tile,slide,patient}]\n"
        "       soma prepare-croma RAW_ROOT [--rebuild]\n"
        "       soma reproduce NAME [--from-run-dir DIR] [--seeds N] [--raw-root DIR]\n"
        "       soma leaderboard [NAME] --root OUTPUT_ROOT [--vary AXIS] [--fix AXIS=VALUE] [--like DIR]\n"
        "       soma compact-index OUTPUT_ROOT\n"
        "\n"
        "commands:\n"
        "  CONFIG       run a pipeline from a YAML config file\n"
        "  list         list public model/component/benchmark registries\n"
        "  prepare-croma  acquire and decode the pinned PathoROB tile sources\n"
        "  reproduce    curate → run → score a registered benchmark, check its tolerance band\n"
        "  leaderboard  render a faceted view over the run dirs under an output root\n"
        "  compact-index  rewrite the append-only runs.csv with one row per run\n"
        "\n"
        "examples:\n"
        "  soma /path/to/config.yaml\n"
        "  python -m soma /path/to/config.yaml\n"
        "  soma list benchmarks\n"
        "  soma prepare-croma /data/croma\n"
        "  soma reproduce ocelot --from-run-dir /runs/ocelot\n"
        "  soma reproduce eva/bach --encoder uni2 --raw-root /data/eva/bach\n"
        "  soma reproduce eva --raw-root /data/eva   # fan out over the eva/<dataset> family\n"
        "  soma leaderboard --root /runs/sweep --vary encoder"
    )


def _print_list_help() -> None:
    print(
        "usage: soma list {encoders,aggregators,decoders,pixel-classifiers,tasks,benchmarks} [--level {tile,slide,patient}]\n"
        "\n"
        "commands:\n"
        "  encoders     list registered encoder presets\n"
        "  aggregators  list registered aggregator presets\n"
        "  decoders     list registered dense decoder presets\n"
        "  pixel-classifiers  list registered per-pixel classifier presets\n"
        "  tasks        list registered task-head presets\n"
        "  benchmarks   list registered benchmarks\n"
        "\n"
        "options:\n"
        "  --level      restrict encoder listing to one level\n"
        "\n"
        "examples:\n"
        "  soma list encoders\n"
        "  soma list encoders --level tile\n"
        "  soma list tasks"
    )


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        _print_top_level_help()
        raise SystemExit(2)

    if args[0] in {"-h", "--help"}:
        _print_top_level_help()
        return

    if args[0] == "list":
        if len(args) == 1 or args[1] in {"-h", "--help"}:
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

    if args[0] == "compact-index":
        parser = _build_compact_index_parser()
        parsed = parser.parse_args(args[1:])
        raise SystemExit(parsed.func(parsed))

    if args[0] == "prepare-croma":
        parser = _build_prepare_croma_parser()
        parsed = parser.parse_args(args[1:])
        parsed.func(parsed)
        return

    if args[0] == "run":
        print(
            "Error: pass the config path directly.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Config-run form: one config path, then any number of `--set key=value` overrides.
    config_path = args[0]
    if config_path.startswith("-"):
        print(
            "Error: expected one config path or the 'list' subcommand.",
            file=sys.stderr,
        )
        sys.exit(2)

    set_pairs: list[str] = []
    rest = args[1:]
    i = 0
    while i < len(rest):
        token = rest[i]
        if token == "--set":
            if i + 1 >= len(rest):
                print("Error: --set requires a key=value argument", file=sys.stderr)
                sys.exit(2)
            set_pairs.append(rest[i + 1])
            i += 2
        elif token.startswith("--set="):
            set_pairs.append(token[len("--set="):])
            i += 1
        else:
            print(
                f"Error: unexpected argument {token!r} "
                "(expected one config path and optional --set key=value)",
                file=sys.stderr,
            )
            sys.exit(2)

    overrides = _parse_set_overrides(set_pairs) if set_pairs else None
    _run_config_path(Path(config_path), overrides)


if __name__ == "__main__":
    main()
