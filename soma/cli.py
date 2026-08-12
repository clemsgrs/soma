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


def _reproduce_seeds(benchmark, requested: int | None) -> tuple[int, ...]:
    """Seeds to run: the benchmark's canonical set by default, or the first ``requested``.

    ``--seeds N`` is a single-seed-style smoke knob: it runs seeds ``0..N-1`` so the check
    is a fast subset of the canonical set (``--seeds 1`` is the quickest smoke).
    """
    if requested is None:
        return tuple(benchmark.canonical_seeds)
    if requested < 1:
        print("Error: --seeds must be >= 1", file=sys.stderr)
        sys.exit(2)
    return tuple(range(requested))


def _require_reported_scores(benchmark, scores: dict[str, float]) -> dict[str, float]:
    """Fail one scoring attempt if any required Reported metric is absent."""
    from soma.benchmarks import get_reported_metrics

    missing = [metric for metric in get_reported_metrics(benchmark) if metric not in scores]
    if missing:
        names = ", ".join(repr(metric) for metric in missing)
        print(
            f"Error: benchmark {benchmark.name!r} score is missing Reported metric(s): "
            f"{names}.",
            file=sys.stderr,
        )
        sys.exit(2)
    return scores


def _reproduce_reference_row(benchmark, axes: dict[str, Any]):
    """The single comparable reference row for the primary metric, if one exists.

    ``kind="gate"`` rows carry a tolerance band that can highlight potential drift. External
    rows remain contextual comparisons without a tolerance. No matching row is valid: the
    benchmark protocol still runs and the reference comparison is explicitly skipped.
    """
    matching = [r for r in benchmark.expected(**axes) if r.metric == benchmark.primary_metric]
    gate_rows = [r for r in matching if not r.is_external]
    if len(gate_rows) > 1:
        print(
            f"Error: {len(gate_rows)} gate reference rows match metric "
            f"{benchmark.primary_metric!r} (axes={axes}); refine the axes.",
            file=sys.stderr,
        )
        sys.exit(2)
    if gate_rows:
        return gate_rows[0]
    if any(r.is_external for r in matching):
        # External-only: render Measured beside the Reference without a drift status (#260).
        return None
    return None


def _report_tolerance(benchmark, measured: float, row) -> bool:
    ok = row.within_tolerance(measured)
    status = "REFERENCE OK" if ok else "POTENTIAL DRIFT"
    delta = measured - row.expected
    # Show the effective absolute band; annotate the fraction when the row is relative.
    band = f"±{row.tolerance_band():.4f}" + (
        f" ({row.tolerance:.0%} relative)" if row.relative else ""
    )
    print(
        f"[{status}] {benchmark.name} {row.metric} = {measured:.4f}  "
        f"(reference {row.expected:.4f}, Δ {delta:+.4f}, tolerance {band})"
    )
    return ok


def _render_reference_comparison(row, measured: float) -> None:
    """Render one measured value beside a matching reference row."""
    delta = measured - row.expected
    if row.is_external:
        label = row.label or "external reference"
        link = f"  <{row.url}>" if row.url else ""
        print(
            f"  reference [{label}]: {row.metric} = {row.expected:.4f}  "
            f"(Δ {delta:+.4f}, external — context only){link}"
        )
    else:
        print(f"  reference: {row.metric} = {row.expected:.4f}  (Δ {delta:+.4f})")


def _report_external(benchmark, measured: float, axes: dict[str, Any]) -> int:
    """Render the Measured value beside external contextual Reference row(s) (#260).

    An external-only benchmark has no tolerance band. Reproduce prints the Measured value
    and, for each external Reference row at these axes, the published number and signed
    delta. The comparison is context only (e.g. HEST's slide2vec↔TRIDENT gap).
    """
    print(f"[MEASURED] {benchmark.name} {benchmark.primary_metric} = {measured:.4f}")
    external = [
        r
        for r in benchmark.expected(**axes)
        if r.metric == benchmark.primary_metric and r.is_external
    ]
    if not external:
        print(
            f"[REFERENCE SKIPPED] {benchmark.name} {benchmark.primary_metric} — "
            f"no packaged reference matches axes={axes}."
        )
        return 0
    for row in external:
        _render_reference_comparison(row, measured)
    return 0


def _report_secondary(benchmark, metric: str, measured: float, axes: dict[str, Any]) -> None:
    """Render one non-gating Reported metric beside any matching references."""
    print(f"[MEASURED] {benchmark.name} {metric} = {measured:.4f}")
    references = [row for row in benchmark.expected(**axes) if row.metric == metric]
    if not references:
        print(f"  {benchmark.name} {metric} — no reference matches axes={axes}.")
        return
    for row in references:
        _render_reference_comparison(row, measured)


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


def _resolve_run_dir(path: str | Path) -> Path:
    """Resolve a run directory directly or through its newest nested summary."""
    candidate = Path(path)
    if (candidate / "summary.json").is_file() or (candidate / "run.yaml").is_file():
        return candidate
    summaries = sorted(
        candidate.glob("**/summary.json"), key=lambda summary: summary.stat().st_mtime
    )
    return summaries[-1].parent if summaries else candidate


def _from_run_dir_axes(benchmark, from_run_dir: str | Path) -> dict[str, Any]:
    """The benchmark's varied axes (encoder) read from a run's OWN recorded spec.

    ``reproduce --from-run-dir`` must compare against the reference row for the
    encoder the run actually used — otherwise empty axes fall back to the benchmark
    default and a ``uni`` run is silently compared against the ``uni2`` reference. Reads the
    tolerant ``canonical_spec`` (the same source the leaderboard projects), so it still
    resolves on runs whose full config no longer round-trips through ``load_config``.
    Unresolved axes are skipped; a missing/unrankable run dir yields ``{}``.
    """
    from soma.leaderboard import _MISSING, axis_value, load_run_record

    run_dir = _resolve_run_dir(from_run_dir)
    record = load_run_record(run_dir)
    if record is None:
        return {}
    resolved: dict[str, Any] = {}
    for axis in benchmark.facet.varied:
        value = axis_value(record.canonical_spec, axis)
        if value is not _MISSING:
            resolved[axis] = value
    return resolved


def _results_table_name(benchmark) -> str:
    """The results/reference table a benchmark records under (its family prefix).

    Family members share one keyed table: ``eva/bach`` records under ``eva`` (keyed by
    ``dataset``), matching ``reference/eva.csv``. A standalone benchmark (``ocelot``) maps
    to itself.
    """
    return benchmark.name.split("/", 1)[0]


def _git_commit() -> str:
    """Short ``HEAD`` SHA of the soma checkout (``-dirty`` if *tracked* code has changes).

    Resolved from the installed package location, not the CWD, so it pins the code that
    actually produced the number. Only tracked modifications count as dirty — untracked
    scratch (run outputs, notes) does not affect the code, so ``--untracked-files=no``
    keeps it out of the provenance. Returns ``"unknown"`` outside a git checkout.
    """
    import subprocess

    import soma

    repo = Path(soma.__file__).resolve().parents[1]
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _provenance() -> tuple[str, str, str]:
    """``(date, soma_commit, slide2vec_version)`` pinning the environment of a recorded row."""
    import datetime
    from importlib import metadata

    date = datetime.date.today().isoformat()
    try:
        slide2vec_version = metadata.version("slide2vec")
    except metadata.PackageNotFoundError:
        slide2vec_version = "unknown"
    return date, _git_commit(), slide2vec_version


def _runtime_croma_version() -> str:
    """Installed Croma version used by a representation-evaluation run."""
    from importlib import metadata

    try:
        return metadata.version("croma")
    except metadata.PackageNotFoundError:
        return "unknown"


def _recorded_croma_version(
    run_dir: str | Path | None, *, historical_run: bool
) -> str:
    """Croma producer version from run metadata or the current execution environment."""
    if run_dir is not None:
        path = _resolve_run_dir(run_dir)
        metadata_path = path / "run.yaml"
        if metadata_path.is_file():
            metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            provenance = metadata.get("representation_provenance") or {}
            version = provenance.get("croma")
            if version not in (None, ""):
                return str(version)
    return "unknown" if historical_run else _runtime_croma_version()


def _record_reference_row(benchmark, axes: dict[str, Any], gate_row):
    """The reference row whose key + metric key a ``--record`` ledger entry.

    A recorded measurement must join its reference table, so the ledger entry copies a
    reference row's key. Prefer the tolerance-bearing **gate** row; for an **external-only**
    benchmark (no gate row — e.g. ``hest/<task>``) fall back to the single external row that
    matches the primary metric, so reproduced numbers are still recorded (they join the HEST
    reference for the per-cell delta + rank concordance the docs render). Returns ``None`` when
    neither resolves — nothing to key on, so ``--record`` is a no-op for that cell.
    """
    if gate_row is not None:
        return gate_row
    external = [
        r
        for r in benchmark.expected(**axes)
        if r.metric == benchmark.primary_metric and r.is_external
    ]
    return external[0] if len(external) == 1 else None


def _record_axes(benchmark, axes: dict[str, Any], run_dir: str | Path | None) -> dict[str, Any]:
    """The varied-axis values a ``--record`` row must be attributable by.

    A **broad** reference row states one config-agnostic band (OCELOT's) and so carries an
    *empty* key. Copying it verbatim appends a measurement no reader can attribute, and
    which :func:`~soma.benchmarks.reproduction.latest_measurements` drops outright because
    its encoder is ``None`` — recorded in appearance only. So resolve the benchmark's varied
    axes independently: what the CLI made explicit, falling back to the run's own recorded
    spec for anything left implicit (an omitted ``--encoder`` means the benchmark's anchor,
    which only the run knows). Axes that resolve nowhere are left out rather than guessed.
    """
    resolved = {
        axis: axes[axis] for axis in benchmark.facet.varied if axes.get(axis) not in (None, "")
    }
    missing = [axis for axis in benchmark.facet.varied if axis not in resolved]
    if missing and run_dir is not None:
        for axis, value in _from_run_dir_axes(benchmark, run_dir).items():
            if axis in missing:
                resolved[axis] = value
    return resolved


def _record_result(
    benchmark,
    row,
    measured: float,
    std: float | None,
    n_seeds: int | None,
    key_axes: dict[str, Any] | None = None,
    metric: str | None = None,
    run_dir: str | Path | None = None,
    historical_run: bool = False,
) -> None:
    """Append a reproduced-measurement row to the benchmark's results ledger (``--record``).

    Keys are copied from the primary reference anchor ``row`` (a **gate** row, or an
    **external** row for an external-only benchmark). ``metric`` selects the Reported metric
    being recorded, so secondary metrics share the primary cell key even when they have no
    reference row. Provenance is captured at run time.

    ``key_axes`` fills key columns the reference leaves empty — see :func:`_record_axes`. A
    keyed reference (EVA, HEST) already states them, so this only bites for a broad one;
    values the reference does state always win, since that is the cell being joined.
    """
    from soma.benchmarks import MeasuredRow, append_result

    key = dict(row.key)
    for axis, value in (key_axes or {}).items():
        if not key.get(axis):
            key[axis] = value

    date, commit, slide2vec_version = _provenance()
    measured_row = MeasuredRow(
        key=key,
        metric=metric or row.metric,
        measured=measured,
        std=std,
        n_seeds=n_seeds,
        date=date,
        soma_commit=commit,
        slide2vec_version=slide2vec_version,
        croma_version=(
            _recorded_croma_version(run_dir, historical_run=historical_run)
            if getattr(benchmark, "records_croma_version", False)
            else ""
        ),
        source="soma reproduce --record",
    )
    path = append_result(_results_table_name(benchmark), measured_row, key_order=list(key))
    print(f"  recorded → {path}")


def _curated_manifest_from_dir(curated_dir: Path):
    """Reconstruct a :class:`CuratedManifest` from an already-curated directory.

    The ``--curated-dir`` fast path: curation is a deterministic ``raw -> manifest`` step
    that can cost minutes (HEST-bench IDC explodes 35k spots to lossless PNGs). When the
    caller already has a manifest, skip curation and point the pipeline straight at it. We
    only require the two CSVs; the ``spatial_expression`` sidecars (``targets.npy`` /
    ``genes.json``) are resolved by the Manifest loader relative to ``dataset.csv``, so we
    just populate the paths that exist and let the loader fail-fast on a malformed dir.
    """
    from soma.curation.manifest import CuratedManifest
    from soma.dataset import GENES_FILENAME, TARGET_MATRIX_FILENAME

    curated_dir = Path(curated_dir)
    dataset_csv = curated_dir / "dataset.csv"
    splits_csv = curated_dir / "splits.csv"
    missing = [p.name for p in (dataset_csv, splits_csv) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"--curated-dir {curated_dir} is not a curated manifest: missing {missing}. "
            f"Point it at a directory holding dataset.csv + splits.csv (as written by a "
            f"prior full-mode run under <raw-root>/curated), or use --raw-root to curate."
        )

    def _opt(name: str) -> Path | None:
        p = curated_dir / name
        return p if p.is_file() else None

    return CuratedManifest(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        summary_json=_opt("summary.json"),
        target_matrix_path=_opt(TARGET_MATRIX_FILENAME),
        genes_path=_opt(GENES_FILENAME),
    )


class _MissingReproduceSourceError(ValueError):
    """The CLI was not given a source from which to obtain a curated Manifest."""


def _reproduce_manifest(benchmark, args: argparse.Namespace, *, family_root: str | None = None):
    """Resolve one concrete Benchmark's invariant curated Manifest."""
    curated_dir = getattr(args, "curated_dir", None)
    if curated_dir is None and args.raw_root is None:
        raise _MissingReproduceSourceError(
            "reproduce needs --raw-root <dir> (curate from raw), "
            "--curated-dir <dir> (reuse a pre-curated manifest, skip curation), or "
            "--from-run-dir <dir> (re-score an existing run)."
        )

    sub = benchmark.name.split("/", 1)[1] if family_root else None
    if curated_dir is not None:
        curated = Path(curated_dir) / sub if sub else Path(curated_dir)
        manifest = _curated_manifest_from_dir(curated)
        print(f"Using pre-curated manifest (skipping curation): {curated}", flush=True)
        return manifest

    family_raw_root = Path(args.raw_root)
    shares_family_root = bool(
        sub and getattr(benchmark, "family_uses_shared_raw_root", False)
    )
    raw_root = (
        family_raw_root
        if shares_family_root
        else family_raw_root / sub
        if sub
        else family_raw_root
    )
    if args.out_dir:
        out_dir = Path(args.out_dir) / sub if sub else Path(args.out_dir)
    elif shares_family_root:
        out_dir = family_raw_root / "curated" / sub
    else:
        out_dir = raw_root / "curated"
    return benchmark.curate(raw_root, out_dir)


def _reproduce_output_root(
    benchmark, args: argparse.Namespace, *, family_root: str | None = None
) -> Path:
    """The existing output-root contract for one concrete Benchmark."""
    sub = benchmark.name.split("/", 1)[1] if family_root else None
    if args.output_root:
        return Path(args.output_root) / sub if sub else Path(args.output_root)
    return Path.cwd() / "soma_reproduce" / benchmark.name


def _reproduce_one(
    benchmark,
    args: argparse.Namespace,
    *,
    family_root: str | None = None,
    manifest=None,
) -> int:
    """Curate → run → score one benchmark and report any reference comparison.

    ``family_root`` is the family name when this benchmark is one member of a fanned-out
    family (e.g. ``eva`` for ``eva/bach``); it nests the member's raw/curated/output paths
    under a per-dataset subdirectory so sibling members do not collide.
    """
    axes: dict[str, Any] = {}
    if args.encoder is not None:
        axes["encoder"] = args.encoder

    if args.from_run_dir is not None:
        # Constrain the reference lookup to the run's OWN axes; an explicit --encoder
        # still wins (setdefault only fills what the CLI left unset).
        for axis, value in _from_run_dir_axes(benchmark, args.from_run_dir).items():
            axes.setdefault(axis, value)

    seeds = _reproduce_seeds(benchmark, args.seeds)
    # ON START: surface the fast paths so a time-conscious user isn't surprised by the
    # full canonical-seed training cost.
    print(
        f"Reproducing benchmark '{benchmark.name}' — canonical seeds "
        f"{list(benchmark.canonical_seeds)}, running {list(seeds)}.\n"
        "  Fast paths: --seeds 1 (single-seed smoke) | "
        "--from-run-dir <dir> (re-score an existing run, no training).\n"
        "  Cache-aware: feature extraction is cached and shared across seeds and repeat "
        "runs, so it runs once per encoder.",
        flush=True,
    )

    row = _reproduce_reference_row(benchmark, axes)

    from soma.benchmarks import get_reported_metrics

    reported_metrics = get_reported_metrics(benchmark)

    def _report_measured_metrics(measured: dict[str, float]) -> int:
        for metric in reported_metrics:
            value = measured[metric]
            if metric != benchmark.primary_metric:
                _report_secondary(benchmark, metric, value, axes)
            elif row is None:
                _report_external(benchmark, value, axes)
            else:
                _report_tolerance(benchmark, value, row)
        return 0

    if args.from_run_dir is not None:
        metrics = _require_reported_scores(benchmark, benchmark.score(args.from_run_dir))
        measured = float(metrics[benchmark.primary_metric])
        if getattr(args, "record", False):
            # A re-scored single run has no seed spread, so std is None — but it *is* one
            # seed, and an empty n_seeds reads as "unknown" rather than "one". Key the ledger
            # entry off the gate row, or the external row for an external-only benchmark,
            # with the varied axes filled from the run itself.
            record_row = _record_reference_row(benchmark, axes, row)
            if record_row is not None:
                key_axes = _record_axes(benchmark, axes, args.from_run_dir)
                for metric in reported_metrics:
                    _record_result(
                        benchmark,
                        record_row,
                        float(metrics[metric]),
                        std=None,
                        n_seeds=1,
                        key_axes=key_axes,
                        metric=metric,
                        run_dir=args.from_run_dir,
                        historical_run=True,
                    )
            else:
                print("  (no reference row to key --record on; nothing recorded)")
        return _report_measured_metrics({benchmark.primary_metric: measured, **metrics})

    import statistics

    if manifest is None:
        try:
            manifest = _reproduce_manifest(benchmark, args, family_root=family_root)
        except _MissingReproduceSourceError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    output_root = _reproduce_output_root(benchmark, args, family_root=family_root)
    # Feature extraction is seed-independent (the encoder is frozen and the cache key is
    # derived from encoder + tile content, not the seed), so every seed shares one cache
    # root and extraction runs once. --cache-root relocates it (e.g. to fast local storage);
    # by default it sits beside the run outputs, shared across seeds. Without a shared root
    # each seed's per-seed output_root would get its own empty cache and re-extract.
    cache_root = Path(args.cache_root) if args.cache_root else output_root / "feature_cache"
    overrides = {"cache": {"enabled": True, "root_dir": str(cache_root)}}
    print(f"Feature cache (shared across seeds): {cache_root}", flush=True)

    measured_values: dict[str, list[float]] = {metric: [] for metric in reported_metrics}
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        config = benchmark.build_config(
            **axes,
            dataset_csv=manifest.dataset_csv,
            splits_csv=manifest.splits_csv,
            output_root=seed_root,
            seed=seed,
            overrides=overrides,
        )
        Pipeline(config).run()
        metrics = _require_reported_scores(benchmark, benchmark.score(seed_root))
        for metric in reported_metrics:
            measured_values[metric].append(float(metrics[metric]))

    measured = {
        metric: statistics.fmean(values) for metric, values in measured_values.items()
    }
    if getattr(args, "record", False):
        # Key off the gate row, or the external row for an external-only benchmark (hest).
        record_row = _record_reference_row(benchmark, axes, row)
        if record_row is not None:
            key_axes = _record_axes(benchmark, axes, seed_root)
            for metric in reported_metrics:
                values = measured_values[metric]
                std = statistics.stdev(values) if len(values) > 1 else 0.0
                _record_result(
                    benchmark,
                    record_row,
                    measured[metric],
                    std=std,
                    n_seeds=len(seeds),
                    # seed_root is the last seed's output; every seed shares the varied
                    # axes, so it identifies the cell for every Reported metric.
                    key_axes=key_axes,
                    metric=metric,
                    run_dir=seed_root,
                )
        else:
            print("  (no reference row to key --record on; nothing recorded)")
    return _report_measured_metrics(measured)


def _preflight_reproduce_panel(benchmark, encoders: list[str]) -> None:
    """Resolve one concrete Benchmark's healthy Encoder panel before curation."""
    import tempfile

    from slide2vec.encoders import resolve_encoder_capabilities

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
            capabilities = resolve_encoder_capabilities(encoder)
            config = benchmark.build_config(
                encoder=encoder,
                dataset_csv=dataset_csv,
                splits_csv=splits_csv,
                output_root=root / encoder,
                seed=benchmark.canonical_seeds[0],
                overrides=overrides,
            )
            if config.encoder is None or config.encoder.name != capabilities.name:
                resolved = None if config.encoder is None else config.encoder.name
                raise ValueError(
                    f"Benchmark builder resolved encoder {resolved!r}, expected "
                    f"{capabilities.name!r}."
                )


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


def _panel_runtime_failure_context(exc: RuntimeError) -> str:
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
        if is_family:
            print(
                "Error: --encoders requires one concrete registered Benchmark, "
                f"not the {args.name!r} family.",
                file=sys.stderr,
            )
            return 2
        benchmark = targets[0]
        try:
            _preflight_reproduce_panel(benchmark, encoders)
        except (KeyError, ValueError) as exc:
            print(f"Error: Encoder panel preflight failed: {exc}", file=sys.stderr)
            return 2
        try:
            manifest = _reproduce_manifest(benchmark, args)
        except _MissingReproduceSourceError as exc:
            print(f"Error: {benchmark.name}: {exc}", file=sys.stderr)
            return 2

        output_root = _reproduce_output_root(benchmark, args)
        completed_before = _completed_run_dirs(output_root)
        failures: list[tuple[str, str]] = []
        completed_cells = 0
        for encoder in encoders:
            cell_args = argparse.Namespace(**vars(args))
            cell_args.encoder = encoder
            cell_args.encoders = None
            try:
                code = _reproduce_one(
                    benchmark,
                    cell_args,
                    manifest=manifest,
                )
            except RuntimeError as exc:
                failures.append((encoder, _panel_runtime_failure_context(exc)))
                continue
            if code:
                return code
            completed_cells += 1

        leaderboard_code = 0
        completed_during_panel = _completed_run_dirs(output_root) - completed_before
        if completed_cells or completed_during_panel:
            if failures:
                print(
                    f"PARTIAL Encoder panel: {completed_cells}/{len(encoders)} cells "
                    "completed; rendering the canonical Leaderboard from completed "
                    "Runs. Completed Runs remain valid.",
                    flush=True,
                )
            leaderboard_code = _cmd_leaderboard(
                _plural_leaderboard_args(benchmark, output_root)
            )
        elif failures:
            print(
                f"Encoder panel: 0/{len(encoders)} cells completed; no Leaderboard "
                "was written.",
                flush=True,
            )

        if failures:
            print(
                f"Encoder panel runtime failures ({len(failures)}):",
                file=sys.stderr,
            )
            for encoder, context in failures:
                print(f"  - {encoder}: {context}", file=sys.stderr)
            return max(1, leaderboard_code)
        return leaderboard_code

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
        help="Run an ordered Encoder panel, then write the cross-encoder Leaderboard.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Append the measured number + provenance to the results ledger "
        "(soma/benchmarks/results/<name>.csv) so 'reproduced' becomes a committed fact.",
    )
    parser.set_defaults(func=_cmd_reproduce)
    return parser


def _cmd_prepare_pathorob(args: argparse.Namespace) -> None:
    from soma.pathorob import prepare_pathorob

    prepared = prepare_pathorob(args.raw_root, rebuild=args.rebuild)
    counts = ", ".join(f"{cohort.name}: {cohort.rows}" for cohort in prepared)
    print(f"Prepared PathoROB data under {args.raw_root} ({counts}).")


def _build_prepare_pathorob_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soma prepare-pathorob",
        description="Acquire and decode the pinned PathoROB tile sources.",
    )
    parser.add_argument("raw_root", type=Path, help="Destination prepared-data root.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Deliberately replace a partial or revision-mismatched destination.",
    )
    parser.set_defaults(func=_cmd_prepare_pathorob)
    return parser


def _print_top_level_help() -> None:
    print(
        "usage: soma CONFIG\n"
        "       soma list {encoders,aggregators,decoders,pixel-classifiers,tasks,benchmarks} [--level {tile,slide,patient}]\n"
        "       soma prepare-pathorob RAW_ROOT [--rebuild]\n"
        "       soma reproduce NAME [--from-run-dir DIR] [--seeds N] [--raw-root DIR]\n"
        "       soma leaderboard [NAME] --root OUTPUT_ROOT [--vary AXIS] [--fix AXIS=VALUE] [--like DIR]\n"
        "\n"
        "commands:\n"
        "  CONFIG       run a pipeline from a YAML config file\n"
        "  list         list public model/component/benchmark registries\n"
        "  prepare-pathorob  acquire and decode the pinned PathoROB tile sources\n"
        "  reproduce    curate → run → score a registered benchmark, check its tolerance band\n"
        "  leaderboard  render a faceted view over the run dirs under an output root\n"
        "\n"
        "examples:\n"
        "  soma /path/to/config.yaml\n"
        "  python -m soma /path/to/config.yaml\n"
        "  soma list benchmarks\n"
        "  soma prepare-pathorob /data/pathorob\n"
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

    if args[0] == "prepare-pathorob":
        parser = _build_prepare_pathorob_parser()
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
