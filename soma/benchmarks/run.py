"""Importable benchmark-reproduction orchestration (issue #370).

:func:`run_benchmark` is the public entry point carrying everything ``soma reproduce``
provides — the canonical-seed loop, reference-row resolution and tolerance status,
provenance stamping (git commit, slide2vec/croma versions), and the results-ledger
append — as an importable API. The CLI is a thin caller of the same code, so an external
harness (e.g. the BigPicture FMTF benchmark leaderboard) calling :func:`run_benchmark`
gets byte-identical orchestration guarantees, and can pass ``results_root`` to append
:class:`~soma.benchmarks.registry.MeasuredRow` rows to its own committed ledger instead
of the in-package one.

The helpers here keep the CLI's reporting contract verbatim (stdout/stderr text, exit
codes): a return value of ``0`` is success, ``2`` a usage/protocol error, and — exactly
like the CLI — protocol violations inside one scoring attempt may raise ``SystemExit``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import yaml

from soma.benchmarks.registry import get_benchmark, get_reported_metrics
from soma.benchmarks.spec import BenchmarkSpec


def _pipeline_cls():
    """The training Pipeline class, resolved lazily (importing soma.benchmarks stays light)."""
    from soma.pipeline import Pipeline

    return Pipeline


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


class ReportedScoreError(RuntimeError):
    """A Benchmark scorer omitted metrics required by its public protocol."""


@dataclass(frozen=True)
class MetricResult:
    """One Reported metric aggregated across a benchmark's completed seeds.

    The measurement half of the ledger's :class:`MeasuredRow` (same measured/std/n_seeds
    semantics, minus the reference key and provenance columns) — keep the two in step.
    Mirroring the ledger convention, ``std`` is ``None`` for a ``from_run_dir`` re-score
    (a single historical run has no seed spread) and ``0.0`` for a single-seed loop.
    """

    metric: str
    measured: float
    std: float | None
    n_seeds: int


@dataclass(frozen=True)
class BenchmarkRunResult:
    """Measurements and evidence locations produced by :func:`run_benchmark`.

    ``seed_roots`` holds the per-seed output roots of a canonical-seed run; for a
    ``from_run_dir`` re-score it holds the single resolved run directory that was scored.
    """

    status: int
    metrics: tuple[MetricResult, ...]
    seed_roots: tuple[Path, ...]


def _run_return(
    status: int,
    *,
    return_result: bool,
    metrics: tuple[MetricResult, ...] = (),
    seed_roots: tuple[Path, ...] = (),
) -> int | BenchmarkRunResult:
    """Project the outcome to the CLI int status unless a structured result was asked for."""
    if not return_result:
        return status
    return BenchmarkRunResult(status=status, metrics=metrics, seed_roots=seed_roots)


def _require_reported_scores(
    benchmark,
    scores: dict[str, float],
    *,
    raise_error: bool = False,
) -> dict[str, float]:
    """Fail one scoring attempt if any required Reported metric is absent."""
    missing = [metric for metric in get_reported_metrics(benchmark) if metric not in scores]
    if missing:
        names = ", ".join(repr(metric) for metric in missing)
        message = (
            f"benchmark {benchmark.name!r} score is missing Reported metric(s): {names}."
        )
        if raise_error:
            raise ReportedScoreError(message)
        print(f"Error: {message}", file=sys.stderr)
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
    run_dir: str | Path | None,
    *,
    historical_run: bool,
    runtime_version: Callable[[], str] | None = None,
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
    return "unknown" if historical_run else (runtime_version or _runtime_croma_version)()


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
    *,
    provenance: Callable[[], tuple[str, str, str]] | None = None,
    recorded_croma_version: Callable[..., str] | None = None,
    results_root: str | Path | None = None,
) -> None:
    """Append a reproduced-measurement row to the benchmark's results ledger (``--record``).

    Keys are copied from the primary reference anchor ``row`` (a **gate** row, or an
    **external** row for an external-only benchmark). ``metric`` selects the Reported metric
    being recorded, so secondary metrics share the primary cell key even when they have no
    reference row. Provenance is captured at run time.

    ``key_axes`` fills key columns the reference leaves empty — see :func:`_record_axes`. A
    keyed reference (EVA, HEST) already states them, so this only bites for a broad one;
    values the reference does state always win, since that is the cell being joined.

    ``results_root`` appends to a ledger hosted outside the soma checkout; the
    ``provenance`` / ``recorded_croma_version`` seams let the CLI keep its patchable
    module-level collaborators while defaulting to the implementations here.
    """
    from soma.benchmarks import MeasuredRow, append_result

    key = dict(row.key)
    for axis, value in (key_axes or {}).items():
        if not key.get(axis):
            key[axis] = value

    date, commit, slide2vec_version = (provenance or _provenance)()
    croma_resolver = recorded_croma_version or _recorded_croma_version
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
            croma_resolver(run_dir, historical_run=historical_run)
            if getattr(benchmark, "records_croma_version", False)
            else ""
        ),
        source="soma reproduce --record",
    )
    path = append_result(
        _results_table_name(benchmark),
        measured_row,
        key_order=list(key),
        results_root=results_root,
    )
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
    """The caller did not provide a source from which to obtain a curated Manifest."""


def _reproduce_manifest(benchmark, args, *, family_root: str | None = None):
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


def _reproduce_output_root(benchmark, args, *, family_root: str | None = None) -> Path:
    """The existing output-root contract for one concrete Benchmark."""
    sub = benchmark.name.split("/", 1)[1] if family_root else None
    if args.output_root:
        return Path(args.output_root) / sub if sub else Path(args.output_root)
    return Path.cwd() / "soma_reproduce" / benchmark.name


class _PanelCellRuntimeFailure(RuntimeError):
    """An expected operational failure at a running panel cell's public boundaries."""

    def __init__(self, cause: OSError | RuntimeError | ValueError):
        super().__init__(str(cause))
        self.cause = cause


@contextmanager
def _panel_runtime_boundary(enabled: bool):
    """Type expected Pipeline/scorer failures for plural-panel orchestration only."""
    try:
        yield
    except (OSError, RuntimeError, ValueError) as exc:
        if enabled:
            raise _PanelCellRuntimeFailure(exc) from exc
        raise


def _reproduce_one(
    benchmark,
    args,
    *,
    family_root: str | None = None,
    manifest=None,
    isolate_runtime_failures: bool = False,
    pipeline_cls=None,
    provenance: Callable[[], tuple[str, str, str]] | None = None,
    resolve_manifest=None,
    recorded_croma_version: Callable[..., str] | None = None,
    results_root: str | Path | None = None,
    return_result: bool = False,
) -> int | BenchmarkRunResult:
    """Curate → run → score one benchmark and report any reference comparison.

    ``family_root`` is the family name when this benchmark is one member of a fanned-out
    family (e.g. ``eva`` for ``eva/bach``); it nests the member's raw/curated/output paths
    under a per-dataset subdirectory so sibling members do not collide.

    The keyword seams (``pipeline_cls``, ``provenance``, ``resolve_manifest``,
    ``recorded_croma_version``) default to the implementations in this module; the CLI
    threads its own module-level names through them so its behavior — including
    monkeypatched collaborators — stays byte-identical. ``results_root`` relocates the
    ``--record`` ledger outside the soma checkout.
    """
    resolve_manifest = resolve_manifest or _reproduce_manifest
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
                        provenance=provenance,
                        recorded_croma_version=recorded_croma_version,
                        results_root=results_root,
                    )
            else:
                print("  (no reference row to key --record on; nothing recorded)")
        status = _report_measured_metrics({benchmark.primary_metric: measured, **metrics})
        return _run_return(
            status,
            return_result=return_result,
            metrics=tuple(
                MetricResult(
                    metric=metric, measured=float(metrics[metric]), std=None, n_seeds=1
                )
                for metric in reported_metrics
            ),
            seed_roots=(_resolve_run_dir(args.from_run_dir),),
        )

    import statistics

    if manifest is None:
        try:
            manifest = resolve_manifest(benchmark, args, family_root=family_root)
        except _MissingReproduceSourceError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return _run_return(2, return_result=return_result)
    output_root = _reproduce_output_root(benchmark, args, family_root=family_root)
    # Feature extraction is seed-independent (the encoder is frozen and the cache key is
    # derived from encoder + tile content, not the seed), so every seed shares one cache
    # root and extraction runs once. --cache-root relocates it (e.g. to fast local storage);
    # by default it sits beside the run outputs, shared across seeds. Without a shared root
    # each seed's per-seed output_root would get its own empty cache and re-extract.
    cache_root = Path(args.cache_root) if args.cache_root else output_root / "feature_cache"
    overrides = {"cache": {"enabled": True, "root_dir": str(cache_root)}}
    print(f"Feature cache (shared across seeds): {cache_root}", flush=True)

    pipeline_cls = pipeline_cls or _pipeline_cls()
    measured_values: dict[str, list[float]] = {metric: [] for metric in reported_metrics}
    seed_roots: list[Path] = []
    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"
        seed_roots.append(seed_root)
        config = benchmark.build_config(
            **axes,
            dataset_csv=manifest.dataset_csv,
            splits_csv=manifest.splits_csv,
            output_root=seed_root,
            seed=seed,
            overrides=overrides,
        )
        with _panel_runtime_boundary(isolate_runtime_failures):
            pipeline_cls(config).run()
            metrics = _require_reported_scores(
                benchmark,
                benchmark.score(seed_root),
                raise_error=isolate_runtime_failures,
            )
            for metric in reported_metrics:
                measured_values[metric].append(float(metrics[metric]))

    measured = {
        metric: statistics.fmean(values) for metric, values in measured_values.items()
    }
    spread = {
        metric: statistics.stdev(values) if len(values) > 1 else 0.0
        for metric, values in measured_values.items()
    }
    if getattr(args, "record", False):
        # Key off the gate row, or the external row for an external-only benchmark (hest).
        record_row = _record_reference_row(benchmark, axes, row)
        if record_row is not None:
            key_axes = _record_axes(benchmark, axes, seed_root)
            for metric in reported_metrics:
                _record_result(
                    benchmark,
                    record_row,
                    measured[metric],
                    std=spread[metric],
                    n_seeds=len(seeds),
                    # seed_root is the last seed's output; every seed shares the varied
                    # axes, so it identifies the cell for every Reported metric.
                    key_axes=key_axes,
                    metric=metric,
                    run_dir=seed_root,
                    provenance=provenance,
                    recorded_croma_version=recorded_croma_version,
                    results_root=results_root,
                )
        else:
            print("  (no reference row to key --record on; nothing recorded)")
    status = _report_measured_metrics(measured)
    return _run_return(
        status,
        return_result=return_result,
        metrics=tuple(
            MetricResult(
                metric=metric,
                measured=measured[metric],
                std=spread[metric],
                n_seeds=len(measured_values[metric]),
            )
            for metric in reported_metrics
        ),
        seed_roots=tuple(seed_roots),
    )


def run_benchmark(
    name: str,
    *,
    encoder: str | None = None,
    seeds: int | None = None,
    record: bool = False,
    results_root: str | Path | None = None,
    from_run_dir: str | Path | None = None,
    raw_root: str | Path | None = None,
    curated_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    return_result: bool = False,
) -> int | BenchmarkRunResult:
    """Reproduce one registered benchmark exactly as ``soma reproduce`` does (issue #370).

    The importable counterpart of the CLI's reproduce path, carrying the same guarantees:

    - **Seeds** — runs the benchmark's canonical seed set by default; ``seeds=N`` runs the
      fast subset ``0..N-1`` (``seeds=1`` is the quickest smoke), like ``--seeds N``.
    - **Reference comparison** — resolves the packaged reference row for the primary
      metric at the requested axes and reports ``REFERENCE OK`` / ``POTENTIAL DRIFT``
      within the row's tolerance band (external rows stay contextual, never gating).
    - **Provenance** — ``record=True`` stamps each ledger row with the date, the soma git
      commit, and the installed slide2vec (and, for representation benchmarks, croma)
      versions, exactly as ``--record`` does.
    - **Record** — appends :class:`~soma.benchmarks.registry.MeasuredRow` rows to the
      benchmark's results ledger; ``results_root`` points that ledger at a directory
      outside the soma checkout (``<results_root>/<table>.csv``) so an external repository
      can host its own committed ledger. Without it, the in-package table is used.

    ``name`` must be a single registered benchmark (``ocelot``, ``eva/bach`` …); an unknown
    name raises ``KeyError`` listing the registered names. The remaining keywords mirror
    the CLI flags of the same name (``from_run_dir`` re-scores an existing run with no
    training; one of ``raw_root`` / ``curated_dir`` / ``from_run_dir`` must be given).

    By default, returns the CLI's exit status (``0`` success, ``2`` usage/protocol error).
    Pass ``return_result=True`` to receive the aggregated Reported metrics and seed output
    roots as a :class:`BenchmarkRunResult`. Both modes print the same report; like the CLI,
    a scorer omitting a required Reported metric terminates via ``SystemExit``.
    """
    benchmark = get_benchmark(name)
    options = SimpleNamespace(
        encoder=encoder,
        seeds=seeds,
        from_run_dir=from_run_dir,
        raw_root=raw_root,
        curated_dir=curated_dir,
        out_dir=out_dir,
        output_root=output_root,
        cache_root=cache_root,
        record=record,
    )
    return _reproduce_one(
        benchmark,
        options,
        results_root=results_root,
        return_result=return_result,
    )


def run_benchmark_spec(
    spec: BenchmarkSpec,
    *,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    encoder: str,
    output_root: str | Path,
    cache_root: str | Path | None = None,
    seeds: Sequence[int] | None = None,
) -> BenchmarkRunResult:
    """Execute an external benchmark specification without registry concerns."""
    import statistics

    if seeds is not None and (
        not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes))
    ):
        raise ValueError("seeds must be a non-empty sequence of unique integers")
    selected_seeds = tuple(spec.canonical_seeds if seeds is None else seeds)
    if (
        not selected_seeds
        or not all(isinstance(seed, int) for seed in selected_seeds)
        or len(set(selected_seeds)) != len(selected_seeds)
    ):
        raise ValueError("seeds must be a non-empty sequence of unique integers")
    output_root = Path(output_root)
    shared_cache_root = (
        Path(cache_root) if cache_root is not None else output_root / "feature_cache"
    )
    overrides = {"cache": {"enabled": True, "root_dir": str(shared_cache_root)}}
    reported_metrics = get_reported_metrics(spec)
    measured_values: dict[str, list[float]] = {metric: [] for metric in reported_metrics}
    seed_roots: list[Path] = []
    pipeline_cls = _pipeline_cls()

    for seed in selected_seeds:
        seed_root = output_root / f"seed_{seed}"
        seed_roots.append(seed_root)
        config = spec.build_config(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=seed_root,
            seed=seed,
            overrides=overrides,
            encoder=encoder,
        )
        pipeline_cls(config).run()
        scores = spec.score(seed_root)
        missing = [metric for metric in reported_metrics if metric not in scores]
        if missing:
            names = ", ".join(repr(metric) for metric in missing)
            raise ValueError(
                f"benchmark spec {spec.name!r} score is missing Reported metric(s): "
                f"{names}."
            )
        for metric in reported_metrics:
            measured_values[metric].append(float(scores[metric]))

    return BenchmarkRunResult(
        status=0,
        metrics=tuple(
            MetricResult(
                metric=metric,
                measured=statistics.fmean(values),
                std=statistics.stdev(values) if len(values) > 1 else 0.0,
                n_seeds=len(values),
            )
            for metric, values in measured_values.items()
        ),
        seed_roots=tuple(seed_roots),
    )
