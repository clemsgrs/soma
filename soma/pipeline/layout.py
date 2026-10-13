"""The managed run layout: where a run writes, what it records, and when it may resume.

Absorbs the former ``soma.output_layout`` (experiment identity, run metadata, run index,
test-identity registry) and ``soma.provenance`` (git state of the soma checkout), and
adds the pieces :class:`~soma.pipeline.Pipeline` composes: :class:`RunRecorder` (the
run's status lifecycle), the plan and manifest resume drift guards, the fold
directory layout and the fold-completeness rule (``metrics.json`` + ``bundle/``).

Everything here reads the run's :class:`~soma.plan.Plan`: the experiment identity is a
digest of the plan (minus where and how a launch runs) and of the train/tune content of
the cohort files, and a resume must match the saved ``plan.yaml``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from getpass import getuser
from pathlib import Path
from typing import Any

import yaml

from soma._internal.atomic_io import atomic_write_text
from soma.config import ExperimentConfig
from soma.plan import Plan, comparable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Git provenance of the soma code that produced a run or a recorded benchmark row
# ---------------------------------------------------------------------------
#
# Resolved from the installed package directory, never from the working directory: a run
# launched from some project folder must record the soma checkout it imported, and a
# wheel install (site-packages, no checkout) records nothing rather than the state of
# whatever repository happens to enclose the environment.


@dataclass(frozen=True)
class GitState:
    sha: str | None
    dirty: bool | None

    @property
    def short(self) -> str | None:
        return self.sha[:7] if self.sha else None


def _soma_package_dir() -> Path:
    import soma

    return Path(soma.__file__).resolve().parent


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip()


def soma_git_state() -> GitState:
    """``GitState`` of the checkout the imported ``soma`` package lives in.

    ``sha`` is the full ``HEAD`` commit; ``dirty`` is ``True`` when *tracked* files under
    the checkout have modifications (untracked scratch such as run outputs does not
    count). Both are ``None`` when the package is not inside a git checkout that owns
    it — a wheel install, or a package directory copied out of its repository.
    """
    package_dir = _soma_package_dir()
    toplevel = _git(package_dir, "rev-parse", "--show-toplevel")
    if not toplevel:
        return GitState(sha=None, dirty=None)
    # The checkout must be soma's own: an environment created inside an unrelated
    # repository would otherwise stamp that repository's commit onto soma runs.
    try:
        package_dir.relative_to(Path(toplevel).resolve())
    except ValueError:
        return GitState(sha=None, dirty=None)
    sha = _git(package_dir, "rev-parse", "HEAD")
    if not sha:
        return GitState(sha=None, dirty=None)
    status = _git(package_dir, "status", "--porcelain", "--untracked-files=no")
    return GitState(sha=sha, dirty=None if status is None else bool(status))


_SEMANTIC_IDENTITY_EXCLUDED_PATH_COLUMN_ALLOWLIST = frozenset(
    {
        "image_path",
        "mask_path",
        "label_mask_path",
        "label_mask_crop_path",
        "points_path",
        "ignore_mask_path",
        "coordinates_path",
    }
)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slugify(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        elif cleaned and cleaned[-1] != "-":
            cleaned.append("-")
    slug = "".join(cleaned).strip("-")
    return slug or "unknown"


def _stable_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


#: Bump when the canonical payload changes shape. Recorded on every run and
#: experiment manifest so ids minted under different rules are never compared as equal.
#: v8 replaced the extraction blocks with one ``features`` block holding the slide2vec
#: and hs2p dataclasses (ADR 0019); v9 hashes the resolved plan (ADR 0020).
IDENTITY_VERSION = 9

#: Plan fields left out of the experiment identity, by section: where and how a launch
#: runs (the whole ``run`` section, so the seed too: one experiment, several seeded
#: runs), the cohort files (their train/tune content is hashed instead), loader
#: throughput, the device, and what is saved rather than measured.
_IDENTITY_EXCLUDED: dict[str, tuple[str, ...]] = {
    "cohort": ("records", "splits"),
    "loaders": ("num_workers", "pin_memory", "persistent_workers"),
    "trainer": ("device",),
    "evaluation": ("overwrite_test", "save_overlays", "save_probabilities", "save_heatmaps"),
}


def _plan_identity(plan: Plan) -> dict[str, Any]:
    """The experiment-defining part of a plan (see :data:`_IDENTITY_EXCLUDED`)."""
    payload = plan.to_dict()
    payload.pop("run", None)
    payload["features"] = _features_identity(plan)
    for section, names in _IDENTITY_EXCLUDED.items():
        for name in names:
            payload[section].pop(name, None)
    return payload


def _split_assignments(splits_path: Path, keep) -> list[list[str]]:
    """Sorted ``[fold, split, sample_id]`` rows of a splits.csv whose split ``keep(split)``.

    The byte-stable, order-independent projection of a split manifest used by both the
    training and test manifest digests (issue #247).
    """
    rows = _read_csv_rows(splits_path)
    selected = [
        [str(row.get("fold", "")), str(row.get("split", "")), str(row.get("sample_id", ""))]
        for row in rows
        if keep(str(row.get("split", "")))
    ]
    selected.sort()
    return selected


def _dataset_rows_digest(dataset_path: Path, sample_ids: set[str]) -> str:
    """sha256 over semantic dataset rows for ``sample_ids`` (sorted by id)."""
    rows = [
        {
            str(k): ("" if v is None else str(v))
            for k, v in row.items()
            if str(k) not in _SEMANTIC_IDENTITY_EXCLUDED_PATH_COLUMN_ALLOWLIST
        }
        for row in _read_csv_rows(dataset_path)
        if str(row.get("sample_id", "")) in sample_ids
    ]
    rows.sort(key=lambda row: row.get("sample_id", ""))
    return hashlib.sha256(_stable_json({"rows": rows}).encode("utf-8")).hexdigest()


def _manifest_slice_digests(dataset_path: Path, splits_path: Path, keep) -> tuple[str, str]:
    """Return ``(dataset_digest, splits_digest)`` for the split rows selected by ``keep``.

    The dataset digest covers only the dataset rows referenced by the selected split
    assignments, so it is invariant to rows for samples outside the slice.
    """
    assignments = _split_assignments(splits_path, keep)
    sample_ids = {row[2] for row in assignments}
    dataset_digest = _dataset_rows_digest(dataset_path, sample_ids)
    splits_digest = hashlib.sha256(
        _stable_json({"assignments": assignments}).encode("utf-8")
    ).hexdigest()
    return dataset_digest, splits_digest


def _is_training_split(name: str) -> bool:
    return name in ("train", "tune")


def _is_test_split(name: str) -> bool:
    return name.startswith("test")


def test_identity_digest(plan: Plan) -> str:
    """Digest identifying WHICH test set a run is scored against (issue #247).

    The test-slice counterpart of the training digest: a byte-stable sha256 over the
    ``test*`` split assignments and the dataset rows they reference. Two runs of the same
    model (same ``experiment_id``) scored on different test sets share an experiment dir
    but differ here, so their results can be namespaced and a re-score of an already-scored
    test set refused rather than silently clobbered. Empty when a run has no ``test*`` rows.
    """
    dataset_path = Path(plan.cohort.records).resolve()
    splits_path = Path(plan.cohort.splits).resolve()
    dataset_digest, splits_digest = _manifest_slice_digests(
        dataset_path, splits_path, _is_test_split
    )
    return hashlib.sha256(
        _stable_json({"dataset": dataset_digest, "splits": splits_digest}).encode("utf-8")
    ).hexdigest()


def _features_identity(plan: Plan, *, cache_dtype: bool = False) -> dict[str, Any]:
    """The experiment-defining part of a run's features: every field of the slide2vec /
    hs2p dataclasses but the execution resources (only the precision counts).

    The extraction and a grid run's dense options are hashed as the run resolves them
    (the geometry left to the encoder preset or to ``dense`` filled in), so two plans that
    extract the same features share one identity however the geometry was spelled.
    """
    from soma.extract.cache.keys import _jsonable
    from soma.pipeline.extraction import resolve_pipeline_dense, resolve_pipeline_preprocessing

    features = plan.features
    encoder = None
    if features.encoder is not None:
        encoder = {"name": features.encoder.name, "params": _jsonable(features.encoder.params)}
    extraction = resolve_pipeline_preprocessing(plan)
    dense = None
    if plan.shape == "grid":
        dense = features.dense
        if dense is None and extraction.tiling is not None:
            dense = resolve_pipeline_dense(plan)
    cache: dict[str, Any] = {
        "enabled": features.cache.enabled,
        "reuse_policy": features.cache.reuse_policy,
    }
    if cache_dtype:
        cache["dtype"] = features.cache.dtype
    return {
        "kind": features.kind,
        "encoder": encoder,
        "extraction": _jsonable(extraction),
        "dense": None if dense is None else _jsonable(dense),
        "spacing_policy": features.spacing_policy,
        "precision": features.execution.precision,
        "cache": cache,
    }


def canonical_experiment_payload(plan: Plan) -> dict[str, Any]:
    """What the experiment id digests: the plan's experiment-defining fields and the
    content of the cohort files' {train, tune} slice (the representation split of a
    CRoMa run).

    Experiment identity is invariant to the *test* set (issue #247) and to the cohort
    files' paths: dropping in a new or official test set leaves the experiment id, and the
    leaderboard triple, unchanged, so a frozen checkpoint is re-scored rather than
    retrained. Full-file and test-slice provenance is recorded on the run metadata.
    """
    dataset_path = Path(plan.cohort.records).resolve()
    splits_path = Path(plan.cohort.splits).resolve()
    representation = plan.evaluation.representation
    if representation is not None:
        dataset_digest, splits_digest = _manifest_slice_digests(
            dataset_path, splits_path, lambda name: name == representation.split
        )
        return {
            "identity_version": IDENTITY_VERSION,
            "dataset": {"checksum": dataset_digest},
            "splits": {"checksum": splits_digest},
            "features": _features_identity(plan, cache_dtype=True),
            "representation": asdict(representation),
        }
    dataset_digest, splits_digest = _manifest_slice_digests(dataset_path, splits_path, _is_training_split)
    return {
        "identity_version": IDENTITY_VERSION,
        "dataset": {"checksum": dataset_digest},
        "splits": {"checksum": splits_digest},
        **_plan_identity(plan),
    }


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    slug: str
    short_hash: str
    experiment_dirname: str
    dataset_path: Path
    splits_path: Path
    dataset_checksum: str
    splits_checksum: str
    canonical_spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "identity_version": IDENTITY_VERSION,
            "slug": self.slug,
            "short_hash": self.short_hash,
            "experiment_dirname": self.experiment_dirname,
            "dataset_path": str(self.dataset_path),
            "splits_path": str(self.splits_path),
            "dataset_checksum": self.dataset_checksum,
            "splits_checksum": self.splits_checksum,
            "canonical_spec": self.canonical_spec,
        }


def build_experiment_spec(plan: Plan) -> ExperimentSpec:
    canonical_spec = canonical_experiment_payload(plan)
    digest = hashlib.sha256(_stable_json(canonical_spec).encode("utf-8")).hexdigest()
    short_hash = digest[:12]
    dataset_name = _slugify(Path(plan.cohort.records).stem)
    encoder = plan.features.encoder
    encoder_name = _slugify(encoder.name if encoder is not None else "precomputed")
    params = plan.learner.params if plan.learner is not None else {}
    trained = params.get("aggregator") or params.get("decoder")
    aggregator_name = _slugify(trained.name if trained is not None else "slide")
    head = params.get("head")
    task_name = _slugify(head.name if head is not None else (plan.learner.name if plan.learner else "croma"))
    slug = f"{dataset_name}-{encoder_name}-{aggregator_name}-{task_name}_{short_hash}"
    return ExperimentSpec(
        experiment_id=digest,
        slug=slug,
        short_hash=short_hash,
        experiment_dirname=slug,
        dataset_path=Path(plan.cohort.records).resolve(),
        splits_path=Path(plan.cohort.splits).resolve(),
        dataset_checksum=str(canonical_spec["dataset"]["checksum"]),
        splits_checksum=str(canonical_spec["splits"]["checksum"]),
        canonical_spec=canonical_spec,
    )


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    experiment_id: str
    status: str
    started_at: str
    finished_at: str | None
    seed: int
    wandb_id: str | None
    wandb_url: str | None
    git_sha: str | None
    git_dirty: bool | None
    hostname: str | None
    username: str | None
    resolved_output_dir: Path
    summary_metrics: dict[str, float]
    error: str | None = None
    # Bounded provenance: EXACTLY {soma, torch, cuda} (issue #213). Deeper env/GPU-model
    # capture is deliberately out of scope; a benchmark may show its own reference env.
    environment: dict[str, str] = field(default_factory=dict)
    # Test-set provenance (issue #247). The experiment_id / leaderboard triple are
    # test-invariant, so which test set THIS run scored is recorded here, not in the
    # shared experiment.json: ``test_checksum`` is the test-identity digest and the
    # ``*_file_checksum`` are the whole-file digests (which DO see test rows).
    test_checksum: str = ""
    dataset_file_checksum: str = ""
    splits_file_checksum: str = ""
    comparison_key: dict[str, Any] = field(default_factory=dict)
    representation_provenance: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "identity_version": IDENTITY_VERSION,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at or "",
            "seed": self.seed,
            "wandb_id": self.wandb_id or "",
            "wandb_url": self.wandb_url or "",
            "git_sha": self.git_sha or "",
            "git_dirty": "" if self.git_dirty is None else str(self.git_dirty).lower(),
            "hostname": self.hostname or "",
            "username": self.username or "",
            "resolved_output_dir": str(self.resolved_output_dir),
            "summary_metrics": self.summary_metrics,
            "environment": dict(self.environment),
            "test_checksum": self.test_checksum,
            "dataset_file_checksum": self.dataset_file_checksum,
            "splits_file_checksum": self.splits_file_checksum,
            "error": self.error or "",
        }
        payload["comparison_key"] = dict(self.comparison_key)
        if self.representation_provenance:
            payload["representation_provenance"] = dict(self.representation_provenance)
        return payload

    def with_updates(self, **updates: Any) -> "RunMetadata":
        return replace(self, **updates)


def capture_environment() -> dict[str, str]:
    """Bounded provenance stamp: EXACTLY ``{soma, torch, cuda}`` (issue #213).

    Nothing further is captured — no OS/GPU-model probing, no clean-tree gate — so the
    stamp stays cheap and deterministic. ``torch``/``cuda`` fall back to empty strings when
    torch is not importable (e.g. a curation-only environment).
    """
    import soma

    environment = {"soma": str(getattr(soma, "__version__", ""))}
    try:
        import torch

        environment["torch"] = str(torch.__version__)
        environment["cuda"] = str(torch.version.cuda or "")
    except Exception:  # torch optional at metadata time
        environment["torch"] = ""
        environment["cuda"] = ""
    return environment


def _timestamp_now() -> str:
    return datetime.now().astimezone().isoformat()


def make_run_id(*, when: datetime | None = None, wandb_id: str | None = None) -> str:
    current = when.astimezone() if when is not None else datetime.now().astimezone()
    suffix = wandb_id or "local"
    return f"{current:%Y-%m-%d_%H-%M-%S}__{suffix}"


def latest_existing_run_id(experiment_dir: Path) -> str | None:
    """Return the most recent run id under ``experiment_dir/runs``, or None.

    Run ids are ``YYYY-MM-DD_HH-MM-SS__<suffix>`` so a lexical max is the newest
    run — the one a bare ``resume`` reuses (issue #244).
    """
    runs_dir = experiment_dir / "runs"
    if not runs_dir.is_dir():
        return None
    ids = sorted(child.name for child in runs_dir.iterdir() if child.is_dir())
    return ids[-1] if ids else None


def _resolve_run_id(
    plan: Plan, experiment_dir: Path, *, when: datetime | None = None
) -> str:
    """Pick the run id for this launch, honoring the resume directives (issue #244).

    Precedence: an explicit ``run_id`` pins that run (resume into it if present,
    else create it under that name); otherwise ``resume`` reuses the latest
    existing run for this experiment (error if none); otherwise a fresh
    timestamped id — the unchanged default.
    """
    if plan.run.run_id:
        return plan.run.run_id
    if plan.run.resume:
        latest = latest_existing_run_id(experiment_dir)
        if latest is None:
            raise FileNotFoundError(
                f"resume=True but no prior run exists under {experiment_dir / 'runs'}; "
                "nothing to resume. Launch without resume to start a fresh run."
            )
        return latest
    return make_run_id(when=when)


@dataclass(frozen=True)
class ManagedOutputPaths:
    output_root: Path
    experiment: ExperimentSpec
    experiment_dir: Path
    run_id: str
    run_dir: Path
    feature_dir: Path
    index_dir: Path


def resolve_managed_output_paths(
    plan: Plan,
    *,
    run_id: str | None = None,
    when: datetime | None = None,
) -> ManagedOutputPaths:
    output_root = Path(plan.run.output_dir).resolve()
    experiment = build_experiment_spec(plan)
    experiment_dir = output_root / "experiments" / experiment.experiment_dirname
    # An explicit run_id arg (e.g. tests, leaderboard replay) pins the run as-is;
    # otherwise consult the config's resume directives (issue #244).
    resolved_run_id = run_id or _resolve_run_id(plan, experiment_dir, when=when)
    run_dir = experiment_dir / "runs" / resolved_run_id
    return ManagedOutputPaths(
        output_root=output_root,
        experiment=experiment,
        experiment_dir=experiment_dir,
        run_id=resolved_run_id,
        run_dir=run_dir,
        feature_dir=run_dir / "features",
        index_dir=output_root / "indexes",
    )


def create_run_metadata(
    *,
    plan: Plan,
    experiment: ExperimentSpec,
    run_dir: Path,
    run_id: str,
    status: str,
    started_at: str | None = None,
    finished_at: str | None = None,
    summary_metrics: dict[str, float] | None = None,
    error: str | None = None,
    wandb_id: str | None = None,
    wandb_url: str | None = None,
) -> RunMetadata:
    git_state = soma_git_state()
    dataset_path = Path(plan.cohort.records)
    splits_path = Path(plan.cohort.splits)
    if plan.evaluation.representation is not None:
        comparison_key = {
            "kind": "representation",
            "dataset_checksum": experiment.dataset_checksum,
            "splits_checksum": experiment.splits_checksum,
            "representation": asdict(plan.evaluation.representation),
        }
        from importlib.metadata import version

        croma_version = version("croma")
        representation_provenance = {"croma": croma_version}
    else:
        head = plan.learner.params.get("head") if plan.learner is not None else None
        comparison_key = {
            "kind": "task",
            "dataset_checksum": experiment.dataset_checksum,
            "splits_checksum": experiment.splits_checksum,
            "task": head.name if head is not None else "",
        }
        representation_provenance = {}
    return RunMetadata(
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        status=status,
        started_at=started_at or _timestamp_now(),
        finished_at=finished_at,
        seed=plan.run.seed,
        wandb_id=wandb_id,
        wandb_url=wandb_url,
        git_sha=git_state.sha,
        git_dirty=git_state.dirty,
        hostname=socket.gethostname() or None,
        username=getuser() or None,
        resolved_output_dir=run_dir.resolve(),
        summary_metrics=summary_metrics or {},
        error=error,
        environment=capture_environment(),
        test_checksum=test_identity_digest(plan),
        dataset_file_checksum=_sha256_file(dataset_path) if dataset_path.is_file() else "",
        splits_file_checksum=_sha256_file(splits_path) if splits_path.is_file() else "",
        comparison_key=comparison_key,
        representation_provenance=representation_provenance,
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=False))


def write_experiment_metadata(path: Path, experiment: ExperimentSpec) -> None:
    _write_yaml(path / "experiment.yaml", experiment.to_dict())
    _write_json(path / "experiment.json", experiment.to_dict())


def write_run_metadata(path: Path, metadata: RunMetadata) -> None:
    payload = metadata.to_dict()
    payload["summary_metrics"] = metadata.summary_metrics
    _write_yaml(path / "run.yaml", payload)


def _ensure_csv_field_size_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10
            if limit <= 0:
                return


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    _ensure_csv_field_size_limit()
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue(), newline="")


RUN_INDEX_FIELDNAMES = [
    "run_id",
    "experiment_id",
    "status",
    "started_at",
    "finished_at",
    "seed",
    "wandb_id",
    "git_sha",
    "run_dir",
    "error",
]


def _run_index_row(metadata: RunMetadata) -> dict[str, str]:
    return {
        "run_id": metadata.run_id,
        "experiment_id": metadata.experiment_id,
        "status": metadata.status,
        "started_at": metadata.started_at,
        "finished_at": metadata.finished_at or "",
        "seed": str(metadata.seed),
        "wandb_id": metadata.wandb_id or "",
        "git_sha": metadata.git_sha or "",
        "run_dir": str(metadata.resolved_output_dir),
        "error": metadata.error or "",
    }


def update_run_index(path: Path, metadata: RunMetadata) -> None:
    """Append one row per status change; readers keep the last row per ``run_id``.

    Appending a single line in ``O_APPEND`` mode is atomic for practical line sizes, so
    concurrent runs under one output root no longer race a read-modify-rewrite of the
    whole file (which could drop each other's rows). :func:`read_run_index` dedupes and
    :func:`compact_run_index` rewrites the file on demand.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = _run_index_row(metadata)
    if path.is_file() and path.stat().st_size > 0:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        if header != RUN_INDEX_FIELDNAMES:
            # Schema changed under an existing index: rewrite once so the appended
            # rows share the new header.
            compact_run_index(path)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_INDEX_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def read_run_index(path: Path) -> list[dict[str, str]]:
    """Rows of ``runs.csv`` deduplicated by ``run_id`` (last appended row wins), sorted."""
    latest: dict[str, dict[str, str]] = {}
    for row in _read_csv_rows(path):
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        latest[run_id] = {name: str(row.get(name, "") or "") for name in RUN_INDEX_FIELDNAMES}
    return [latest[run_id] for run_id in sorted(latest)]


def compact_run_index(path: Path) -> int:
    """Rewrite ``runs.csv`` with one row per ``run_id``; returns the rows kept."""
    rows = read_run_index(path)
    _write_csv_rows(path, RUN_INDEX_FIELDNAMES, rows)
    return len(rows)


def has_successful_run(experiment_dir: Path) -> bool:
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        return False
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        run_yaml = run_dir / "run.yaml"
        if not run_yaml.exists():
            continue
        try:
            payload = yaml.safe_load(run_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if payload.get("status") == "completed":
            return True
    return False


def update_latest_pointer(experiment_dir: Path, run_dir: Path) -> None:
    latest_path = experiment_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    target = os.path.relpath(run_dir, experiment_dir)
    latest_path.symlink_to(target)


# ---------------------------------------------------------------------------
# Test-results registry (issue #247) — namespace results by test identity so a
# checkpoint scored against several test sets does not silently clobber, and refuse a
# re-score of an already-scored test set unless overwrite is requested.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestResultDecision:
    """Outcome of reserving a test-identity slot in a run's test-results registry."""

    test_digest: str
    skipped: bool  # True ⇒ a prior result exists and overwrite was not requested
    prior: dict[str, Any] | None


def test_results_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "test_results.json"


def read_test_results(run_dir: str | Path) -> dict[str, Any]:
    """Load a run's ``test_results.json`` registry (``{test_digest: {...}}``), or ``{}``."""
    path = test_results_path(run_dir)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def register_test_result(
    run_dir: str | Path,
    test_digest: str,
    *,
    split_names: list[str] | tuple[str, ...] = (),
    run_id: str | None = None,
    overwrite: bool = False,
) -> TestResultDecision:
    """Reserve/record the ``test_digest`` slot for a run, guarding against clobber (#247).

    A run dir may score several test sets over its life (same split *name* ``test`` but
    different rows). Each is recorded under its ``test_digest`` so distinct test sets
    coexist. Re-scoring an already-recorded test identity is refused (``skipped=True``,
    the caller logs loudly and skips) unless ``overwrite`` is set, so a prior result is
    never silently overwritten. A distinct ``test_digest`` is always recorded alongside
    any existing ones.
    """
    run_dir = Path(run_dir)
    registry = read_test_results(run_dir)
    prior = registry.get(test_digest)
    if prior is not None and not overwrite:
        return TestResultDecision(test_digest=test_digest, skipped=True, prior=prior)
    registry[test_digest] = {
        "split_names": sorted(split_names),
        "run_id": run_id or "",
        "recorded_at": _timestamp_now(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    test_results_path(run_dir).write_text(
        json.dumps(registry, indent=2, sort_keys=True), encoding="utf-8"
    )
    return TestResultDecision(test_digest=test_digest, skipped=False, prior=prior)


# ---------------------------------------------------------------------------
# Opening a run: layout, resume drift guards, recorded config, artifact mirror
# ---------------------------------------------------------------------------


#: What a run dir records: the config as given, and the plan it resolved to.
CONFIG_FILENAME = "config.yaml"
PLAN_FILENAME = "plan.yaml"


def open_run(plan: Plan, *, config: ExperimentConfig | None = None, num_folds: int):
    """Create (or reopen) the run directory a plan writes to.

    Restores completed folds from the artifact mirror, resolves the managed layout,
    writes the experiment metadata, refuses a resume whose plan or manifests drifted,
    records ``config.yaml`` (``config`` as given; the plan's config when none is given)
    and ``plan.yaml``, and returns ``(layout, mirror)``; ``mirror`` is ``None`` without
    ``run.mirror_root``.
    """
    from soma.pipeline.mirror import ArtifactMirror, restore_run_from_mirror

    restored_run_id = restore_run_from_mirror(plan, num_folds=num_folds)
    layout = resolve_managed_output_paths(plan, run_id=restored_run_id)
    for directory in (
        layout.output_root,
        layout.experiment_dir,
        layout.experiment_dir / "runs",
        layout.run_dir,
        layout.index_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    write_experiment_metadata(layout.experiment_dir, layout.experiment)
    # Runs before plan.yaml is overwritten with the record it compares against.
    guard_resume_plan_drift(layout.run_dir, plan)
    record_plan(layout.run_dir, plan, config=config)
    mirror = None
    if plan.run.mirror_root is not None:
        mirror = ArtifactMirror(
            run_dir=layout.run_dir,
            mirror_run_dir=Path(plan.run.mirror_root) / layout.run_dir.relative_to(layout.output_root),
        )
        mirror.retry_pending()
    return layout, mirror


def record_plan(run_dir: Path, plan: Plan, *, config: ExperimentConfig | None = None) -> None:
    """Write ``config.yaml`` (as given) and ``plan.yaml`` into ``run_dir``."""
    (config if config is not None else plan.config).to_yaml(Path(run_dir) / CONFIG_FILENAME)
    plan.to_yaml(Path(run_dir) / PLAN_FILENAME)


def guard_resume_plan_drift(run_dir: Path, plan: Plan) -> None:
    """Refuse to resume into a run dir whose saved plan differs from the current one.

    Only a resume (``run.resume`` / ``run.run_id``) that lands on an existing
    ``plan.yaml`` is checked: a fresh run, or a pinned run id naming a not-yet-created
    dir, has nothing to compare against. The plans are compared without where and how a
    launch runs (``run.output_dir``, ``run.run_id``, ``run.resume``, ``run.mirror_root``,
    and ``run.folds``, the folds this launch takes on) and the execution fields slide2vec
    resolves for the machine (:meth:`Plan.comparable`), so an otherwise identical relaunch
    passes while a changed seed, encoder, head or evaluation protocol is caught before it
    mixes incompatible folds (issue #244). The manifest guard then checks the cohort
    files' content."""
    if not (plan.run.resume or plan.run.run_id):
        return
    saved_path = Path(run_dir) / PLAN_FILENAME
    if not saved_path.is_file():
        return
    saved = comparable(yaml.safe_load(saved_path.read_text(encoding="utf-8")) or {})
    # Round-trip the current plan through YAML so both sides are the same primitives.
    current = comparable(yaml.safe_load(plan.to_yaml()))
    if saved == current:
        guard_resume_manifest_drift(run_dir, plan)
        return
    differing = sorted(key for key in set(saved) | set(current) if saved.get(key) != current.get(key))
    raise ValueError(
        f"Refusing to resume run '{Path(run_dir).name}': its saved plan.yaml differs from the "
        f"current plan (differing sections: {differing}). Resuming would mix folds trained "
        "under different recipes. Start a fresh run, or reconcile the config to match the "
        "existing run."
    )


def guard_resume_manifest_drift(run_dir: Path, plan: Plan) -> None:
    """Refuse to resume when the train/tune manifest *content* changed under the same paths.

    ``plan.yaml`` only records the cohort file paths. The run's ``run.yaml`` carries the
    ``experiment_id`` minted from the train+tune slice content, so recompute that identity
    for the current files and compare: an edited label or reshuffled fold with unchanged
    paths is caught here rather than mixing folds trained on different data.
    """
    run_yaml = Path(run_dir) / "run.yaml"
    if not run_yaml.is_file():
        return
    saved = yaml.safe_load(run_yaml.read_text(encoding="utf-8")) or {}
    saved_experiment_id = saved.get("experiment_id")
    if not saved_experiment_id:
        return
    current_experiment_id = build_experiment_spec(plan).experiment_id
    if saved_experiment_id == current_experiment_id:
        return
    raise ValueError(
        f"Refusing to resume run '{Path(run_dir).name}': the train/tune content of "
        f"{plan.cohort.records} / {plan.cohort.splits} differs from what the run was trained "
        f"on (experiment identity {str(saved_experiment_id)[:12]} → "
        f"{current_experiment_id[:12]}). Resuming would mix folds trained on different "
        "data. Start a fresh run, or restore the original manifests."
    )


class RunRecorder:
    """Context manager that tracks run metadata lifecycle (running → completed/failed).

    On enter: writes initial "running" metadata and appends to the per-run index.
    On success: call ``complete(summary_metrics)`` (or ``partial(pending_folds)``) then
    exit normally. On exception: writes "failed" metadata and (if no prior successful run
    exists) advances the latest pointer before re-raising.
    """

    def __init__(self, layout: ManagedOutputPaths, plan: Plan) -> None:
        self._layout = layout
        self._plan = plan
        self._metadata: RunMetadata | None = None
        self._summary_metrics: dict[str, float] | None = None
        self._pending_folds: tuple[int, ...] = ()

    def __enter__(self) -> "RunRecorder":
        layout = self._layout
        self._metadata = create_run_metadata(
            plan=self._plan,
            experiment=layout.experiment,
            run_dir=layout.run_dir,
            run_id=layout.run_id,
            status="running",
            started_at=datetime.now().astimezone().isoformat(),
        )
        write_run_metadata(layout.run_dir, self._metadata)
        self._write_indexes(self._metadata)
        return self

    def complete(self, summary_metrics: dict[str, float]) -> None:
        self._summary_metrics = summary_metrics

    def partial(self, pending_folds: tuple[int, ...]) -> None:
        """This launch trained its ``run.folds`` share; other folds are still pending."""
        self._pending_folds = pending_folds

    def with_identity(self, result):
        """``result`` (a ``PipelineResult``) with the manifest checksums the run's
        metadata files record."""
        experiment = self._layout.experiment
        return replace(
            result,
            dataset_checksum=experiment.dataset_checksum,
            splits_checksum=experiment.splits_checksum,
            test_checksum=self._metadata.test_checksum,
        )

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        layout = self._layout
        if exc_type is not None:
            metadata = self._metadata.with_updates(
                status="failed",
                finished_at=datetime.now().astimezone().isoformat(),
                error=str(exc_val),
            )
            write_run_metadata(layout.run_dir, metadata)
            self._write_indexes(metadata)
            if not has_successful_run(layout.experiment_dir):
                update_latest_pointer(layout.experiment_dir, layout.run_dir)
        elif self._pending_folds:
            # Not "completed": the leaderboard ranks completed runs, and this one has no
            # summary yet. The latest pointer stays where it was for the same reason.
            metadata = self._metadata.with_updates(
                status="partial",
                finished_at=datetime.now().astimezone().isoformat(),
            )
            write_run_metadata(layout.run_dir, metadata)
            self._write_indexes(metadata)
        else:
            metadata = self._metadata.with_updates(
                status="completed",
                finished_at=datetime.now().astimezone().isoformat(),
                summary_metrics=self._summary_metrics or {},
            )
            write_run_metadata(layout.run_dir, metadata)
            self._write_indexes(metadata)
            update_latest_pointer(layout.experiment_dir, layout.run_dir)
        return False

    def _write_indexes(self, metadata: RunMetadata) -> None:
        # Each run touches ONLY its own dir + the append-friendly per-run index. The
        # experiment-level projection is not written here: it was an unlocked
        # read-modify-rewrite of a shared CSV that silently lost rows when concurrent
        # sweep runs finished together (ADR 0003). The leaderboard rebuilds that
        # projection on demand by scanning the self-describing run dirs.
        update_run_index(self._layout.index_dir / "runs.csv", metadata)


# ---------------------------------------------------------------------------
# Fold layout and fold-level resume
# ---------------------------------------------------------------------------

#: Where a fold's model lands (``soma.save``); with ``metrics.json`` it marks the fold done.
BUNDLE_DIRNAME = "bundle"


def fold_dir(run_dir: Path, fold: int, num_folds: int) -> Path:
    """A single fold owns ``run_dir`` directly; multi-fold runs get ``fold_N/`` subdirs."""
    return Path(run_dir) if num_folds == 1 else Path(run_dir) / f"fold_{fold}"


def fold_is_complete(directory: Path) -> bool:
    """A fold is complete when its ``metrics.json`` and its bundle both exist.

    Resume is fold-level (ADR 0016): a fold missing either restarts from epoch 0.
    """
    directory = Path(directory)
    return (directory / "metrics.json").is_file() and (directory / BUNDLE_DIRNAME).is_dir()


def reject_undeclared_fold_dirs(run_dir: Path, num_folds: int) -> None:
    """Refuse a run dir holding ``fold_*`` dirs beyond what the current splits declare.

    Such a dir comes from a previous launch with a different split file; folding its
    ``metrics.json`` into the summary would silently mix two protocols."""
    stray = sorted(
        path.name
        for path in Path(run_dir).glob("fold_*")
        if path.is_dir()
        and path.name[len("fold_"):].isdigit()
        and int(path.name[len("fold_"):]) >= num_folds
    )
    if stray:
        raise ValueError(
            f"Run dir {run_dir} holds fold dir(s) {stray} that the current splits do not "
            f"declare ({num_folds} fold(s)). They were written under a different split "
            "file; start a fresh run or remove them before resuming."
        )


def source_identity(source: object) -> dict[str, Any]:
    """What a feature source is, never where it lives.

    A source reading a soma feature cache is identified by the cache's own record
    (``cache_metadata.json``: cache key, artifact kind, encoder, feature type, dtype and
    the slide2vec feature identity); any other source by its type.
    """
    cls = type(source)
    identity: dict[str, Any] = {"type": f"{cls.__module__}.{cls.__qualname__}"}
    feature_dir = getattr(source, "feature_dir", None)
    if feature_dir is None:
        return identity
    feature_dir = Path(feature_dir)
    for directory in (feature_dir, *list(feature_dir.parents)[:2]):
        record_path = directory / "cache_metadata.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            keys = ("cache_key", "artifact_kind", "encoder_name", "feature_type", "dtype", "feature_identity")
            identity.update({key: record[key] for key in keys if key in record})
            break
    return identity
