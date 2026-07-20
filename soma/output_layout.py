"""Managed experiment output layout helpers."""

from __future__ import annotations

import csv
import hashlib
import json
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

from soma.config import PipelineConfig


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


def _training_identity(config: PipelineConfig) -> dict[str, Any]:
    training = asdict(config.training)
    training.pop("seed", None)
    # ``checkpoint_selection`` folds in only when non-default (issue #282), the same
    # guard the dtype/annotation-sampling knobs use: the evaluation protocol *is* part
    # of what the experiment is, but every pre-existing run was implicitly ``best``, so
    # emitting the key unconditionally would re-mint every legacy experiment_id.
    if config.training.checkpoint_selection == "best":
        training.pop("checkpoint_selection", None)
    return training


def _evaluation_identity(config: PipelineConfig) -> dict[str, Any]:
    evaluation = asdict(config.evaluation)
    # ``overwrite_test`` is an operational clobber-guard flag (issue #247), not part of
    # what the experiment *is* — toggling it must never mint a new experiment_id.
    evaluation.pop("overwrite_test", None)
    return evaluation


def _heatmap_identity(config: PipelineConfig) -> dict[str, Any]:
    heatmaps = asdict(config.heatmaps)
    if not config.heatmaps.enabled:
        return {"enabled": False}
    return heatmaps


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
    """sha256 over the dataset.csv rows for ``sample_ids`` (sorted by id, byte-stable)."""
    rows = [
        {str(k): ("" if v is None else str(v)) for k, v in row.items()}
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


def test_identity_digest(config: PipelineConfig) -> str:
    """Digest identifying WHICH test set a run is scored against (issue #247).

    The test-slice counterpart of the training digest: a byte-stable sha256 over the
    ``test*`` split assignments and the dataset rows they reference. Two runs of the same
    model (same ``experiment_id``) scored on different test sets share an experiment dir
    but differ here, so their results can be namespaced and a re-score of an already-scored
    test set refused rather than silently clobbered. Empty when a run has no ``test*`` rows.
    """
    dataset_path = Path(config.dataset_csv).resolve()
    splits_path = Path(config.splits_csv).resolve()
    dataset_digest, splits_digest = _manifest_slice_digests(
        dataset_path, splits_path, _is_test_split
    )
    return hashlib.sha256(
        _stable_json({"dataset": dataset_digest, "splits": splits_digest}).encode("utf-8")
    ).hexdigest()


def canonical_experiment_payload(config: PipelineConfig) -> dict[str, Any]:
    dataset_path = Path(config.dataset_csv).resolve()
    splits_path = Path(config.splits_csv).resolve()
    # Experiment identity is invariant to the *test* set (issue #247): only the {train,
    # tune} slice of the dataset + splits manifest enters the digest, so dropping in a
    # new/official test set (which only adds test rows) leaves experiment_id — and the
    # leaderboard triple — unchanged, letting a frozen checkpoint be re-scored rather than
    # retrained. Full-file + test-slice provenance is recorded on the RunMetadata instead.
    dataset_digest, splits_digest = _manifest_slice_digests(
        dataset_path, splits_path, _is_training_split
    )
    return {
        # No manifest ``path`` here: identity is the {train,tune} content, not the machine-
        # local file location, so a checkpoint is reused even when the official test set
        # arrives as a *new* splits/dataset file rather than appended rows (issue #247).
        # The resolved paths are still kept on ExperimentSpec + RunMetadata for provenance.
        "dataset": {
            "checksum": dataset_digest,
        },
        "splits": {
            "checksum": splits_digest,
        },
        "dataset_type": config.dataset_type,
        "feature_mode": config.feature_mode,
        "preprocessing": asdict(config.preprocessing),
        "cache": {
            "enabled": config.cache.enabled,
            "reuse_policy": config.cache.reuse_policy,
        },
        "encoder": asdict(config.encoder) if config.encoder is not None else None,
        "composite": asdict(config.composite) if config.composite is not None else None,
        "aggregator": asdict(config.aggregator) if config.aggregator is not None else None,
        "decoder": asdict(config.decoder) if config.decoder is not None else None,
        "pixel_classifier": (
            asdict(config.pixel_classifier) if config.pixel_classifier is not None else None
        ),
        "task": asdict(config.task),
        "evaluation": _evaluation_identity(config),
        "heatmaps": _heatmap_identity(config),
        "augmentation": asdict(config.augmentation),
        "training": _training_identity(config),
        "tags": list(config.tags),
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
            "slug": self.slug,
            "short_hash": self.short_hash,
            "experiment_dirname": self.experiment_dirname,
            "dataset_path": str(self.dataset_path),
            "splits_path": str(self.splits_path),
            "dataset_checksum": self.dataset_checksum,
            "splits_checksum": self.splits_checksum,
            "canonical_spec": self.canonical_spec,
        }


def build_experiment_spec(config: PipelineConfig) -> ExperimentSpec:
    canonical_spec = canonical_experiment_payload(config)
    digest = hashlib.sha256(_stable_json(canonical_spec).encode("utf-8")).hexdigest()
    short_hash = digest[:12]
    dataset_name = _slugify(Path(config.dataset_csv).stem)
    encoder_name = _slugify(config.encoder.name if config.encoder is not None else "precomputed")
    aggregator_name = _slugify(config.aggregator.name if config.aggregator is not None else "slide")
    task_name = _slugify(config.task.name)
    slug = f"{dataset_name}-{encoder_name}-{aggregator_name}-{task_name}_{short_hash}"
    return ExperimentSpec(
        experiment_id=digest,
        slug=slug,
        short_hash=short_hash,
        experiment_dirname=slug,
        dataset_path=Path(config.dataset_csv).resolve(),
        splits_path=Path(config.splits_csv).resolve(),
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
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


def _git_sha(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _git_dirty(cwd: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return bool(result.stdout.strip())


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
    config: PipelineConfig, experiment_dir: Path, *, when: datetime | None = None
) -> str:
    """Pick the run id for this launch, honoring the resume directives (issue #244).

    Precedence: an explicit ``run_id`` pins that run (resume into it if present,
    else create it under that name); otherwise ``resume`` reuses the latest
    existing run for this experiment (error if none); otherwise a fresh
    timestamped id — the unchanged default.
    """
    if config.run_id:
        return config.run_id
    if config.resume:
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
    config: PipelineConfig,
    *,
    run_id: str | None = None,
    when: datetime | None = None,
) -> ManagedOutputPaths:
    output_root = Path(config.output_root).resolve()
    experiment = build_experiment_spec(config)
    experiment_dir = output_root / "experiments" / experiment.experiment_dirname
    # An explicit run_id arg (e.g. tests, leaderboard replay) pins the run as-is;
    # otherwise consult the config's resume directives (issue #244).
    resolved_run_id = run_id or _resolve_run_id(config, experiment_dir, when=when)
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
    config: PipelineConfig,
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
    cwd = Path.cwd()
    dataset_path = Path(config.dataset_csv)
    splits_path = Path(config.splits_csv)
    return RunMetadata(
        run_id=run_id,
        experiment_id=experiment.experiment_id,
        status=status,
        started_at=started_at or _timestamp_now(),
        finished_at=finished_at,
        seed=config.training.seed,
        wandb_id=wandb_id,
        wandb_url=wandb_url,
        git_sha=_git_sha(cwd),
        git_dirty=_git_dirty(cwd),
        hostname=socket.gethostname() or None,
        username=getuser() or None,
        resolved_output_dir=run_dir.resolve(),
        summary_metrics=summary_metrics or {},
        error=error,
        environment=capture_environment(),
        test_checksum=test_identity_digest(config),
        dataset_file_checksum=_sha256_file(dataset_path) if dataset_path.is_file() else "",
        splits_file_checksum=_sha256_file(splits_path) if splits_path.is_file() else "",
    )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_run_index(path: Path, metadata: RunMetadata) -> None:
    fieldnames = [
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
    rows = _read_csv_rows(path)
    row = {
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
    rows = [existing for existing in rows if existing.get("run_id") != metadata.run_id]
    rows.append(row)
    rows.sort(key=lambda item: item["run_id"])
    _write_csv_rows(path, fieldnames, rows)


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
