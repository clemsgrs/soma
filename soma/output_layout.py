"""Managed experiment output layout helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
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


def _training_without_seed(config: PipelineConfig) -> dict[str, Any]:
    training = asdict(config.training)
    training.pop("seed", None)
    return training


def canonical_experiment_payload(config: PipelineConfig) -> dict[str, Any]:
    dataset_path = Path(config.dataset_csv).resolve()
    splits_path = Path(config.splits_csv).resolve()
    return {
        "dataset": {
            "path": str(dataset_path),
            "checksum": _sha256_file(dataset_path),
        },
        "splits": {
            "path": str(splits_path),
            "checksum": _sha256_file(splits_path),
        },
        "dataset_type": config.dataset_type,
        "preprocessing": asdict(config.preprocessing),
        "cache": {
            "enabled": config.cache.enabled,
            "reuse_policy": config.cache.reuse_policy,
            "save_tile_features_for_slide": config.cache.save_tile_features_for_slide,
        },
        "encoder": asdict(config.encoder) if config.encoder is not None else None,
        "aggregator": asdict(config.aggregator) if config.aggregator is not None else None,
        "task": asdict(config.task),
        "training": _training_without_seed(config),
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
            "error": self.error or "",
        }

    def with_updates(self, **updates: Any) -> "RunMetadata":
        return replace(self, **updates)


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
    resolved_run_id = run_id or make_run_id(when=when)
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


def update_experiment_index(
    path: Path,
    experiment: ExperimentSpec,
    *,
    num_runs: int,
    latest_run_id: str,
    latest_status: str,
) -> None:
    fieldnames = [
        "experiment_id",
        "slug",
        "dataset_path",
        "dataset_checksum",
        "splits_path",
        "splits_checksum",
        "encoder",
        "aggregator",
        "task",
        "num_runs",
        "latest_run_id",
        "latest_status",
        "experiment_dir",
    ]
    rows = _read_csv_rows(path)
    encoded = experiment.canonical_spec["encoder"]
    aggregator = experiment.canonical_spec["aggregator"]
    row = {
        "experiment_id": experiment.experiment_id,
        "slug": experiment.slug,
        "dataset_path": str(experiment.dataset_path),
        "dataset_checksum": experiment.dataset_checksum,
        "splits_path": str(experiment.splits_path),
        "splits_checksum": experiment.splits_checksum,
        "encoder": "" if encoded is None else str(encoded.get("name", "")),
        "aggregator": "" if aggregator is None else str(aggregator.get("name", "")),
        "task": str(experiment.canonical_spec["task"]["name"]),
        "num_runs": str(num_runs),
        "latest_run_id": latest_run_id,
        "latest_status": latest_status,
        "experiment_dir": experiment.experiment_dirname,
    }
    rows = [existing for existing in rows if existing.get("experiment_id") != experiment.experiment_id]
    rows.append(row)
    rows.sort(key=lambda item: item["slug"])
    _write_csv_rows(path, fieldnames, rows)


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


def count_run_directories(experiment_dir: Path) -> int:
    runs_dir = experiment_dir / "runs"
    if not runs_dir.exists():
        return 0
    return sum(1 for child in runs_dir.iterdir() if child.is_dir())


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
