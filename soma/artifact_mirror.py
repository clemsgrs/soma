"""Atomic publication and restoration of completed fold artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any

import yaml

from soma.config import PipelineConfig, config_yaml_dict

logger = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_files(root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        relative = path.relative_to(root).as_posix()
        if relative != "manifest.json":
            files[relative] = {"sha256": _sha256(path), "size": path.stat().st_size}
    return files


def _verified_manifest(
    root: Path, *, raise_io_errors: bool = False
) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        if not isinstance(files, dict) or not files:
            return None
        actual_files: set[str] = set()
        for path in root.rglob("*"):
            try:
                mode = path.stat().st_mode
            except FileNotFoundError:
                return None
            if stat.S_ISREG(mode) and path != manifest_path:
                actual_files.add(path.relative_to(root).as_posix())
        if actual_files != set(files):
            return None
        for relative, expected in files.items():
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return None
            path = root / relative_path
            try:
                file_stat = path.stat()
            except FileNotFoundError:
                return None
            if not stat.S_ISREG(file_stat.st_mode):
                return None
            if (
                file_stat.st_size != expected["size"]
                or _sha256(path) != expected["sha256"]
            ):
                return None
    except FileNotFoundError:
        return None
    except OSError:
        if raise_io_errors:
            raise
        return None
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return manifest


def _resume_recipe(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    run = normalized.get("run")
    if isinstance(run, dict):
        run.pop("mirror_root", None)
        run.pop("resume", None)
        run.pop("run_id", None)
    evaluation = normalized.get("evaluation")
    if isinstance(evaluation, dict):
        evaluation.pop("save_segmentation_confusion_evidence", None)
    return normalized


def _published_fold_bundles(run_dir: Path) -> list[Path]:
    root = run_dir / "recovery" / "folds"
    try:
        with os.scandir(root) as entries:
            return sorted(
                Path(entry.path)
                for entry in entries
                if entry.is_dir(follow_symlinks=False)
                and re.fullmatch(r"fold_\d+", entry.name)
            )
    except FileNotFoundError:
        return []


def _compatible_fold_bundle(
    bundle: Path, config: PipelineConfig
) -> tuple[dict[str, Any], Path] | None:
    manifest = _verified_manifest(bundle, raise_io_errors=True)
    if manifest is None or manifest.get("kind") != "fold":
        return None
    fold_match = re.fullmatch(r"fold_(\d+)", bundle.name)
    if fold_match is None or int(fold_match.group(1)) != manifest.get("fold"):
        return None
    try:
        saved = (
            yaml.safe_load((bundle / "config.yaml").read_text(encoding="utf-8")) or {}
        )
    except OSError:
        raise
    except (ValueError, yaml.YAMLError):
        return None
    if _resume_recipe(saved) != _resume_recipe(config_yaml_dict(config)):
        return None
    return manifest, bundle


def _compatible_fold_bundles(
    run_dir: Path, config: PipelineConfig
) -> list[tuple[dict[str, Any], Path]]:
    return [
        compatible
        for bundle in _published_fold_bundles(run_dir)
        if (compatible := _compatible_fold_bundle(bundle, config)) is not None
    ]


def _fold_sources(fold_dir: Path, run_dir: Path) -> list[tuple[str, Path]]:
    sources: list[tuple[str, Path]] = []
    for source in sorted(path for path in fold_dir.rglob("*") if path.is_file()):
        relative = source.relative_to(fold_dir)
        if relative.parts and relative.parts[0] == "recovery":
            continue
        sources.append((relative.as_posix(), source))
    if not any(relative == "config.yaml" for relative, _ in sources):
        sources.append(("config.yaml", run_dir / "config.yaml"))
    return sources


def _publish_fold(
    *, run_dir: Path, mirror_run_dir: Path, fold: int, fold_dir: Path
) -> None:
    destination = mirror_run_dir / "recovery" / "folds" / f"fold_{fold}"
    # Final directories are only exposed by the atomic replace below. Their presence
    # is therefore the retry marker; restore performs the expensive checksum audit.
    if destination.is_dir():
        return
    if destination.exists():
        raise OSError(f"fold mirror destination is not a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent)
    )
    try:
        for relative, source in _fold_sources(fold_dir, run_dir):
            if not source.is_file():
                raise FileNotFoundError(f"required fold artifact is missing: {source}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        manifest = {
            "schema_version": 1,
            "kind": "fold",
            "fold": fold,
            "files": _manifest_files(staging),
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        if _verified_manifest(staging) is None:
            raise OSError(f"staged fold publication failed verification: {staging}")
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _completed_local_folds(run_dir: Path) -> dict[int, Path]:
    completed: dict[int, Path] = {}
    if (run_dir / "metrics.json").is_file():
        completed[0] = run_dir
    for fold_dir in sorted(run_dir.glob("fold_*")):
        match = re.fullmatch(r"fold_(\d+)", fold_dir.name)
        if match is not None and (fold_dir / "metrics.json").is_file():
            completed[int(match.group(1))] = fold_dir
    return completed


def _atomic_copy_verified_file(
    source: Path, destination: Path, expected: dict[str, Any]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.restore.", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        if (
            temporary_path.stat().st_size != expected["size"]
            or _sha256(temporary_path) != expected["sha256"]
        ):
            raise OSError(f"restored artifact failed verification: {source.name}")
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _materialize_fold(source: Path, run_dir: Path, *, num_folds: int) -> None:
    manifest = _verified_manifest(source, raise_io_errors=True)
    if manifest is None:
        raise OSError(f"completed fold bundle failed verification: {source}")
    fold = int(manifest["fold"])
    target = run_dir if num_folds == 1 else run_dir / f"fold_{fold}"
    if (target / "metrics.json").is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.restore.", dir=target.parent)
    )
    try:
        for relative, expected in sorted(manifest["files"].items()):
            if relative == "config.yaml" and num_folds > 1:
                continue
            _atomic_copy_verified_file(source / relative, staging / relative, expected)
        if not (staging / "metrics.json").is_file():
            raise OSError(f"completed fold bundle has no metrics.json: {source}")
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if num_folds > 1:
        config_target = run_dir / "config.yaml"
        if not config_target.is_file():
            _atomic_copy_verified_file(
                source / "config.yaml",
                config_target,
                manifest["files"]["config.yaml"],
            )


def restore_run_from_mirror(config: PipelineConfig, *, num_folds: int) -> str | None:
    """Restore verified completed folds before local resume resolution."""
    if config.mirror_root is None or not (config.resume or config.run_id):
        return None

    from soma.output_layout import build_experiment_spec, latest_existing_run_id

    experiment = build_experiment_spec(config)
    local_experiment_dir = (
        Path(config.output_root).resolve()
        / "experiments"
        / experiment.experiment_dirname
    )
    local_runs_dir = local_experiment_dir / "runs"
    existing_local_run_id = latest_existing_run_id(local_experiment_dir)
    if config.run_id is None and existing_local_run_id is not None:
        return None
    pinned_local_run = (
        local_runs_dir / config.run_id if config.run_id is not None else None
    )
    if pinned_local_run is not None and pinned_local_run.is_dir():
        return None

    mirror_runs_dir = (
        Path(config.mirror_root).resolve()
        / "experiments"
        / experiment.experiment_dirname
        / "runs"
    )
    try:
        if config.run_id is not None:
            candidate_paths = [mirror_runs_dir / config.run_id]
        elif mirror_runs_dir.is_dir():
            candidate_paths = sorted(mirror_runs_dir.iterdir())
        else:
            candidate_paths = []
        candidates: list[
            tuple[Path, list[tuple[dict[str, Any], Path]]]
        ] = []
        for candidate in candidate_paths:
            if not candidate.is_dir():
                continue
            fold_bundles = _compatible_fold_bundles(candidate, config)
            if fold_bundles:
                candidates.append((candidate, fold_bundles))
        if config.run_id is None and len(candidates) > 1:
            raise ValueError(
                "resume=True found multiple compatible mirrored runs; set "
                "run.run_id to choose one explicitly."
            )
        if not candidates:
            if config.resume:
                raise FileNotFoundError(
                    "no compatible verified mirror run exists and no local run is "
                    "available to resume"
                )
            return None
        mirror_run_dir, fold_bundles = candidates[0]
        local_run_dir = local_runs_dir / mirror_run_dir.name
        for _, fold_bundle in fold_bundles:
            _materialize_fold(fold_bundle, local_run_dir, num_folds=num_folds)
        return mirror_run_dir.name
    except OSError as exc:
        raise OSError(
            "mirror recovery failed and no local run is available to resume"
        ) from exc


class ArtifactMirror:
    """Fail-soft publisher for immutable completed-fold bundles."""

    def __init__(self, *, run_dir: Path, mirror_run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.mirror_run_dir = Path(mirror_run_dir)

    def fold_completed(self, *, fold: int, fold_dir: Path) -> None:
        completed = _completed_local_folds(self.run_dir)
        completed[fold] = Path(fold_dir)
        self._publish(completed)

    def retry_pending(self) -> None:
        """Publish completed local folds whose atomic destination is still absent."""
        self._publish(_completed_local_folds(self.run_dir))

    def _publish(self, completed: dict[int, Path]) -> None:
        for fold, fold_dir in sorted(completed.items()):
            try:
                _publish_fold(
                    run_dir=self.run_dir,
                    mirror_run_dir=self.mirror_run_dir,
                    fold=fold,
                    fold_dir=fold_dir,
                )
            except Exception as exc:  # shared storage must never interrupt training
                logger.warning(
                    "Artifact mirror publication remains pending for fold %d: %s",
                    fold,
                    exc,
                    exc_info=True,
                )
