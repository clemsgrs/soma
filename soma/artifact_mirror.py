"""Recoverable, content-addressed publication of long-running fold artifacts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Iterable, Mapping, Sequence
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
        if relative == "manifest.json":
            continue
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


def _verified_bundle(root: Path) -> bool:
    return _verified_manifest(root) is not None


def _matching_bundles(source: Path, destination: Path) -> bool:
    """Return whether both bundles verify and describe exactly the same bytes."""
    if not _verified_bundle(source) or not _verified_bundle(destination):
        return False
    try:
        source_manifest = json.loads(
            (source / "manifest.json").read_text(encoding="utf-8")
        )
        destination_manifest = json.loads(
            (destination / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return source_manifest == destination_manifest


def _resume_recipe(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload))
    run = normalized.get("run")
    if isinstance(run, dict):
        run.pop("mirror_root", None)
        run.pop("resume", None)
        run.pop("run_id", None)
    evaluation = normalized.get("evaluation")
    if isinstance(evaluation, dict):
        # Evidence export is an operational selected-checkpoint pass, not part of the
        # training recipe that produced the checkpoint.
        evaluation.pop("save_segmentation_confusion_evidence", None)
    return normalized


def _canonical_mirror_bundles(run_dir: Path, kind: str) -> list[Path]:
    """Return only published bundle directories at the canonical recovery depth."""
    def directories(root: Path) -> list[Path]:
        try:
            with os.scandir(root) as entries:
                return sorted(
                    Path(entry.path)
                    for entry in entries
                    if entry.is_dir(follow_symlinks=False)
                )
        except FileNotFoundError:
            return []

    recovery = run_dir / "recovery"
    if kind == "folds":
        root = recovery / "folds"
        return sorted(
            child
            for child in directories(root)
            if re.fullmatch(r"fold_\d+", child.name)
        )
    if kind == "checkpoints":
        root = recovery / "checkpoints"
        bundles: list[Path] = []
        for fold_dir in directories(root):
            if not re.fullmatch(r"fold_\d+", fold_dir.name):
                continue
            bundles.extend(
                child
                for child in directories(fold_dir)
                if re.fullmatch(r"epoch_\d+(?:__[0-9a-f]{12})?", child.name)
            )
        return bundles
    raise ValueError(f"unknown recovery bundle kind: {kind}")


def _compatible_bundle(
    bundle: Path, config: PipelineConfig, *, kind: str
) -> tuple[dict[str, Any], Path] | None:
    manifest = _verified_manifest(bundle, raise_io_errors=True)
    if manifest is None or manifest.get("kind") != kind:
        return None
    fold_match = re.fullmatch(
        r"fold_(\d+)", bundle.parent.name if kind == "checkpoint" else bundle.name
    )
    if fold_match is None or int(fold_match.group(1)) != manifest.get("fold"):
        return None
    if kind == "checkpoint":
        epoch_match = re.fullmatch(r"epoch_(\d+)(?:__[0-9a-f]{12})?", bundle.name)
        if epoch_match is None or int(epoch_match.group(1)) != manifest.get("epoch"):
            return None
    expected = _resume_recipe(config_yaml_dict(config))
    try:
        saved = (
            yaml.safe_load((bundle / "config.yaml").read_text(encoding="utf-8")) or {}
        )
    except OSError:
        raise
    except (ValueError, yaml.YAMLError):
        return None
    if _resume_recipe(saved) != expected:
        return None
    return manifest, bundle


def _compatible_mirror_bundles(
    run_dir: Path, config: PipelineConfig, *, kind: str
) -> list[tuple[dict[str, Any], Path]]:
    manifest_kind = "fold" if kind == "folds" else "checkpoint"
    return [
        compatible
        for bundle in _canonical_mirror_bundles(run_dir, kind)
        if (compatible := _compatible_bundle(bundle, config, kind=manifest_kind))
        is not None
    ]


def _compatible_mirror_run(run_dir: Path, config: PipelineConfig) -> bool:
    return bool(
        _compatible_mirror_bundles(run_dir, config, kind="folds")
        or _compatible_mirror_bundles(run_dir, config, kind="checkpoints")
    )


def _copy_verified_bundle(source: Path, destination: Path) -> None:
    """Atomically install one exact bundle beside its final destination."""
    if not _verified_bundle(source):
        raise OSError(f"recovery source failed verification: {source}")
    if _matching_bundles(source, destination):
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp.", dir=destination.parent)
    )
    try:
        for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
            target = staging / source_file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
        if not _matching_bundles(source, staging):
            raise OSError(f"staged recovery bundle does not match source: {staging}")
        if destination.exists():
            quarantine = destination.with_name(
                f".{destination.name}.corrupt.{uuid.uuid4().hex}"
            )
            os.replace(destination, quarantine)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _atomic_copy_verified_file(
    source: Path, destination: Path, expected: dict[str, Any]
) -> None:
    """Copy one file through a verified same-parent temporary file."""
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


def _checkpoint_transaction_path(run_dir: Path, fold: int) -> Path:
    return (
        run_dir.parent
        / ".restore_transactions"
        / run_dir.name
        / f"fold_{fold}.json"
    )


def _transaction_path(runs_dir: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise OSError(f"unsafe checkpoint restore transaction path: {relative}")
    return runs_dir / relative_path


def _checkpoint_files_match(
    target: Path, expected_files: dict[str, dict[str, Any]]
) -> bool:
    for relative, expected in expected_files.items():
        path = target / relative
        try:
            if (
                path.stat().st_size != expected["size"]
                or _sha256(path) != expected["sha256"]
            ):
                return False
        except (OSError, KeyError, TypeError):
            return False
    return True


def _cleanup_transaction_marker(marker_path: Path) -> None:
    marker_path.unlink(missing_ok=True)
    for directory in (marker_path.parent, marker_path.parent.parent):
        try:
            directory.rmdir()
        except OSError:
            break


def _recover_checkpoint_transaction(marker_path: Path, runs_dir: Path) -> None:
    """Finish or roll back an interrupted directory-level checkpoint install."""
    try:
        transaction = json.loads(marker_path.read_text(encoding="utf-8"))
        target = _transaction_path(runs_dir, transaction["target"])
        staging = _transaction_path(runs_dir, transaction["staging"])
        quarantine = _transaction_path(runs_dir, transaction["quarantine"])
        expected_files = transaction["files"]
        run_dir = runs_dir / transaction["run_id"]
    except FileNotFoundError:
        return
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise OSError(f"invalid checkpoint restore transaction: {marker_path}") from exc
    if run_dir.parent != runs_dir or not isinstance(expected_files, dict):
        raise OSError(f"unsafe checkpoint restore transaction: {marker_path}")

    if target.is_dir():
        if _checkpoint_files_match(target, expected_files):
            shutil.rmtree(quarantine, ignore_errors=True)
        elif quarantine.is_dir():
            rejected = target.with_name(
                f".{target.name}.checkpoint-rejected.{uuid.uuid4().hex}"
            )
            os.replace(target, rejected)
            os.replace(quarantine, target)
            shutil.rmtree(rejected, ignore_errors=True)
        else:
            # The ledger is durable before the first swap. With no quarantine, the
            # target is therefore still the authoritative pre-transaction directory.
            # Discard an uncommitted staged replacement rather than blocking resume.
            pass
        shutil.rmtree(staging, ignore_errors=True)
    elif quarantine.is_dir():
        os.replace(quarantine, target)
        shutil.rmtree(staging, ignore_errors=True)
    elif staging.is_dir() and _checkpoint_files_match(staging, expected_files):
        os.replace(staging, target)
    else:
        raise OSError(
            f"checkpoint restore transaction has no recoverable state: {marker_path}"
        )
    restore_error = transaction.get("restore_error")
    _cleanup_transaction_marker(marker_path)
    if isinstance(restore_error, str):
        _append_restore_error(run_dir, restore_error)


def _recover_checkpoint_transactions(runs_dir: Path) -> None:
    markers = sorted((runs_dir / ".restore_transactions").glob("*/fold_*.json"))
    for marker in markers:
        _recover_checkpoint_transaction(marker, runs_dir)


def _record_transaction_restore_error(run_dir: Path, message: str) -> bool:
    transaction_dir = run_dir.parent / ".restore_transactions" / run_dir.name
    markers = sorted(transaction_dir.glob("fold_*.json"))
    if not markers:
        return False
    for marker in markers:
        transaction = json.loads(marker.read_text(encoding="utf-8"))
        transaction["restore_error"] = message
        _write_json_atomic(marker, transaction)
    return True


def _install_checkpoint_set(
    source: Path,
    run_dir: Path,
    target: Path,
    manifest: dict[str, Any],
    *,
    fold: int,
    include_config: bool,
) -> None:
    """Install an active checkpoint by one recoverable directory-level swap."""
    target.parent.mkdir(parents=True, exist_ok=True)
    runs_dir = run_dir.parent
    marker_path = _checkpoint_transaction_path(run_dir, fold)
    _recover_checkpoint_transaction(marker_path, runs_dir)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.checkpoint-stage.", dir=target.parent)
    )
    quarantine = target.with_name(
        f".{target.name}.checkpoint-previous.{uuid.uuid4().hex}"
    )
    try:
        if target.is_dir():
            shutil.copytree(
                target,
                staging,
                dirs_exist_ok=True,
                copy_function=os.link,
            )
        raw_restore_files = manifest.get("restore_files")
        if raw_restore_files is None:
            # Schema-v1 bundles written before task-supplied checkpoint snapshots.
            raw_restore_files = [
                "best_model.pt",
                "training_history.json",
                "sampler_audit.json",
            ]
        if not isinstance(raw_restore_files, list) or not raw_restore_files:
            raise OSError("checkpoint manifest has no restorable fold artifacts")
        relatives = [str(relative) for relative in raw_restore_files]
        for relative in relatives:
            relative_path = Path(relative)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative not in manifest["files"]
                or relative == "config.yaml"
            ):
                raise OSError(
                    f"checkpoint manifest has an unsafe restore file: {relative!r}"
                )
        if include_config:
            relatives.append("config.yaml")
        for relative in relatives:
            _atomic_copy_verified_file(
                source / relative,
                staging / relative,
                manifest["files"][relative],
            )

        transaction = {
            "run_id": run_dir.name,
            "target": target.relative_to(runs_dir).as_posix(),
            "staging": staging.relative_to(runs_dir).as_posix(),
            "quarantine": quarantine.relative_to(runs_dir).as_posix(),
            "files": {relative: manifest["files"][relative] for relative in relatives},
        }
        _write_json_atomic(marker_path, transaction)
        moved_existing = False
        try:
            if target.exists():
                os.replace(target, quarantine)
                moved_existing = True
            os.replace(staging, target)
        except OSError as install_error:
            if moved_existing and not target.exists() and quarantine.is_dir():
                try:
                    os.replace(quarantine, target)
                except OSError as rollback_error:
                    transaction["restore_error"] = (
                        f"{type(install_error).__name__}: {install_error}; rollback: "
                        f"{type(rollback_error).__name__}: {rollback_error}"
                    )
                    _write_json_atomic(marker_path, transaction)
                else:
                    _cleanup_transaction_marker(marker_path)
            raise
        else:
            shutil.rmtree(quarantine, ignore_errors=True)
            _cleanup_transaction_marker(marker_path)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _materialize_fold(source: Path, run_dir: Path, *, num_folds: int) -> None:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    fold = int(manifest["fold"])
    target = run_dir if num_folds == 1 else run_dir / f"fold_{fold}"
    if (target / "metrics.json").is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.restore.", dir=target.parent)
    )
    try:
        for relative in sorted(manifest["files"]):
            source_file = source / relative
            if relative == "config.yaml" and num_folds > 1:
                continue
            destination_file = staging / relative
            destination_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination_file)
        if not (staging / "metrics.json").is_file():
            raise OSError(f"completed fold bundle has no metrics.json: {source}")
        for relative, expected in manifest["files"].items():
            if relative == "config.yaml" and num_folds > 1:
                continue
            restored_file = staging / relative
            if (
                not restored_file.is_file()
                or restored_file.stat().st_size != expected["size"]
                or _sha256(restored_file) != expected["sha256"]
            ):
                raise OSError(f"restored fold artifact failed verification: {relative}")
        if target.exists():
            quarantine = target.with_name(f".{target.name}.partial.{uuid.uuid4().hex}")
            os.replace(target, quarantine)
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    if num_folds > 1:
        config_source = source / "config.yaml"
        config_target = run_dir / "config.yaml"
        if not config_target.is_file():
            _atomic_copy_verified_file(
                config_source, config_target, manifest["files"]["config.yaml"]
            )


def _materialize_checkpoint(source: Path, run_dir: Path, *, num_folds: int) -> None:
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    fold = int(manifest["fold"])
    target_dir = run_dir if num_folds == 1 else run_dir / f"fold_{fold}"
    if (target_dir / "metrics.json").is_file():
        return
    _install_checkpoint_set(
        source,
        run_dir,
        target_dir,
        manifest,
        fold=fold,
        include_config=num_folds == 1,
    )
    config_target = run_dir / "config.yaml"
    if not config_target.is_file():
        _atomic_copy_verified_file(
            source / "config.yaml", config_target, manifest["files"]["config.yaml"]
        )


def _load_mirror_state(state_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == 1 and isinstance(
            payload.get("entries"), dict
        ):
            return payload
    except (OSError, ValueError, AttributeError, json.JSONDecodeError):
        pass
    return {"schema_version": 1, "entries": {}}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_mirror_state(state_path: Path, state: dict[str, Any]) -> None:
    _write_json_atomic(state_path, state)


def _append_restore_error(run_dir: Path, message: str) -> None:
    state_path = run_dir / "recovery" / "mirror_state.json"
    state = _load_mirror_state(state_path)
    restore_errors = state.setdefault("restore_errors", [])
    pending = next(
        (
            error
            for error in restore_errors
            if error.get("status") == "pending" and error.get("error") == message
        ),
        None,
    )
    if pending is None:
        restore_errors.append({"status": "pending", "error": message, "attempts": 1})
    else:
        pending["attempts"] = int(pending.get("attempts", 1)) + 1
    _write_mirror_state(state_path, state)


def _record_restore_error(run_dir: Path, exc: OSError) -> None:
    message = f"{type(exc).__name__}: {exc}"
    if not run_dir.is_dir() and _record_transaction_restore_error(run_dir, message):
        return
    _append_restore_error(run_dir, message)


def _resolve_restore_errors(run_dir: Path) -> None:
    state_path = run_dir / "recovery" / "mirror_state.json"
    if not state_path.is_file():
        return
    state = _load_mirror_state(state_path)
    changed = False
    for error in state.get("restore_errors", []):
        if error.get("status") == "pending":
            error["status"] = "resolved"
            changed = True
    if changed:
        _write_mirror_state(state_path, state)


def restore_run_from_mirror(config: PipelineConfig, *, num_folds: int) -> str | None:
    """Restore a pinned or uniquely compatible mirrored run before resume resolution."""
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
    try:
        _recover_checkpoint_transactions(local_runs_dir)
    except OSError as exc:
        local_run_id = config.run_id or existing_local_run_id
        local_run = local_runs_dir / local_run_id if local_run_id is not None else None
        if local_run is None or not local_run.is_dir():
            raise OSError(
                "mirror recovery failed and no local run is available to resume"
            ) from exc
        try:
            _record_restore_error(local_run, exc)
        except OSError:
            logger.warning("Could not persist mirror restore failure state", exc_info=True)
        logger.warning(
            "Checkpoint transaction recovery remains pending; continuing from local "
            "run %s: %s",
            local_run,
            exc,
        )
        return None
    existing_local_run_id = latest_existing_run_id(local_experiment_dir)
    if config.run_id is None and existing_local_run_id is not None:
        return None

    pinned_local_run = (
        local_runs_dir / config.run_id
        if config.run_id is not None
        else None
    )
    had_local_recovery_path = pinned_local_run is not None and pinned_local_run.is_dir()
    completed_local_run = pinned_local_run is not None and _completed_local_run(
        pinned_local_run, num_folds=num_folds
    )

    mirror_runs_dir = (
        Path(config.mirror_root).resolve()
        / "experiments"
        / experiment.experiment_dirname
        / "runs"
    )
    try:
        if config.run_id is not None:
            candidates = [mirror_runs_dir / config.run_id]
        elif mirror_runs_dir.is_dir():
            candidates = [
                child
                for child in sorted(mirror_runs_dir.iterdir())
                if child.is_dir() and _compatible_mirror_run(child, config)
            ]
            if len(candidates) > 1:
                raise ValueError(
                    "resume=True found multiple compatible mirrored runs; set run.run_id "
                    "to choose one explicitly."
                )
        else:
            candidates = []

        candidates = [
            candidate
            for candidate in candidates
            if candidate.is_dir() and _compatible_mirror_run(candidate, config)
        ]
        if not candidates:
            if had_local_recovery_path and pinned_local_run is not None:
                _resolve_restore_errors(pinned_local_run)
                return None
            if config.resume:
                raise FileNotFoundError(
                    "no compatible verified mirror run exists and no local run is "
                    "available to resume"
                )
            return None
        mirror_run_dir = candidates[0]
        run_id = mirror_run_dir.name
        local_run_dir = local_experiment_dir / "runs" / run_id
        fold_bundles = _compatible_mirror_bundles(mirror_run_dir, config, kind="folds")
        checkpoint_bundles = _compatible_mirror_bundles(
            mirror_run_dir, config, kind="checkpoints"
        )
        if completed_local_run and pinned_local_run is not None:
            # Probe the mirror above so traversal failures remain observable, but a
            # completed local run is authoritative. In particular, never let a
            # self-consistent foreign mirror replace its recovery spool;
            # ArtifactMirror.retry_pending() reconciles the mirror from local bytes.
            _resolve_restore_errors(pinned_local_run)
            return None

        for _, fold_bundle in fold_bundles:
            _materialize_fold(fold_bundle, local_run_dir, num_folds=num_folds)

        for kind, bundles in (
            ("checkpoints", checkpoint_bundles),
            ("folds", fold_bundles),
        ):
            source_root = mirror_run_dir / "recovery" / kind
            for _, source_bundle in bundles:
                relative = source_bundle.relative_to(source_root)
                local_bundle = local_run_dir / "recovery" / "spool" / kind / relative
                _copy_verified_bundle(source_bundle, local_bundle)

        latest_checkpoints: dict[int, tuple[tuple[int, str], Path]] = {}
        for manifest, bundle in checkpoint_bundles:
            fold = int(manifest["fold"])
            key = (int(manifest["epoch"]), bundle.name)
            if fold not in latest_checkpoints or key > latest_checkpoints[fold][0]:
                latest_checkpoints[fold] = (key, bundle)
        for _, checkpoint_bundle in latest_checkpoints.values():
            _materialize_checkpoint(
                checkpoint_bundle, local_run_dir, num_folds=num_folds
            )
        _resolve_restore_errors(local_run_dir)
        return run_id
    except OSError as exc:
        if had_local_recovery_path and pinned_local_run is not None:
            try:
                _record_restore_error(pinned_local_run, exc)
            except OSError:
                logger.warning(
                    "Could not persist mirror restore failure state", exc_info=True
                )
            logger.warning(
                "Mirror restore remains pending; continuing from local run %s: %s",
                pinned_local_run,
                exc,
            )
            return None
        raise OSError(
            "mirror recovery failed and no local run is available to resume"
        ) from exc


def _completed_local_run(run_dir: Path, *, num_folds: int) -> bool:
    if num_folds == 1:
        return (run_dir / "metrics.json").is_file()
    return all(
        (run_dir / f"fold_{fold}" / "metrics.json").is_file()
        for fold in range(num_folds)
    )


class ArtifactMirror:
    """Spool immutable local bundles and publish verified copies to shared storage.

    Public methods are deliberately fail-soft: local training is authoritative, so an
    unavailable mirror records a pending attempt and returns without raising.
    """

    def __init__(self, *, run_dir: Path, mirror_run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.mirror_run_dir = Path(mirror_run_dir)
        self.recovery_dir = self.run_dir / "recovery"
        self.spool_dir = self.recovery_dir / "spool"
        self.state_path = self.recovery_dir / "mirror_state.json"
        self._state: dict[str, Any] = self._load_state()
        self._discover_spools()

    def checkpoint_improved(
        self,
        *,
        fold: int,
        epoch: int,
        artifacts: Mapping[str, Path],
        restore_files: Sequence[str],
    ) -> None:
        """Spool and publish a caller-described, restorable checkpoint snapshot."""
        try:
            sources = {
                str(relative): Path(source)
                for relative, source in artifacts.items()
            }
            if not sources or "config.yaml" in sources:
                raise ValueError(
                    "checkpoint artifacts must be non-empty and must not override "
                    "the run config.yaml"
                )
            for relative in sources:
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(
                        f"unsafe checkpoint artifact name: {relative!r}"
                    )
            restore = [str(relative) for relative in restore_files]
            if not restore or len(set(restore)) != len(restore):
                raise ValueError("restore_files must be a non-empty unique sequence")
            unknown_restore = sorted(set(restore) - set(sources))
            if unknown_restore:
                raise ValueError(
                    "restore_files reference unknown checkpoint artifacts: "
                    f"{unknown_restore}"
                )
            sources["config.yaml"] = self.run_dir / "config.yaml"
            spool = (
                self.spool_dir / "checkpoints" / f"fold_{fold}" / f"epoch_{epoch:04d}"
            )
            spool = self._spool(
                spool,
                kind="checkpoint",
                fold=fold,
                epoch=epoch,
                sources=sources.items(),
                preserve_collision=True,
                manifest_metadata={"restore_files": restore},
            )
            self._register(spool)
            self.retry_pending()
        except Exception as exc:  # mirror bookkeeping must never terminate training
            self._record_unhandled("checkpoint", fold, exc)

    def fold_completed(self, *, fold: int, fold_dir: Path) -> None:
        """Spool and attempt publication of one coherent completed-fold bundle."""
        try:
            fold_dir = Path(fold_dir)
            sources: list[tuple[str, Path]] = []
            for source in sorted(
                path for path in fold_dir.rglob("*") if path.is_file()
            ):
                relative = source.relative_to(fold_dir)
                if relative.parts and relative.parts[0] == "recovery":
                    continue
                sources.append((relative.as_posix(), source))
            if not any(relative == "config.yaml" for relative, _ in sources):
                sources.append(("config.yaml", self.run_dir / "config.yaml"))
            spool = self.spool_dir / "folds" / f"fold_{fold}"
            spool = self._spool(
                spool,
                kind="fold",
                fold=fold,
                epoch=None,
                sources=sources,
                preserve_collision=False,
            )
            self._register(spool)
            self.retry_pending()
        except Exception as exc:  # mirror bookkeeping must never terminate training
            self._record_unhandled("fold", fold, exc)

    def retry_pending(self) -> None:
        """Retry every locally spooled bundle not already verified at its destination."""
        try:
            self._discover_spools()
            for bundle_id in sorted(self._state["entries"]):
                entry = self._state["entries"][bundle_id]
                source = self.run_dir / entry["source"]
                destination = self.mirror_run_dir / entry["destination"]
                if _matching_bundles(source, destination):
                    entry.update(status="verified", last_error=None)
                    continue
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                try:
                    _copy_verified_bundle(source, destination)
                except Exception as exc:
                    entry.update(
                        status="pending", last_error=f"{type(exc).__name__}: {exc}"
                    )
                    self._state.setdefault("failures", []).append(
                        {
                            "bundle": bundle_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempt": entry["attempts"],
                        }
                    )
                    logger.warning(
                        "Artifact mirror publication remains pending for %s: %s",
                        bundle_id,
                        exc,
                    )
                else:
                    entry.update(status="verified", last_error=None)
            self._save_state()
        except Exception as exc:
            logger.warning(
                "Artifact mirror retry bookkeeping failed: %s", exc, exc_info=True
            )

    def _spool(
        self,
        spool: Path,
        *,
        kind: str,
        fold: int,
        epoch: int | None,
        sources: Iterable[tuple[str, Path]],
        preserve_collision: bool,
        manifest_metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        if not preserve_collision and _verified_bundle(spool):
            return spool
        spool.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{spool.name}.tmp.", dir=spool.parent))
        try:
            for relative, source in sources:
                if not source.is_file():
                    raise FileNotFoundError(
                        f"required recovery artifact is missing: {source}"
                    )
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            manifest = {
                "schema_version": 1,
                "kind": kind,
                "fold": fold,
                "epoch": epoch,
                "files": _manifest_files(staging),
            }
            if manifest_metadata is not None:
                overlap = set(manifest) & set(manifest_metadata)
                if overlap:
                    raise ValueError(
                        f"manifest metadata cannot override reserved keys: {sorted(overlap)}"
                    )
                manifest.update(manifest_metadata)
            (staging / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            if not _verified_bundle(staging):
                raise OSError(f"local recovery spool failed verification: {staging}")
            if _matching_bundles(staging, spool):
                return spool
            if preserve_collision and _verified_bundle(spool):
                digest = _sha256(staging / "manifest.json")[:12]
                spool = spool.with_name(f"{spool.name}__{digest}")
                if _matching_bundles(staging, spool):
                    return spool
            if spool.exists():
                quarantine = spool.with_name(
                    f".{spool.name}.corrupt.{uuid.uuid4().hex}"
                )
                os.replace(spool, quarantine)
            os.replace(staging, spool)
            return spool
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    def _register(self, spool: Path) -> None:
        relative_source = spool.relative_to(self.run_dir)
        relative_spool = spool.relative_to(self.spool_dir)
        destination = Path("recovery") / relative_spool
        bundle_id = relative_spool.as_posix()
        self._state["entries"].setdefault(
            bundle_id,
            {
                "source": relative_source.as_posix(),
                "destination": destination.as_posix(),
                "status": "pending",
                "attempts": 0,
                "last_error": None,
            },
        )
        self._save_state()

    def _discover_spools(self) -> None:
        if not self.spool_dir.is_dir():
            return
        for manifest in sorted(self.spool_dir.rglob("manifest.json")):
            spool = manifest.parent
            if _verified_bundle(spool):
                self._register(spool)

    def _load_state(self) -> dict[str, Any]:
        return _load_mirror_state(self.state_path)

    def _save_state(self) -> None:
        _write_mirror_state(self.state_path, self._state)

    def _record_unhandled(self, kind: str, fold: int, exc: Exception) -> None:
        logger.warning(
            "Artifact mirror %s handling failed for fold %d: %s",
            kind,
            fold,
            exc,
            exc_info=True,
        )
        try:
            failures = self._state.setdefault("failures", [])
            failures.append(
                {
                    "kind": kind,
                    "fold": fold,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self._save_state()
        except Exception:
            logger.warning(
                "Could not persist artifact mirror failure state", exc_info=True
            )
