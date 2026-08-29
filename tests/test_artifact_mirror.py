from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import shutil

import pytest

from soma.artifact_mirror import ArtifactMirror, _resume_recipe
from soma.training.trainer import Trainer


def test_resume_recipe_ignores_operational_evidence_export_setting() -> None:
    with_export = {"evaluation": {"save_segmentation_confusion_evidence": True}}
    without_export = {"evaluation": {}}

    assert _resume_recipe(with_export) == _resume_recipe(without_export) == {
        "evaluation": {}
    }


def test_resume_recipe_treats_other_evaluation_settings_as_recipe() -> None:
    payload = {"evaluation": {"patient_oof": {"arm": "legacy"}}}

    assert _resume_recipe(payload) == payload


def test_completed_fold_is_published_directly_with_verified_manifest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "local" / "run"
    fold_dir = run_dir / "fold_0"
    fold_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("run:\n  seed: 0\n", encoding="utf-8")
    (fold_dir / "metrics.json").write_text('{"dice": 0.75}', encoding="utf-8")
    (fold_dir / "best_model.pt").write_bytes(b"selected-checkpoint")
    mirror_run_dir = tmp_path / "shared" / "run"

    ArtifactMirror(
        run_dir=run_dir, mirror_run_dir=mirror_run_dir
    ).fold_completed(fold=0, fold_dir=fold_dir)

    bundle = mirror_run_dir / "recovery" / "folds" / "fold_0"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest == {
        "schema_version": 1,
        "kind": "fold",
        "fold": 0,
        "files": {
            "best_model.pt": {
                "sha256": (
                    "411a6035c7d6e222474a08b93c7f4f75"
                    "d5ce6263652a1787c2e852242b8438a4"
                ),
                "size": 19,
            },
            "config.yaml": {
                "sha256": (
                    "93764599da0d274d1cbbbed0c41825ae"
                    "fbe8ab3720ebbc2cfc8c38f041fb2b8a"
                ),
                "size": 15,
            },
            "metrics.json": {
                "sha256": (
                    "84877ca75a1b504bd8b28c30e7e006cc"
                    "f98c2331626af1fad8d77b3d968a4ccf"
                ),
                "size": 14,
            },
        },
    }
    assert (bundle / "best_model.pt").read_bytes() == b"selected-checkpoint"
    assert not (run_dir / "recovery").exists()


def test_trainer_has_no_per_improvement_publication_hook() -> None:
    assert "on_checkpoint_improved" not in inspect.signature(Trainer).parameters


def _completed_run(tmp_path: Path) -> tuple[Path, Path, ArtifactMirror]:
    run_dir = tmp_path / "local" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("run:\n  seed: 0\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    mirror_run_dir = tmp_path / "shared" / "run"
    return (
        run_dir,
        mirror_run_dir,
        ArtifactMirror(run_dir=run_dir, mirror_run_dir=mirror_run_dir),
    )


def _shared_storage_outage(mirror_run_dir: Path):
    real_copy2 = shutil.copy2

    def fail_shared_copy(source, destination, *args, **kwargs):
        if Path(destination).is_relative_to(mirror_run_dir):
            raise OSError("shared storage unavailable")
        return real_copy2(source, destination, *args, **kwargs)

    return fail_shared_copy


def test_outage_does_not_interrupt_fold_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, mirror_run_dir, mirror = _completed_run(tmp_path)
    monkeypatch.setattr(
        "soma.artifact_mirror.shutil.copy2",
        _shared_storage_outage(mirror_run_dir),
    )
    mirror.fold_completed(fold=0, fold_dir=run_dir)

    bundle = mirror_run_dir / "recovery" / "folds" / "fold_0"
    assert not bundle.exists()
    assert not (run_dir / "recovery" / "mirror_state.json").exists()


def test_retry_publishes_a_completed_fold_after_an_outage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, mirror_run_dir, mirror = _completed_run(tmp_path)
    monkeypatch.setattr(
        "soma.artifact_mirror.shutil.copy2",
        _shared_storage_outage(mirror_run_dir),
    )
    mirror.fold_completed(fold=0, fold_dir=run_dir)
    monkeypatch.undo()
    mirror.retry_pending()

    bundle = mirror_run_dir / "recovery" / "folds" / "fold_0"
    assert (bundle / "manifest.json").is_file()


def test_retry_does_not_rehash_an_already_published_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir, _, mirror = _completed_run(tmp_path)
    mirror.fold_completed(fold=0, fold_dir=run_dir)
    rehashed: list[Path] = []

    def record_rehash(path: Path) -> str:
        rehashed.append(path)
        return ""

    monkeypatch.setattr("soma.artifact_mirror._sha256", record_rehash)
    mirror.retry_pending()

    assert rehashed == []


def test_interrupted_publish_never_exposes_a_completed_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "local" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("run:\n  seed: 0\n", encoding="utf-8")
    (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    mirror_run_dir = tmp_path / "shared" / "run"
    destination = mirror_run_dir / "recovery" / "folds" / "fold_0"
    real_replace = os.replace

    def interrupt_final_swap(source, target, *args, **kwargs):
        if Path(target) == destination:
            raise OSError("interrupted before atomic publication")
        return real_replace(source, target, *args, **kwargs)

    monkeypatch.setattr("soma.artifact_mirror.os.replace", interrupt_final_swap)
    ArtifactMirror(
        run_dir=run_dir, mirror_run_dir=mirror_run_dir
    ).fold_completed(fold=0, fold_dir=run_dir)

    assert not destination.exists()
    assert not list(destination.parent.glob(".fold_0.tmp.*"))
