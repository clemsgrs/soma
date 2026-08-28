from __future__ import annotations

import json
from pathlib import Path

from soma.artifact_mirror import ArtifactMirror, _install_checkpoint_set, _resume_recipe


def test_resume_recipe_ignores_operational_evidence_export_setting() -> None:
    with_export = {"evaluation": {"save_segmentation_confusion_evidence": True}}
    without_export = {"evaluation": {}}

    assert _resume_recipe(with_export) == _resume_recipe(without_export) == {"evaluation": {}}


def test_resume_recipe_treats_other_evaluation_settings_as_recipe() -> None:
    payload = {"evaluation": {"patient_oof": {"arm": "legacy"}}}

    assert _resume_recipe(payload) == payload


def _write_checkpoint_inputs(run_dir: Path, *, checkpoint: bytes, epoch: int) -> Path:
    fold_dir = run_dir / "fold_0"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text("run:\n  seed: 0\n", encoding="utf-8")
    (fold_dir / "best_model.pt").write_bytes(checkpoint)
    (fold_dir / "training_history.json").write_text(
        json.dumps({"epochs": [{"epoch": epoch}]}), encoding="utf-8"
    )
    (fold_dir / "sampler_audit.json").write_text(
        json.dumps({"strategy": "class_conditioned", "epochs": [epoch]}),
        encoding="utf-8",
    )
    return fold_dir


def test_same_epoch_different_checkpoint_content_is_preserved(tmp_path: Path) -> None:
    run_dir = tmp_path / "local" / "run"
    mirror_run_dir = tmp_path / "shared" / "run"
    fold_dir = _write_checkpoint_inputs(run_dir, checkpoint=b"attempt-one", epoch=0)
    mirror = ArtifactMirror(run_dir=run_dir, mirror_run_dir=mirror_run_dir)

    mirror.checkpoint_improved(
        fold=0,
        epoch=0,
        artifacts={
            "best_model.pt": fold_dir / "best_model.pt",
            "training_history.json": fold_dir / "training_history.json",
            "sampler_audit.json": fold_dir / "sampler_audit.json",
        },
        restore_files=[
            "best_model.pt",
            "training_history.json",
            "sampler_audit.json",
        ],
    )
    _write_checkpoint_inputs(run_dir, checkpoint=b"attempt-two", epoch=0)
    mirror.checkpoint_improved(
        fold=0,
        epoch=0,
        artifacts={
            "best_model.pt": fold_dir / "best_model.pt",
            "training_history.json": fold_dir / "training_history.json",
            "sampler_audit.json": fold_dir / "sampler_audit.json",
        },
        restore_files=[
            "best_model.pt",
            "training_history.json",
            "sampler_audit.json",
        ],
    )

    local_checkpoints = sorted(
        path.read_bytes()
        for path in (run_dir / "recovery" / "spool" / "checkpoints").rglob(
            "best_model.pt"
        )
    )
    mirrored_checkpoints = sorted(
        path.read_bytes()
        for path in (mirror_run_dir / "recovery" / "checkpoints").rglob(
            "best_model.pt"
        )
    )
    state = json.loads((run_dir / "recovery" / "mirror_state.json").read_text())

    assert local_checkpoints == mirrored_checkpoints == [b"attempt-one", b"attempt-two"]
    assert len(state["entries"]) == 2
    assert all(entry["status"] == "verified" for entry in state["entries"].values())


def test_checkpoint_snapshot_accepts_task_supplied_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "local" / "run"
    mirror_run_dir = tmp_path / "shared" / "run"
    fold_dir = run_dir / "fold_0"
    fold_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text("run:\n  seed: 0\n", encoding="utf-8")
    (fold_dir / "weights.bin").write_bytes(b"weights")
    (fold_dir / "optimizer.json").write_text("{}", encoding="utf-8")
    mirror = ArtifactMirror(run_dir=run_dir, mirror_run_dir=mirror_run_dir)

    mirror.checkpoint_improved(
        fold=0,
        epoch=2,
        artifacts={
            "weights.bin": fold_dir / "weights.bin",
            "optimizer.json": fold_dir / "optimizer.json",
        },
        restore_files=["weights.bin", "optimizer.json"],
    )

    bundle = next((mirror_run_dir / "recovery" / "checkpoints").rglob("epoch_*"))
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["restore_files"] == ["weights.bin", "optimizer.json"]
    assert set(manifest["files"]) == {
        "config.yaml",
        "optimizer.json",
        "weights.bin",
    }
    restored_run = tmp_path / "restored" / "run"
    restored_fold = restored_run / "fold_0"
    _install_checkpoint_set(
        bundle,
        restored_run,
        restored_fold,
        manifest,
        fold=0,
        include_config=False,
    )
    assert (restored_fold / "weights.bin").read_bytes() == b"weights"
    assert (restored_fold / "optimizer.json").read_text() == "{}"
