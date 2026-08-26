from __future__ import annotations

import json
from pathlib import Path

from soma.artifact_mirror import ArtifactMirror


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
        fold_dir=fold_dir,
        checkpoint_path=fold_dir / "best_model.pt",
    )
    _write_checkpoint_inputs(run_dir, checkpoint=b"attempt-two", epoch=0)
    mirror.checkpoint_improved(
        fold=0,
        epoch=0,
        fold_dir=fold_dir,
        checkpoint_path=fold_dir / "best_model.pt",
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
