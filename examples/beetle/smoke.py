"""Run the complete BEETLE development chain against tiny offline cached grids."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from examples.beetle.launch import (
    ENCODER_LOCK_PATH,
    resolve_arm_configs,
)
from examples.beetle.protocol import ARM_NAMES, BATCH_SIZE_CANDIDATES, NUM_FOLDS
from examples.beetle.report_oof import BeetleCohort, assemble_beetle_oof_report
from examples.beetle.select_arm import select_development_arm
from soma.config import load_config
from soma.dense.geometry import compute_dense_geometry
from soma.dense.store import dense_grid_metadata, write_dense_grid
from soma.pipeline import Pipeline


FEATURE_DIM = 1280
TARGET_SIZE = 8
PATCH_SIZE = 4


def _write_offline_preflight(path: Path) -> Path:
    lock = json.loads(ENCODER_LOCK_PATH.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "status": "completed",
        "scope": "offline_smoke",
        "batch_size_candidates": list(BATCH_SIZE_CANDIDATES),
        "batch_size_attempts": [
            {"batch_size": batch_size, "passed": batch_size == 4}
            for batch_size in BATCH_SIZE_CANDIDATES
        ],
        "selected_batch_size": 4,
        "same_batch_every_arm_and_fold": True,
        "encoder": {**lock, "weight_checksum_verified": True},
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_cached_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    feature_dir = root / "cached_grids"
    masks_dir = root / "masks"
    feature_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    geometry = compute_dense_geometry(target_size=TARGET_SIZE, patch_size=PATCH_SIZE)
    metadata = dense_grid_metadata(
        geometry,
        feature_dim=FEATURE_DIM,
        pad_mode="reflect",
        spacing_um=0.5,
    )

    dataset_rows = ["sample_id,image_path,label_mask_path,patient_id"]
    patient_rows = ["sample_id,patient_id"]
    sample_ids: list[str] = []
    base_grid = torch.linspace(
        -1.0,
        1.0,
        FEATURE_DIM * geometry.grid_shape[0] * geometry.grid_shape[1],
        dtype=torch.float16,
    ).reshape(FEATURE_DIM, *geometry.grid_shape)
    mask = np.tile(np.arange(4, dtype=np.uint8), (TARGET_SIZE, TARGET_SIZE // 4))
    for patient in range(NUM_FOLDS):
        for roi in range(2):
            sample_id = f"p{patient}_roi{roi}"
            sample_ids.append(sample_id)
            write_dense_grid(feature_dir, sample_id, base_grid + patient / 100, metadata)
            mask_path = masks_dir / f"{sample_id}.png"
            Image.fromarray(mask).save(mask_path)
            dataset_rows.append(
                f"{sample_id},{root / 'unused_images' / (sample_id + '.png')},"
                f"{mask_path},p{patient}"
            )
            patient_rows.append(f"{sample_id},p{patient}")

    dataset_csv = root / "dataset.csv"
    dataset_csv.write_text("\n".join(dataset_rows) + "\n", encoding="utf-8")
    sample_patient_csv = root / "sample_patient.csv"
    sample_patient_csv.write_text("\n".join(patient_rows) + "\n", encoding="utf-8")
    split_rows = ["sample_id,split,fold"]
    for fold in range(NUM_FOLDS):
        tune_patient = (fold + 1) % NUM_FOLDS
        for sample_id in sample_ids:
            patient = int(sample_id[1])
            split = "test" if patient == fold else "tune" if patient == tune_patient else "train"
            split_rows.append(f"{sample_id},{split},{fold}")
    splits_csv = root / "splits.csv"
    splits_csv.write_text("\n".join(split_rows) + "\n", encoding="utf-8")
    return dataset_csv, splits_csv, sample_patient_csv, feature_dir


def run_offline_smoke(output_dir: str | Path) -> Path:
    """Exercise both project configs without slides, model weights, CUDA, or a network."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight = _write_offline_preflight(output_dir / "offline_preflight.json")
    resolved_paths = resolve_arm_configs(
        preflight_path=preflight,
        output_dir=output_dir / "resolved_configs",
        allow_offline_smoke=True,
    )
    dataset_csv, splits_csv, sample_patient_csv, feature_dir = _write_cached_fixture(
        output_dir / "fixture"
    )

    evidence_by_arm: dict[str, list[Path]] = {}
    arm_artifacts: dict[str, dict[str, list[str]]] = {}
    for arm in ARM_NAMES:
        production = load_config(resolved_paths[arm])
        smoke_preprocessing = replace(
            production.preprocessing,
            requested_tile_size_px=TARGET_SIZE,
            dense_window_size=None,
            dense_window_overlap=0.0,
            masks=None,
            sampling=None,
        )
        smoke_training = replace(
            production.training,
            epochs=1,
            patience=1,
            roi_draws_per_epoch=4,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        config = replace(
            production,
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_dir / "runs" / arm,
            mirror_root=output_dir / "mirror" / arm,
            preprocessing=smoke_preprocessing,
            training=smoke_training,
        )
        # This project smoke is a CPU-only contract even when invoked on a GPU node.
        # Cached grids need no encoder device, and masking availability before Pipeline
        # resolves the decoder device prevents accidental CUDA use and false provenance.
        with patch.object(torch.cuda, "is_available", return_value=False):
            result = Pipeline(config, feature_dir=feature_dir).run()
        run_dir = result.run_dir
        mirrored_run = config.mirror_root / run_dir.relative_to(config.output_root)
        evidence = [
            run_dir / f"fold_{fold}" / "confusion_evidence_tune.json"
            for fold in range(NUM_FOLDS)
        ]
        audits = [
            run_dir / f"fold_{fold}" / "roi_batch_sampling.json"
            for fold in range(NUM_FOLDS)
        ]
        fold_manifests = [
            mirrored_run / "recovery" / "folds" / f"fold_{fold}" / "manifest.json"
            for fold in range(NUM_FOLDS)
        ]
        checkpoint_manifests = [
            next(
                (mirrored_run / "recovery" / "checkpoints" / f"fold_{fold}").glob(
                    "epoch_*/manifest.json"
                )
            )
            for fold in range(NUM_FOLDS)
        ]
        evidence_by_arm[arm] = evidence
        arm_artifacts[arm] = {
            "confusion_evidence": [str(path) for path in evidence],
            "sampling_audits": [str(path) for path in audits],
            "recovery_fold_manifests": [str(path) for path in fold_manifests],
            "recovery_checkpoint_manifests": [str(path) for path in checkpoint_manifests],
        }

    report = assemble_beetle_oof_report(
        evidence_by_arm=evidence_by_arm,
        sample_patient_csv=sample_patient_csv,
        cohort=BeetleCohort(
            primary_patient_count=5,
            sensitivity_patient_count=2,
            spacing_exception_patient_ids=("p2", "p3", "p4"),
        ),
    )
    report_path = output_dir / "oof_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    selection_path = output_dir / "arm_selection.json"
    selection_path.write_text(
        json.dumps(select_development_arm(report_path), indent=2) + "\n",
        encoding="utf-8",
    )
    smoke_manifest = {
        "schema_version": 1,
        "offline": True,
        "used_gated_weights": False,
        "used_wsis": False,
        "used_gpu": False,
        "resolved_configs": {arm: str(path) for arm, path in resolved_paths.items()},
        "cache": {"path": str(feature_dir), "feature_dim": FEATURE_DIM, "dtype": "float16"},
        "arms": arm_artifacts,
        "oof_report": str(report_path),
        "arm_selection": str(selection_path),
    }
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(run_offline_smoke(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_offline_smoke", "main"]
