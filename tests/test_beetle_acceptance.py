"""Deterministic tiny-artifact fixtures for the BEETLE handoff acceptance gate."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image
import pytest
import yaml

from examples.beetle import launch
from examples.beetle.acceptance import (
    assemble_archive,
    main as acceptance_main,
    validate_archive,
    write_acceptance_report,
    write_artifact_manifest,
)

ARMS = ("uniform", "class_conditioned")
FOLDS = 5
EXPECTED_MASKS = 170
EXPECTED_PATIENTS = 527
EXTERNAL_PATIENTS = 54
SPACING_EXCEPTIONS = ("TCGA-OL-A66I", "TCGA-OL-A66P", "TCGA-OL-A6VO")
GIT_SHA = "1234567890abcdef1234567890abcdef12345678"
BATCH_SIZE = 8


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _checkpoint_bytes(arm: str, fold: int) -> bytes:
    return f"tiny fake decoder checkpoint {arm} fold {fold}".encode("utf-8")


def _mask_names() -> list[str]:
    return [f"roi_{index:03d}.png" for index in range(EXPECTED_MASKS)]


def _patient_ids() -> list[str]:
    return list(SPACING_EXCEPTIONS) + [
        f"P{index:03d}" for index in range(EXPECTED_PATIENTS - len(SPACING_EXCEPTIONS))
    ]


def _merged_arm_config(arm: str, cache_root: str) -> dict:
    """Merge the tracked base recipe with one tracked arm overlay, test-side."""
    base = yaml.safe_load((launch.CONFIG_DIR / "base.yaml").read_text(encoding="utf-8"))
    overlay = yaml.safe_load(
        (launch.CONFIG_DIR / f"{arm}.yaml").read_text(encoding="utf-8")
    )
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    merged["training"]["batch_size"] = BATCH_SIZE
    merged["cache"]["root_dir"] = cache_root
    return merged


def _arm_oof_payload() -> dict:
    patients = _patient_ids()
    identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    return {
        "coverage": {
            "expected_patient_count": EXPECTED_PATIENTS,
            "observed_patient_count": EXPECTED_PATIENTS,
            "patient_ids": patients,
            "folds": [0, 1, 2, 3, 4],
            "exactly_once": True,
        },
        "patient_confusions": [
            {
                "patient_id": patient_id,
                "fold": index % FOLDS,
                "confusion_matrix": identity,
            }
            for index, patient_id in enumerate(patients)
        ],
        "spacing_sensitivity": {
            "evaluation_only": True,
            "excluded_patient_ids": list(SPACING_EXCEPTIONS),
        },
    }


def _build_inputs(root: Path) -> dict:
    """Write every campaign artifact the assembler consumes, with real schemas."""
    lock = json.loads(launch.ENCODER_LOCK_PATH.read_text(encoding="utf-8"))
    cache_root = (
        f"data/beetle/cache/virchow2_{lock['revision']}_{lock['weight_sha256']}"
        "_dense_fp16"
    )

    resolved_dir = root / "resolved"
    resolved_dir.mkdir(parents=True)
    for arm in ARMS:
        (resolved_dir / f"{arm}.yaml").write_text(
            yaml.safe_dump(_merged_arm_config(arm, cache_root), sort_keys=False),
            encoding="utf-8",
        )

    preflight_path = _write_json(
        root / "hardware_preflight.json",
        {
            "schema_version": 1,
            "status": "completed",
            "scope": "campaign",
            "batch_size_candidates": [16, 8, 4],
            "batch_size_attempts": [
                {"batch_size": candidate, "passed": candidate <= BATCH_SIZE}
                for candidate in (16, 8, 4)
            ],
            "selected_batch_size": BATCH_SIZE,
            "same_batch_every_arm_and_fold": True,
            "encoder": {
                **lock,
                "weight_checksum_verified": True,
                "snapshot_path": "/maindisk/clement/hf/models--paige-ai--Virchow2/"
                f"snapshots/{lock['revision']}",
            },
        },
    )
    _write_json(
        resolved_dir / "protocol_resolution.json",
        {
            "schema_version": 1,
            "hardware_preflight": {
                "path": str(preflight_path),
                "sha256": _sha256_file(preflight_path),
                "selected_batch_size": BATCH_SIZE,
            },
            "encoder_lock": lock,
            "encoder_preflight": {**lock, "weight_checksum_verified": True},
            "encoder_runtime": {
                "mode": "offline_immutable_hub_snapshot",
                "snapshot_path": "/maindisk/clement/hf/snapshot",
                "hf_hub_cache": "/maindisk/clement/hf/hub",
            },
            "arms": {arm: str(resolved_dir / f"{arm}.yaml") for arm in ARMS},
        },
    )

    run_dirs: dict[str, Path] = {}
    for arm in ARMS:
        run_dir = root / "runs" / arm
        run_dirs[arm] = run_dir
        run_dir.mkdir(parents=True)
        (run_dir / "run.yaml").write_text(
            yaml.safe_dump(
                {
                    "run_id": f"2026-08-27_00-00-00_{arm}",
                    "status": "completed",
                    "git_sha": GIT_SHA,
                    "git_dirty": "false",
                    "environment": {"soma": "1.0.0", "torch": "2.4.0", "cuda": "12.4"},
                    "dataset_file_checksum": "a" * 64,
                    "splits_file_checksum": "b" * 64,
                }
            ),
            encoding="utf-8",
        )
        for fold in range(FOLDS):
            fold_dir = run_dir / f"fold_{fold}"
            fold_dir.mkdir()
            (fold_dir / "best_model.pt").write_bytes(_checkpoint_bytes(arm, fold))
            _write_json(
                fold_dir / "training_history.json",
                {"fold": fold, "history": [{"epoch": 0, "tune_dice": 0.9}]},
            )
            _write_json(
                fold_dir / "roi_batch_sampling.json",
                {"schema_version": 1, "arm": arm, "fold": fold, "draws": 4},
            )

    oof_report = _write_json(
        root / "oof_report.json",
        {
            "schema_version": 1,
            "protocol": {"folds": FOLDS, "arms": list(ARMS)},
            "arms": {arm: _arm_oof_payload() for arm in ARMS},
        },
    )
    arm_selection = _write_json(
        root / "arm_selection.json",
        {
            "schema_version": 1,
            "criterion": "fold_macro_class_dice",
            "development_evidence_only": True,
            "tie_breaker": "uniform",
            "selected_arm": "uniform",
            "arms": {
                arm: {
                    "fold_scores": [0.9, 0.91, 0.92, 0.93, 0.94],
                    "mean": 0.92,
                    "standard_deviation": 0.0158,
                }
                for arm in ARMS
            },
        },
    )

    mask_names = _mask_names()
    roi_sidecar = _write_json(
        root / "roi_to_wsi.json",
        {
            "schema_version": 1,
            "rois": [
                {
                    "roi_filename": name,
                    "patient_id": f"EXT{index % EXTERNAL_PATIENTS:02d}",
                    "source_wsi": f"wsi_{index:03d}",
                    "native_spacing_um": 0.5,
                    "width": 4,
                    "height": 3,
                }
                for index, name in enumerate(mask_names)
            ],
        },
    )
    masks_dir = root / "submission_pngs"
    masks_dir.mkdir()
    for index, name in enumerate(mask_names):
        pixels = np.full((3, 4), (index % 4) + 1, dtype=np.uint8)
        Image.fromarray(pixels, mode="L").save(masks_dir / name)
    submission_zip = root / "submission.zip"
    with zipfile.ZipFile(submission_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(mask_names):
            archive.write(masks_dir / name, arcname=name)

    submission_audit = _write_json(
        root / "submission_audit.json",
        {
            "schema_version": 1,
            "selected_arm": "uniform",
            "probability_ensemble": "mean_of_five_fold_softmaxes",
            "hidden_external_labels_used": False,
            "checkpoints": [
                {
                    "fold": fold,
                    "path": str(run_dirs["uniform"] / f"fold_{fold}" / "best_model.pt"),
                    "sha256": _sha256_bytes(_checkpoint_bytes("uniform", fold)),
                }
                for fold in range(FOLDS)
            ],
            "roi_decisions": [
                {"roi_filename": name, "spacing_decision": "native_within_tolerance"}
                for name in mask_names
            ],
            "submission_zip": {
                "path": str(submission_zip),
                "sha256": _sha256_file(submission_zip),
            },
        },
    )

    return {
        "run_dirs": run_dirs,
        "resolved_dir": resolved_dir,
        "hardware_preflight": preflight_path,
        "oof_report": oof_report,
        "arm_selection": arm_selection,
        "roi_sidecar": roi_sidecar,
        "masks_dir": masks_dir,
        "submission_zip": submission_zip,
        "submission_audit": submission_audit,
    }


@pytest.fixture(scope="module")
def pristine_archive(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("beetle-handoff")
    inputs = _build_inputs(root)
    return assemble_archive(**inputs, archive_dir=root / "archive")


@pytest.fixture
def archive(pristine_archive: Path, tmp_path: Path) -> Path:
    target = tmp_path / "archive"
    shutil.copytree(pristine_archive, target)
    return target


def _failures(archive_dir: Path) -> list[str]:
    return list(validate_archive(archive_dir).failures)


def _edit_json(path: Path, mutate) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_assembled_archive_validates_clean_and_report_accepts(
    archive: Path, tmp_path: Path
) -> None:
    outcome = validate_archive(archive)
    assert outcome.passed, outcome.failures
    assert acceptance_main(["validate", "--archive-dir", str(archive)]) == 0

    report = write_acceptance_report(archive, tmp_path / "report")
    assert report["passed"] is True
    assert report["failure_count"] == 0
    assert report["soma_commit_sha"] == GIT_SHA
    assert report["selected_arm"] == "uniform"
    assert report["external_metrics"] == "pending"
    assert all(group["complete"] for group in report["artifact_groups"])
    text = (tmp_path / "report" / "acceptance_report.txt").read_text(encoding="utf-8")
    assert "ACCEPTED" in text
    assert GIT_SHA in text

    readme = (archive / "README.md").read_text(encoding="utf-8")
    assert "uniform" in readme
    assert GIT_SHA in readme


def test_missing_file_is_reported_and_report_still_writes(
    archive: Path, tmp_path: Path
) -> None:
    victim = "histories/uniform/fold_2/training_history.json"
    (archive / victim).unlink()

    failures = _failures(archive)
    assert any(
        victim in failure and "listed in artifact_manifest.json" in failure
        for failure in failures
    ), failures
    assert any(
        "training history" in failure and "fold 2" in failure for failure in failures
    ), failures
    assert acceptance_main(["validate", "--archive-dir", str(archive)]) == 1

    report = write_acceptance_report(archive, tmp_path / "report")
    assert report["passed"] is False
    assert any(victim in failure for failure in report["checks"]["manifest_coverage"]["failures"])
    text = (tmp_path / "report" / "acceptance_report.txt").read_text(encoding="utf-8")
    assert "REJECTED" in text
    assert victim in text


def test_extra_file_not_listed_in_manifest_is_reported(archive: Path) -> None:
    (archive / "development" / "notes.txt").write_text("stray\n", encoding="utf-8")

    failures = _failures(archive)
    assert any(
        "development/notes.txt is present in the archive but not listed in "
        "artifact_manifest.json" in failure
        for failure in failures
    ), failures


def test_checksum_mismatch_names_file_and_digests(archive: Path) -> None:
    victim = archive / "checkpoints" / "class_conditioned" / "fold_1" / "best_model.pt"
    original = victim.read_bytes()
    victim.write_bytes(b"X" * len(original))

    failures = _failures(archive)
    assert any(
        "checkpoints/class_conditioned/fold_1/best_model.pt sha256" in failure
        and "does not match artifact_manifest.json" in failure
        for failure in failures
    ), failures


def test_nine_checkpoints_fail_the_exact_count(archive: Path) -> None:
    (archive / "checkpoints" / "class_conditioned" / "fold_4" / "best_model.pt").unlink()
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "expected exactly 10 decoder checkpoints (2 arms x 5 folds), found 9"
        in failure
        for failure in failures
    ), failures
    assert any(
        "decoder checkpoint checkpoints/class_conditioned/fold_4/best_model.pt is "
        "missing" in failure
        for failure in failures
    ), failures


def test_eleven_checkpoints_fail_the_exact_count(archive: Path) -> None:
    stray = archive / "checkpoints" / "uniform" / "fold_5" / "best_model.pt"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"eleventh checkpoint")
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "expected exactly 10 decoder checkpoints (2 arms x 5 folds), found 11"
        in failure
        for failure in failures
    ), failures
    assert any(
        "unexpected file checkpoints/uniform/fold_5/best_model.pt" in failure
        for failure in failures
    ), failures


def test_169_mask_manifest_entries_are_rejected(archive: Path) -> None:
    _edit_json(
        archive / "external" / "roi_to_wsi.json",
        lambda payload: payload["rois"].pop(),
    )
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "requires exactly 170 ROIs, got 169" in failure for failure in failures
    ), failures


def test_171_mask_manifest_entries_are_rejected(archive: Path) -> None:
    def _append(payload: dict) -> None:
        payload["rois"].append(
            {
                "roi_filename": "roi_170.png",
                "patient_id": "EXT00",
                "source_wsi": "wsi_170",
                "native_spacing_um": 0.5,
                "width": 4,
                "height": 3,
            }
        )

    _edit_json(archive / "external" / "roi_to_wsi.json", _append)
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "requires exactly 170 ROIs, got 171" in failure for failure in failures
    ), failures


def test_zip_member_mismatch_names_the_members(archive: Path) -> None:
    zip_path = archive / "external" / "submission.zip"
    masks_dir = archive / "external" / "masks"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as rebuilt:
        for path in sorted(masks_dir.iterdir()):
            if path.name != "roi_000.png":
                rebuilt.write(path, arcname=path.name)
        rebuilt.writestr("stray.png", b"not a sidecar member")
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "external/submission.zip is missing mask member roi_000.png" in failure
        for failure in failures
    ), failures
    assert any(
        "external/submission.zip contains unexpected member stray.png" in failure
        for failure in failures
    ), failures


def test_wrong_selected_model_count_is_rejected(archive: Path) -> None:
    _edit_json(
        archive / "external" / "submission_audit.json",
        lambda payload: payload.update(checkpoints=payload["checkpoints"][:4]),
    )
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "must record exactly 5 selected fold checkpoints, got 4" in failure
        for failure in failures
    ), failures


def test_selected_models_must_match_the_decision_artifact(archive: Path) -> None:
    _edit_json(
        archive / "development" / "arm_selection.json",
        lambda payload: payload.update(selected_arm="class_conditioned"),
    )
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "external/submission_audit.json records selected arm 'uniform' but "
        "development/arm_selection.json selected 'class_conditioned'" in failure
        for failure in failures
    ), failures


def test_patient_missing_from_one_arm_is_named(archive: Path) -> None:
    def _drop(payload: dict) -> None:
        arm = payload["arms"]["uniform"]
        arm["coverage"]["patient_ids"].remove("P010")
        arm["patient_confusions"] = [
            entry
            for entry in arm["patient_confusions"]
            if entry["patient_id"] != "P010"
        ]

    _edit_json(archive / "development" / "oof_report.json", _drop)
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "patient(s) ['P010'] are missing from arm 'uniform'" in failure
        for failure in failures
    ), failures
    assert any(
        "pools 526 patients" in failure and "527" in failure for failure in failures
    ), failures


def test_duplicated_patient_is_named(archive: Path) -> None:
    def _duplicate(payload: dict) -> None:
        arm = payload["arms"]["class_conditioned"]
        arm["coverage"]["patient_ids"].append("P011")
        arm["patient_confusions"].append(
            {
                "patient_id": "P011",
                "fold": 1,
                "confusion_matrix": [[1, 0, 0, 0]] * 4,
            }
        )

    _edit_json(archive / "development" / "oof_report.json", _duplicate)
    write_artifact_manifest(archive)

    failures = _failures(archive)
    assert any(
        "repeats patient(s) ['P011']" in failure
        and "exactly once" in failure
        for failure in failures
    ), failures


def test_virchow2_weight_like_file_is_refused(archive: Path) -> None:
    (archive / "provenance" / "model.safetensors").write_bytes(b"gated weight bytes")

    failures = _failures(archive)
    assert any(
        "provenance/model.safetensors matches gated Virchow2 weight pattern" in failure
        and "must not be redistributed" in failure
        for failure in failures
    ), failures
