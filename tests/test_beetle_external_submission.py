"""Command-level fixtures for the BEETLE External submission protocol."""

from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")

from examples.beetle.external_submission import (  # noqa: E402
    ExternalCohort,
    encoder_runtime_environment,
    main as external_main,
)
from examples.beetle.external_runtime import validate_selected_run_recipe  # noqa: E402
from soma.dense.geometry import compute_dense_geometry  # noqa: E402
from soma.dense.predict import SlidingWindowSegmentationPredictor  # noqa: E402


class _ConstantFoldModel:
    """A stand-in fold decoder loaded from a tiny checkpoint probability vector."""

    def __init__(self, probabilities: list[float], *, target_size: int) -> None:
        self._logits = torch.tensor(
            [math.log(probability) for probability in probabilities],
            dtype=torch.float32,
        )
        self._target_size = target_size
        self.task_head = SimpleNamespace(num_classes=4)

    def __call__(self, batch):
        logits = self._logits[None, :, None, None].expand(
            batch.shape[0], 4, self._target_size, self._target_size
        )
        return SimpleNamespace(logits=logits)


def _write_rgb(path: Path, *, width: int, height: int) -> None:
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = 127
    Image.fromarray(pixels, mode="RGB").save(path)


def _write_sidecar(
    path: Path, specs: list[tuple[str, str, str, float, int, int]]
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rois": [
                    {
                        "roi_filename": filename,
                        "patient_id": patient,
                        "source_wsi": wsi,
                        "native_spacing_um": spacing,
                        "width": width,
                        "height": height,
                    }
                    for filename, patient, wsi, spacing, width, height in specs
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_selected_recipe(
    tmp_path: Path,
    *,
    pixel_mapping: dict[str, int] | None = None,
    num_classes: int = 4,
):
    from examples.beetle.protocol import PIXEL_MAPPING
    from soma.config import (
        DecoderConfig,
        EncoderConfig,
        MasksConfig,
        PipelineConfig,
        PreprocessingConfig,
        TaskConfig,
        save_config,
    )

    config = PipelineConfig(
        dataset_csv="development.csv",
        splits_csv="development_splits.csv",
        output_root="runs/uniform",
        dataset_type="segmentation",
        preprocessing=PreprocessingConfig(
            requested_tile_size_px=8,
            requested_spacing_um=0.5,
            masks=MasksConfig(pixel_mapping=pixel_mapping or PIXEL_MAPPING),
        ),
        encoder=EncoderConfig(name="virchow2", precision="fp32", output_variant="cls"),
        decoder=DecoderConfig(name="lightweight_conv"),
        task=TaskConfig(name="segmentation", params={"num_classes": num_classes}),
        tags=["beetle", "project_protocol", "uniform"],
    )
    resolved = tmp_path / "resolved_uniform.yaml"
    save_config(config, resolved)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    save_config(config, run_dir / "config.yaml")
    resolution = tmp_path / "protocol_resolution.json"
    resolution.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "arms": {"uniform": str(resolved.resolve())},
            }
        ),
        encoding="utf-8",
    )
    return config, resolved, run_dir, resolution


def test_encoder_runtime_environment_revalidates_and_binds_locked_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from examples.beetle import launch

    revision = "a" * 40
    weights = b"stand-in gated encoder bytes"
    digest = __import__("hashlib").sha256(weights).hexdigest()
    snapshot = (
        tmp_path / "source_hub" / "models--paige-ai--Virchow2" / "snapshots" / revision
    )
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors").write_bytes(weights)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    lock = {
        "repository": "paige-ai/Virchow2",
        "revision": revision,
        "weight_file": "model.safetensors",
        "weight_sha256": digest,
        "patch_size": 14,
        "feature_channels": 1280,
    }
    lock_path = tmp_path / "encoder_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(launch, "ENCODER_LOCK_PATH", lock_path)
    runtime_hub = tmp_path / "runtime_hub"
    bound = runtime_hub / "models--paige-ai--Virchow2" / "snapshots" / revision
    refs = bound.parent.parent / "refs"
    resolution = tmp_path / "protocol_resolution.json"
    resolution.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "encoder_lock": lock,
                "encoder_runtime": {
                    "mode": "offline_immutable_hub_snapshot",
                    "snapshot_path": str(snapshot),
                    "hf_hub_cache": str(runtime_hub),
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    with encoder_runtime_environment(resolution) as binding:
        assert binding["revision"] == revision
        assert os.environ["HF_HUB_CACHE"] == str(runtime_hub.resolve())
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert bound.is_symlink() and bound.resolve() == snapshot.resolve()
        assert (refs / "main").read_text(encoding="utf-8") == revision

    assert "HF_HUB_OFFLINE" not in os.environ
    (refs / "main").write_text("b" * 40 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="main ref does not bind"):
        with encoder_runtime_environment(resolution):
            pass


def test_selected_run_recipe_must_equal_protocol_resolved_arm(tmp_path: Path) -> None:
    from soma.config import save_config

    config, resolved, run_dir, resolution = _write_selected_recipe(tmp_path)

    binding = validate_selected_run_recipe(resolution, "uniform", run_dir)
    assert binding["resolved_arm_config"] == str(resolved.resolve())

    drifted = replace(
        config,
        preprocessing=replace(config.preprocessing, requested_spacing_um=0.6),
    )
    save_config(drifted, run_dir / "config.yaml")
    with pytest.raises(ValueError, match="disagrees with the protocol-resolved"):
        validate_selected_run_recipe(resolution, "uniform", run_dir)


def test_selected_run_recipe_rejects_wrong_pixel_mapping(tmp_path: Path) -> None:
    wrong_mapping = {
        "background": 0,
        "other": 1,
        "non_invasive_epithelium": 2,
        "necrosis": 3,
        "invasive_epithelium": 4,
    }
    _config, _resolved, run_dir, resolution = _write_selected_recipe(
        tmp_path, pixel_mapping=wrong_mapping
    )

    with pytest.raises(ValueError, match="protocol's exact pixel vocabulary"):
        validate_selected_run_recipe(resolution, "uniform", run_dir)


def test_selected_run_recipe_rejects_wrong_number_of_classes(tmp_path: Path) -> None:
    _config, _resolved, run_dir, resolution = _write_selected_recipe(
        tmp_path, num_classes=3
    )

    with pytest.raises(ValueError, match="four annotated pixel classes"):
        validate_selected_run_recipe(resolution, "uniform", run_dir)


def test_external_infer_command_ensembles_spacing_maps_pngs_and_flat_zip(
    tmp_path: Path,
) -> None:
    roi_dir = tmp_path / "rois"
    roi_dir.mkdir()
    specs = [
        ("patient_a_roi_1.png", "patient_a", "wsi_a", 0.51, 11, 9),
        ("patient_a_roi_2.png", "patient_a", "wsi_a", 0.25, 12, 10),
        ("patient_b_roi_1.png", "patient_b", "wsi_b", 0.60, 13, 11),
    ]
    for filename, _patient, _wsi, _spacing, width, height in specs:
        _write_rgb(roi_dir / filename, width=width, height=height)

    sidecar = tmp_path / "roi_to_wsi.json"
    _write_sidecar(sidecar, specs)
    selection = tmp_path / "arm_selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "development_evidence_only": True,
                "selected_arm": "uniform",
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "uniform_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(
        "run:\n  tags: [beetle, project_protocol, uniform]\n", encoding="utf-8"
    )

    # Three weak class-1 folds versus two strong class-3 folds. Majority voting would emit
    # label 1; averaging the five per-pixel probability vectors must emit BEETLE label 3.
    fold_probabilities = [
        [0.40, 0.20, 0.30, 0.10],
        [0.40, 0.20, 0.30, 0.10],
        [0.40, 0.20, 0.30, 0.10],
        [0.01, 0.01, 0.97, 0.01],
        [0.01, 0.01, 0.97, 0.01],
    ]
    for fold, probabilities in enumerate(fold_probabilities):
        fold_dir = run_dir / f"fold_{fold}"
        fold_dir.mkdir()
        torch.save({"probabilities": probabilities}, fold_dir / "best_model.pt")

    loaded_checkpoints: list[Path] = []

    def _standin_loader(_run_dir: Path, checkpoint_paths: tuple[Path, ...]):
        loaded_checkpoints.extend(checkpoint_paths)
        geometry = compute_dense_geometry(target_size=8, patch_size=4)
        models = [
            _ConstantFoldModel(
                torch.load(path, weights_only=True)["probabilities"], target_size=8
            )
            for path in checkpoint_paths
        ]
        return SlidingWindowSegmentationPredictor(
            models=models,
            geometry=geometry,
            preprocessor=lambda pixels: pixels.to(torch.float32) / 255.0,
            device=torch.device("cpu"),
            spacing_um=0.5,
            tolerance=0.05,
        )

    output_dir = tmp_path / "submission_pngs"
    audit_path = tmp_path / "submission_audit.json"
    zip_path = tmp_path / "submission.zip"
    assert (
        external_main(
            [
                "infer",
                "--selection",
                str(selection),
                "--run-dir",
                str(run_dir),
                "--roi-dir",
                str(roi_dir),
                "--roi-sidecar",
                str(sidecar),
                "--output-dir",
                str(output_dir),
                "--audit",
                str(audit_path),
                "--zip",
                str(zip_path),
            ],
            predictor_loader=_standin_loader,
            cohort=ExternalCohort(roi_count=3, patient_count=2),
        )
        == 0
    )

    assert loaded_checkpoints == [
        run_dir / f"fold_{fold}" / "best_model.pt" for fold in range(5)
    ]
    for filename, _patient, _wsi, _spacing, width, height in specs:
        with Image.open(output_dir / filename) as prediction:
            assert prediction.mode == "L"
            assert prediction.size == (width, height)
            assert set(np.asarray(prediction).ravel()) == {3}

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["selected_arm"] == "uniform"
    assert [decision["spacing_decision"] for decision in audit["roi_decisions"]] == [
        "native_within_tolerance",
        "downsample_finer_input",
        "native_coarse_no_upsample",
    ]
    assert all(
        decision["output_matches_input_dimensions"]
        for decision in audit["roi_decisions"]
    )

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == sorted(filename for filename, *_ in specs)
        assert all("/" not in name for name in archive.namelist())
        assert all(
            info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist()
        )
        assert all(info.external_attr == 0o100644 << 16 for info in archive.infolist())
    second_zip = tmp_path / "submission_again.zip"
    assert (
        external_main(
            [
                "validate",
                "--roi-sidecar",
                str(sidecar),
                "--output-dir",
                str(output_dir),
                "--zip",
                str(second_zip),
            ],
            cohort=ExternalCohort(roi_count=3, patient_count=2),
        )
        == 0
    )
    assert second_zip.read_bytes() == zip_path.read_bytes()


def test_validate_command_rejects_missing_or_extra_filenames(tmp_path: Path) -> None:
    specs = [
        ("roi_a.png", "patient_a", "wsi_a", 0.5, 4, 3),
        ("roi_b.png", "patient_b", "wsi_b", 0.5, 5, 4),
    ]
    sidecar = tmp_path / "roi_to_wsi.json"
    _write_sidecar(sidecar, specs)
    output_dir = tmp_path / "predictions"
    output_dir.mkdir()
    Image.fromarray(np.ones((3, 4), dtype=np.uint8), mode="L").save(
        output_dir / "roi_a.png"
    )
    Image.fromarray(np.ones((4, 5), dtype=np.uint8), mode="L").save(
        output_dir / "extra.png"
    )

    with pytest.raises(
        ValueError, match=r"missing=\['roi_b.png'\].*extra=\['extra.png'\]"
    ):
        external_main(
            [
                "validate",
                "--roi-sidecar",
                str(sidecar),
                "--output-dir",
                str(output_dir),
                "--zip",
                str(tmp_path / "submission.zip"),
            ],
            cohort=ExternalCohort(roi_count=2, patient_count=2),
        )


def test_evaluate_command_groups_nested_rois_for_fixed_patient_bootstrap(
    tmp_path: Path,
) -> None:
    specs = [
        ("patient_a_wsi_1_roi_1.png", "patient_a", "wsi_a_1", 0.5, 2, 2),
        ("patient_a_wsi_2_roi_1.png", "patient_a", "wsi_a_2", 0.5, 2, 2),
        ("patient_b_wsi_1_roi_1.png", "patient_b", "wsi_b_1", 0.5, 2, 2),
    ]
    sidecar = tmp_path / "roi_to_wsi.json"
    _write_sidecar(sidecar, specs)
    predictions = tmp_path / "predictions"
    labels = tmp_path / "sequestered_labels"
    predictions.mkdir()
    labels.mkdir()
    arrays = {
        "patient_a_wsi_1_roi_1.png": (
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
        ),
        "patient_a_wsi_2_roi_1.png": (
            [[1, 1], [2, 0]],
            [[2, 1], [2, 4]],
        ),
        "patient_b_wsi_1_roi_1.png": (
            [[1, 2], [3, 4]],
            [[4, 3], [2, 1]],
        ),
    }
    for filename, (truth, prediction) in arrays.items():
        Image.fromarray(np.asarray(truth, dtype=np.uint8), mode="L").save(
            labels / filename
        )
        Image.fromarray(np.asarray(prediction, dtype=np.uint8), mode="L").save(
            predictions / filename
        )

    report_path = tmp_path / "external_report.json"
    assert (
        external_main(
            [
                "evaluate",
                "--roi-sidecar",
                str(sidecar),
                "--predictions-dir",
                str(predictions),
                "--labels-dir",
                str(labels),
                "--output",
                str(report_path),
            ],
            cohort=ExternalCohort(roi_count=3, patient_count=2),
        )
        == 0
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["hidden_external_labels_supplied"] is True
    assert report["coverage"] == {
        "expected_patient_count": 2,
        "observed_patient_count": 2,
        "roi_count": 3,
        "patient_ids": ["patient_a", "patient_b"],
        "rois_per_patient": {"patient_a": 2, "patient_b": 1},
        "source_wsis_per_patient": {"patient_a": 2, "patient_b": 1},
    }
    assert report["pooled"]["confusion_matrix"] == [
        [2, 1, 0, 1],
        [0, 2, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1],
    ]
    assert report["bootstrap"]["seed"] == 0
    assert report["bootstrap"]["draws"] == 10_000
    assert report["bootstrap"]["sampling_unit"] == "patient"
