"""Executable protocol contract for the BEETLE two-arm development campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from huggingface_hub import hf_hub_download
import pytest
import torch
import yaml

from examples.beetle import launch
from examples.beetle.launch import resolve_arm_configs
from examples.beetle.protocol import ARM_NAMES, BATCH_SIZE_CANDIDATES, PIXEL_MAPPING
from examples.beetle.select_arm import main as select_main, select_development_arm
from examples.beetle.smoke import run_offline_smoke
from soma.config import Component, ExperimentConfig
from soma.plan import check

REPOSITORY = "paige-ai/Virchow2"
REVISION = "3158645804b69e3f3bc4439d4116edddf0840a72"
WEIGHT_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
WEIGHT_SHA256 = "8d6cea947eb2418c3b0dff48cfb9b238e47744ab0dfca21b2b0637b140769b4b"
EXPECTED_ARMS = ("uniform", "class_conditioned")
EXPECTED_BATCH_SIZE_CANDIDATES = (16, 8, 4)


@pytest.fixture
def local_encoder_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, str]:
    snapshot = (
        tmp_path / "hf" / "hub" / "models--paige-ai--Virchow2" / "snapshots" / REVISION
    )
    snapshot.mkdir(parents=True)
    weight_bytes = b"offline Virchow2 test fixture; not model weights"
    (snapshot / WEIGHT_FILE).write_bytes(weight_bytes)
    (snapshot / CONFIG_FILE).write_text(
        '{"architecture": "virchow2-fixture"}\n', encoding="utf-8"
    )
    weight_sha256 = hashlib.sha256(weight_bytes).hexdigest()
    refs = snapshot.parent.parent / "refs"
    refs.mkdir()
    (refs / "main").write_text(REVISION + "\n", encoding="utf-8")
    lock_path = tmp_path / "encoder_lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": weight_sha256,
                "patch_size": 14,
                "feature_channels": 1280,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(launch, "ENCODER_LOCK_PATH", lock_path)
    return snapshot, weight_sha256


def _write_completed_preflight(
    path: Path,
    local_encoder_snapshot: tuple[Path, str],
    *,
    batch_size: int = 8,
) -> Path:
    snapshot, weight_sha256 = local_encoder_snapshot
    attempts = [
        {"batch_size": candidate, "passed": candidate <= batch_size}
        for candidate in EXPECTED_BATCH_SIZE_CANDIDATES
    ]
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "completed",
                "scope": "campaign",
                "batch_size_candidates": [16, 8, 4],
                "batch_size_attempts": attempts,
                "selected_batch_size": batch_size,
                "same_batch_every_arm_and_fold": True,
                "encoder": {
                    "repository": REPOSITORY,
                    "revision": REVISION,
                    "weight_file": WEIGHT_FILE,
                    "weight_sha256": weight_sha256,
                    "patch_size": 14,
                    "feature_channels": 1280,
                    "weight_checksum_verified": True,
                    "snapshot_path": str(snapshot),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_encoder_lock_records_the_exact_published_virchow2_object() -> None:
    lock = json.loads(
        Path("examples/beetle/encoder_lock.json").read_text(encoding="utf-8")
    )

    assert lock == {
        "repository": REPOSITORY,
        "revision": REVISION,
        "weight_file": WEIGHT_FILE,
        "weight_sha256": WEIGHT_SHA256,
        "patch_size": 14,
        "feature_channels": 1280,
    }


def test_protocol_axes_are_the_predeclared_literal_values() -> None:
    assert ARM_NAMES == ("uniform", "class_conditioned")
    assert BATCH_SIZE_CANDIDATES == (16, 8, 4)


def test_resolved_arms_encode_the_locked_virchow2_recipe(
    tmp_path: Path, local_encoder_snapshot: tuple[Path, str]
) -> None:
    paths = resolve_arm_configs(
        preflight_path=_write_completed_preflight(
            tmp_path / "preflight.json", local_encoder_snapshot
        ),
        output_dir=tmp_path / "resolved",
    )

    assert tuple(paths) == EXPECTED_ARMS
    for arm, path in paths.items():
        config = ExperimentConfig.from_yaml(path)
        check(config)
        features = config.features
        assert features.encoder is not None
        assert features.encoder.name == "virchow2"
        assert features.encoder.output_variant == "cls"
        assert features.execution.precision == "fp16"
        assert features.cache.dtype == "fp16"
        assert REVISION in str(features.cache.root_dir)
        assert str(local_encoder_snapshot[0]) in path.read_text(encoding="utf-8")
        assert features.dense.feature_kind == "patch_features"
        assert features.spacing_policy == "native_if_coarser"
        assert features.extraction.tiling.tolerance == pytest.approx(0.1)
        assert features.extraction.tiling.mask_backend == "openslide"
        assert features.dense.window_size == 224
        assert features.dense.overlap == 0.5
        assert features.extraction.masks.pixel_mapping == {
            label: (value,) for label, value in PIXEL_MAPPING.items()
        }
        params = config.learner.params
        assert params["decoder"] == Component(
            "lightweight_conv", {"hidden_dim": 256, "num_upsample_blocks": 2, "num_groups": 32}
        )
        assert config.loaders.augment is None
        assert config.loaders.batch_size == 8
        assert config.trainer.max_epochs == 30
        assert params["optimizer"] == Component("adam", {"lr": 1e-4, "weight_decay": 1e-5})
        assert params["scheduler"] == [Component("cosine")]
        monitor = {"monitor": "dataset_global_mean_dice", "mode": "max"}
        assert [(leaf.name, leaf.params) for leaf in config.trainer.callbacks] == [
            ("checkpoint", monitor),
            ("early_stopping", {**monitor, "patience": 8}),
        ]
        assert config.trainer.accumulate_grad_batches == 1
        assert config.run.seed == 0
        assert "dataset_global_mean_dice" in config.evaluation.metrics
        assert config.loaders.sampler.name == arm
        assert arm in str(config.run.output_dir)


def test_arm_overlays_contain_only_sampling_and_output_identity() -> None:
    config_dir = Path("examples/beetle/configs")

    assert yaml.safe_load(
        (config_dir / "uniform.yaml").read_text(encoding="utf-8")
    ) == {
        "run": {
            "output_dir": "data/beetle/runs/uniform",
            "mirror_root": "data/beetle/recovery/uniform",
        },
        "loaders": {"sampler": {"name": "uniform"}},
    }
    assert yaml.safe_load(
        (config_dir / "class_conditioned.yaml").read_text(encoding="utf-8")
    ) == {
        "run": {
            "output_dir": "data/beetle/runs/class_conditioned",
            "mirror_root": "data/beetle/recovery/class_conditioned",
        },
        "loaders": {
            "sampler": {"name": "class_conditioned", "params": {"class_request_ratios": [1, 1, 1, 1]}}
        },
    }


def test_resolution_rejects_unverified_or_drifted_encoder_weights(
    tmp_path: Path, local_encoder_snapshot: tuple[Path, str]
) -> None:
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    payload = json.loads(preflight.read_text())
    payload["encoder"]["weight_sha256"] = "0" * 64
    preflight.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="weight_sha256.*does not match"):
        resolve_arm_configs(preflight_path=preflight, output_dir=tmp_path / "resolved")


def test_resolution_uses_the_largest_passing_preflight_batch(
    tmp_path: Path, local_encoder_snapshot: tuple[Path, str]
) -> None:
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot, batch_size=4
    )
    payload = json.loads(preflight.read_text())
    payload["selected_batch_size"] = 8
    preflight.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="largest passing batch size is 4"):
        resolve_arm_configs(preflight_path=preflight, output_dir=tmp_path / "resolved")


def test_resolution_accepts_configurable_descending_decoder_batch_candidates(
    tmp_path: Path, local_encoder_snapshot: tuple[Path, str]
) -> None:
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    payload = json.loads(preflight.read_text())
    payload["batch_size_candidates"] = [64, 32, 16, 8, 4]
    payload["batch_size_attempts"] = [
        {"batch_size": candidate, "passed": True}
        for candidate in payload["batch_size_candidates"]
    ]
    payload["selected_batch_size"] = 64
    preflight.write_text(json.dumps(payload), encoding="utf-8")

    paths = resolve_arm_configs(
        preflight_path=preflight,
        output_dir=tmp_path / "resolved",
    )

    assert {
        arm: ExperimentConfig.from_yaml(path).loaders.batch_size for arm, path in paths.items()
    } == {"uniform": 64, "class_conditioned": 64}


def test_launch_binds_offline_hub_to_the_validated_immutable_snapshot(
    tmp_path: Path,
    local_encoder_snapshot: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    launch.main(
        [
            "run",
            "--preflight",
            str(preflight),
            "--output-dir",
            str(tmp_path / "resolved"),
        ]
    )

    assert [command for command, _ in calls] == [
        [sys.executable, "-m", "soma", str(tmp_path / "resolved" / "uniform.yaml")],
        [
            sys.executable,
            "-m",
            "soma",
            str(tmp_path / "resolved" / "class_conditioned.yaml"),
        ],
    ]
    for _, call in calls:
        assert call["check"] is True
        assert call["cwd"] == launch.REPOSITORY_ROOT
        assert call["env"]["HF_HUB_OFFLINE"] == "1"
        assert call["env"]["TRANSFORMERS_OFFLINE"] == "1"
        runtime_hub = Path(call["env"]["HF_HUB_CACHE"])
        bound_snapshot = (
            runtime_hub / "models--paige-ai--Virchow2" / "snapshots" / REVISION
        )
        assert bound_snapshot.resolve() == local_encoder_snapshot[0].resolve()
        resolved_runtime_files = {
            filename: Path(
                hf_hub_download(
                    repo_id=REPOSITORY,
                    filename=filename,
                    cache_dir=runtime_hub,
                    local_files_only=True,
                )
            )
            for filename in ("config.json", "model.safetensors")
        }
        assert resolved_runtime_files["config.json"].read_text() == (
            '{"architecture": "virchow2-fixture"}\n'
        )
        assert resolved_runtime_files["model.safetensors"].read_bytes() == (
            b"offline Virchow2 test fixture; not model weights"
        )


def test_resolution_rejects_a_snapshot_missing_runtime_config(
    tmp_path: Path, local_encoder_snapshot: tuple[Path, str]
) -> None:
    snapshot, _ = local_encoder_snapshot
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    (snapshot / CONFIG_FILE).unlink()

    with pytest.raises(ValueError, match="snapshot is missing config.json"):
        resolve_arm_configs(preflight_path=preflight, output_dir=tmp_path / "resolved")


def test_pinned_single_arm_launch_forwards_resume_override(
    tmp_path: Path,
    local_encoder_snapshot: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    calls: list[tuple[list[str], dict]] = []
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    launch.main(
        [
            "run-arm",
            "--preflight",
            str(preflight),
            "--output-dir",
            str(tmp_path / "resolved"),
            "--arm",
            "uniform",
            "--set",
            "run.run_id=beetle-uniform-resume",
        ]
    )

    assert [command for command, _ in calls] == [
        [
            sys.executable,
            "-m",
            "soma",
            str((tmp_path / "resolved" / "uniform.yaml").resolve()),
            "--set",
            "run.run_id=beetle-uniform-resume",
        ]
    ]
    assert calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"


def test_pinned_single_arm_launch_rejects_scientific_recipe_overrides(
    tmp_path: Path,
    local_encoder_snapshot: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    calls: list[object] = []
    monkeypatch.setattr(
        launch.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )

    with pytest.raises(SystemExit):
        launch.main(
            [
                "run-arm",
                "--preflight",
                str(preflight),
                "--output-dir",
                str(tmp_path / "resolved"),
                "--arm",
                "uniform",
                "--set",
                "training.epochs=99",
            ]
        )

    assert calls == []


def test_launch_uses_absolute_config_paths_from_a_non_repository_cwd(
    tmp_path: Path,
    local_encoder_snapshot: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        launch.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command),
    )

    launch.main(["run", "--preflight", str(preflight), "--output-dir", "resolved"])

    assert calls == [
        [
            sys.executable,
            "-m",
            "soma",
            str((caller / "resolved/uniform.yaml").resolve()),
        ],
        [
            sys.executable,
            "-m",
            "soma",
            str((caller / "resolved/class_conditioned.yaml").resolve()),
        ],
    ]


@pytest.mark.parametrize("source_main_state", ["moved", "missing"])
def test_launch_ignores_mutable_source_main_and_keeps_runtime_main_locked(
    tmp_path: Path,
    local_encoder_snapshot: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    source_main_state: str,
) -> None:
    snapshot, _ = local_encoder_snapshot
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    calls: list[object] = []
    monkeypatch.setattr(
        launch.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )
    source_main = snapshot.parent.parent / "refs" / "main"
    if source_main_state == "moved":
        source_main.write_text("f" * 40 + "\n", encoding="utf-8")
    else:
        source_main.unlink()

    launch.main(
        [
            "run",
            "--preflight",
            str(preflight),
            "--output-dir",
            str(tmp_path / "resolved"),
        ]
    )

    runtime_main = (
        tmp_path
        / "resolved"
        / ".hf-home"
        / "hub"
        / "models--paige-ai--Virchow2"
        / "refs"
        / "main"
    )
    assert runtime_main.read_text(encoding="utf-8") == REVISION
    assert len(calls) == 2


def test_launch_rehashes_weights_and_rejects_snapshot_drift_before_spawning_soma(
    tmp_path: Path,
    local_encoder_snapshot: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, _ = local_encoder_snapshot
    preflight = _write_completed_preflight(
        tmp_path / "preflight.json", local_encoder_snapshot
    )
    calls: list[object] = []
    monkeypatch.setattr(
        launch.subprocess, "run", lambda *args, **kwargs: calls.append(args)
    )
    (snapshot / WEIGHT_FILE).write_bytes(b"drifted after preflight")

    with pytest.raises(ValueError, match="snapshot weight checksum does not match"):
        launch.main(
            [
                "run",
                "--preflight",
                str(preflight),
                "--output-dir",
                str(tmp_path / "resolved"),
            ]
        )

    assert calls == []


def _write_oof_report(path: Path) -> Path:
    fold_scores = {
        "uniform": [0.70, 0.80, 0.90, 0.60, 1.00],
        "class_conditioned": [0.80, 0.90, 0.90, 0.85, 0.95],
    }
    payload = {
        "schema_version": 1,
        "protocol": {"folds": 5, "arms": ["uniform", "class_conditioned"]},
        "arms": {
            arm: {
                "primary": {
                    "folds": {
                        str(fold): {"mean_dice": score}
                        for fold, score in enumerate(scores)
                    },
                    # Deliberately wrong: selection must derive the equal fold average.
                    "fold_macro_class_dice": -1,
                }
            }
            for arm, scores in fold_scores.items()
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_selector_reports_both_arms_and_uses_equal_five_fold_average(
    tmp_path: Path,
) -> None:
    selection = select_development_arm(_write_oof_report(tmp_path / "oof.json"))

    assert selection["selected_arm"] == "class_conditioned"
    assert selection["criterion"] == "fold_macro_class_dice"
    assert selection["development_evidence_only"] is True
    assert selection["arms"]["uniform"]["fold_scores"] == [0.70, 0.80, 0.90, 0.60, 1.00]
    assert selection["arms"]["uniform"]["mean"] == pytest.approx(0.8)
    assert selection["arms"]["class_conditioned"]["mean"] == pytest.approx(0.88)


def test_selector_uses_uniform_as_the_exact_tie_fallback(tmp_path: Path) -> None:
    report = _write_oof_report(tmp_path / "oof.json")
    payload = json.loads(report.read_text(encoding="utf-8"))
    uniform_folds = payload["arms"]["uniform"]["primary"]["folds"]
    payload["arms"]["class_conditioned"]["primary"]["folds"] = uniform_folds
    report.write_text(json.dumps(payload), encoding="utf-8")

    selection = select_development_arm(report)

    assert (
        selection["arms"]["uniform"]["mean"]
        == selection["arms"]["class_conditioned"]["mean"]
    )
    assert selection["tie_breaker"] == "uniform"
    assert selection["selected_arm"] == "uniform"


def test_selector_has_no_external_score_interface(tmp_path: Path) -> None:
    report = _write_oof_report(tmp_path / "oof.json")

    with pytest.raises(SystemExit):
        select_main(
            [
                "--oof-report",
                str(report),
                "--output",
                str(tmp_path / "selection.json"),
                "--external-score",
                "0.99",
            ]
        )


def test_selector_rejects_incomplete_fold_evidence(tmp_path: Path) -> None:
    report = _write_oof_report(tmp_path / "oof.json")
    payload = json.loads(report.read_text())
    del payload["arms"]["uniform"]["primary"]["folds"]["4"]
    report.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="folds 0 through 4"):
        select_development_arm(report)


def test_complete_offline_smoke_exercises_the_publication_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    manifest_path = run_offline_smoke(tmp_path / "smoke")

    manifest = json.loads(manifest_path.read_text())
    assert manifest["offline"] is True
    assert manifest["used_gated_weights"] is False
    assert manifest["used_wsis"] is False
    assert manifest["used_gpu"] is False
    assert set(manifest["resolved_configs"]) == {"uniform", "class_conditioned"}
    assert manifest["cache"]["feature_dim"] == 1280
    assert manifest["cache"]["dtype"] == "float16"
    for arm in EXPECTED_ARMS:
        artifacts = manifest["arms"][arm]
        assert len(artifacts["confusion_evidence"]) == 5
        assert len(artifacts["sampling_audits"]) == 5
        assert len(artifacts["recovery_fold_manifests"]) == 5
        assert "recovery_checkpoint_manifests" not in artifacts
        assert all(
            Path(path).is_file() for paths in artifacts.values() for path in paths
        )
    report = json.loads(Path(manifest["oof_report"]).read_text())
    assert report["protocol"]["bootstrap_draws"] == 10_000
    assert set(report["arms"]) == {"uniform", "class_conditioned"}
    selection = json.loads(Path(manifest["arm_selection"]).read_text())
    assert selection["selected_arm"] in {"uniform", "class_conditioned"}
    assert set(selection["arms"]) == {"uniform", "class_conditioned"}
