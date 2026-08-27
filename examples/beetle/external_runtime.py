"""Bind BEETLE's selected run, checkpoints, and immutable encoder runtime."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import yaml

from examples.beetle.protocol import ARM_NAMES, NUM_FOLDS, PIXEL_MAPPING

PredictorLoader = Callable[[Path, tuple[Path, ...]], object]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def encoder_runtime_environment(protocol_resolution: str | Path):
    """Revalidate and temporarily bind the immutable encoder runtime used for training."""
    from examples.beetle import launch

    resolution_path = Path(protocol_resolution)
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    lock = json.loads(launch.ENCODER_LOCK_PATH.read_text(encoding="utf-8"))
    if resolution.get("schema_version") != 1 or resolution.get("encoder_lock") != lock:
        raise ValueError("BEETLE protocol resolution disagrees with encoder_lock.json")
    runtime = resolution.get("encoder_runtime") or {}
    if runtime.get("mode") != "offline_immutable_hub_snapshot":
        raise ValueError(
            "BEETLE External inference requires an immutable encoder binding"
        )
    snapshot_value = runtime.get("snapshot_path")
    hub_value = runtime.get("hf_hub_cache")
    if not isinstance(snapshot_value, str) or not isinstance(hub_value, str):
        raise ValueError(
            "BEETLE encoder runtime requires snapshot_path and hf_hub_cache"
        )
    snapshot = Path(snapshot_value)
    runtime_hub = Path(hub_value)
    if not snapshot.is_absolute() or not runtime_hub.is_absolute():
        raise ValueError("BEETLE encoder runtime paths must be absolute")
    snapshot = snapshot.resolve()
    runtime_hub = runtime_hub.resolve()
    repository_dir = "models--" + str(lock["repository"]).replace("/", "--")
    if (
        snapshot.name != lock["revision"]
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != repository_dir
    ):
        raise ValueError(
            "BEETLE encoder runtime snapshot does not match the locked object"
        )
    weight_path = snapshot / str(lock["weight_file"])
    if not (snapshot / "config.json").is_file() or not weight_path.is_file():
        raise ValueError("BEETLE encoder runtime snapshot is incomplete")
    if sha256_file(weight_path) != lock["weight_sha256"]:
        raise ValueError(
            "BEETLE encoder runtime weight checksum does not match the lock"
        )

    bound_snapshot = runtime_hub / repository_dir / "snapshots" / str(lock["revision"])
    bound_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if bound_snapshot.is_symlink():
        if bound_snapshot.resolve() != snapshot:
            raise ValueError("BEETLE encoder runtime Hub binds another snapshot")
    elif bound_snapshot.exists():
        raise ValueError(
            "BEETLE encoder runtime Hub snapshot binding must be a symlink"
        )
    else:
        bound_snapshot.symlink_to(snapshot, target_is_directory=True)
    main_ref = runtime_hub / repository_dir / "refs" / "main"
    main_ref.parent.mkdir(parents=True, exist_ok=True)
    if main_ref.exists() and (
        not main_ref.is_file()
        or main_ref.read_text(encoding="utf-8").strip() != lock["revision"]
    ):
        raise ValueError(
            "BEETLE encoder runtime Hub main ref does not bind the locked revision"
        )
    if not main_ref.exists():
        main_ref.write_text(str(lock["revision"]), encoding="utf-8")

    environment = {
        "HF_HOME": str(runtime_hub.parent),
        "HF_HUB_CACHE": str(runtime_hub),
        "HUGGINGFACE_HUB_CACHE": str(runtime_hub),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        yield {
            "mode": runtime["mode"],
            "revision": lock["revision"],
            "weight_sha256": lock["weight_sha256"],
            "snapshot_path": str(snapshot),
            "hf_hub_cache": str(runtime_hub),
            "protocol_resolution": str(resolution_path.resolve()),
            "protocol_resolution_sha256": sha256_file(resolution_path),
        }
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def selected_arm_from_file(selection_path: str | Path) -> str:
    payload = json.loads(Path(selection_path).read_text(encoding="utf-8"))
    arm = str(payload.get("selected_arm", ""))
    if (
        payload.get("schema_version") != 1
        or payload.get("development_evidence_only") is not True
        or arm not in ARM_NAMES
    ):
        raise ValueError(
            "BEETLE External inference requires a schema-v1 development-only arm selection"
        )
    return arm


def selected_checkpoints(run_dir: str | Path, selected_arm: str) -> tuple[Path, ...]:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"BEETLE selected run has no config.yaml: {config_path}"
        )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    tags = set((config.get("run") or {}).get("tags") or ())
    if "beetle" not in tags or selected_arm not in tags:
        raise ValueError(
            f"BEETLE selected run config must carry beetle and {selected_arm!r} tags"
        )
    checkpoints = tuple(
        run_dir / f"fold_{fold}" / "best_model.pt" for fold in range(NUM_FOLDS)
    )
    missing = [str(path) for path in checkpoints if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"BEETLE selected arm requires five fold-selected checkpoints; missing={missing}"
        )
    return checkpoints


def validate_selected_run_recipe(
    protocol_resolution: str | Path, selected_arm: str, run_dir: str | Path
) -> dict[str, str]:
    """Bind the selected run config to the selected arm in protocol resolution."""
    from soma.config import config_yaml_dict, load_config

    resolution_path = Path(protocol_resolution)
    run_dir = Path(run_dir)
    resolution = json.loads(resolution_path.read_text(encoding="utf-8"))
    arms = resolution.get("arms")
    if resolution.get("schema_version") != 1 or not isinstance(arms, dict):
        raise ValueError(
            "BEETLE protocol resolution must declare its resolved arm configs"
        )
    resolved_value = arms.get(selected_arm)
    if not isinstance(resolved_value, str) or not resolved_value:
        raise ValueError(
            f"BEETLE protocol resolution has no config for selected arm {selected_arm!r}"
        )
    resolved_path = Path(resolved_value)
    if not resolved_path.is_absolute() or not resolved_path.is_file():
        raise ValueError(
            "BEETLE resolved selected-arm config must be an existing absolute path"
        )
    run_config_path = run_dir / "config.yaml"
    resolved_config = load_config(resolved_path)
    run_config = load_config(run_config_path)
    if config_yaml_dict(run_config) != config_yaml_dict(resolved_config):
        raise ValueError(
            "BEETLE selected run config disagrees with the protocol-resolved selected arm"
        )
    masks = resolved_config.preprocessing.masks
    if masks is None or masks.pixel_mapping != PIXEL_MAPPING:
        raise ValueError(
            "BEETLE resolved selected arm must use the protocol's exact pixel vocabulary"
        )
    task = resolved_config.task
    if (
        task is None
        or int(task.params.get("num_classes", -1)) != len(PIXEL_MAPPING) - 1
    ):
        raise ValueError(
            "BEETLE resolved selected arm must predict the four annotated pixel classes"
        )
    return {
        "resolved_arm_config": str(resolved_path.resolve()),
        "resolved_arm_config_sha256": sha256_file(resolved_path),
        "run_config_sha256": sha256_file(run_config_path),
    }


def load_selected_fold_predictor(run_dir: Path, checkpoint_paths: tuple[Path, ...]):
    """Load the selected run recipe, one frozen encoder, and all five fold decoders."""
    from soma.config import load_config
    from soma.dense.live import build_live_segmentation_source
    from soma.dense.predict import (
        SlidingWindowSegmentationPredictor,
        build_live_segmentation_models,
    )

    config = load_config(run_dir / "config.yaml")
    if config.decoder is None or config.task is None or config.encoder is None:
        raise ValueError(
            "BEETLE External inference requires encoder, decoder, and task config"
        )
    source = build_live_segmentation_source(config)
    models = build_live_segmentation_models(
        source,
        decoder_name=config.decoder.name,
        decoder_params=config.decoder.params,
        num_classes=int(config.task.params["num_classes"]),
        ckpt_paths=checkpoint_paths,
        normalization=config.normalization,
        projection=config.projection,
        encoder_identity=config.encoder.name,
    )
    return SlidingWindowSegmentationPredictor.from_source(source, models)


__all__ = [
    "PredictorLoader",
    "encoder_runtime_environment",
    "load_selected_fold_predictor",
    "selected_arm_from_file",
    "selected_checkpoints",
    "sha256_file",
    "validate_selected_run_recipe",
]
