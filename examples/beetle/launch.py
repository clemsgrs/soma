"""Resolve and launch the two-arm BEETLE development campaign.

The tracked YAML files are protocol templates, not guesses about GPU memory. A completed
hardware-preflight record supplies one batch size for every arm and fold and proves that
the locally available gated Virchow2 object matches the tracked lock.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import yaml

from examples.beetle.protocol import ARM_NAMES, BATCH_SIZE_CANDIDATES
from soma.config import load_config


PROJECT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
CONFIG_DIR = PROJECT_DIR / "configs"
ENCODER_LOCK_PATH = PROJECT_DIR / "encoder_lock.json"
BATCH_SIZE_TOKEN = "${BEETLE_CAMPAIGN_BATCH_SIZE}"
ENCODER_CACHE_ID_TOKEN = "${BEETLE_ENCODER_CACHE_ID}"
RUNTIME_REQUIRED_FILES = ("config.json",)
RUN_LIFECYCLE_OVERRIDE_PREFIXES = ("run.run_id=", "run.resume=")


@dataclass(frozen=True)
class FrozenEncoderSource:
    snapshot_path: Path
    revision: str
    weight_sha256: str


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge a small project-owned arm overlay into the common recipe."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validated_preflight(
    preflight_path: Path, *, allow_offline_smoke: bool
) -> tuple[dict, int, FrozenEncoderSource | None]:
    payload = _read_json(preflight_path)
    if payload.get("schema_version") != 1 or payload.get("status") != "completed":
        raise ValueError("BEETLE launch requires a schema-v1 completed hardware preflight")
    scope = payload.get("scope")
    if scope != "campaign" and not (allow_offline_smoke and scope == "offline_smoke"):
        raise ValueError("BEETLE production launch requires a campaign hardware preflight")
    candidates = payload.get("batch_size_candidates")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(
            isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0
            for candidate in candidates
        )
        or any(left <= right for left, right in zip(candidates, candidates[1:]))
    ):
        raise ValueError(
            "BEETLE decoder batch-size candidates must be positive integers in "
            "strictly descending order"
        )
    attempts = payload.get("batch_size_attempts")
    if not isinstance(attempts, list) or [row.get("batch_size") for row in attempts] != candidates:
        raise ValueError("BEETLE hardware preflight must record one ordered result per batch")
    passing = [int(row["batch_size"]) for row in attempts if row.get("passed") is True]
    if not passing:
        raise ValueError("BEETLE hardware preflight found no feasible campaign batch size")
    largest_passing = max(passing)
    if payload.get("selected_batch_size") != largest_passing:
        raise ValueError(
            f"BEETLE largest passing batch size is {largest_passing}; "
            f"selected {payload.get('selected_batch_size')!r}"
        )
    if payload.get("same_batch_every_arm_and_fold") is not True:
        raise ValueError("BEETLE preflight must freeze one batch size for every arm and fold")

    lock = _read_json(ENCODER_LOCK_PATH)
    observed = payload.get("encoder")
    if not isinstance(observed, dict):
        raise ValueError("BEETLE hardware preflight is missing encoder provenance")
    for field in (
        "repository",
        "revision",
        "weight_file",
        "weight_sha256",
        "patch_size",
        "feature_channels",
    ):
        if observed.get(field) != lock[field]:
            raise ValueError(
                f"BEETLE encoder {field} {observed.get(field)!r} does not match "
                f"the protocol lock {lock[field]!r}"
            )
    if observed.get("weight_checksum_verified") is not True:
        raise ValueError("BEETLE Virchow2 weight checksum was not verified")

    if scope == "offline_smoke":
        return payload, largest_passing, None

    snapshot_value = observed.get("snapshot_path")
    if not isinstance(snapshot_value, str) or not snapshot_value:
        raise ValueError("BEETLE production preflight requires encoder.snapshot_path")
    snapshot_path = Path(snapshot_value)
    if not snapshot_path.is_absolute():
        raise ValueError("BEETLE encoder.snapshot_path must be absolute")
    snapshot_path = snapshot_path.resolve()
    repository_cache_name = f"models--{lock['repository'].replace('/', '--')}"
    if (
        snapshot_path.name != lock["revision"]
        or snapshot_path.parent.name != "snapshots"
        or snapshot_path.parent.parent.name != repository_cache_name
    ):
        raise ValueError(
            "BEETLE encoder snapshot_path must identify the locked Hugging Face "
            f"snapshot {repository_cache_name}/snapshots/{lock['revision']}"
        )
    for filename in (*RUNTIME_REQUIRED_FILES, lock["weight_file"]):
        if not (snapshot_path / filename).is_file():
            raise ValueError(f"BEETLE encoder snapshot is missing {filename}")
    weight_path = snapshot_path / lock["weight_file"]
    actual_sha256 = _sha256(weight_path)
    if actual_sha256 != lock["weight_sha256"]:
        raise ValueError(
            "BEETLE encoder snapshot weight checksum does not match the protocol lock: "
            f"{actual_sha256}"
        )
    return (
        payload,
        largest_passing,
        FrozenEncoderSource(
            snapshot_path=snapshot_path,
            revision=lock["revision"],
            weight_sha256=lock["weight_sha256"],
        ),
    )


def _bind_frozen_encoder(source: FrozenEncoderSource, output_dir: Path) -> Path:
    """Build a weight-free offline Hub view whose main ref can only resolve the lock."""
    runtime_hub = output_dir / ".hf-home" / "hub"
    repository_dir = runtime_hub / source.snapshot_path.parent.parent.name
    bound_snapshot = repository_dir / "snapshots" / source.revision
    bound_snapshot.parent.mkdir(parents=True, exist_ok=True)
    if bound_snapshot.is_symlink():
        if bound_snapshot.resolve() != source.snapshot_path:
            raise ValueError("BEETLE runtime encoder binding points at another snapshot")
    elif bound_snapshot.exists():
        raise ValueError("BEETLE runtime encoder binding must be a snapshot symlink")
    else:
        bound_snapshot.symlink_to(source.snapshot_path, target_is_directory=True)
    refs_dir = repository_dir / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    main_ref = refs_dir / "main"
    if main_ref.exists() and main_ref.read_text(encoding="utf-8").strip() != source.revision:
        raise ValueError("BEETLE runtime encoder main ref points at another revision")
    # huggingface_hub reads refs verbatim when resolving symbolic revisions.
    main_ref.write_text(source.revision, encoding="utf-8")
    return runtime_hub.resolve()


def resolve_arm_configs(
    *,
    preflight_path: str | Path,
    output_dir: str | Path,
    allow_offline_smoke: bool = False,
) -> Mapping[str, Path]:
    """Validate one preflight and write the two runnable resolved arm configs."""
    preflight_path = Path(preflight_path)
    output_dir = Path(output_dir)
    preflight, batch_size, encoder_source = _validated_preflight(
        preflight_path, allow_offline_smoke=allow_offline_smoke
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_hub = (
        _bind_frozen_encoder(encoder_source, output_dir)
        if encoder_source is not None
        else None
    )

    paths: dict[str, Path] = {}
    base_path = CONFIG_DIR / "base.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    for arm in ARM_NAMES:
        template_path = CONFIG_DIR / f"{arm}.yaml"
        overlay = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        payload = _merge(base, overlay)
        configured = payload.get("training", {}).get("batch_size")
        if configured != BATCH_SIZE_TOKEN:
            raise ValueError(
                f"BEETLE arm template {template_path} must leave batch size to preflight"
            )
        payload["training"]["batch_size"] = batch_size
        cache_root = payload.get("cache", {}).get("root_dir")
        if cache_root != ENCODER_CACHE_ID_TOKEN:
            raise ValueError("BEETLE base config must namespace its cache by encoder lock")
        lock = _read_json(ENCODER_LOCK_PATH)
        payload["cache"]["root_dir"] = (
            "data/beetle/cache/virchow2_"
            f"{lock['revision']}_{lock['weight_sha256']}_dense_fp16"
        )
        path = output_dir / f"{arm}.yaml"
        source_header = (
            f"# validated_encoder_snapshot: {encoder_source.snapshot_path}\n"
            f"# validated_encoder_revision: {encoder_source.revision}\n"
            f"# validated_encoder_weight_sha256: {encoder_source.weight_sha256}\n"
            if encoder_source is not None
            else "# encoder_source: offline_smoke_cached_grid_fixture\n"
        )
        path.write_text(
            source_header + yaml.safe_dump(payload, sort_keys=False),
            encoding="utf-8",
        )
        load_config(path)
        paths[arm] = path

    resolution = {
        "schema_version": 1,
        "hardware_preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": _sha256(preflight_path),
            "selected_batch_size": batch_size,
        },
        "encoder_lock": _read_json(ENCODER_LOCK_PATH),
        "encoder_preflight": preflight["encoder"],
        "encoder_runtime": (
            {
                "mode": "offline_immutable_hub_snapshot",
                "snapshot_path": str(encoder_source.snapshot_path),
                "hf_hub_cache": str(runtime_hub),
            }
            if encoder_source is not None
            else {"mode": "offline_smoke_cached_grid_fixture"}
        ),
        "arms": {arm: str(path.resolve()) for arm, path in paths.items()},
    }
    (output_dir / "protocol_resolution.json").write_text(
        json.dumps(resolution, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("resolve", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--preflight", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
    run_arm = subparsers.add_parser("run-arm")
    run_arm.add_argument("--preflight", type=Path, required=True)
    run_arm.add_argument("--output-dir", type=Path, required=True)
    run_arm.add_argument("--arm", choices=ARM_NAMES, required=True)
    run_arm.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)

    overrides = getattr(args, "overrides", [])
    invalid_overrides = [
        value
        for value in overrides
        if not value.startswith(RUN_LIFECYCLE_OVERRIDE_PREFIXES)
    ]
    if invalid_overrides:
        parser.error(
            "run-arm --set accepts only run.run_id=... or run.resume=... overrides"
        )

    paths = resolve_arm_configs(
        preflight_path=args.preflight,
        output_dir=args.output_dir,
    )
    if args.command in {"run", "run-arm"}:
        runtime = _read_json(Path(args.output_dir) / "protocol_resolution.json")[
            "encoder_runtime"
        ]
        if runtime.get("mode") != "offline_immutable_hub_snapshot":
            raise ValueError("BEETLE production launch requires an immutable encoder binding")
        runtime_hub = str(Path(runtime["hf_hub_cache"]).resolve())
        environment = {
            **os.environ,
            "HF_HOME": str(Path(runtime_hub).parent),
            "HF_HUB_CACHE": runtime_hub,
            "HUGGINGFACE_HUB_CACHE": runtime_hub,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
        arms = ARM_NAMES if args.command == "run" else (args.arm,)
        for arm in arms:
            command = [sys.executable, "-m", "soma", str(paths[arm].resolve())]
            for override in overrides:
                command.extend(("--set", override))
            subprocess.run(
                command,
                check=True,
                cwd=REPOSITORY_ROOT,
                env=environment,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "BATCH_SIZE_CANDIDATES",
    "resolve_arm_configs",
]
