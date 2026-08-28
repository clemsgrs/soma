"""Production preflight for the BEETLE frozen-Virchow2 campaign."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from time import perf_counter
from typing import Sequence

import torch

from soma.decoders.registry import build_decoder_for_grid
from soma.dense.geometry import compute_dense_geometry
from soma.tasks.segmentation import SegmentationHead
from soma.training.model import SegmentationModel

from examples.beetle.launch import (
    ENCODER_LOCK_PATH,
    FrozenEncoderSource,
    _bind_frozen_encoder,
)
from examples.beetle.preflight_extraction import run_representative_extraction_parity


DEFAULT_DATASET_CSV = Path("data/beetle/curated_slide_manifest/dataset.csv")
DEFAULT_SPLITS_CSV = Path("data/beetle/curated_slide_manifest/splits.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_encoder_snapshot(
    snapshot_path: str | Path,
) -> tuple[dict, FrozenEncoderSource]:
    """Verify the local Virchow2 object against the tracked immutable lock."""
    lock = json.loads(ENCODER_LOCK_PATH.read_text(encoding="utf-8"))
    snapshot = Path(snapshot_path).resolve()
    repository_cache_name = f"models--{lock['repository'].replace('/', '--')}"
    if (
        snapshot.name != lock["revision"]
        or snapshot.parent.name != "snapshots"
        or snapshot.parent.parent.name != repository_cache_name
    ):
        raise ValueError(
            "snapshot-path does not identify the locked Hugging Face Virchow2 revision"
        )
    for filename in ("config.json", lock["weight_file"]):
        if not (snapshot / filename).is_file():
            raise ValueError(f"locked Virchow2 snapshot is missing {filename}")
    actual_sha256 = _sha256(snapshot / lock["weight_file"])
    if actual_sha256 != lock["weight_sha256"]:
        raise ValueError(
            "locked Virchow2 weight checksum mismatch: "
            f"expected {lock['weight_sha256']}, observed {actual_sha256}"
        )
    source = FrozenEncoderSource(
        snapshot_path=snapshot,
        revision=lock["revision"],
        weight_sha256=actual_sha256,
    )
    return (
        {
            **lock,
            "weight_checksum_verified": True,
            "snapshot_path": str(snapshot),
        },
        source,
    )


def bind_frozen_encoder(source: FrozenEncoderSource, output_dir: str | Path) -> Path:
    """Expose the validated snapshot as the only offline Hub `main` revision."""
    return _bind_frozen_encoder(source, Path(output_dir))


def collect_node_observations(*, output_dir: str | Path, device: str) -> dict:
    """Capture the GPU/CUDA and writable-storage facts used by this gate."""
    if not torch.cuda.is_available():
        raise RuntimeError("BEETLE production preflight requires visible CUDA GPUs")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError("BEETLE production preflight decoder device must be CUDA")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_probe"
    probe.write_bytes(b"beetle-preflight\n")
    probe.unlink()
    gpus = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": props.total_memory,
                "compute_capability": [props.major, props.minor],
                "multiprocessor_count": props.multi_processor_count,
            }
        )
    usage = shutil.disk_usage(output_dir)
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu_count": len(gpus),
        "gpus": gpus,
        "storage": {
            "path": str(output_dir),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "write_verified": True,
        },
    }


@contextmanager
def _offline_encoder_environment(runtime_hub: Path):
    keys = {
        "HF_HOME": str(runtime_hub.parent),
        "HF_HUB_CACHE": str(runtime_hub),
        "HUGGINGFACE_HUB_CACHE": str(runtime_hub),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_campaign_preflight(
    *,
    snapshot_path: str | Path,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    batch_size_candidates: list[int],
    output_path: str | Path,
    device: str = "cuda:0",
) -> dict:
    """Run all production gates and atomically publish the launcher's schema-v1 record."""
    candidates = list(validate_decoder_batch_candidates(batch_size_candidates))
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoder, source = validate_frozen_encoder_snapshot(snapshot_path)
    artifacts = output_path.parent / "preflight_artifacts"
    runtime_hub = bind_frozen_encoder(source, artifacts / "encoder_runtime")
    node = collect_node_observations(output_dir=output_path.parent, device=device)
    with _offline_encoder_environment(runtime_hub):
        extraction = run_representative_extraction_parity(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=artifacts / "representative_extraction",
        ).to_dict()
        decoder = probe_decoder_batch_candidates(
            candidates,
            device=device,
            steps=2,
        )
    payload = {
        "schema_version": 1,
        "status": "completed",
        "scope": "campaign",
        **decoder,
        "same_batch_every_arm_and_fold": True,
        "decoder_probe_device": str(torch.device(device)),
        "encoder_batch_size": 8,
        "encoder": encoder,
        "encoder_runtime": {
            "mode": "offline_immutable_hub_snapshot",
            "hf_hub_cache": str(runtime_hub),
        },
        "node": node,
        "representative_extraction": extraction,
    }
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return payload


def validate_decoder_batch_candidates(candidates: list[int]) -> tuple[int, ...]:
    """Validate the ordered downstream-decoder batch sizes to probe."""
    if (
        not candidates
        or any(
            isinstance(candidate, bool) or not isinstance(candidate, int) or candidate <= 0
            for candidate in candidates
        )
        or any(left <= right for left, right in zip(candidates, candidates[1:]))
    ):
        raise ValueError(
            "decoder batch-size candidates must be positive integers in strictly "
            "descending order"
        )
    return tuple(candidates)


def probe_decoder_batch_size(
    *, batch_size: int, device: str | torch.device, steps: int = 2
) -> dict:
    """Run complete optimizer steps for the tracked downstream decoder recipe."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer, got {batch_size!r}")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError(f"steps must be an integer >= 2, got {steps!r}")

    resolved_device = torch.device(device)
    torch.manual_seed(0)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    geometry = compute_dense_geometry(target_size=512, patch_size=14)
    decoder = build_decoder_for_grid(
        "lightweight_conv",
        {"hidden_dim": 256, "num_upsample_blocks": 2, "num_groups": 32},
        geometry=geometry,
        input_dim=1280,
        num_classes=4,
    )
    head = SegmentationHead(num_classes=4, geometry=geometry)
    model = SegmentationModel(decoder=decoder, task_head=head).to(resolved_device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    first_parameter = next(model.parameters())
    before = first_parameter.detach().clone()
    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)

    features = torch.randn(
        batch_size,
        1280,
        *geometry.grid_shape,
        device=resolved_device,
        dtype=torch.float32,
    )
    base_mask = (
        torch.arange(512 * 512, device=resolved_device, dtype=torch.long)
        .remainder(4)
        .reshape(512, 512)
    )
    masks = base_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    masks[:, -1, :] = 255

    started = perf_counter()
    final_loss = None
    logits_shape: list[int] | None = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model(features)
        loss = head.compute_loss(output.logits, {"mask": masks})
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        logits_shape = list(output.logits.shape)
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
    elapsed = perf_counter() - started

    result = {
        "batch_size": batch_size,
        "passed": True,
        "steps": steps,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype).removeprefix("torch."),
        "logits_shape": logits_shape,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "optimizer": {
            "name": "Adam",
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
        },
        "parameters_changed": not torch.equal(before, first_parameter.detach()),
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
        "samples_per_second": batch_size * steps / elapsed,
    }
    if resolved_device.type == "cuda":
        props = torch.cuda.get_device_properties(resolved_device)
        result.update(
            {
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(resolved_device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(resolved_device),
                "total_memory_bytes": props.total_memory,
                "headroom_bytes": props.total_memory
                - torch.cuda.max_memory_reserved(resolved_device),
            }
        )
    return result


def decoder_batch_attempt(
    *, batch_size: int, device: str | torch.device, steps: int = 2
) -> dict:
    """Return one serializable attempt, including CUDA OOM as an ordinary failure."""
    try:
        return probe_decoder_batch_size(
            batch_size=batch_size,
            device=device,
            steps=steps,
        )
    except Exception as exc:
        return {
            "batch_size": batch_size,
            "passed": False,
            "steps": steps,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _failed_subprocess_attempt(
    *, batch_size: int, completed: subprocess.CompletedProcess[str]
) -> dict:
    detail = completed.stderr.strip() or completed.stdout.strip()
    return {
        "batch_size": batch_size,
        "passed": False,
        "error_type": "SubprocessError",
        "error": detail or f"decoder probe worker exited with status {completed.returncode}",
    }


def _read_worker_attempt(
    *, batch_size: int, completed: subprocess.CompletedProcess[str]
) -> dict:
    if completed.returncode != 0:
        return _failed_subprocess_attempt(batch_size=batch_size, completed=completed)
    try:
        attempt = json.loads(completed.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        return {
            "batch_size": batch_size,
            "passed": False,
            "error_type": "SubprocessProtocolError",
            "error": f"decoder probe worker returned invalid JSON: {exc}",
        }
    if not isinstance(attempt, dict) or attempt.get("batch_size") != batch_size:
        return {
            "batch_size": batch_size,
            "passed": False,
            "error_type": "SubprocessProtocolError",
            "error": "decoder probe worker returned the wrong batch size",
        }
    return attempt


def probe_decoder_batch_candidates(
    candidates: list[int],
    *,
    device: str | torch.device,
    steps: int = 2,
) -> dict:
    """Probe every candidate in its own process and select the largest passing size."""
    ordered = validate_decoder_batch_candidates(candidates)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 2:
        raise ValueError(f"steps must be an integer >= 2, got {steps!r}")

    attempts: list[dict] = []
    for batch_size in ordered:
        command = [
            sys.executable,
            "-m",
            "examples.beetle.preflight",
            "--decoder-batch-worker",
            "--probe-decoder-batch-size",
            str(batch_size),
            "--probe-decoder-device",
            str(torch.device(device)),
            "--probe-decoder-steps",
            str(steps),
        ]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            attempts.append(
                {
                    "batch_size": batch_size,
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            continue
        attempts.append(_read_worker_attempt(batch_size=batch_size, completed=completed))

    passing = [
        int(attempt["batch_size"])
        for attempt in attempts
        if attempt.get("passed") is True
    ]
    if not passing:
        raise RuntimeError("no decoder batch-size candidate passed the optimizer-step probe")
    return {
        "batch_size_candidates": list(ordered),
        "batch_size_attempts": attempts,
        "selected_batch_size": max(passing),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-path", type=Path)
    parser.add_argument("--dataset-csv", type=Path, default=DEFAULT_DATASET_CSV)
    parser.add_argument("--splits-csv", type=Path, default=DEFAULT_SPLITS_CSV)
    parser.add_argument(
        "--batch-size-candidates", type=int, nargs="+", default=[16, 8, 4]
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--decoder-batch-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--probe-decoder-batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--probe-decoder-device", help=argparse.SUPPRESS)
    parser.add_argument("--probe-decoder-steps", type=int, default=2, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.decoder_batch_worker:
        if args.probe_decoder_batch_size is None or args.probe_decoder_device is None:
            raise SystemExit("decoder batch worker requires batch size and device")
        attempt = decoder_batch_attempt(
            batch_size=args.probe_decoder_batch_size,
            device=args.probe_decoder_device,
            steps=args.probe_decoder_steps,
        )
        print(json.dumps(attempt, sort_keys=True))
        return 0
    if args.snapshot_path is None or args.output is None:
        raise SystemExit("production preflight requires --snapshot-path and --output")
    run_campaign_preflight(
        snapshot_path=args.snapshot_path,
        dataset_csv=args.dataset_csv,
        splits_csv=args.splits_csv,
        batch_size_candidates=args.batch_size_candidates,
        output_path=args.output,
        device=args.device,
    )
    return 0


__all__ = [
    "decoder_batch_attempt",
    "run_campaign_preflight",
    "probe_decoder_batch_candidates",
    "probe_decoder_batch_size",
    "validate_frozen_encoder_snapshot",
    "validate_decoder_batch_candidates",
]


if __name__ == "__main__":
    main()
