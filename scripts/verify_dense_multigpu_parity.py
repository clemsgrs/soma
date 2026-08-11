#!/usr/bin/env python3
"""Run the real-GPU dense parity gate for issue #305.

The five-image case produces contiguous two-rank shards of 3 + 2 items at batch size 2,
so rank 0 has a partial tail. The region case flattens three slides with 3 + 3 + 1 ROIs;
the resulting 4 + 3 shards split ``slide-b`` after its first ROI. Its one-GPU batches are
2 + 1 while its two-GPU batches are 1 and 2 across ranks. Payloads and a JSON report are
written beneath the caller-supplied output directory, which must not already exist.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Sequence, TypeVar

import numpy as np
import torch
from PIL import Image
from slide2vec import (
    DenseImageOptions,
    DenseOptions,
    ExecutionOptions,
    ImageSpec,
    Model,
    SlideRegions,
    __version__ as slide2vec_version,
)

ArtifactT = TypeVar("ArtifactT")

IMAGE_IDS = ("image-a", "image-b", "image-c", "image-d", "image-e")
REGION_COORDINATES = {
    "slide-a": ((64, 7104), (64, 7548), (64, 7992)),
    "slide-b": ((508, 7104), (508, 7548), (508, 7992)),
    "slide-c": ((952, 7104),),
}
BATCH_SIZE = 2
COSINE_MINIMUM = 0.9999


@dataclass(frozen=True)
class Comparison:
    semantic_order: list[Any]
    shapes: list[list[int]]
    dtype: str
    minimum_cosine: float
    maximum_cosine_distance: float
    maximum_absolute_delta: float
    mean_absolute_delta: float


def _write_images(root: Path) -> list[ImageSpec]:
    y, x = np.indices((224, 224), dtype=np.uint16)
    specs = []
    for index, sample_id in enumerate(IMAGE_IDS):
        pixels = np.stack(
            (
                (x + 3 * y + 11 * index) % 256,
                (7 * x + y + 13 * index) % 256,
                (5 * x + 9 * y + 17 * index) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        path = root / "inputs" / f"{sample_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(path)
        specs.append(
            ImageSpec(
                sample_id=sample_id,
                image_path=path,
                spacing_at_level_0=0.5,
            )
        )
    return specs


def _regions(slide_path: Path) -> list[SlideRegions]:
    return [
        SlideRegions(
            sample_id=slide_id,
            image_path=slide_path,
            coordinates=coordinates,
            spacing_at_level_0=0.25200000393750005,
        )
        for slide_id, coordinates in REGION_COORDINATES.items()
    ]


def _execution(output_dir: Path, *, num_gpus: int, precision: str) -> ExecutionOptions:
    return ExecutionOptions(
        output_dir=output_dir,
        num_gpus=num_gpus,
        batch_size=BATCH_SIZE,
        num_workers_per_gpu=0,
        precision=precision,
        output_dtype=precision,
    )


def _load_metadata(artifact: Any) -> dict[str, Any]:
    return json.loads(artifact.metadata_path.read_text(encoding="utf-8"))


def _compare_gpu_runs(
    one_gpu: Sequence[ArtifactT],
    two_gpu: Sequence[ArtifactT],
    *,
    semantic_key: Callable[[ArtifactT], Hashable],
) -> Comparison:
    one_gpu_keys = [semantic_key(artifact) for artifact in one_gpu]
    two_gpu_keys = [semantic_key(artifact) for artifact in two_gpu]
    assert one_gpu_keys == two_gpu_keys

    one_gpu_by_key = {semantic_key(artifact): artifact for artifact in one_gpu}
    two_gpu_by_key = {semantic_key(artifact): artifact for artifact in two_gpu}
    assert list(one_gpu_by_key) == one_gpu_keys
    assert list(two_gpu_by_key) == two_gpu_keys

    shapes: list[list[int]] = []
    dtypes: set[torch.dtype] = set()
    cosines: list[float] = []
    absolute_deltas: list[torch.Tensor] = []
    for key in one_gpu_keys:
        one_gpu_artifact = one_gpu_by_key[key]
        two_gpu_artifact = two_gpu_by_key[key]
        one_gpu_grid = torch.load(
            one_gpu_artifact.path, weights_only=True, map_location="cpu"
        )
        two_gpu_grid = torch.load(
            two_gpu_artifact.path, weights_only=True, map_location="cpu"
        )
        assert one_gpu_grid.shape == two_gpu_grid.shape
        assert one_gpu_grid.dtype == two_gpu_grid.dtype
        assert _load_metadata(one_gpu_artifact) == _load_metadata(two_gpu_artifact)
        assert torch.isfinite(one_gpu_grid).all()
        assert torch.isfinite(two_gpu_grid).all()

        cosine = float(
            torch.nn.functional.cosine_similarity(
                one_gpu_grid.float().reshape(-1),
                two_gpu_grid.float().reshape(-1),
                dim=0,
            )
        )
        assert cosine >= COSINE_MINIMUM
        shapes.append(list(one_gpu_grid.shape))
        dtypes.add(one_gpu_grid.dtype)
        cosines.append(cosine)
        absolute_deltas.append(
            (one_gpu_grid.float() - two_gpu_grid.float()).abs().reshape(-1)
        )

    assert len(dtypes) == 1
    all_deltas = torch.cat(absolute_deltas)
    minimum_cosine = min(cosines)
    return Comparison(
        semantic_order=[
            list(key) if isinstance(key, tuple) else key for key in one_gpu_keys
        ],
        shapes=shapes,
        dtype=str(dtypes.pop()).removeprefix("torch."),
        minimum_cosine=minimum_cosine,
        maximum_cosine_distance=1.0 - minimum_cosine,
        maximum_absolute_delta=float(all_deltas.max()),
        mean_absolute_delta=float(all_deltas.mean()),
    )


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _run_gpu_pair(
    output_dir: Path,
    *,
    precision: str,
    embed: Callable[[Model, ExecutionOptions], Sequence[ArtifactT]],
    semantic_key: Callable[[ArtifactT], Hashable],
) -> Comparison:
    one_gpu = embed(
        Model.from_preset("lunit", device="cuda:0"),
        _execution(output_dir / "one-gpu", num_gpus=1, precision=precision),
    )
    _clear_cuda()
    two_gpu = embed(
        Model.from_preset("lunit"),
        _execution(output_dir / "two-gpu", num_gpus=2, precision=precision),
    )
    comparison = _compare_gpu_runs(
        one_gpu,
        two_gpu,
        semantic_key=semantic_key,
    )
    _clear_cuda()
    return comparison


def _slide2vec_commit(source: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()


def run(output_dir: Path, slide_path: Path, slide2vec_source: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("this gate requires at least two visible CUDA devices")

    image_specs = _write_images(output_dir)
    slide_regions = _regions(slide_path.resolve())
    image_dense = DenseImageOptions(target_size=224, spacing_um=0.5)
    region_dense = DenseOptions(
        target_size=224,
        spacing_um=0.5,
        backend="openslide",
    )
    results: dict[str, Any] = {}

    for precision in ("fp32", "fp16"):
        results[f"embed_images_dense_{precision}"] = asdict(
            _run_gpu_pair(
                output_dir / "images" / precision,
                precision=precision,
                embed=lambda model, execution: model.embed_images_dense(
                    image_specs,
                    dense=image_dense,
                    execution=execution,
                ),
                semantic_key=lambda artifact: artifact.sample_id,
            )
        )
        results[f"embed_regions_dense_{precision}"] = asdict(
            _run_gpu_pair(
                output_dir / "regions" / precision,
                precision=precision,
                embed=lambda model, execution: model.embed_regions_dense(
                    slide_regions,
                    dense=region_dense,
                    execution=execution,
                ),
                semantic_key=lambda artifact: (artifact.sample_id, artifact.x, artifact.y),
            )
        )

    report = {
        "contract": {
            "encoder": "lunit",
            "spacing_um": 0.5,
            "target_size": 224,
            "window_size": None,
            "overlap": 0.0,
            "batch_size": BATCH_SIZE,
            "image_count": len(image_specs),
            "image_two_gpu_shard_sizes": [3, 2],
            "region_counts_by_slide": {
                slide_id: len(coordinates)
                for slide_id, coordinates in REGION_COORDINATES.items()
            },
            "region_two_gpu_shard_sizes": [4, 3],
            "region_boundary_crossing_slide": "slide-b",
            "region_boundary_split_after_roi": 1,
            "cosine_minimum": COSINE_MINIMUM,
        },
        "environment": {
            "slide2vec_version": slide2vec_version,
            "slide2vec_commit": _slide2vec_commit(slide2vec_source),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpus": [torch.cuda.get_device_name(index) for index in range(2)],
            "slide_path": str(slide_path.resolve()),
        },
        "results": results,
        "cache_key_implication": (
            "GPU count remains outside soma's dense cache key: rank-dependent batch "
            "composition can change bytes, but all measured grids remain semantically "
            "equivalent within the public cosine contract. A resumed cache may therefore "
            "contain tolerance-equivalent grids produced by different rank batch shapes."
        ),
    }
    report_path = output_dir / "parity-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--slide-path", type=Path, required=True)
    parser.add_argument("--slide2vec-source", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir, args.slide_path, args.slide2vec_source), indent=2))


if __name__ == "__main__":
    main()
