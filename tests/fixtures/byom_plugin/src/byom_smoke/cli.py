"""Test-owned Benchmark wrapper around soma's real CLI."""

from pathlib import Path
import os
import sys
from types import SimpleNamespace

from PIL import Image
import timm
import torch

from soma.benchmarks import (
    Facet,
    ReferenceRow,
    register_benchmark,
    score_from_summary,
)
from soma.cli import main
from soma.config import (
    CacheConfig,
    EncoderConfig,
    EvalConfig,
    ExecutionConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
)
from soma.curation.manifest import write_manifest
from slide2vec.encoders import encoder_registry


class _DeterministicDino(torch.nn.Module):
    """No-weight stand-in at timm's external model-construction boundary."""

    num_features = 768
    pretrained_cfg = {
        "input_size": (3, 224, 224),
        "mean": (0.0, 0.0, 0.0),
        "std": (1.0, 1.0, 1.0),
        "interpolation": "bilinear",
        "crop_pct": 1.0,
    }

    def __init__(self):
        super().__init__()
        self.patch_embed = SimpleNamespace(patch_size=(14, 14))

    def forward(self, batch):
        pooled = batch.mean(dim=(-2, -1)).mean(dim=1, keepdim=True)
        return pooled.repeat(1, self.num_features)


_real_create_model = timm.create_model


def _offline_create_model(name, **kwargs):
    if name == "vit_base_patch14_dinov2.lvd142m":
        Path(os.environ["BYOM_PUBLIC_FACTORY_SENTINEL"]).write_text(
            f"{name}:pretrained={kwargs['pretrained']}\n"
        )
        return _DeterministicDino()
    return _real_create_model(name, **kwargs)


timm.create_model = _offline_create_model

_public_encoder_class = encoder_registry.require("dinov2-vitb14")
Path(os.environ["BYOM_PUBLIC_CLASS_REPORT"]).write_text(
    f"{_public_encoder_class.__module__}:{_public_encoder_class.__qualname__}\n"
)


class ByomSmokeBenchmark:
    name = "byom-smoke"
    facet = Facet(fixed={}, varied=("encoder",))
    canonical_seeds = (0,)
    primary_metric = "test/accuracy"
    reference_environment = {}

    def curate(self, raw_root, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        splits = []
        assignments = ["train"] * 4 + ["tune"] * 2 + ["test"] * 2
        for index, split in enumerate(assignments):
            label = index % 2
            image_path = out_dir / f"tile-{index}.png"
            value = 32 if label == 0 else 224
            Image.new("RGB", (8, 8), (value, value, value)).save(image_path)
            rows.append(
                {"sample_id": f"tile-{index}", "image_path": image_path, "label": label}
            )
            splits.append(
                {"sample_id": f"tile-{index}", "split": split, "fold": 0}
            )
        return write_manifest(
            out_dir,
            dataset_type="tile",
            dataset_rows=rows,
            split_rows=splits,
            summary={"samples": len(rows)},
        )

    def build_config(
        self,
        *,
        encoder="dinov2-vitb14",
        dataset_csv=None,
        splits_csv=None,
        output_root=None,
        seed=None,
        overrides=None,
    ):
        return PipelineConfig(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_root=output_root,
            dataset_type="tile",
            encoder=EncoderConfig(name=encoder, batch_size=4),
            execution=ExecutionConfig(num_gpus=1, num_workers_per_gpu=0),
            cache=CacheConfig(**(overrides or {}).get("cache", {})),
            task=TaskConfig(name="binary_classification"),
            evaluation=EvalConfig(metrics=["accuracy"]),
            training=TrainingConfig(
                seed=0 if seed is None else seed,
                epochs=1,
                patience=1,
                batch_size=2,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            ),
            tags=["byom-smoke"],
        )

    def expected(self, **axes):
        if axes.get("encoder", "dinov2-vitb14") != "dinov2-vitb14":
            return []
        return [
            ReferenceRow(
                key={"encoder": "dinov2-vitb14"},
                metric=self.primary_metric,
                expected=0.123,
                tolerance=1.0,
                source="fixture reference",
            )
        ]

    def score(self, run_dir):
        return score_from_summary(run_dir)


register_benchmark(ByomSmokeBenchmark())
main(sys.argv[1:])
