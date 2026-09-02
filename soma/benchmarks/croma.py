"""The CRoMa robustness benchmark family and its Croma 0.3 encoder panel."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Iterable
from typing import Any, Mapping

from slide2vec.encoders import encoder_registry, resolve_encoder_output

from pathlib import Path
from typing import Any

from soma.benchmarks.registry import (
    Facet,
    ReferenceRow,
    expected_rows,
    register_benchmark,
    score_from_summary,
)
from soma.config import (
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    RepresentationConfig,
    TrainingConfig,
)
from soma.benchmarks.reproduction import ResolvabilityPolicy
from soma.curation.croma import curate_croma_view
from soma.curation.manifest import CuratedManifest


@dataclass(frozen=True, slots=True)
class EncoderOutputSpec:
    """The soma encoder output corresponding to one published model name."""

    soma_encoder: str
    output_variant: str
    dimension: int


CROMA_0_3_ENCODER_PANEL: Mapping[str, EncoderOutputSpec] = MappingProxyType(
    {
        "Virchow2": EncoderOutputSpec("virchow2", "cls_patch_mean", 2560),
        "CONCH": EncoderOutputSpec("conch", "default", 512),
        "GenBio-PathFM": EncoderOutputSpec("genbio-pathfm", "default", 4608),
        "CONCHv1.5": EncoderOutputSpec("conchv15", "default", 768),
        "H0-mini": EncoderOutputSpec("h0-mini", "cls_patch_mean", 1536),
        "Virchow": EncoderOutputSpec("virchow", "cls_patch_mean", 2560),
        "Midnight-12k": EncoderOutputSpec("midnight", "default", 3072),
        "H-optimus-1": EncoderOutputSpec("h-optimus-1", "default", 1536),
        "DINOv2-B": EncoderOutputSpec("dinov2-vitb14", "default", 768),
        "H-optimus-0": EncoderOutputSpec("h-optimus-0", "default", 1536),
        "UNI2-h": EncoderOutputSpec("uni2", "default", 1536),
        "MUSK": EncoderOutputSpec("musk", "ms_aug", 2048),
        "mSTAR": EncoderOutputSpec("mstar", "default", 1024),
        "Prov-GigaPath": EncoderOutputSpec("gigapath", "default", 1536),
        "UNI": EncoderOutputSpec("uni", "default", 1024),
        "Hibou-B": EncoderOutputSpec("hibou-b", "default", 768),
        "GPFM": EncoderOutputSpec("gpfm", "default", 1024),
        "Phikon": EncoderOutputSpec("phikon", "default", 768),
        "Phikon-v2": EncoderOutputSpec("phikonv2", "default", 1024),
        "Prost40M": EncoderOutputSpec("prost40m", "default", 384),
        "Hibou-L": EncoderOutputSpec("hibou-l", "default", 1024),
        "Mascaret": EncoderOutputSpec("mascaret", "default", 1536),
        "Phaet": EncoderOutputSpec("phaet", "default", 1024),
        "RudolfV-2": EncoderOutputSpec("rudolfv2", "cls_patch_mean", 3072),
        "RudolfV-2-B": EncoderOutputSpec("rudolfv2-b", "cls_patch_mean", 1536),
        "RudolfV-2-S": EncoderOutputSpec("rudolfv2-s", "cls_patch_mean", 768),
    }
)


def _metadata_mismatch(
    spec: EncoderOutputSpec,
    metadata: Mapping[str, Any] | None,
    reason: str,
) -> ValueError:
    return ValueError(
        "Croma 0.3 encoder metadata mismatch: "
        f"encoder={spec.soma_encoder!r}, "
        f"requested_variant={spec.output_variant!r}, "
        f"expected_dimension={spec.dimension}, "
        f"observed_metadata={metadata!r}, "
        f"reason={reason}"
    )


def validate_croma_0_3_encoder_panel(
    *,
    metadata_by_encoder: Mapping[str, Mapping[str, Any]] | None = None,
    encoders: Iterable[str] | None = None,
) -> None:
    """Validate the pinned panel against slide2vec metadata without loading weights.

    ``encoders`` restricts the check to those soma encoder names (the whole panel when
    ``None``), so building a config for one panel encoder does not fail on an unrelated
    panel member whose registry metadata drifted.
    """
    if encoders is None:
        specs = list(CROMA_0_3_ENCODER_PANEL.values())
    else:
        wanted = set(encoders)
        specs = [spec for spec in CROMA_0_3_ENCODER_PANEL.values() if spec.soma_encoder in wanted]
    for spec in specs:
        try:
            metadata = (
                encoder_registry.info(spec.soma_encoder)
                if metadata_by_encoder is None
                else dict(metadata_by_encoder[spec.soma_encoder])
            )
        except Exception as error:
            raise _metadata_mismatch(spec, None, str(error)) from error
        level = metadata.get("level")
        if level != "tile":
            raise _metadata_mismatch(
                spec,
                metadata,
                f"expected tile-level encoder, observed level={level!r}",
            )
        try:
            resolved = resolve_encoder_output(
                spec.soma_encoder,
                requested_output_variant=spec.output_variant,
                metadata=metadata,
            )
        except Exception as error:
            raise _metadata_mismatch(spec, metadata, str(error)) from error
        if resolved.get("encode_dim") != spec.dimension:
            raise _metadata_mismatch(
                spec,
                metadata,
                f"observed dimension={resolved.get('encode_dim')!r}",
            )


COHORTS = ("camelyon", "tcga-4x4", "tolkach-esca")
CANONICAL_SEEDS = (0,)
PRIMARY_METRIC = "test/croma_median"
REPORTED_METRICS = (
    PRIMARY_METRIC,
    "test/croma_f0",
    "test/croma_ltm10",
)
RANKING_METRICS = (PRIMARY_METRIC, "test/croma_ltm10")
DEFAULT_ENCODER = "uni2"
REPRESENTATION_PROTOCOL = {
    "kind": "croma",
    "confounder_column": "medical_center",
    "split": "test",
    "evaluation_design": "all",
    "m": 5,
    "alpha": 0.10,
}

_PANEL_BY_ENCODER = {
    spec.soma_encoder: spec for spec in CROMA_0_3_ENCODER_PANEL.values()
}


class CromaBenchmark:
    """One cohort in the CRoMa robustness benchmark family."""

    canonical_seeds = CANONICAL_SEEDS
    primary_metric = PRIMARY_METRIC
    reported_metrics = REPORTED_METRICS
    ranking_metrics = RANKING_METRICS
    records_croma_version = True
    # CRoMa's ranking metrics live on a signed scale that crosses zero, so pair
    # resolvability uses the near-zero-safe hybrid rule decided in soma#321:
    # |a−b| > max(0.005, 0.02·max(|a|,|b|)), strict boundary, on unrounded references.
    # Other families keep the absolute rule they were published under.
    resolvability = ResolvabilityPolicy.hybrid(abs_floor=0.005, rel=0.02)
    family_uses_shared_raw_root = True
    reference_environment: dict[str, str] = {}

    def __init__(self, cohort: str) -> None:
        self.cohort = cohort
        self.name = f"croma/{cohort}"
        self.facet = Facet(
            fixed={
                "dataset": cohort,
                "dataset_type": "tile",
                **{
                    f"representation.{field}": value
                    for field, value in REPRESENTATION_PROTOCOL.items()
                },
            },
            varied=("encoder",),
        )

    def curate(self, raw_root: str | Path, out_dir: str | Path) -> CuratedManifest:
        return curate_croma_view(raw_root, out_dir, cohort=self.cohort)

    def build_config(
        self,
        *,
        encoder: str = DEFAULT_ENCODER,
        dataset_csv: str | Path | None = None,
        splits_csv: str | Path | None = None,
        output_root: str | Path | None = None,
        seed: int | None = None,
        overrides: dict[str, Any] | None = None,
        encoder_batch_size: int = 32,
    ) -> PipelineConfig:
        panel_spec = _PANEL_BY_ENCODER.get(encoder)
        if panel_spec is not None:
            validate_croma_0_3_encoder_panel(encoders=[encoder])
        cache_overrides = (overrides or {}).get("cache")
        return PipelineConfig(
            dataset_csv=str(dataset_csv) if dataset_csv is not None else "dataset.csv",
            splits_csv=str(splits_csv) if splits_csv is not None else "splits.csv",
            output_root=(
                Path(output_root)
                if output_root is not None
                else Path("output/croma")
            ),
            dataset_type="tile",
            cache=(
                CacheConfig(**cache_overrides)
                if cache_overrides
                else CacheConfig(enabled=True)
            ),
            encoder=EncoderConfig(
                name=encoder,
                output_variant=(
                    panel_spec.output_variant if panel_spec is not None else None
                ),
                batch_size=encoder_batch_size,
            ),
            task=None,
            representation=RepresentationConfig(**REPRESENTATION_PROTOCOL),
            training=TrainingConfig(seed=0 if seed is None else int(seed)),
            tags=["croma", self.cohort, encoder],
        )

    def expected(self, **axes: Any) -> list[ReferenceRow]:
        merged = {"dataset": self.cohort, "encoder": DEFAULT_ENCODER}
        merged.update({key: value for key, value in axes.items() if value is not None})
        return expected_rows("croma", **merged)

    def is_ranking_eligible(self, **axes: Any) -> bool:
        return axes.get("encoder") != "dinov2-vitb14"

    def score(self, run_dir: str | Path) -> dict[str, float]:
        return score_from_summary(run_dir)


CROMA_BENCHMARKS: dict[str, CromaBenchmark] = {}
for _cohort in COHORTS:
    _benchmark = CromaBenchmark(_cohort)
    CROMA_BENCHMARKS[_benchmark.name] = _benchmark
    register_benchmark(_benchmark)
