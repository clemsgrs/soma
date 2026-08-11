"""PathoROB representation-robustness benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from soma.benchmarks.croma import (
    CROMA_0_3_ENCODER_PANEL,
    validate_croma_0_3_encoder_panel,
)
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
from soma.curation.pathorob import curate_pathorob_ri_view
from soma.curation.manifest import CuratedManifest

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


class PathoROBBenchmark:
    """One cohort in the PathoROB robustness benchmark family."""

    canonical_seeds = CANONICAL_SEEDS
    primary_metric = PRIMARY_METRIC
    reported_metrics = REPORTED_METRICS
    ranking_metrics = RANKING_METRICS
    records_croma_version = True
    family_uses_shared_raw_root = True
    reference_environment: dict[str, str] = {}

    def __init__(self, cohort: str) -> None:
        self.cohort = cohort
        self.name = f"pathorob/{cohort}"
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
        return curate_pathorob_ri_view(raw_root, out_dir, cohort=self.cohort)

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
            validate_croma_0_3_encoder_panel()
        cache_overrides = (overrides or {}).get("cache")
        return PipelineConfig(
            dataset_csv=str(dataset_csv) if dataset_csv is not None else "dataset.csv",
            splits_csv=str(splits_csv) if splits_csv is not None else "splits.csv",
            output_root=(
                Path(output_root)
                if output_root is not None
                else Path("output/pathorob")
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
            tags=["pathorob", self.cohort, encoder],
        )

    def expected(self, **axes: Any) -> list[ReferenceRow]:
        merged = {"dataset": self.cohort, "encoder": DEFAULT_ENCODER}
        merged.update({key: value for key, value in axes.items() if value is not None})
        return expected_rows("pathorob", **merged)

    def is_ranking_eligible(self, **axes: Any) -> bool:
        return axes.get("encoder") != "dinov2-vitb14"

    def score(self, run_dir: str | Path) -> dict[str, float]:
        return score_from_summary(run_dir)


PATHOROB_BENCHMARKS: dict[str, PathoROBBenchmark] = {}
for _cohort in COHORTS:
    _benchmark = PathoROBBenchmark(_cohort)
    PATHOROB_BENCHMARKS[_benchmark.name] = _benchmark
    register_benchmark(_benchmark)
