"""The two rich panels a run prints: what it trains on, and what it scored."""

from __future__ import annotations

from rich import box
from rich.panel import Panel
from rich.table import Table

from slide2vec import PreprocessingConfig

from soma.config import AggregatorConfig, DecoderConfig, EncoderLeaf, TaskConfig

__all__ = ["build_completed_run_panel", "build_run_summary_panel"]


def build_run_summary_panel(
    *,
    encoder: EncoderLeaf | None,
    preprocessing: PreprocessingConfig,
    aggregator: AggregatorConfig | None,
    task: TaskConfig,
    feature_store: object,
    dataset_type: str = "slide",
    decoder: DecoderConfig | None = None,
) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", justify="right", no_wrap=True)
    grid.add_column()

    # Encoder
    if encoder is not None:
        grid.add_row("encoder", encoder.name)
    else:
        grid.add_row("encoder", "[dim]pre-extracted[/dim]")

    # Spacing — only meaningful for slide/WSI pipelines
    if dataset_type != "tile":
        spacing = None if preprocessing.tiling is None else preprocessing.tiling.requested_spacing_um
        grid.add_row("spacing", f"{spacing} µm" if spacing is not None else "[dim]—[/dim]")

    # Aggregator (set shape) or decoder (grid shape) — the trainable component.
    if dataset_type in ("segmentation", "detection"):
        grid.add_row("decoder", decoder.name if decoder is not None else "[dim]—[/dim]")
    elif aggregator is not None:
        grid.add_row("aggregator", aggregator.name)
    else:
        grid.add_row("aggregator", "[dim]—[/dim]")

    # Task
    grid.add_row("task", task.name.replace("_", " "))

    # Feature level + dim
    if dataset_type == "patient":
        level = "patient"
    elif dataset_type == "tile":
        level = "tile (encoded)"
    elif dataset_type in ("segmentation", "detection"):
        # A grid source is not a set source (no rank); branch before rank is touched.
        level = "dense (segmentation)" if dataset_type == "segmentation" else "dense (detection)"
    elif feature_store.rank == 1:
        level = "slide"
    elif feature_store.rank == 3:
        level = "hierarchical"
    else:
        level = "tile"
    feature_dim = getattr(feature_store, "feature_dim", None)
    if feature_dim is None:
        grid.add_row("features", f"{level}  ·  live re-encoding")
    else:
        grid.add_row("features", f"{level}  ·  dim={feature_dim}")

    return Panel.fit(
        grid,
        title="[bold]run summary[/bold]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def build_completed_run_panel(*, summary_metrics: dict[str, float]) -> Panel:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold", justify="right", no_wrap=True)
    grid.add_column()

    def _split_metric(split_name: str, metric: str) -> float | None:
        for suffix in ("_mean", ""):
            key = f"{split_name}/{metric}{suffix}"
            if key in summary_metrics:
                return summary_metrics[key]
        return None

    split_names = sorted({key.split("/", 1)[0] for key in summary_metrics if "/" in key})
    for split_name in split_names:
        primary_metric_name: str | None = None
        primary_metric_mean: float | None = None
        for key, value in summary_metrics.items():
            prefix = f"{split_name}/"
            if not key.startswith(prefix):
                continue
            if key.endswith("_mean"):
                metric_name = key[len(prefix):-5]
            else:
                metric_name = key[len(prefix):]
            if metric_name.endswith("_std"):
                continue
            if metric_name in {
                "coverage",
                "num_samples",
                "num_real_samples",
                "num_placeholder_samples",
            }:
                continue
            primary_metric_name = metric_name
            primary_metric_mean = value
            break

        coverage = _split_metric(split_name, "coverage")
        num_samples = _split_metric(split_name, "num_samples")
        num_real = _split_metric(split_name, "num_real_samples")
        num_placeholder = _split_metric(split_name, "num_placeholder_samples")
        if (
            coverage is None
            or num_samples is None
            or num_real is None
            or num_placeholder is None
        ):
            continue
        metric_text = (
            f"{primary_metric_name}={primary_metric_mean:.4f}  ·  "
            if primary_metric_name is not None and primary_metric_mean is not None
            else ""
        )
        coverage_text = (
            f"{metric_text}{coverage * 100:.1f}%"
            f" ({int(round(num_real))}/{int(round(num_samples))})"
            f"  ·  placeholder={int(round(num_placeholder))}"
        )
        grid.add_row(f"{split_name} coverage", coverage_text)

    if not grid.rows:
        grid.add_row("coverage", "[dim]—[/dim]")

    return Panel.fit(
        grid,
        title="[bold]completed run[/bold]",
        border_style="dim",
        box=box.ROUNDED,
        padding=(0, 1),
    )
