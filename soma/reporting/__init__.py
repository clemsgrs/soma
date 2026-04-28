"""Experiment report generation for soma runs.

Usage — from a saved run directory:
    from soma.reporting import generate_report
    path = generate_report("/path/to/run_dir")

Usage — from an in-memory PipelineResult:
    from soma.reporting import generate_report_from_result
    path = generate_report_from_result(result, config)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from soma.output_layout import _slugify, _stable_json
from soma.reporting.data import (
    ComparisonData,
    FoldData,
    RunData,
    load_comparison_data,
    load_run_data,
    run_data_from_result,
)
from soma.reporting.html import render_comparison_report, render_report

if TYPE_CHECKING:
    from soma.config import PipelineConfig
    from soma.pipeline import PipelineResult

__all__ = [
    "compare_runs",
    "generate_report",
    "generate_report_from_result",
    "load_comparison_data",
    "load_run_data",
    "run_data_from_result",
    "render_comparison_report",
    "render_report",
    "ComparisonData",
    "RunData",
    "FoldData",
]


def generate_report(
    run_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Generate an HTML report from a saved run directory.

    Reads config.yaml, run.yaml, summary.json, and each fold_N/ subdirectory
    from disk.

    Args:
        run_dir: Path to a completed run directory.
        output_path: Destination for the HTML file.
            Defaults to run_dir/report.html.

    Returns:
        Path to the generated report file.
    """
    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path else run_dir / "report.html"

    run_data = load_run_data(run_dir)
    html = render_report(run_data)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def compare_runs(
    run_dirs: list[str | Path],
    *,
    output_dir: str | Path | None = None,
    labels: list[str] | None = None,
) -> Path:
    """Generate an HTML comparison report for multiple runs.

    Args:
        run_dirs: Paths to completed run directories.
        output_dir: Destination directory for the comparison report bundle.
            Defaults to ``<shared output_root>/comparisons/<comparison-id>``.
        labels: Short labels for each run. When None, labels are auto-derived
            from the config diff (e.g., aggregator name when that's the only
            difference), falling back to run_id.

    Returns:
        Path to the generated comparison report.
    """
    run_dirs = [Path(d) for d in run_dirs]
    comparison_data = load_comparison_data(run_dirs, labels=labels)
    if output_dir is None:
        output_dir = _default_comparison_output_dir(comparison_data, run_dirs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html = render_comparison_report(comparison_data)
    report_path = output_dir / "index.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _default_comparison_output_dir(comparison_data: ComparisonData, run_dirs: list[Path]) -> Path:
    output_roots = {
        Path(run.config["output_root"]).expanduser().resolve()
        for run in comparison_data.runs
        if run.config.get("output_root")
    }
    if len(output_roots) != 1:
        raise ValueError(
            "compare_runs requires an explicit output_dir when the compared runs do not "
            "share the same output_root."
        )

    comparison_root = output_roots.pop() / "comparisons"
    label_slug = "-vs-".join(_slugify(label) for label in comparison_data.labels)
    digest_payload = {
        "labels": list(comparison_data.labels),
        "run_dirs": [str(Path(d).expanduser().resolve()) for d in run_dirs],
    }
    digest_input = _stable_json(digest_payload).encode("utf-8")
    digest = hashlib.sha256(digest_input).hexdigest()[:12]
    return comparison_root / f"{label_slug}__{digest}"


def generate_report_from_result(
    result: PipelineResult,
    config: PipelineConfig,
    *,
    dataset: object = None,
    output_path: str | Path | None = None,
) -> Path:
    """Generate an HTML report from an in-memory PipelineResult.

    Avoids redundant disk reads for heavy data (predictions, metrics, history)
    by using the already-computed in-memory objects.

    Args:
        result: PipelineResult returned by Pipeline.run() or train().
        config: The PipelineConfig used for the run.
        dataset: Optional Dataset used to enrich configured subgroup reports.
        output_path: Destination for the HTML file.
            Defaults to result.run_dir/report.html.

    Returns:
        Path to the generated report file.
    """
    output_path = Path(output_path) if output_path else result.run_dir / "report.html"

    run_data = run_data_from_result(result, config, dataset=dataset)
    html = render_report(run_data)
    output_path.write_text(html, encoding="utf-8")
    return output_path
