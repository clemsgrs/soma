"""Experiment report generation for soma runs.

Usage — from a saved run directory:
    from soma.reporting import generate_report
    path = generate_report("/path/to/run_dir")

Usage — from an in-memory PipelineResult:
    from soma.reporting import generate_report_from_result
    path = generate_report_from_result(result, config)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from soma.reporting.data import RunData, FoldData, load_run_data, run_data_from_result
from soma.reporting.html import render_report

if TYPE_CHECKING:
    from soma.config import PipelineConfig
    from soma.pipeline import PipelineResult

__all__ = [
    "generate_report",
    "generate_report_from_result",
    "load_run_data",
    "run_data_from_result",
    "render_report",
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


def generate_report_from_result(
    result: PipelineResult,
    config: PipelineConfig,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Generate an HTML report from an in-memory PipelineResult.

    Avoids redundant disk reads for heavy data (predictions, metrics, history)
    by using the already-computed in-memory objects.

    Args:
        result: PipelineResult returned by Pipeline.run() or train().
        config: The PipelineConfig used for the run.
        output_path: Destination for the HTML file.
            Defaults to result.run_dir/report.html.

    Returns:
        Path to the generated report file.
    """
    output_path = Path(output_path) if output_path else result.run_dir / "report.html"

    run_data = run_data_from_result(result, config)
    html = render_report(run_data)
    output_path.write_text(html, encoding="utf-8")
    return output_path
