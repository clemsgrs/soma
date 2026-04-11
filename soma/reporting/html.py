"""HTML report assembly.

render_report(run_data) -> str  produces a fully self-contained HTML page.
Plotly JS is embedded inline so the report opens without network access.
"""

from __future__ import annotations

import plotly
import plotly.graph_objects as go

from soma.reporting.charts import (
    comparison_loss_curves,
    comparison_metric_curves,
    confusion_matrix_chart,
    loss_curves,
    lr_curve,
    metric_curves,
    pr_curve_chart,
    residual_plot,
    roc_curve_chart,
    scatter_predicted_vs_actual,
    score_distribution_chart,
)
from soma.reporting.data import ComparisonData, RunData

_CLASSIFICATION_FAMILIES = {"binary_classification", "multiclass_classification"}
_ORDINAL_FAMILIES = {"ordinal_classification"}
_REGRESSION_FAMILIES = {"regression"}


def render_report(run_data: RunData) -> str:
    """Generate a self-contained HTML report string.

    Args:
        run_data: Populated RunData from load_run_data() or run_data_from_result().

    Returns:
        Full HTML string with embedded Plotly JS and all report sections.
    """
    sections: list[str] = []

    sections.append(_section_header(run_data))
    sections.append(_section_config(run_data))
    sections.append(_section_results_summary(run_data))
    sections.append(_section_training_curves(run_data))
    sections.append(_section_prediction_analysis(run_data))

    body = "\n".join(sections)
    plotly_js = plotly.offline.get_plotlyjs()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Experiment Report — {_run_id(run_data)}</title>
  <script>{plotly_js}</script>
  <style>{_css()}</style>
</head>
<body>
{body}
</body>
</html>"""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _section_header(run_data: RunData) -> str:
    meta = run_data.run_metadata
    run_id = _run_id(run_data)
    status = meta.get("status", "—")
    started = meta.get("started_at", "—")
    finished = meta.get("finished_at", "—")
    git_sha = meta.get("git_sha") or "—"
    seed = run_data.config.get("training", {}).get("seed", "—")
    num_folds = len(run_data.folds)

    status_class = {"completed": "status-ok", "failed": "status-err"}.get(status, "status-run")

    return f"""
<div class="page-header">
  <h1>Experiment Report</h1>
  <div class="run-id">{run_id}</div>
  <div class="header-meta">
    <span class="badge {status_class}">{status}</span>
    <span>{num_folds} fold{"s" if num_folds != 1 else ""}</span>
    <span>seed&nbsp;{seed}</span>
    <span>started&nbsp;{started}</span>
    {f'<span>finished&nbsp;{finished}</span>' if finished != "—" else ""}
    <span>git&nbsp;<code>{git_sha[:10] if git_sha != "—" else "—"}</code></span>
  </div>
</div>"""


def _section_config(run_data: RunData) -> str:
    cfg = run_data.config
    rows = _config_rows(cfg)
    table_html = "".join(
        f'<tr><td class="cfg-key">{k}</td><td class="cfg-val">{v}</td></tr>'
        for k, v in rows
    )
    return f"""
<div class="section">
  <h2>Configuration</h2>
  <table class="cfg-table"><tbody>{table_html}</tbody></table>
</div>"""


def _section_results_summary(run_data: RunData) -> str:
    if not run_data.folds:
        return ""

    # Collect all test metric names
    all_metric_names: list[str] = []
    for fd in run_data.folds:
        for name in fd.test_metrics:
            if name not in all_metric_names:
                all_metric_names.append(name)

    fold_headers = "".join(f"<th>Fold {fd.fold}</th>" for fd in run_data.folds)
    header_row = f"<tr><th>Metric</th>{fold_headers}<th>Mean ± Std</th></tr>"

    rows_html = ""
    for metric in all_metric_names:
        fold_vals = [fd.test_metrics.get(metric) for fd in run_data.folds]
        fold_cells = "".join(
            f'<td>{v:.4f}</td>' if v is not None else "<td>—</td>"
            for v in fold_vals
        )
        mean_std = run_data.summary.get(f"{metric}_mean")
        std = run_data.summary.get(f"{metric}_std")
        if mean_std is not None and std is not None:
            summary_cell = f"<td><strong>{mean_std:.4f}</strong> ± {std:.4f}</td>"
        elif mean_std is not None:
            summary_cell = f"<td><strong>{mean_std:.4f}</strong></td>"
        else:
            summary_cell = "<td>—</td>"
        rows_html += f"<tr><td class='metric-name'>{metric}</td>{fold_cells}{summary_cell}</tr>"

    return f"""
<div class="section">
  <h2>Test Results</h2>
  <table class="results-table">
    <thead>{header_row}</thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


def _section_training_curves(run_data: RunData) -> str:
    has_history = any(fd.training_history for fd in run_data.folds)
    if not has_history:
        return """
<div class="section">
  <h2>Training Curves</h2>
  <p class="muted">No training history available (fold_N/training_history.json not found).</p>
</div>"""

    chart_divs: list[str] = []

    # Loss curves (full width)
    chart_divs.append(_chart_div(loss_curves(run_data.folds), width="full"))

    # Metric curves for user-requested metrics (2-column grid)
    for metric_name in run_data.metrics:
        fig = metric_curves(run_data.folds, metric_name)
        chart_divs.append(_chart_div(fig, width="half"))

    # LR curve
    chart_divs.append(_chart_div(lr_curve(run_data.folds), width="half"))

    inner = "\n".join(chart_divs)
    return f"""
<div class="section">
  <h2>Training Curves</h2>
  <div class="chart-grid">{inner}</div>
</div>"""


def _section_prediction_analysis(run_data: RunData) -> str:
    has_preds = any(not fd.predictions.empty for fd in run_data.folds)
    if not has_preds:
        return ""

    family = run_data.task_family
    chart_divs: list[str] = []

    if family in _CLASSIFICATION_FAMILIES:
        # ROC + PR (binary only)
        chart_divs.append(_chart_div(roc_curve_chart(run_data.folds), width="half"))
        if family == "binary_classification":
            chart_divs.append(_chart_div(pr_curve_chart(run_data.folds), width="half"))

        # Confusion matrix + score distribution
        chart_divs.append(_chart_div(confusion_matrix_chart(run_data.folds), width="half"))
        chart_divs.append(_chart_div(score_distribution_chart(run_data.folds), width="half"))

    elif family in _ORDINAL_FAMILIES:
        chart_divs.append(_chart_div(confusion_matrix_chart(run_data.folds), width="half"))
        chart_divs.append(_chart_div(score_distribution_chart(run_data.folds), width="half"))

    elif family in _REGRESSION_FAMILIES:
        chart_divs.append(_chart_div(scatter_predicted_vs_actual(run_data.folds), width="half"))
        chart_divs.append(_chart_div(residual_plot(run_data.folds), width="half"))

    if not chart_divs:
        return ""

    inner = "\n".join(chart_divs)
    return f"""
<div class="section">
  <h2>Prediction Analysis</h2>
  <div class="chart-grid">{inner}</div>
</div>"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chart_div(fig: go.Figure, *, width: str = "full") -> str:
    """Render a Plotly figure as an HTML div fragment."""
    chart_html = fig.to_html(full_html=False, include_plotlyjs=False)
    css_class = "chart-full" if width == "full" else "chart-half"
    return f'<div class="{css_class}">{chart_html}</div>'


def _run_id(run_data: RunData) -> str:
    return run_data.run_metadata.get("run_id") or "—"


def _config_rows(cfg: dict) -> list[tuple[str, str]]:
    """Extract key config fields as (label, value) pairs for display."""
    rows: list[tuple[str, str]] = []

    def _get(obj: dict, *keys: str, default: str = "—") -> str:
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return str(obj) if obj is not None else default

    rows.append(("Task", _get(cfg, "task", "name")))
    rows.append(("Metrics", ", ".join(cfg.get("task", {}).get("metrics") or ["(defaults)"])))

    encoder = cfg.get("encoder")
    if encoder:
        rows.append(("Encoder", _get(cfg, "encoder", "name")))
        rows.append(("Encoder precision", _get(cfg, "encoder", "precision")))
    else:
        rows.append(("Encoder", "— (pre-extracted features)"))

    agg = cfg.get("aggregator")
    if agg:
        rows.append(("Aggregator", _get(cfg, "aggregator", "name")))
        agg_params = cfg.get("aggregator", {}).get("params") or {}
        if agg_params:
            rows.append(("Aggregator params", str(agg_params)))
    else:
        rows.append(("Aggregator", "— (slide-level)"))

    rows.append(("Dataset", _get(cfg, "dataset_csv")))
    rows.append(("Splits", _get(cfg, "splits_csv")))
    rows.append(("Output root", _get(cfg, "output_root")))

    training = cfg.get("training", {})
    rows.append(("Epochs", str(training.get("epochs", "—"))))
    rows.append(("Learning rate", str(training.get("learning_rate", "—"))))
    rows.append(("Optimizer", str(training.get("optimizer", "—"))))
    rows.append(("Scheduler", str(training.get("scheduler", "—"))))
    rows.append(("Patience", str(training.get("patience", "—"))))
    rows.append(("Batch size", str(training.get("batch_size", "—"))))
    rows.append(("Seed", str(training.get("seed", "—"))))

    tags = cfg.get("tags") or []
    if tags:
        rows.append(("Tags", ", ".join(tags)))

    return [(k, v) for k, v in rows if v and v != "—"]


def render_comparison_report(comparison_data: ComparisonData) -> str:
    """Generate a self-contained HTML comparison report string.

    Args:
        comparison_data: Populated ComparisonData from load_comparison_data().

    Returns:
        Full HTML string with embedded Plotly JS and all comparison sections.
    """
    sections: list[str] = []
    sections.append(_comparison_section_header(comparison_data))
    sections.append(_comparison_section_config_diff(comparison_data))
    sections.append(_comparison_section_metrics(comparison_data))
    sections.append(_comparison_section_curves(comparison_data))

    body = "\n".join(s for s in sections if s)
    plotly_js = plotly.offline.get_plotlyjs()
    n = len(comparison_data.runs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Run Comparison ({n} runs)</title>
  <script>{plotly_js}</script>
  <style>{_css()}{_comparison_css()}</style>
</head>
<body>
{body}
</body>
</html>"""


def _comparison_section_header(cd: ComparisonData) -> str:
    n = len(cd.runs)
    label_badges = "".join(
        f'<span class="run-badge" style="background:{_badge_color(i)}">{label}</span>'
        for i, label in enumerate(cd.labels)
    )
    return f"""
<div class="page-header">
  <h1>Run Comparison</h1>
  <div class="header-meta">
    <span>{n} runs</span>
    {label_badges}
  </div>
</div>"""


def _comparison_section_config_diff(cd: ComparisonData) -> str:
    if not any(cd.config_diffs):
        # All configs identical
        return """
<div class="section">
  <h2>Configuration</h2>
  <p class="muted">All runs share identical configurations.</p>
</div>"""

    # Collect all varying keys
    varying_keys: list[str] = []
    for diff in cd.config_diffs:
        for k in diff:
            if k not in varying_keys:
                varying_keys.append(k)

    col_headers = "".join(
        f'<th><span class="run-badge" style="background:{_badge_color(i)}">{label}</span></th>'
        for i, label in enumerate(cd.labels)
    )
    header_row = f"<tr><th>Config field</th>{col_headers}</tr>"

    rows_html = ""
    for key in varying_keys:
        cells = "".join(
            f"<td><code>{diff.get(key, '—')}</code></td>"
            for diff in cd.config_diffs
        )
        rows_html += f"<tr><td class='cfg-key'>{key}</td>{cells}</tr>"

    # Shared config in a collapsible block
    shared_rows = "".join(
        f"<tr><td class='cfg-key'>{k}</td><td class='cfg-val'><code>{v}</code></td></tr>"
        for k, v in sorted(cd.shared_config.items())
    )
    shared_html = f"""
<details class="shared-cfg">
  <summary>Shared configuration ({len(cd.shared_config)} fields)</summary>
  <table class="cfg-table" style="margin-top:10px"><tbody>{shared_rows}</tbody></table>
</details>""" if cd.shared_config else ""

    return f"""
<div class="section">
  <h2>Configuration differences</h2>
  <table class="results-table">
    <thead>{header_row}</thead>
    <tbody>{rows_html}</tbody>
  </table>
  {shared_html}
</div>"""


def _comparison_section_metrics(cd: ComparisonData) -> str:
    if not cd.metric_names:
        return ""

    col_headers = "".join(
        f'<th><span class="run-badge" style="background:{_badge_color(i)}">{label}</span></th>'
        for i, label in enumerate(cd.labels)
    )
    header_row = f"<tr><th>Metric</th>{col_headers}</tr>"

    rows_html = ""
    for metric in cd.metric_names:
        # Collect mean value per run for best-highlighting
        means: list[float | None] = []
        for run in cd.runs:
            mean_key = f"{metric}_mean"
            val = run.summary.get(mean_key)
            if val is None:
                # Fall back to first fold's test metric
                val = run.folds[0].test_metrics.get(metric) if run.folds else None
            means.append(val)

        valid_means = [v for v in means if v is not None]
        best_val = max(valid_means) if valid_means else None

        cells = ""
        for val in means:
            if val is None:
                cells += "<td>—</td>"
                continue
            std_key = f"{metric}_std"
            run_idx = means.index(val)
            std = cd.runs[run_idx].summary.get(std_key)
            is_best = best_val is not None and abs(val - best_val) < 1e-9
            cell_class = " class='best-val'" if is_best else ""
            std_str = f" ± {std:.4f}" if std is not None else ""
            cells += f"<td{cell_class}><strong>{val:.4f}</strong>{std_str}</td>"

        rows_html += f"<tr><td class='metric-name'>{metric}</td>{cells}</tr>"

    return f"""
<div class="section">
  <h2>Metrics comparison</h2>
  <table class="results-table">
    <thead>{header_row}</thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


def _comparison_section_curves(cd: ComparisonData) -> str:
    has_history = any(
        any(fd.training_history for fd in run.folds)
        for run in cd.runs
    )
    if not has_history:
        return """
<div class="section">
  <h2>Training Curves</h2>
  <p class="muted">No training history available.</p>
</div>"""

    chart_divs: list[str] = []
    chart_divs.append(_chart_div(
        comparison_loss_curves(cd.runs, cd.labels), width="full"
    ))
    for metric_name in cd.metric_names:
        chart_divs.append(_chart_div(
            comparison_metric_curves(cd.runs, cd.labels, metric_name), width="half"
        ))

    inner = "\n".join(chart_divs)
    return f"""
<div class="section">
  <h2>Training Curves</h2>
  <div class="chart-grid">{inner}</div>
</div>"""


def _badge_color(idx: int) -> str:
    palette = [
        "#E63946", "#2A9D8F", "#E9C46A", "#264653",
        "#F4A261", "#A8DADC", "#457B9D", "#1D3557",
    ]
    return palette[idx % len(palette)]


def _comparison_css() -> str:
    return """
.run-badge {
  display: inline-block;
  padding: 2px 10px; border-radius: 12px;
  font-size: 12px; font-weight: 600; color: #fff;
  margin-right: 6px;
}
.best-val { background: #d4edda; }
.shared-cfg { margin-top: 16px; }
.shared-cfg summary {
  cursor: pointer; color: #6c757d; font-size: 13px;
  padding: 6px 0;
}
details[open] summary { margin-bottom: 4px; }
"""


def _css() -> str:
    return """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: Inter, system-ui, sans-serif;
  font-size: 14px;
  background: #f8f9fa;
  color: #1a1a2e;
  line-height: 1.5;
}
.page-header {
  background: #1a1a2e;
  color: #fff;
  padding: 28px 40px 20px;
}
.page-header h1 { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
.run-id { font-family: monospace; font-size: 13px; color: #adb5bd; margin-bottom: 10px; }
.header-meta {
  display: flex; flex-wrap: wrap; gap: 16px;
  font-size: 13px; color: #ced4da;
}
.badge {
  padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
}
.status-ok  { background: #198754; color: #fff; }
.status-err { background: #dc3545; color: #fff; }
.status-run { background: #ffc107; color: #000; }
.section {
  background: #fff;
  border-radius: 8px;
  margin: 20px 40px;
  padding: 24px 28px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.section h2 {
  font-size: 17px; font-weight: 600; margin-bottom: 16px;
  border-bottom: 1px solid #e9ecef; padding-bottom: 10px;
}
.muted { color: #868e96; font-style: italic; }
/* Config table */
.cfg-table { border-collapse: collapse; width: 100%; max-width: 680px; }
.cfg-table tr:nth-child(even) td { background: #f8f9fa; }
.cfg-key {
  padding: 6px 12px 6px 0; font-weight: 600; color: #495057;
  white-space: nowrap; width: 200px; vertical-align: top;
}
.cfg-val { padding: 6px 0; color: #1a1a2e; word-break: break-all; }
/* Results table */
.results-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.results-table th {
  background: #f1f3f5; padding: 8px 14px;
  text-align: right; font-weight: 600; border-bottom: 2px solid #dee2e6;
}
.results-table th:first-child { text-align: left; }
.results-table td { padding: 7px 14px; text-align: right; border-bottom: 1px solid #f1f3f5; }
.results-table td.metric-name { text-align: left; font-weight: 500; }
.results-table tr:hover td { background: #f8f9fa; }
/* Chart grid */
.chart-grid { display: flex; flex-wrap: wrap; gap: 16px; }
.chart-full  { flex: 1 1 100%; min-width: 0; }
.chart-half  { flex: 1 1 calc(50% - 8px); min-width: 320px; }
code { font-family: monospace; font-size: 12px; }
@media (max-width: 768px) {
  .section { margin: 12px; padding: 16px; }
  .chart-half { flex: 1 1 100%; }
  .page-header { padding: 20px; }
}
"""
