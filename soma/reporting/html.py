"""HTML report assembly.

render_report(run_data) -> str  produces a fully self-contained HTML page.
Plotly JS is embedded inline so the report opens without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
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
    subgroup_metric_chart,
    subgroup_stats_heatmap,
)
from soma.evaluation.metrics import bh_correct, compare_run_metrics, compute_subgroup_metrics, compute_subgroup_stats
from soma.reporting.data import ComparisonData, RunData, aggregate_fold_predictions

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
    sections.append(_section_subgroup_analysis(run_data))

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


def _section_subgroup_analysis(run_data: RunData) -> str:
    """Subgroup analysis section — only rendered when subgroup columns are configured."""
    if not run_data.subgroup_columns:
        return ""

    all_preds = aggregate_fold_predictions(run_data.folds)
    if all_preds.empty:
        return ""

    # Overall (non-subgroup) test metrics: average across folds
    overall: dict[str, float] = {}
    for metric in run_data.metrics:
        vals = [fd.test_metrics.get(metric) for fd in run_data.folds if metric in fd.test_metrics]
        if vals:
            overall[metric] = float(np.mean(vals))

    sg_metrics = compute_subgroup_metrics(
        run_data.task_family, run_data.metrics, all_preds, run_data.subgroup_columns
    )
    # Always compute stats from the concatenated predictions (more robust than per-fold)
    sg_stats = compute_subgroup_stats(
        run_data.task_family, run_data.metrics, all_preds, run_data.subgroup_columns
    )

    # Apply BH FDR correction per (column, metric): groups are the family.
    sg_stats_adj: dict[str, dict[str, dict[str, float]]] = {}
    for col, col_stats in sg_stats.items():
        sg_stats_adj[col] = {}
        for metric in run_data.metrics:
            tested = [(g, p) for g in sorted(col_stats) if (p := col_stats[g].get(metric)) is not None]
            if not tested:
                continue
            groups_tested, raw_p = zip(*tested)
            for g, p_adj in zip(groups_tested, bh_correct(list(raw_p))):
                sg_stats_adj[col].setdefault(g, {})[metric] = p_adj

    html_parts: list[str] = []

    for col in run_data.subgroup_columns:
        col_data = sg_metrics.get(col, {})
        if not col_data:
            continue
        col_stats_adj = sg_stats_adj.get(col, {})

        # Table: groups × metrics
        # Cell highlight classes (p-values are BH-adjusted):
        #   subgroup-sig  — p_adj < 0.05 AND deviation ≥ 10% (significant and large)
        #   subgroup-flag — deviation ≥ 10% but not statistically significant
        #   subgroup-sig-small — p_adj < 0.05 but small deviation
        header_cells = "<th>Group</th><th>n</th>" + "".join(
            f"<th>{m}</th>" for m in run_data.metrics
        )
        rows_html = ""
        for group_val, group_metrics in sorted(col_data.items()):
            n = group_metrics.get("n", "—")
            cells = f"<td><strong>{group_val}</strong></td><td>{n}</td>"
            for metric in run_data.metrics:
                val = group_metrics.get(metric)
                overall_val = overall.get(metric)
                if val is None:
                    cells += "<td>—</td>"
                    continue
                deviation = abs(val - overall_val) / max(abs(overall_val), 1e-9) if overall_val is not None else 0
                p_adj = col_stats_adj.get(group_val, {}).get(metric)
                significant = p_adj is not None and p_adj < 0.05
                large = deviation >= 0.10
                if significant and large:
                    css = " class=\"subgroup-sig\""
                    tip = f" title=\"p_adj={p_adj:.3f}, Δ={deviation:.0%}\""
                elif large:
                    css = " class=\"subgroup-flag\""
                    tip = f" title=\"Δ={deviation:.0%} (n.s.)\""
                elif significant:
                    css = " class=\"subgroup-sig-small\""
                    tip = f" title=\"p_adj={p_adj:.3f}\""
                else:
                    css, tip = "", ""
                cells += f"<td{css}{tip}>{val:.3f}</td>"
            rows_html += f"<tr>{cells}</tr>"

        table_html = f"""
<table class="metrics-table">
  <thead><tr>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p class="subgroup-legend">
  <span class="legend-sig">■</span> p_adj&lt;0.05 &amp; Δ≥10% &nbsp;
  <span class="legend-flag">■</span> Δ≥10% (not significant) &nbsp;
  <span class="legend-sig-small">■</span> p_adj&lt;0.05 (small Δ)
</p>"""

        # Bar charts (one per metric)
        chart_divs = "".join(
            _chart_div(subgroup_metric_chart(col_data, m, overall.get(m, 0), col), width="half")
            for m in run_data.metrics
            if any(m in col_data[g] for g in col_data)
        )

        html_parts.append(f"""
<h3>{col}</h3>
{table_html}
<div class="chart-grid">{chart_divs}</div>""")

    inner = "\n".join(html_parts)
    return f"""
<div class="section">
  <h2>Subgroup Analysis</h2>
  {inner}
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
    rows.append(("Metrics", ", ".join(cfg.get("eval", {}).get("metrics") or ["(defaults)"])))

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

    # First pass: collect raw p-values for all (metric, run) pairs so we can
    # apply BH FDR correction globally across all comparisons.
    per_metric_data: list[tuple[list[float | None], list[float | None], list[list[float]]]] = []
    for metric in cd.metric_names:
        means: list[float | None] = []
        fold_values: list[list[float]] = []
        for run in cd.runs:
            mean_key = f"{metric}_mean"
            val = run.summary.get(mean_key)
            if val is None:
                val = run.folds[0].test_metrics.get(metric) if run.folds else None
            means.append(val)
            fold_values.append([
                fd.test_metrics[metric]
                for fd in run.folds
                if metric in fd.test_metrics
            ])
        raw_p = compare_run_metrics(fold_values)
        per_metric_data.append((means, raw_p, fold_values))

    # Collect all non-None raw p-values, correct, then redistribute.
    all_keys: list[tuple[int, int]] = []  # (metric_idx, run_idx)
    all_raw_p: list[float] = []
    for m_idx, (_, raw_p, _) in enumerate(per_metric_data):
        for r_idx, p in enumerate(raw_p):
            if p is not None:
                all_keys.append((m_idx, r_idx))
                all_raw_p.append(p)
    corrected = bh_correct(all_raw_p)
    adj_p: dict[tuple[int, int], float] = dict(zip(all_keys, corrected))

    rows_html = ""
    for m_idx, metric in enumerate(cd.metric_names):
        means, _, _ = per_metric_data[m_idx]
        valid_means = [v for v in means if v is not None]
        best_val = max(valid_means) if valid_means else None

        cells = ""
        for i, val in enumerate(means):
            if val is None:
                cells += "<td>—</td>"
                continue
            std_key = f"{metric}_std"
            std = cd.runs[i].summary.get(std_key)
            is_best = best_val is not None and abs(val - best_val) < 1e-9
            p_adj = adj_p.get((m_idx, i))
            sig_worse = not is_best and p_adj is not None and p_adj < 0.05
            if is_best:
                cell_class = " class='best-val'"
                tip = ""
            elif sig_worse:
                cell_class = " class='sig-worse'"
                tip = f" title='significantly worse than best (p_adj={p_adj:.3f})'"
            else:
                cell_class = ""
                tip = f" title='p_adj={p_adj:.3f}'" if p_adj is not None else ""
            std_str = f" ± {std:.4f}" if std is not None else ""
            cells += f"<td{cell_class}{tip}><strong>{val:.4f}</strong>{std_str}</td>"

        rows_html += f"<tr><td class='metric-name'>{metric}</td>{cells}</tr>"

    # Note when statistical comparison was not possible
    all_fold_counts = [len(run.folds) for run in cd.runs]
    stats_note = ""
    if any(n < 2 for n in all_fold_counts):
        stats_note = """
  <p class="muted" style="margin-top:8px;font-size:12px;">
    Statistical comparison requires ≥ 2 folds per run — not available for single-fold runs.
  </p>"""
    else:
        stats_note = """
  <p class="muted" style="margin-top:8px;font-size:12px;">
    <span style="background:#f8d7da;padding:2px 6px;border-radius:3px;">■</span>
    significantly worse than best run (p&lt;0.05, paired permutation test)
  </p>"""

    return f"""
<div class="section">
  <h2>Metrics comparison</h2>
  <table class="results-table">
    <thead>{header_row}</thead>
    <tbody>{rows_html}</tbody>
  </table>
  {stats_note}
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
.sig-worse { background: #f8d7da; }
.subgroup-sig { background: #f8d7da; font-weight: 600; }
.subgroup-flag { background: #fff3cd; font-weight: 600; }
.subgroup-sig-small { background: #e8d5f5; }
.subgroup-legend { font-size: 12px; color: #6c757d; margin-top: 6px; }
.legend-sig { color: #f8d7da; text-shadow: 0 0 1px #666; }
.legend-flag { color: #fff3cd; text-shadow: 0 0 1px #666; }
.legend-sig-small { color: #e8d5f5; text-shadow: 0 0 1px #666; }
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
/* Subgroup analysis */
.subgroup-sig { background: #f8d7da; font-weight: 600; }
.subgroup-flag { background: #fff3cd; font-weight: 600; }
.subgroup-sig-small { background: #e8d5f5; }
.subgroup-legend { font-size: 12px; color: #6c757d; margin-top: 6px; }
@media (max-width: 768px) {
  .section { margin: 12px; padding: 16px; }
  .chart-half { flex: 1 1 100%; }
  .page-header { padding: 20px; }
}
"""
