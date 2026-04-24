"""HTML report assembly.

render_report(run_data) -> str  produces a fully self-contained HTML page.
Reports are self-contained and offline-capable (no external JS or CSS).
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from soma.reporting.charts import (
    SOMA_PALETTE,
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
    subgroup_metric_chart,
    subgroup_stats_heatmap,
)
from soma.evaluation.metrics import bh_correct, compare_run_predictions, compute_subgroup_metrics, compute_subgroup_stats
from soma.output_layout import _slugify
from soma.reporting.data import ComparisonData, FoldSlice, RunData, aggregate_fold_predictions, fold_slices_for_split

_CLASSIFICATION_FAMILIES = {"binary_classification", "multiclass_classification"}
_ORDINAL_FAMILIES = {"ordinal_classification"}
_REGRESSION_FAMILIES = {"regression"}


def render_report(run_data: RunData) -> str:
    """Generate a self-contained HTML report string.

    Args:
        run_data: Populated RunData from load_run_data() or run_data_from_result().

    Returns:
        Full HTML string with all report sections.
    """
    body = "\n".join(
        part for part in [
            _section_header(run_data),
            _single_run_tabs(run_data),
        ] if part
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SOMA — {_run_id(run_data)}</title>
  <style>{_css()}</style>
  <script>
  function somaTab(groupId, idx) {{
    var g = document.getElementById(groupId);
    var panelsRoot = g.querySelector('.tab-panels');
    if (panelsRoot) {{
      Array.from(panelsRoot.children).forEach(function(p) {{
        if (p.classList && p.classList.contains('tab-panel')) {{
          p.style.display = 'none';
        }}
      }});
    }}
    var barRoot = g.querySelector('.tab-bar');
    if (barRoot) {{
      Array.from(barRoot.children).forEach(function(b) {{
        if (b.classList && b.classList.contains('tab-btn')) {{
          b.classList.remove('active');
        }}
      }});
    }}
    var panel = document.getElementById(idx);
    if (panel) {{
      panel.style.display = '';
    }}
    var btn = g.querySelector('[data-target=\"' + idx + '\"]');
    if (btn) {{
      btn.classList.add('active');
    }}
  }}
  </script>
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
    status_class = {"completed": "status-ok", "failed": "status-err"}.get(status, "status-run")
    git_sha = meta.get("git_sha") or "—"
    seed = run_data.config.get("run", {}).get("seed", run_data.config.get("training", {}).get("seed", "—"))
    num_folds = len(run_data.folds)

    return f"""
<header class="site-header">
  <span class="soma-wordmark">SOMA</span>
  <span class="header-run-id">{run_id}</span>
  <div class="header-right">
    <span class="badge {status_class}">{status}</span>
    <span class="header-meta-item">{num_folds} fold{"s" if num_folds != 1 else ""}</span>
    <span class="header-meta-item">seed {seed}</span>
    <span class="header-meta-item">git <code>{git_sha[:10] if git_sha != "—" else "—"}</code></span>
  </div>
</header>"""


def _single_run_tabs(run_data: RunData) -> str:
    tabs: list[tuple[str, str]] = [
        ("Overview", _single_run_overview_tab(run_data)),
        ("Training results", _single_run_train_tab(run_data)),
        ("Test results", _single_run_test_tab(run_data)),
        ("Configuration", _single_run_config_tab(run_data)),
    ]

    return _tab_shell(tabs, group_id="single-run-tabs")


def _single_run_overview_tab(run_data: RunData) -> str:
    status = run_data.run_metadata.get("status", "—")
    fold_label = f"{len(run_data.folds)} fold{'s' if len(run_data.folds) != 1 else ''}"

    return f"""
<div class="overview-layout">
  <section class="overview-banner">
    <div class="overview-banner-copy">
      <div class="overview-kicker">Single run</div>
      <h2>Run summary</h2>
      <p>Key configuration and headline metrics, arranged for quick scanning.</p>
    </div>
    <div class="overview-banner-meta">
      <span>{fold_label}</span>
      <span>{status}</span>
    </div>
  </section>
  <div class="overview-grid">
    <section class="overview-panel">
      <div class="overview-panel-head">
        <h3>Main results</h3>
        <p>Headline metrics from the primary evaluation split.</p>
      </div>
      {_overview_results_panel(run_data)}
    </section>
    <section class="overview-panel">
      <div class="overview-panel-head">
        <h3>Main configuration</h3>
        <p>Dataset, spacing, encoder, and task head.</p>
      </div>
      {_overview_config_panel(_overview_config_items(run_data))}
    </section>
  </div>
</div>"""


def _single_run_train_tab(run_data: RunData) -> str:
    parts = [_training_summary_panel(run_data)]
    timing = _section_training_timing(run_data)
    if timing:
        parts.append(timing)
    curves = _section_training_curves(run_data)
    if curves:
        parts.append(curves)
    return "".join(parts) or '<div class="section"><p class="muted">No training history available.</p></div>'


def _single_run_test_tab(run_data: RunData) -> str:
    parts = []
    summary = _section_results_summary(run_data)
    if summary:
        parts.append(summary)
    preds = _section_prediction_analysis(run_data)
    if preds:
        parts.append(preds)
    stats = _section_subgroup_analysis(run_data)
    if stats:
        parts.append(stats)
    return "".join(parts) or '<div class="section"><p class="muted">No test results available.</p></div>'


def _single_run_config_tab(run_data: RunData) -> str:
    return _section_config(run_data) or '<div class="section"><p class="muted">No configuration available.</p></div>'


def _overview_config_items(run_data: RunData) -> list[tuple[str, str]]:
    cfg = run_data.config

    def _spacing(value: object) -> str:
        if value is None:
            return "Not set"
        try:
            return f"{float(value):.2f} um"
        except (TypeError, ValueError):
            return str(value)

    def _nested_value(*keys: str) -> str:
        obj: object = cfg
        for key in keys:
            if not isinstance(obj, dict) or key not in obj:
                return "—"
            obj = obj[key]
        if obj in (None, "", "None"):
            return "—"
        return str(obj)

    items: list[tuple[str, str]] = []
    dataset_csv = _basename(cfg.get("dataset_csv"))
    if dataset_csv != "—":
        items.append(("Dataset CSV", dataset_csv))

    splits_csv = _basename(cfg.get("splits_csv"))
    if splits_csv != "—":
        items.append(("Splits CSV", splits_csv))

    items.append(("Spacing", _spacing(cfg.get("preprocessing", {}).get("requested_spacing_um"))))

    encoder = _nested_value("encoder", "name")
    if encoder != "—":
        items.append(("Encoder", encoder))

    aggregator = _nested_value("aggregator", "name")
    if aggregator != "—":
        items.append(("Aggregator", aggregator))

    task_head = _nested_value("task", "name")
    if task_head != "—":
        items.append(("Task head", task_head))

    return items


def _overview_config_panel(items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f"""
        <div class="overview-spec">
          <div class="overview-spec-label">{label}</div>
          <div class="overview-spec-value">{value}</div>
        </div>"""
        for label, value in items
    )
    return f'<div class="overview-spec-grid">{rows}</div>'


def _overview_results_panel(run_data: RunData) -> str:
    cards = _hero_metric_cards(run_data)
    return f'<div class="overview-metrics">{cards}</div>' if cards else '<p class="muted">No summary metrics available.</p>'


def _training_summary_panel(run_data: RunData) -> str:
    cfg = run_data.config.get("training", {})
    items = [
        ("Epochs", str(cfg.get("epochs", "—"))),
        ("Batch size", str(cfg.get("batch_size", "—"))),
        ("Learning rate", str(cfg.get("learning_rate", "—"))),
        ("Optimizer", str(cfg.get("optimizer", "—"))),
        ("Scheduler", str(cfg.get("scheduler", "—"))),
        ("Patience", str(cfg.get("patience", "—"))),
    ]
    grad_accum = cfg.get("gradient_accumulation")
    if grad_accum not in (None, "", 1, "1"):
        items.append(("Gradient accumulation", str(grad_accum)))
    cells = "".join(
        f"""
        <div class="training-spec">
          <div class="training-spec-label">{label}</div>
          <div class="training-spec-value">{value}</div>
        </div>"""
        for label, value in items
    )
    return f"""
<div class="section section-compact">
  <h2>Training configuration</h2>
  <div class="training-spec-grid">{cells}</div>
</div>"""


def _hero_metric_cards(run_data: RunData) -> str:
    if not run_data.folds:
        return ""

    all_split_names = _test_split_names(run_data.folds)
    if not all_split_names or not run_data.metrics:
        return ""

    single_fold = len(run_data.folds) == 1
    split_name = all_split_names[0]
    cards = []
    for metric in run_data.metrics:
        if single_fold:
            value = run_data.summary.get(f"{split_name}/{metric}")
            if value is None:
                continue
            std_html = ""
        else:
            value = run_data.summary.get(f"{split_name}/{metric}_mean")
            std = run_data.summary.get(f"{split_name}/{metric}_std")
            if value is None:
                continue
            std_html = f'<span class="hero-std">± {std:.3f}</span>' if std is not None else ""
        cards.append(f"""
  <div class="hero-card">
    <div class="hero-value">{value:.3f}{std_html}</div>
    <div class="hero-label">{metric}</div>
  </div>""")

    return "".join(cards)


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
  <p class="section-note">Complete flattened configuration for reproducibility and auditing.</p>
  <table class="cfg-table"><tbody>{table_html}</tbody></table>
</div>"""


def _section_results_summary(run_data: RunData) -> str:
    if not run_data.folds:
        return ""

    all_split_names = _test_split_names(run_data.folds)
    single_fold = len(run_data.folds) == 1

    if single_fold:
        header_row = "<tr><th>Metric</th><th>Value</th></tr>"
    else:
        fold_headers = "".join(f"<th>Fold {fd.fold}</th>" for fd in run_data.folds)
        header_row = f"<tr><th>Metric</th>{fold_headers}<th>Mean ± Std</th></tr>"

    sections = ""
    for split_name in all_split_names:
        coverage_table = _coverage_table(run_data, split_name)
        all_metric_names: list[str] = []
        for fd in run_data.folds:
            for name in fd.test_metrics.get(split_name, {}):
                if name in {
                    "coverage",
                    "num_samples",
                    "num_real_samples",
                    "num_placeholder_samples",
                }:
                    continue
                if name not in all_metric_names:
                    all_metric_names.append(name)

        rows_html = ""
        for metric in all_metric_names:
            if single_fold:
                v = run_data.folds[0].test_metrics.get(split_name, {}).get(metric)
                value_cell = f"<td><strong>{v:.4f}</strong></td>" if v is not None else "<td>—</td>"
                rows_html += f"<tr><td class='metric-name'>{metric}</td>{value_cell}</tr>"
            else:
                fold_vals = [fd.test_metrics.get(split_name, {}).get(metric) for fd in run_data.folds]
                fold_cells = "".join(
                    f'<td>{v:.4f}</td>' if v is not None else "<td>—</td>"
                    for v in fold_vals
                )
                mean_std = run_data.summary.get(f"{split_name}/{metric}_mean")
                std = run_data.summary.get(f"{split_name}/{metric}_std")
                if mean_std is not None and std is not None:
                    summary_cell = f"<td><strong>{mean_std:.4f}</strong> ± {std:.4f}</td>"
                elif mean_std is not None:
                    summary_cell = f"<td><strong>{mean_std:.4f}</strong></td>"
                else:
                    summary_cell = "<td>—</td>"
                rows_html += f"<tr><td class='metric-name'>{metric}</td>{fold_cells}{summary_cell}</tr>"

        heading = "Test Results" if len(all_split_names) == 1 else f"Test Results — {split_name}"
        sections += f"""
<div class="section">
  <h2>{heading}</h2>
  {coverage_table}
  <table class="results-table">
    <thead>{header_row}</thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""

    return sections


def _coverage_table(run_data: RunData, split_name: str) -> str:
    rows: list[str] = []
    total_real = 0
    total_samples = 0
    total_placeholder = 0
    has_any = False
    single_fold = len(run_data.folds) == 1

    for fd in run_data.folds:
        metrics = fd.test_metrics.get(split_name, {})
        num_samples = metrics.get("num_samples")
        num_real = metrics.get("num_real_samples")
        num_placeholder = metrics.get("num_placeholder_samples")
        coverage = metrics.get("coverage")
        if num_samples is None or num_real is None or num_placeholder is None or coverage is None:
            continue
        has_any = True
        total_real += int(num_real)
        total_samples += int(num_samples)
        total_placeholder += int(num_placeholder)
        if not single_fold:
            rows.append(
                "<tr>"
                f"<td>Fold {fd.fold}</td>"
                f"<td>{coverage * 100:.1f}% ({int(num_real)}/{int(num_samples)})</td>"
                f"<td>{int(num_real)}</td>"
                f"<td>{int(num_placeholder)}</td>"
                "</tr>"
            )

    if not has_any:
        return ""

    total_coverage = (total_real / total_samples) if total_samples else 0.0
    if single_fold:
        summary_row = (
            "<tr>"
            f"<td><strong>{total_coverage * 100:.1f}% ({total_real}/{total_samples})</strong></td>"
            f"<td><strong>{total_real}</strong></td>"
            f"<td><strong>{total_placeholder}</strong></td>"
            "</tr>"
        )
        return f"""
  <table class="results-table coverage-table">
    <thead><tr><th>Coverage</th><th>Real-feature predictions</th><th>Placeholder predictions</th></tr></thead>
    <tbody>{summary_row}</tbody>
  </table>"""

    rows.append(
        "<tr>"
        "<td><strong>Overall</strong></td>"
        f"<td><strong>{total_coverage * 100:.1f}% ({total_real}/{total_samples})</strong></td>"
        f"<td><strong>{total_real}</strong></td>"
        f"<td><strong>{total_placeholder}</strong></td>"
        "</tr>"
    )

    return f"""
  <table class="results-table coverage-table">
    <thead><tr><th>Fold</th><th>Coverage</th><th>Real-feature predictions</th><th>Placeholder predictions</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
"""


def _section_training_timing(run_data: RunData) -> str:
    rows: list[str] = []
    for fd in run_data.folds:
        if not fd.training_history:
            continue
        final = fd.training_history[-1]
        rows.append(
            "<tr>"
            f"<td>Fold {fd.fold}</td>"
            f"<td>{_format_duration_cell(final.get('elapsed_seconds'))}</td>"
            f"<td>{_format_duration_cell(final.get('avg_epoch_seconds'))}</td>"
            "</tr>"
        )

    if not rows:
        return ""

    return f"""
<div class="section">
  <h2>Elapsed Time</h2>
  <table class="results-table">
    <thead><tr><th>Fold</th><th>Elapsed</th><th>Average epoch</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
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
    chart_divs.append(_chart_div(loss_curves(run_data.folds), width="full"))

    for metric_name in run_data.metrics:
        chart_divs.append(_chart_div(metric_curves(run_data.folds, metric_name), width="half"))

    chart_divs.append(_chart_div(lr_curve(run_data.folds), width="half"))

    inner = "\n".join(chart_divs)
    return f"""
<div class="section">
  <h2>Training Curves</h2>
  <div class="chart-grid">{inner}</div>
</div>"""


def _section_prediction_analysis(run_data: RunData) -> str:
    all_split_names = _test_split_names(run_data.folds)
    if not all_split_names:
        return ""

    family = run_data.task_family
    sections = ""

    for split_name in all_split_names:
        slices: list[FoldSlice] = fold_slices_for_split(run_data.folds, split_name)
        has_preds = any(not s.predictions.empty for s in slices)
        if not has_preds:
            continue

        tabs: list[tuple[str, str]] = []  # (label, svg)
        if family in _CLASSIFICATION_FAMILIES:
            tabs.append(("ROC curve", roc_curve_chart(slices)))
            if family == "binary_classification":
                tabs.append(("PR curve", pr_curve_chart(slices)))
            tabs.append(("Confusion matrix", confusion_matrix_chart(slices)))
        elif family in _ORDINAL_FAMILIES:
            tabs.append(("Confusion matrix", confusion_matrix_chart(slices)))
        elif family in _REGRESSION_FAMILIES:
            tabs.append(("Predicted vs. actual", scatter_predicted_vs_actual(slices)))
            tabs.append(("Residuals", residual_plot(slices)))

        if not tabs:
            continue

        heading = f"Test Performance — {split_name}"
        default_label = tabs[0][0]
        sections += f"""
<div class="section">
  <h2>{heading}</h2>
  <p class="section-note">{default_label} is shown first by default. Use the tabs to inspect alternate diagnostics.</p>
  {_tab_group(tabs)}
</div>"""

    return sections


def _tab_group(tabs: list[tuple[str, str]]) -> str:
    chart_tabs = [(label, f'<div class="chart-panel"><div class="chart-square">{svg}</div></div>') for label, svg in tabs]
    return _tab_shell(chart_tabs)


def _tab_shell(tabs: list[tuple[str, str]], *, group_id: str | None = None) -> str:
    resolved_tabs = [t for t in tabs if t[1]]
    if not resolved_tabs:
        return ""

    resolved_group_id = group_id or f"tg-{uuid.uuid4().hex[:8]}"
    btn_html = ""
    panels_html = ""
    for i, (label, html) in enumerate(resolved_tabs):
        tab_id = f"{_slugify(label)}-tab" if group_id else f"{resolved_group_id}-{_slugify(label)}"
        active_btn = " active" if i == 0 else ""
        hidden = "" if i == 0 else ' style="display:none"'
        btn_html += (
            f'<button class="tab-btn{active_btn}" data-target="{tab_id}" '
            f'onclick="somaTab(\'{resolved_group_id}\',\'{tab_id}\')">{label}</button>'
        )
        panels_html += f'<div class="tab-panel" id="{tab_id}"{hidden}>{html}</div>'

    return f"""<div class="tab-group" id="{resolved_group_id}">
  <div class="tab-bar">{btn_html}</div>
  <div class="tab-panels">{panels_html}</div>
</div>"""


def _section_subgroup_analysis(run_data: RunData) -> str:
    if not run_data.subgroup_columns:
        return ""

    all_split_names = _test_split_names(run_data.folds)

    sections = ""
    for split_name in all_split_names:
        all_preds = aggregate_fold_predictions(run_data.folds, split_name)
        if all_preds.empty:
            continue

        overall: dict[str, float] = {}
        for metric in run_data.metrics:
            vals = [
                fd.test_metrics.get(split_name, {}).get(metric)
                for fd in run_data.folds
                if metric in fd.test_metrics.get(split_name, {})
            ]
            if vals:
                overall[metric] = float(np.mean(vals))

        sg_metrics = compute_subgroup_metrics(
            run_data.task_family, run_data.metrics, all_preds, run_data.subgroup_columns
        )
        sg_stats = compute_subgroup_stats(
            run_data.task_family, run_data.metrics, all_preds, run_data.subgroup_columns
        )

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
<table class="results-table">
  <thead><tr>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
<p class="subgroup-legend">
  <span class="legend-sig">■</span> p_adj&lt;0.05 &amp; Δ≥10% &nbsp;
  <span class="legend-flag">■</span> Δ≥10% (not significant) &nbsp;
  <span class="legend-sig-small">■</span> p_adj&lt;0.05 (small Δ)
</p>"""

            chart_divs = "".join(
                _chart_div(subgroup_metric_chart(col_data, m, overall.get(m, 0), col), width="half")
                for m in run_data.metrics
                if any(m in col_data[g] for g in col_data)
            )

            html_parts.append(f"""
<h3>{col}</h3>
{table_html}
<div class="chart-grid">{chart_divs}</div>""")

        if html_parts:
            inner = "\n".join(html_parts)
            heading = "Subgroup Analysis" if len(all_split_names) == 1 else f"Subgroup Analysis — {split_name}"
            sections += f"""
<div class="section">
  <h2>{heading}</h2>
  {inner}
</div>"""

    return sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _test_split_names(folds: list) -> list[str]:
    return sorted({split_name for fd in folds for split_name in fd.predictions})


def _chart_div(svg: str, *, width: str = "full") -> str:
    css_class = "chart-full" if width == "full" else "chart-half"
    return f'<div class="{css_class}">{svg}</div>'


def _basename(path: object) -> str:
    return os.path.basename(str(path)) if path else "—"


def _primary_split(run: RunData) -> str:
    splits = sorted({s for fd in run.folds for s in fd.test_metrics})
    return splits[0] if splits else "test"


def _run_id(run_data: RunData) -> str:
    return run_data.run_metadata.get("run_id") or "—"


def _format_duration_cell(value: object) -> str:
    if value is None:
        return "—"
    try:
        total_seconds = max(0, int(float(value)))
    except (TypeError, ValueError):
        return "—"
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _config_rows(cfg: dict) -> list[tuple[str, str]]:
    """Flat dot-notation dump of the entire config dict."""
    rows: list[tuple[str, str]] = []

    def _flatten(obj: object, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{prefix}{k}.")
        elif isinstance(obj, list):
            rows.append((prefix.rstrip("."), ", ".join(str(i) for i in obj) or "—"))
        else:
            rows.append((prefix.rstrip("."), str(obj) if obj is not None else "—"))

    _flatten(cfg)
    return [(k, v) for k, v in rows if v and v != "—"]


def _comparison_tabs(cd: ComparisonData) -> str:
    tabs: list[tuple[str, str]] = [
        ("Overview", _comparison_overview_tab(cd)),
        ("Train plots", _comparison_section_curves(cd)),
        ("Test results", _comparison_test_results_tab(cd)),
        ("Statistical analysis", _comparison_section_metrics(cd)),
        ("Configuration", _comparison_configuration_tab(cd)),
    ]

    if any(len(run.folds) > 1 for run in cd.runs):
        fold_tab = _comparison_fold_tab(cd)
        if fold_tab:
            tabs.insert(3, ("Fold-specific results", fold_tab))

    return _tab_shell(tabs, group_id="comparison-tabs")


def _sorted_metric_names(cd: ComparisonData) -> list[str]:
    metric_scores: list[tuple[float, int, str]] = []
    for idx, metric in enumerate(cd.metric_names):
        values: list[float] = []
        for run in cd.runs:
            split = _primary_split(run)
            val = run.summary.get(f"{split}/{metric}_mean")
            if val is None and run.folds:
                val = run.folds[0].test_metrics.get(split, {}).get(metric)
            if val is not None:
                values.append(float(val))
        metric_scores.append((max(values) if values else float("-inf"), idx, metric))
    metric_scores.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [metric for _, _, metric in metric_scores]


def _comparison_overview_tab(cd: ComparisonData) -> str:
    metric_names = _sorted_metric_names(cd)
    primary_metric = metric_names[0] if metric_names else "—"
    top_run = cd.labels[0] if cd.labels else "—"
    if metric_names:
        ranked = []
        for idx, run in enumerate(cd.runs):
            split_name = _primary_split(run)
            value = run.summary.get(f"{split_name}/{primary_metric}_mean")
            if value is None and run.folds:
                value = run.folds[0].test_metrics.get(split_name, {}).get(primary_metric)
            ranked.append((float("-inf") if value is None else float(value), idx))
        ranked.sort(reverse=True)
        if ranked:
            top_run = cd.labels[ranked[0][1]]

    def _display_path(key: str) -> str:
        if key in cd.shared_config and cd.shared_config[key]:
            return _basename(cd.shared_config[key])
        values: list[str] = []
        for diff in cd.config_diffs:
            value = diff.get(key)
            if value:
                values.append(_basename(value))
        if values:
            return " / ".join(values[:2]) + (" …" if len(values) > 2 else "")
        return "—"

    dataset_items = [
        ("Dataset CSV", _display_path("dataset_csv")),
        ("Splits CSV", _display_path("splits_csv")),
    ]

    dataset_context = _overview_config_panel(dataset_items)

    return f"""
<div class="overview-layout">
  <section class="overview-banner">
    <div class="overview-banner-copy">
      <div class="overview-kicker">Cross-run comparison</div>
      <h2>Run ranking</h2>
      <p>Comparing multiple runs on the same dataset.</p>
    </div>
    <div class="overview-banner-meta">
      <span>{len(cd.runs)} run{"s" if len(cd.runs) != 1 else ""}</span>
      <span>{len(metric_names)} metric{"s" if len(metric_names) != 1 else ""}</span>
    </div>
  </section>
  <div class="overview-grid">
    <section class="overview-panel overview-panel-wide">
      <div class="overview-panel-head">
        <h3>Ranking</h3>
        <p>Sorted by {primary_metric}.</p>
      </div>
      {_comparison_ranking_table(cd)}
    </section>
    <section class="overview-panel">
      <div class="overview-panel-head">
        <h3>Dataset context</h3>
        <p>Shared setup used by the compared runs.</p>
      </div>
      {dataset_context}
    </section>
  </div>
</div>"""


def _comparison_test_results_tab(cd: ComparisonData) -> str:
    matrix = _comparison_metric_matrix(cd)
    return matrix or '<div class="section"><p class="muted">No test results available.</p></div>'


def _comparison_configuration_tab(cd: ComparisonData) -> str:
    varying = _comparison_section_config_varying(cd)
    shared = _comparison_section_config_shared(cd)
    parts = [part for part in [varying, shared] if part]
    return "".join(parts) or '<div class="section"><p class="muted">No configuration details available.</p></div>'


def _comparison_fold_tab(cd: ComparisonData) -> str:
    rows: list[str] = []
    for i, run in enumerate(cd.runs):
        split_names = _test_split_names(run.folds)
        if not split_names:
            continue
        for split_name in split_names:
            rows.append(
                f"<tr><td class='metric-name'>{cd.labels[i]}</td><td>{split_name}</td><td>{len(run.folds)}</td></tr>"
            )
    if not rows:
        return ""
    return f"""
<div class="section">
  <h2>Fold-specific results</h2>
  <table class="results-table">
    <thead><tr><th>Run</th><th>Split</th><th>Folds</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</div>"""


def _comparison_ranking_table(cd: ComparisonData) -> str:
    if not cd.metric_names:
        return '<p class="muted">No comparison metrics available.</p>'

    metric_names = _sorted_metric_names(cd)
    primary_metric = metric_names[0]

    ranked_rows: list[tuple[int, int, list[float | None]]] = []
    for idx, run in enumerate(cd.runs):
        split_name = _primary_split(run)
        values: list[float | None] = []
        for metric in metric_names:
            val = run.summary.get(f"{split_name}/{metric}_mean")
            if val is None and run.folds:
                val = run.folds[0].test_metrics.get(split_name, {}).get(metric)
            values.append(val)
        ranked_rows.append((idx, 0, values))

    def _sort_key(item: tuple[int, int, list[float | None]]) -> list:
        idx, _, values = item
        return [float("-inf") if v is None else float(v) for v in values] + [-idx]

    ranked_rows = sorted(ranked_rows, key=_sort_key, reverse=True)

    header_cells = "".join(
        f'<th class="metric-col" data-metric="{metric}">{metric}</th>'
        for metric in metric_names
    )
    rows_html = ""
    podium_classes = ["podium-gold", "podium-silver", "podium-bronze"]
    for rank, (idx, _, values) in enumerate(ranked_rows, start=1):
        label = cd.labels[idx]
        row_class = podium_classes[rank - 1] if rank <= 3 else "soma-brand-rank"
        rank_class = f"rank-{rank}"
        run_href = _run_report_href(cd.run_dirs[idx])
        run_link = (
            f'<a class="rank-link" href="{run_href}" target="_blank" rel="noopener" '
            f'aria-label="Open report for {label}" title="Open report for {label}">'
            f'<span class="rank-link-label">{label}</span>'
            f'<span class="rank-link-arrow" aria-hidden="true">↗</span>'
            f'</a>'
        )
        cells = [
            f'<td class="rank-cell"><span class="rank-pill {row_class}">{rank}</span></td>',
            f'<td class="rank-name">{run_link}</td>',
        ]
        for metric, value in zip(metric_names, values):
            display = "—" if value is None else f"{value:.3f}"
            cells.append(f'<td class="rank-metric">{display}</td>')
        rows_html += f'<tr class="rank-row {rank_class}" data-rank="{rank}" data-label="{label}">{"".join(cells)}</tr>'

    return f"""
<table class="results-table ranking-table">
  <thead><tr><th>Rank</th><th>Run</th>{header_cells}</tr></thead>
  <tbody>{rows_html}</tbody>
</table>
"""


def _run_report_href(run_dir: Path) -> str:
    return (Path(run_dir) / "report.html").resolve().as_uri()


def _comparison_metric_matrix(cd: ComparisonData) -> str:
    if not cd.metric_names:
        return ""

    metric_names = _sorted_metric_names(cd)
    col_headers = "".join(
        f'<th><span class="run-badge" style="background:{_badge_color(i)}">{label}</span></th>'
        for i, label in enumerate(cd.labels)
    )
    header_row = f"<tr><th>Metric</th>{col_headers}</tr>"

    primary_splits = [_primary_split(run) for run in cd.runs]

    rows_html = ""
    for metric in metric_names:
        cells = []
        vals: list[float | None] = []
        for i, run in enumerate(cd.runs):
            split_name = primary_splits[i]
            val = run.summary.get(f"{split_name}/{metric}_mean")
            if val is None and run.folds:
                val = run.folds[0].test_metrics.get(split_name, {}).get(metric)
            vals.append(val)
        valid = [v for v in vals if v is not None]
        best_val = max(valid) if valid else None
        for i, val in enumerate(vals):
            if val is None:
                cells.append("<td>—</td>")
                continue
            is_best = best_val is not None and abs(val - best_val) < 1e-9
            cell_class = "best-val" if is_best else ""
            std = cd.runs[i].summary.get(f"{primary_splits[i]}/{metric}_std")
            std_str = f" <span class='metric-std'>± {std:.3f}</span>" if std is not None else ""
            cells.append(f"<td class='{cell_class}'><strong>{val:.3f}</strong>{std_str}</td>")
        rows_html += f"<tr><td class='metric-name'>{metric}</td>{''.join(cells)}</tr>"

    return f"""
<div class="section">
  <h2>Test results</h2>
  <table class="results-table">
    <thead>{header_row}</thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


def _comparison_section_metrics(cd: ComparisonData) -> str:
    if not cd.metric_names:
        return ""

    col_headers = "".join(
        f'<th><span class="run-badge" style="background:{_badge_color(i)}">{label}</span></th>'
        for i, label in enumerate(cd.labels)
    )
    header_row = f"<tr><th>Metric</th>{col_headers}</tr>"

    task_family = cd.runs[0].task_family if cd.runs else ""
    primary_splits = [_primary_split(run) for run in cd.runs]

    runs_pooled = [
        aggregate_fold_predictions(run.folds, primary_splits[i])
        for i, run in enumerate(cd.runs)
    ]

    per_metric_data: list[tuple[list[float | None], list[float | None]]] = []
    for metric in cd.metric_names:
        means: list[float | None] = []
        for i, run in enumerate(cd.runs):
            split_name = primary_splits[i]
            mean_key = f"{split_name}/{metric}_mean"
            val = run.summary.get(mean_key)
            if val is None:
                val = run.folds[0].test_metrics.get(split_name, {}).get(metric) if run.folds else None
            means.append(val)
        raw_p = compare_run_predictions(runs_pooled, task_family, metric)
        per_metric_data.append((means, raw_p))

    all_keys: list[tuple[int, int]] = []
    all_raw_p: list[float] = []
    for m_idx, (_, raw_p) in enumerate(per_metric_data):
        for r_idx, p in enumerate(raw_p):
            if p is not None:
                all_keys.append((m_idx, r_idx))
                all_raw_p.append(p)
    corrected = bh_correct(all_raw_p)
    adj_p: dict[tuple[int, int], float] = dict(zip(all_keys, corrected))

    rows_html = ""
    for m_idx, metric in enumerate(cd.metric_names):
        means, _ = per_metric_data[m_idx]
        valid_means = [v for v in means if v is not None]
        best_val = max(valid_means) if valid_means else None

        cells = ""
        for i, val in enumerate(means):
            if val is None:
                cells += "<td>—</td>"
                continue
            std = cd.runs[i].summary.get(f"{primary_splits[i]}/{metric}_std")
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

    stats_note = """
  <p class="muted" style="margin-top:8px;font-size:12px;">
    <span style="background:#fecdd3;padding:2px 6px;border-radius:3px;">■</span>
    significantly worse than best run (p&lt;0.05, sample-level paired permutation test)
  </p>"""

    return f"""
<div class="section">
  <h2>Statistical analysis</h2>
  <table class="results-table">
    <thead>{header_row}</thead>
    <tbody>{rows_html}</tbody>
  </table>
  {stats_note}
</div>"""


def render_comparison_report(comparison_data: ComparisonData) -> str:
    """Generate a self-contained HTML comparison report string."""
    body = "\n".join(
        part for part in [
            _comparison_section_header(comparison_data),
            _comparison_tabs(comparison_data),
        ] if part
    )
    n = len(comparison_data.runs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>SOMA — Run Comparison ({n} runs)</title>
  <style>{_css()}</style>
  <script>
  function somaTab(groupId, idx) {{
    var g = document.getElementById(groupId);
    var panelsRoot = g.querySelector('.tab-panels');
    if (panelsRoot) {{
      panelsRoot.querySelectorAll(':scope > .tab-panel').forEach(function(p) {{ p.style.display = 'none'; }});
    }}
    var bar = g.querySelector('.tab-bar');
    if (bar) {{
      bar.querySelectorAll(':scope > .tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    }}
    var panel = document.getElementById(idx);
    if (panel) {{
      panel.style.display = '';
    }}
    var btn = g.querySelector('[data-target=\"' + idx + '\"]');
    if (btn) {{
      btn.classList.add('active');
    }}
  }}
  </script>
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
<header class="site-header">
  <span class="soma-wordmark">SOMA</span>
  <span class="header-run-id">Run Comparison</span>
  <div class="header-right">
    <span class="header-meta-item">{n} runs</span>
    {label_badges}
  </div>
</header>"""


def _comparison_section_hero_metrics(cd: ComparisonData) -> str:
    if not cd.metric_names:
        return ""

    cards = []
    for metric in cd.metric_names:
        means: list[float | None] = []
        stds: list[float | None] = []
        for run in cd.runs:
            split_name = _primary_split(run)
            val = run.summary.get(f"{split_name}/{metric}_mean")
            if val is None:
                val = run.folds[0].test_metrics.get(split_name, {}).get(metric) if run.folds else None
            means.append(val)
            std_key = f"{split_name}/{metric}_std"
            stds.append(run.summary.get(std_key))

        valid = [v for v in means if v is not None]
        if not valid:
            continue
        best_val = max(valid)

        rows = ""
        for i, (label, val, std) in enumerate(zip(cd.labels, means, stds)):
            if val is None:
                continue
            color = _badge_color(i)
            is_best = abs(val - best_val) < 1e-9
            val_weight = "font-weight:700;" if is_best else "font-weight:400;opacity:0.8;"
            std_html = f'<span style="font-size:0.8rem;color:var(--soma-text-muted)"> ± {std:.3f}</span>' if std is not None else ""
            best_mark = ' <span style="color:var(--soma-success);font-size:0.9rem">▲</span>' if is_best else ""
            rows += f"""
  <div class="comp-hero-row">
    <span class="run-badge" style="background:{color}">{label}</span>
    <span class="comp-hero-val" style="color:{color};{val_weight}">{val:.3f}{std_html}{best_mark}</span>
  </div>"""

        cards.append(f"""
<div class="comp-hero-card">
  <div class="comp-hero-metric">{metric.upper()}</div>
  {rows}
</div>""")

    if not cards:
        return ""
    return f'<div class="hero-strip"><div class="comp-hero-grid">{"".join(cards)}</div></div>'


def _comparison_section_run_context(cd: ComparisonData) -> str:
    shared = cd.shared_config
    diffs = cd.config_diffs

    items: list[str] = []

    # Shared dataset / splits → single chip
    for key, label in [("dataset_csv", "Dataset"), ("splits_csv", "Splits")]:
        val = shared.get(key)
        if val:
            name = _basename(str(val))
            items.append(
                f'<span class="ctx-chip"><span class="ctx-label">{label}</span>'
                f'<span class="ctx-value">{name}</span></span>'
            )
        else:
            # Differs per run → one chip per run
            per_run = [diff.get(key) for diff in diffs]
            if any(v for v in per_run):
                for i, (label_run, val) in enumerate(zip(cd.labels, per_run)):
                    if val:
                        color = _badge_color(i)
                        items.append(
                            f'<span class="ctx-chip">'
                            f'<span class="ctx-label" style="color:{color}">{label_run} {label.lower()}</span>'
                            f'<span class="ctx-value">{_basename(str(val))}</span></span>'
                        )

    if not items:
        return ""
    return f'<div class="run-context">{"".join(items)}</div>'


def _comparison_section_config_varying(cd: ComparisonData) -> str:
    if not any(cd.config_diffs):
        return """
<div class="section">
  <h2>Varying fields</h2>
  <p class="section-note">All runs share identical configurations.</p>
</div>"""

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

    return f"""
<div class="section">
  <h2>Varying fields</h2>
  <p class="section-note">The table highlights only the values that differ across runs.</p>
  <table class="results-table" style="margin-top:12px">
    <thead>{header_row}</thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


def _comparison_section_config_shared(cd: ComparisonData) -> str:
    if not cd.shared_config:
        return ""

    shared_rows = "".join(
        f"<tr><td class='cfg-key'>{k}</td><td class='cfg-val'><code>{v}</code></td></tr>"
        for k, v in sorted(cd.shared_config.items())
    )
    return f"""
<div class="section">
  <h2>Shared fields</h2>
  <p class="section-note">These settings are identical across runs.</p>
  <table class="cfg-table"><tbody>{shared_rows}</tbody></table>
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
    chart_divs.append(_chart_div(comparison_loss_curves(cd.runs, cd.labels), width="full"))
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
    return SOMA_PALETTE[idx % len(SOMA_PALETTE)]


def _css() -> str:
    return """
:root {
  --soma-bg:           #FFFFFF;
  --soma-bg-subtle:    #FFF7FA;
  --soma-bg-card:      #FFF0F5;
  --soma-text:         #0F172A;
  --soma-text-muted:   #64748B;
  --soma-accent:       #E2558A;
  --soma-accent-light: #FCE3ED;
  --soma-accent-deep:  #B52B65;
  --soma-border:       #E9D5DF;
  --soma-header-bg:    #171018;
  --soma-header-text:  #F8FAFC;
  --soma-success:      #10B981;
  --soma-danger:       #EF4444;
  --soma-warning:      #F59E0B;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-size: 14px;
  background:
    radial-gradient(circle at top left, rgba(226, 85, 138, 0.08), transparent 35%),
    linear-gradient(180deg, #fff9fc 0%, #fff6f9 42%, #f8f5f8 100%);
  color: var(--soma-text);
  line-height: 1.5;
}
/* ---- Header ---- */
.site-header {
  background: var(--soma-header-bg);
  color: var(--soma-header-text);
  padding: 0 28px;
  height: 58px;
  display: flex;
  align-items: center;
  gap: 14px;
  position: sticky;
  top: 0;
  z-index: 100;
  box-shadow: 0 10px 30px rgba(15, 23, 42, 0.15);
}
.soma-wordmark {
  font-weight: 800;
  font-size: 15px;
  letter-spacing: 0.08em;
  color: var(--soma-accent);
  flex-shrink: 0;
}
.header-run-id {
  font-family: ui-monospace, 'JetBrains Mono', monospace;
  font-size: 12px;
  color: #94A3B8;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-shrink: 0;
}
.header-meta-item {
  font-size: 12px;
  color: #94A3B8;
}
/* ---- Hero metrics ---- */
.overview-metrics {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 12px;
}
.hero-card {
  background:
    linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,246,250,0.92));
  border: 1px solid rgba(226,85,138,0.18);
  border-left: 4px solid var(--soma-accent);
  border-radius: 20px;
  padding: 14px 16px;
  min-width: 0;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.03);
}
.hero-value {
  font-size: 1.9rem;
  font-weight: 700;
  color: var(--soma-accent-deep);
  line-height: 1.1;
}
.hero-std {
  font-size: 1rem;
  font-weight: 400;
  color: var(--soma-text-muted);
  margin-left: 4px;
}
.hero-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--soma-text-muted);
  margin-top: 4px;
}
/* ---- Badges ---- */
.badge {
  padding: 2px 10px; border-radius: 999px; font-size: 11px; font-weight: 700;
}
.status-ok  { background: #10B981; color: #fff; }
.status-err { background: #EF4444; color: #fff; }
.status-run { background: #F59E0B; color: #000; }
.run-badge {
  display: inline-block;
  padding: 2px 10px; border-radius: 999px;
  font-size: 11px; font-weight: 700; color: #fff;
}
/* ---- Sections ---- */
.section {
  background: var(--soma-bg);
  border-radius: 18px;
  margin: 18px 28px;
  padding: 22px 24px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04), 0 14px 40px rgba(15, 23, 42, 0.04);
  border: 1px solid var(--soma-border);
}
.section-compact {
  margin: 0;
  padding: 0;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  background: transparent;
}
.section h2 {
  font-size: 15px; font-weight: 700; margin-bottom: 16px;
  border-bottom: 1px solid var(--soma-border); padding-bottom: 10px;
  color: var(--soma-text);
}
.section h3 {
  font-size: 13px; font-weight: 600; margin: 16px 0 8px;
  color: var(--soma-text-muted);
}
.tab-stack {
  display: flex;
  flex-direction: column;
}
.overview-layout {
  display: grid;
  gap: 16px;
}
.overview-banner {
  display: flex;
  justify-content: space-between;
  gap: 18px;
  align-items: end;
  padding: 22px 24px;
  border-radius: 26px;
  background:
    radial-gradient(circle at top left, rgba(226, 85, 138, 0.16), transparent 38%),
    linear-gradient(135deg, rgba(255, 255, 255, 0.99), rgba(252, 227, 237, 0.92));
  border: 1px solid rgba(226, 85, 138, 0.18);
  box-shadow: 0 18px 42px rgba(15, 23, 42, 0.06);
}
.overview-banner-copy {
  max-width: 52ch;
}
.overview-banner-meta {
  display: flex;
  gap: 8px;
  align-items: center;
  color: var(--soma-text-muted);
  font-size: 12px;
  font-weight: 700;
}
.overview-banner-meta span {
  background: rgba(255, 255, 255, 0.75);
  border: 1px solid rgba(226, 85, 138, 0.14);
  padding: 6px 10px;
  border-radius: 999px;
}
.overview-kicker {
  color: var(--soma-accent);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 10px;
  font-weight: 800;
  margin-bottom: 8px;
}
.overview-banner h2 {
  font-size: 22px;
  line-height: 1.12;
  margin-bottom: 8px;
  border: 0;
  padding: 0;
}
.overview-banner p {
  max-width: 54ch;
  color: var(--soma-text-muted);
  font-size: 13px;
}
.overview-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 16px;
}
.overview-panel {
  background: linear-gradient(180deg, #ffffff, #fffafc);
  border: 1px solid var(--soma-border);
  border-radius: 22px;
  padding: 20px 22px;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03), 0 12px 28px rgba(15, 23, 42, 0.03);
}
.overview-panel-wide {
  width: 100%;
}
.overview-panel-head {
  margin-bottom: 16px;
}
.overview-inline-head {
  margin-bottom: 12px;
}
.overview-panel-head h3 {
  font-size: 14px;
  font-weight: 800;
  color: var(--soma-text);
  margin-bottom: 4px;
}
.overview-panel-head p {
  color: var(--soma-text-muted);
  font-size: 12px;
}
.overview-panel-foot {
  margin-top: 16px;
}
.overview-spec-grid,
.training-spec-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
.overview-spec,
.training-spec {
  padding: 12px 14px;
  border-radius: 16px;
  background: rgba(226, 85, 138, 0.05);
  border: 1px solid rgba(226, 85, 138, 0.12);
}
.overview-spec-label,
.training-spec-label {
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--soma-text-muted);
  margin-bottom: 6px;
}
.overview-spec-value,
.training-spec-value {
  font-size: 13px;
  font-weight: 700;
  color: var(--soma-text);
  word-break: break-word;
}
.overview-spec-grid > .overview-spec:last-child:nth-child(odd),
.training-spec-grid > .training-spec:last-child:nth-child(odd) {
  grid-column: 1 / -1;
}
.muted { color: var(--soma-text-muted); font-style: italic; }
/* ---- Config collapsible ---- */
.cfg-details summary {
  cursor: pointer;
  font-size: 15px;
  font-weight: 600;
  color: var(--soma-text);
  padding-bottom: 0;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 8px;
}
.cfg-details summary::before {
  content: "▶";
  font-size: 10px;
  color: var(--soma-text-muted);
  transition: transform 0.15s;
}
.cfg-details[open] summary::before { transform: rotate(90deg); }
.cfg-details[open] summary { margin-bottom: 16px; border-bottom: 1px solid var(--soma-border); padding-bottom: 10px; }
.cfg-table {
  border-collapse: collapse;
  width: 100%;
  max-width: none;
  margin-top: 4px;
  table-layout: fixed;
}
.cfg-table tr:nth-child(even) td { background: var(--soma-bg-subtle); }
.cfg-key {
  padding: 6px 12px 6px 0; font-weight: 600; color: var(--soma-text-muted);
  white-space: nowrap; width: 32%; vertical-align: top; font-size: 12px;
}
.cfg-val { padding: 6px 0; color: var(--soma-text); word-break: break-all; font-size: 12px; }
.section-note {
  margin: -6px 0 14px;
  color: var(--soma-text-muted);
  font-size: 12px;
}
/* ---- Results table ---- */
.results-table { border-collapse: collapse; width: 100%; font-size: 13px; }
.results-table th {
  background: var(--soma-bg-subtle); padding: 8px 14px;
  text-align: right; font-weight: 600; border-bottom: 2px solid var(--soma-border);
  color: var(--soma-text-muted); font-size: 12px;
}
.results-table th:first-child { text-align: left; }
.results-table td { padding: 7px 14px; text-align: right; border-bottom: 1px solid var(--soma-bg-subtle); }
.results-table td.metric-name { text-align: left; font-weight: 500; }
.results-table tr:hover td { background: var(--soma-bg-subtle); }
.ranking-table .rank-cell { width: 44px; }
.ranking-table .rank-name { text-align: left; font-weight: 700; }
.ranking-table .rank-metric { font-variant-numeric: tabular-nums; }
.rank-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: var(--soma-text);
  text-decoration: none;
}
.rank-link-label {
  transition: transform 0.16s ease;
}
.rank-link-arrow {
  color: var(--soma-accent);
  font-size: 11px;
  line-height: 1;
  opacity: 1;
  transform: translate(0, 0);
  transition: opacity 0.16s ease, transform 0.16s ease;
}
.rank-link:hover .rank-link-label,
.rank-link:focus-visible .rank-link-label {
  transform: translateX(1px);
}
.rank-link:hover .rank-link-arrow,
.rank-link:focus-visible .rank-link-arrow {
  transform: translate(1px, -1px);
}
.rank-link:focus-visible {
  outline: 2px solid rgba(226, 85, 138, 0.35);
  outline-offset: 3px;
  border-radius: 999px;
}
.ranking-table .rank-row:hover td {
  background: rgba(226, 85, 138, 0.07);
}
.rank-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 30px;
  height: 30px;
  border-radius: 999px;
  padding: 0 10px;
  color: #fff;
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0.01em;
}
.podium-gold { background: linear-gradient(180deg, #F6E08A, #D6A81F); color: #3B2E06; }
.podium-silver { background: linear-gradient(180deg, #E5E7EB, #A8B0BD); color: #1F2937; }
.podium-bronze { background: linear-gradient(180deg, #F1C08B, #B86A24); color: #40220B; }
.soma-brand-rank { background: linear-gradient(180deg, #F7BDD1, var(--soma-accent)); }
.section-overview .results-table tr.rank-row td {
  background: transparent;
}
.section-overview .results-table tr.rank-row.podium-gold td,
.section-overview .results-table tr.rank-row.podium-silver td,
.section-overview .results-table tr.rank-row.podium-bronze td,
.section-overview .results-table tr.rank-row.soma-brand-rank td {
  background: rgba(226, 85, 138, 0.05);
}
/* ---- Chart grid ---- */
.chart-grid { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 8px; }
.chart-full { flex: 1 1 100%; min-width: 0; }
.chart-full svg { width: 100%; height: auto; display: block; }
.chart-half {
  flex: 1 1 calc(50% - 6px);
  max-width: calc(50% - 6px);
  min-width: 280px;
  aspect-ratio: 7 / 4.5;
  overflow: hidden;
}
.chart-half svg { width: 100%; height: 100%; display: block; }
/* ---- Comparison ---- */
.best-val { background: rgba(16, 185, 129, 0.14); }
.sig-worse { background: rgba(239, 68, 68, 0.12); }
/* ---- Subgroup analysis ---- */
.subgroup-sig { background: #FEE2E2; font-weight: 600; }
.subgroup-flag { background: #FEF3C7; font-weight: 600; }
.subgroup-sig-small { background: #EDE9FE; }
.subgroup-legend { font-size: 12px; color: var(--soma-text-muted); margin-top: 6px; }
.legend-sig { color: #FEE2E2; text-shadow: 0 0 1px #666; }
.legend-flag { color: #FEF3C7; text-shadow: 0 0 1px #666; }
.legend-sig-small { color: #EDE9FE; text-shadow: 0 0 1px #666; }
/* ---- Comparison hero ---- */
.comp-hero-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.comp-hero-card {
  background: var(--soma-bg-subtle);
  border: 1px solid var(--soma-border);
  border-radius: 16px;
  padding: 14px 20px;
  min-width: 180px;
}
.comp-hero-metric {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  color: var(--soma-text-muted);
  margin-bottom: 10px;
}
.comp-hero-row {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 6px;
}
.comp-hero-row:last-child { margin-bottom: 0; }
.comp-hero-val {
  font-size: 1.4rem;
  line-height: 1.1;
}
/* ---- Tabs ---- */
.tab-group {
  margin: 18px 28px 0;
  border: 1px solid rgba(226, 85, 138, 0.18);
  border-radius: 24px;
  background:
    linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,247,250,0.88));
  backdrop-filter: blur(10px);
  box-shadow:
    0 12px 30px rgba(15, 23, 42, 0.04),
    0 1px 0 rgba(255,255,255,0.8) inset;
  overflow: hidden;
}
.tab-panels {
  padding: 16px;
}
.tab-panel {
  width: 100%;
}
.tab-bar {
  display: flex;
  gap: 8px;
  border-bottom: 1px solid rgba(226, 85, 138, 0.12);
  margin-bottom: 0;
  padding: 10px 12px 0;
  overflow-x: auto;
  background: linear-gradient(180deg, rgba(255,255,255,0.5), rgba(255,247,250,0.28));
}
.tab-btn {
  padding: 10px 16px 11px;
  border: 1px solid transparent;
  background: rgba(255,255,255,0.45);
  cursor: pointer;
  font-size: 13px;
  font-weight: 700;
  color: var(--soma-text-muted);
  border-bottom: 0;
  margin-bottom: 0;
  border-radius: 16px 16px 0 0;
  transition:
    color 0.15s ease,
    background 0.15s ease,
    border-color 0.15s ease,
    box-shadow 0.15s ease,
    transform 0.15s ease;
  white-space: nowrap;
}
.tab-btn:hover {
  color: var(--soma-text);
  background: rgba(255,255,255,0.92);
  transform: translateY(-1px);
}
.tab-btn.active {
  color: var(--soma-accent-deep);
  background: linear-gradient(180deg, #ffffff, #fff5f9);
  border-color: rgba(226, 85, 138, 0.22);
  box-shadow:
    0 -1px 0 rgba(255,255,255,0.85) inset,
    0 8px 22px rgba(226, 85, 138, 0.08);
  transform: translateY(-1px);
}
.tab-panels .section { margin: 0 0 16px; }
.tab-panels .section:last-child { margin-bottom: 0; }
.tab-panels .overview-layout { padding: 0; }
.tab-panels .overview-panel { margin: 0; }
.tab-panels .overview-grid { margin-top: 0; }
.tab-panels .section-compact {
  padding: 0;
}
.chart-square { max-width: 560px; }
.chart-square svg { width: 100%; height: auto; display: block; }
.chart-panel {
  border: 1px solid rgba(226, 85, 138, 0.14);
  border-radius: 20px;
  padding: 14px;
  background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,247,250,0.8));
}
code { font-family: ui-monospace, 'JetBrains Mono', monospace; font-size: 12px; }
@media (max-width: 768px) {
  .section { margin: 12px; padding: 16px; border-radius: 16px; }
  .chart-half { flex: 1 1 100%; max-width: 100%; aspect-ratio: unset; }
  .site-header { padding: 0 16px; }
  .tab-bar { padding: 8px 8px 0; }
  .tab-group { margin: 12px; border-radius: 18px; }
  .tab-panels { padding: 12px; }
  .overview-grid,
  .overview-spec-grid,
  .training-spec-grid { grid-template-columns: 1fr; }
  .overview-banner { align-items: start; flex-direction: column; }
}
"""
