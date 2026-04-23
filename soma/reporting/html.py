"""HTML report assembly.

render_report(run_data) -> str  produces a fully self-contained HTML page.
Reports are self-contained and offline-capable (no external JS or CSS).
"""

from __future__ import annotations

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
    sections: list[str] = []

    sections.append(_section_header(run_data))
    sections.append(_section_hero_metrics(run_data))
    sections.append(_section_run_context(run_data))
    sections.append(_section_config(run_data))
    sections.append(_section_results_summary(run_data))
    sections.append(_section_training_timing(run_data))
    sections.append(_section_training_curves(run_data))
    sections.append(_section_prediction_analysis(run_data))
    sections.append(_section_subgroup_analysis(run_data))

    body = "\n".join(s for s in sections if s)

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
    g.querySelectorAll('.tab-panel').forEach(function(p) {{ p.style.display = 'none'; }});
    g.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    document.getElementById(groupId + '-' + idx).style.display = '';
    g.querySelectorAll('.tab-btn')[idx].classList.add('active');
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


def _section_hero_metrics(run_data: RunData) -> str:
    if not run_data.folds:
        return ""

    all_split_names = _test_split_names(run_data.folds)
    if not all_split_names or not run_data.metrics:
        return ""

    single_fold = len(run_data.folds) == 1
    # Use the first split for hero display
    split_name = all_split_names[0]
    cards = []
    for metric in run_data.metrics:
        if single_fold:
            value = run_data.summary.get(f"{split_name}/{metric}")
            if value is None:
                continue
            std_html = ""
            value_str = f"{value:.3f}"
        else:
            value = run_data.summary.get(f"{split_name}/{metric}_mean")
            std = run_data.summary.get(f"{split_name}/{metric}_std")
            if value is None:
                continue
            std_html = f'<span class="hero-std">± {std:.3f}</span>' if std is not None else ""
            value_str = f"{value:.3f}"
        split_label = f'<span class="hero-split">{split_name}</span>' if len(all_split_names) > 1 else ""
        cards.append(f"""
  <div class="hero-card">
    <div class="hero-value">{value_str}{std_html}</div>
    <div class="hero-label">{metric}{split_label}</div>
  </div>""")

    if not cards:
        return ""

    return f'<div class="hero-strip"><div class="hero-grid">{"".join(cards)}</div></div>'


def _section_run_context(run_data: RunData) -> str:
    cfg = run_data.config

    def _basename(path: str) -> str:
        if not path or path == "—":
            return "—"
        import os
        return os.path.basename(path)

    def _get(obj: dict, *keys: str, default: str = "—") -> str:
        for key in keys:
            if isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return str(obj) if obj is not None else default

    items = []

    dataset = _basename(_get(cfg, "dataset_csv"))
    if dataset and dataset != "—":
        items.append(("Dataset", dataset))

    splits = _basename(_get(cfg, "splits_csv"))
    if splits and splits != "—":
        items.append(("Splits", splits))

    encoder = cfg.get("encoder")
    if encoder:
        enc_name = _get(cfg, "encoder", "name")
        if enc_name and enc_name != "—":
            items.append(("Encoder", enc_name))

    agg = cfg.get("aggregator")
    if agg:
        agg_name = _get(cfg, "aggregator", "name")
        if agg_name and agg_name != "—":
            items.append(("Aggregator", agg_name))

    if not items:
        return ""

    chips = "".join(
        f'<span class="ctx-chip"><span class="ctx-label">{label}</span><span class="ctx-value">{value}</span></span>'
        for label, value in items
    )
    return f'<div class="run-context">{chips}</div>'


def _section_config(run_data: RunData) -> str:
    cfg = run_data.config
    rows = _config_rows(cfg)
    table_html = "".join(
        f'<tr><td class="cfg-key">{k}</td><td class="cfg-val">{v}</td></tr>'
        for k, v in rows
    )
    return f"""
<div class="section">
  <details class="cfg-details">
    <summary>Configuration</summary>
    <table class="cfg-table"><tbody>{table_html}</tbody></table>
  </details>
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
  <h2>Training Timing</h2>
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
        sections += f"""
<div class="section">
  <h2>{heading}</h2>
  {_tab_group(tabs)}
</div>"""

    return sections


def _tab_group(tabs: list[tuple[str, str]]) -> str:
    import uuid as _uuid
    group_id = f"tg-{_uuid.uuid4().hex[:8]}"

    btn_html = ""
    panels_html = ""
    for i, (label, svg) in enumerate(tabs):
        tab_id = f"{group_id}-{i}"
        active_btn = " active" if i == 0 else ""
        hidden = "" if i == 0 else ' style="display:none"'
        btn_html += f'<button class="tab-btn{active_btn}" onclick="somaTab(\'{group_id}\',{i})">{label}</button>'
        panels_html += f'<div class="tab-panel chart-square" id="{tab_id}"{hidden}>{svg}</div>'

    return f"""<div class="tab-group" id="{group_id}">
  <div class="tab-bar">{btn_html}</div>
  {panels_html}
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


def render_comparison_report(comparison_data: ComparisonData) -> str:
    """Generate a self-contained HTML comparison report string."""
    sections: list[str] = []
    sections.append(_comparison_section_header(comparison_data))
    sections.append(_comparison_section_hero_metrics(comparison_data))
    sections.append(_comparison_section_run_context(comparison_data))
    sections.append(_comparison_section_config_diff(comparison_data))
    sections.append(_comparison_section_metrics(comparison_data))
    sections.append(_comparison_section_curves(comparison_data))

    body = "\n".join(s for s in sections if s)
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
    g.querySelectorAll('.tab-panel').forEach(function(p) {{ p.style.display = 'none'; }});
    g.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    document.getElementById(groupId + '-' + idx).style.display = '';
    g.querySelectorAll('.tab-btn')[idx].classList.add('active');
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

    def _primary_split(run: RunData) -> str:
        splits = sorted({s for fd in run.folds for s in fd.test_metrics})
        return splits[0] if splits else "test"

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
    import os

    def _basename(v: str) -> str:
        return os.path.basename(v) if v else "—"

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


def _comparison_section_config_diff(cd: ComparisonData) -> str:
    if not any(cd.config_diffs):
        return """
<div class="section">
  <details class="cfg-details">
    <summary>Configuration</summary>
    <p class="muted" style="margin-top:12px">All runs share identical configurations.</p>
  </details>
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

    shared_rows = "".join(
        f"<tr><td class='cfg-key'>{k}</td><td class='cfg-val'><code>{v}</code></td></tr>"
        for k, v in sorted(cd.shared_config.items())
    )
    shared_html = f"""
<details class="cfg-details">
  <summary>Shared configuration ({len(cd.shared_config)} fields)</summary>
  <table class="cfg-table" style="margin-top:10px"><tbody>{shared_rows}</tbody></table>
</details>""" if cd.shared_config else ""

    return f"""
<div class="section">
  <details class="cfg-details">
    <summary>Configuration</summary>
    <table class="results-table" style="margin-top:12px">
      <thead>{header_row}</thead>
      <tbody>{rows_html}</tbody>
    </table>
    {shared_html}
  </details>
</div>"""


def _comparison_section_metrics(cd: ComparisonData) -> str:
    if not cd.metric_names:
        return ""

    col_headers = "".join(
        f'<th><span class="run-badge" style="background:{_badge_color(i)}">{label}</span></th>'
        for i, label in enumerate(cd.labels)
    )
    header_row = f"<tr><th>Metric</th>{col_headers}</tr>"

    def _primary_split(run: RunData) -> str:
        splits = sorted({s for fd in run.folds for s in fd.test_metrics})
        return splits[0] if splits else "test"

    task_family = cd.runs[0].task_family if cd.runs else ""

    # Pooled predictions per run (used for both metric means and statistical test)
    runs_pooled = [
        aggregate_fold_predictions(run.folds, _primary_split(run))
        for run in cd.runs
    ]

    per_metric_data: list[tuple[list[float | None], list[float | None]]] = []
    for metric in cd.metric_names:
        means: list[float | None] = []
        for run in cd.runs:
            split_name = _primary_split(run)
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
            split_name = _primary_split(cd.runs[i])
            std = cd.runs[i].summary.get(f"{split_name}/{metric}_std")
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
  --soma-bg-subtle:    #F8FAFC;
  --soma-bg-card:      #F1F5F9;
  --soma-text:         #0F172A;
  --soma-text-muted:   #64748B;
  --soma-accent:       #7C3AED;
  --soma-accent-light: #EDE9FE;
  --soma-border:       #E2E8F0;
  --soma-header-bg:    #0F172A;
  --soma-header-text:  #F8FAFC;
  --soma-success:      #10B981;
  --soma-danger:       #EF4444;
  --soma-warning:      #F59E0B;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
  font-size: 14px;
  background: var(--soma-bg-subtle);
  color: var(--soma-text);
  line-height: 1.5;
}
/* ---- Header ---- */
.site-header {
  background: var(--soma-header-bg);
  color: var(--soma-header-text);
  padding: 0 32px;
  height: 52px;
  display: flex;
  align-items: center;
  gap: 16px;
  position: sticky;
  top: 0;
  z-index: 100;
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
.hero-strip {
  background: var(--soma-bg);
  border-bottom: 1px solid var(--soma-border);
  padding: 20px 32px;
}
.hero-grid {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
}
.hero-card {
  background: var(--soma-accent-light);
  border-left: 4px solid var(--soma-accent);
  border-radius: 8px;
  padding: 14px 20px;
  min-width: 140px;
}
.hero-value {
  font-size: 1.85rem;
  font-weight: 700;
  color: var(--soma-accent);
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
.hero-split {
  font-weight: 400;
  text-transform: none;
  margin-left: 4px;
}
/* ---- Run context strip ---- */
.run-context {
  background: var(--soma-bg);
  border-bottom: 1px solid var(--soma-border);
  padding: 10px 32px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.ctx-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--soma-bg-subtle);
  border: 1px solid var(--soma-border);
  border-radius: 20px;
  padding: 3px 10px 3px 8px;
  font-size: 12px;
}
.ctx-label {
  color: var(--soma-text-muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  font-size: 10px;
}
.ctx-value {
  color: var(--soma-text);
  font-family: ui-monospace, 'JetBrains Mono', monospace;
  font-size: 11px;
}
/* ---- Badges ---- */
.badge {
  padding: 2px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;
}
.status-ok  { background: #10B981; color: #fff; }
.status-err { background: #EF4444; color: #fff; }
.status-run { background: #F59E0B; color: #000; }
.run-badge {
  display: inline-block;
  padding: 2px 10px; border-radius: 12px;
  font-size: 11px; font-weight: 600; color: #fff;
}
/* ---- Sections ---- */
.section {
  background: var(--soma-bg);
  border-radius: 8px;
  margin: 20px 32px;
  padding: 24px 28px;
  box-shadow: 0 1px 3px rgba(0,0,0,.05);
  border: 1px solid var(--soma-border);
}
.section h2 {
  font-size: 15px; font-weight: 600; margin-bottom: 16px;
  border-bottom: 1px solid var(--soma-border); padding-bottom: 10px;
  color: var(--soma-text);
}
.section h3 {
  font-size: 13px; font-weight: 600; margin: 16px 0 8px;
  color: var(--soma-text-muted);
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
.cfg-table { border-collapse: collapse; width: 100%; max-width: 680px; margin-top: 4px; }
.cfg-table tr:nth-child(even) td { background: var(--soma-bg-subtle); }
.cfg-key {
  padding: 6px 12px 6px 0; font-weight: 600; color: var(--soma-text-muted);
  white-space: nowrap; width: 200px; vertical-align: top; font-size: 12px;
}
.cfg-val { padding: 6px 0; color: var(--soma-text); word-break: break-all; font-size: 12px; }
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
.best-val { background: #D1FAE5; }
.sig-worse { background: #FEE2E2; }
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
  border-radius: 8px;
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
.tab-group { margin-top: 4px; }
.tab-bar {
  display: flex;
  gap: 4px;
  border-bottom: 2px solid var(--soma-border);
  margin-bottom: 16px;
}
.tab-btn {
  padding: 7px 16px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 13px;
  font-weight: 500;
  color: var(--soma-text-muted);
  border-bottom: 2px solid transparent;
  margin-bottom: -2px;
  border-radius: 4px 4px 0 0;
  transition: color 0.1s;
}
.tab-btn:hover { color: var(--soma-text); background: var(--soma-bg-subtle); }
.tab-btn.active { color: var(--soma-accent); border-bottom-color: var(--soma-accent); font-weight: 600; }
.chart-square { max-width: 560px; }
.chart-square svg { width: 100%; height: auto; display: block; }
code { font-family: ui-monospace, 'JetBrains Mono', monospace; font-size: 12px; }
@media (max-width: 768px) {
  .section { margin: 12px; padding: 16px; }
  .chart-half { flex: 1 1 100%; max-width: 100%; aspect-ratio: unset; }
  .site-header { padding: 0 16px; }
  .hero-strip, .run-context { padding: 12px 16px; }
}
"""
