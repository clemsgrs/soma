"""Plotly chart builders for experiment reports.

All functions are pure: they take FoldData lists and return plotly Figures.
No I/O, no side effects.

Per-fold vs. aggregated strategy:
- Training curves: one trace per fold overlaid on the same chart.
- Prediction analysis: predictions aggregated across all folds, with per-fold
  curves overlaid on ROC/PR plots to show variance.
- Confusion matrix: summed across all folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from soma.reporting.data import FoldData, FoldSlice, RunData, aggregate_slice_predictions

# Consistent color palette for folds
_FOLD_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA",
    "#FFA15A", "#19D3F3", "#FF6692", "#B6E880",
]


def _fold_color(fold: int) -> str:
    return _FOLD_COLORS[fold % len(_FOLD_COLORS)]


# ---------------------------------------------------------------------------
# Training curves (all tasks)
# ---------------------------------------------------------------------------


def loss_curves(folds: list[FoldData]) -> go.Figure:
    """Train and tune loss per epoch, one trace per fold."""
    fig = go.Figure()
    for fd in folds:
        if not fd.training_history:
            continue
        epochs = [e["epoch"] for e in fd.training_history]
        train_loss = [e["train_loss"] for e in fd.training_history]
        tune_loss = [e["tune_loss"] for e in fd.training_history]
        color = _fold_color(fd.fold)
        label = f"Fold {fd.fold}"
        fig.add_trace(go.Scatter(
            x=epochs, y=train_loss,
            mode="lines", name=f"{label} train",
            line=dict(color=color, dash="solid"),
            legendgroup=label,
        ))
        fig.add_trace(go.Scatter(
            x=epochs, y=tune_loss,
            mode="lines", name=f"{label} tune",
            line=dict(color=color, dash="dot"),
            legendgroup=label,
        ))
    fig.update_layout(
        title="Loss curves",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        legend=dict(groupclick="toggleitem"),
        **_chart_layout(),
    )
    return fig


def metric_curves(folds: list[FoldData], metric_name: str) -> go.Figure:
    """Tune metric per epoch for a given metric, one trace per fold."""
    fig = go.Figure()
    for fd in folds:
        if not fd.training_history:
            continue
        epochs = [e["epoch"] for e in fd.training_history]
        values = [e["tune_metrics"].get(metric_name) for e in fd.training_history]
        if all(v is None for v in values):
            continue
        fig.add_trace(go.Scatter(
            x=epochs, y=values,
            mode="lines", name=f"Fold {fd.fold}",
            line=dict(color=_fold_color(fd.fold)),
        ))
    fig.update_layout(
        title=f"Tune {metric_name} per epoch",
        xaxis_title="Epoch",
        yaxis_title=metric_name,
        **_chart_layout(),
    )
    return fig


def lr_curve(folds: list[FoldData]) -> go.Figure:
    """Learning rate schedule per epoch, one trace per fold."""
    fig = go.Figure()
    for fd in folds:
        if not fd.training_history:
            continue
        epochs = [e["epoch"] for e in fd.training_history]
        lrs = [e["lr"] for e in fd.training_history]
        fig.add_trace(go.Scatter(
            x=epochs, y=lrs,
            mode="lines", name=f"Fold {fd.fold}",
            line=dict(color=_fold_color(fd.fold)),
        ))
    fig.update_layout(
        title="Learning rate",
        xaxis_title="Epoch",
        yaxis_title="LR",
        yaxis_type="log",
        **_chart_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Classification charts (binary + multiclass)
# ---------------------------------------------------------------------------


def roc_curve_chart(folds: list[FoldData]) -> go.Figure:
    """Per-fold ROC curves + pooled ROC curve from aggregated predictions.

    Works for binary (prob_1 column) and multiclass (one-vs-rest on each class,
    macro-averaged to get one curve per fold).
    """
    fig = go.Figure()

    all_preds = _aggregate_predictions(folds)
    prob_cols = [c for c in all_preds.columns if c.startswith("prob_")]

    if not prob_cols:
        return _empty_figure("ROC curve (no probability columns found)")

    is_binary = len(prob_cols) == 2

    # Per-fold curves (thinner, semi-transparent)
    for fd in folds:
        if fd.predictions.empty:
            continue
        color = _fold_color(fd.fold)
        if is_binary:
            fpr, tpr, _ = roc_curve(fd.predictions["true_label"], fd.predictions["prob_1"])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"Fold {fd.fold} (AUC={roc_auc:.3f})",
                line=dict(color=color, width=1.5, dash="dot"),
                opacity=0.7,
                legendgroup=f"fold_{fd.fold}",
            ))
        else:
            # One-vs-rest per class, averaged (macro)
            fprs, tprs, aucs = [], [], []
            y_true = fd.predictions["true_label"].values
            for i, col in enumerate(prob_cols):
                y_score = fd.predictions[col].values
                y_bin = (y_true == i).astype(int)
                if y_bin.sum() == 0:
                    continue
                fpr_i, tpr_i, _ = roc_curve(y_bin, y_score)
                fprs.append(fpr_i)
                tprs.append(tpr_i)
                aucs.append(auc(fpr_i, tpr_i))
            if aucs:
                # Interpolate to common FPR grid for averaging
                mean_fpr = np.linspace(0, 1, 200)
                mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fprs, tprs)], axis=0)
                mean_auc = float(np.mean(aucs))
                fig.add_trace(go.Scatter(
                    x=mean_fpr, y=mean_tpr,
                    mode="lines",
                    name=f"Fold {fd.fold} macro (AUC={mean_auc:.3f})",
                    line=dict(color=color, width=1.5, dash="dot"),
                    opacity=0.7,
                    legendgroup=f"fold_{fd.fold}",
                ))

    # Pooled curve (bold)
    if not all_preds.empty:
        if is_binary:
            fpr, tpr, _ = roc_curve(all_preds["true_label"], all_preds["prob_1"])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"Pooled (AUC={roc_auc:.3f})",
                line=dict(color="black", width=2.5),
            ))
        else:
            fprs, tprs, aucs = [], [], []
            y_true = all_preds["true_label"].values
            for i, col in enumerate(prob_cols):
                y_score = all_preds[col].values
                y_bin = (y_true == i).astype(int)
                if y_bin.sum() == 0:
                    continue
                fpr_i, tpr_i, _ = roc_curve(y_bin, y_score)
                fprs.append(fpr_i)
                tprs.append(tpr_i)
                aucs.append(auc(fpr_i, tpr_i))
            if aucs:
                mean_fpr = np.linspace(0, 1, 200)
                mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fprs, tprs)], axis=0)
                mean_auc = float(np.mean(aucs))
                fig.add_trace(go.Scatter(
                    x=mean_fpr, y=mean_tpr,
                    mode="lines",
                    name=f"Pooled macro (AUC={mean_auc:.3f})",
                    line=dict(color="black", width=2.5),
                ))

    # Chance diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines", name="Chance",
        line=dict(color="gray", dash="dash", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        title="ROC curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        **_chart_layout(),
    )
    return fig


def pr_curve_chart(folds: list[FoldData]) -> go.Figure:
    """Per-fold precision-recall curves + pooled curve (binary classification)."""
    fig = go.Figure()
    all_preds = _aggregate_predictions(folds)

    if all_preds.empty or "prob_1" not in all_preds.columns:
        return _empty_figure("PR curve (binary only)")

    # Per-fold
    for fd in folds:
        if fd.predictions.empty or "prob_1" not in fd.predictions.columns:
            continue
        precision, recall, _ = precision_recall_curve(
            fd.predictions["true_label"], fd.predictions["prob_1"]
        )
        ap = average_precision_score(fd.predictions["true_label"], fd.predictions["prob_1"])
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode="lines",
            name=f"Fold {fd.fold} (AP={ap:.3f})",
            line=dict(color=_fold_color(fd.fold), width=1.5, dash="dot"),
            opacity=0.7,
        ))

    # Pooled
    precision, recall, _ = precision_recall_curve(all_preds["true_label"], all_preds["prob_1"])
    ap = average_precision_score(all_preds["true_label"], all_preds["prob_1"])
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode="lines",
        name=f"Pooled (AP={ap:.3f})",
        line=dict(color="black", width=2.5),
    ))

    # Baseline
    pos_rate = float(all_preds["true_label"].mean())
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[pos_rate, pos_rate],
        mode="lines", name="Baseline",
        line=dict(color="gray", dash="dash", width=1),
        showlegend=False,
    ))
    fig.update_layout(
        title="Precision-Recall curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        **_chart_layout(),
    )
    return fig


def confusion_matrix_chart(folds: list[FoldData]) -> go.Figure:
    """Aggregated confusion matrix summed across all folds."""
    all_preds = _aggregate_predictions(folds)
    if all_preds.empty or "predicted_label" not in all_preds.columns:
        return _empty_figure("Confusion matrix")

    y_true = all_preds["true_label"].astype(int).values
    y_pred = all_preds["predicted_label"].astype(int).values
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    class_names = [f"Class {i}" for i in labels]
    # Normalize for color (but show raw counts as text)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=class_names,
        y=class_names,
        text=cm,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=True,
        colorbar=dict(title="Recall"),
    ))
    fig.update_layout(
        title="Confusion matrix (pooled, row-normalized)",
        xaxis_title="Predicted",
        yaxis_title="True",
        yaxis=dict(autorange="reversed"),
        **_chart_layout(),
    )
    return fig


def score_distribution_chart(folds: list[FoldData]) -> go.Figure:
    """Histogram of predicted probabilities per class, aggregated across folds."""
    all_preds = _aggregate_predictions(folds)
    if all_preds.empty:
        return _empty_figure("Score distribution")

    prob_cols = [c for c in all_preds.columns if c.startswith("prob_")]
    if not prob_cols:
        return _empty_figure("Score distribution (no probability columns)")

    fig = go.Figure()
    is_binary = len(prob_cols) == 2
    col = "prob_1" if is_binary else prob_cols[0]

    classes = sorted(all_preds["true_label"].unique())
    for cls in classes:
        subset = all_preds[all_preds["true_label"] == cls]
        fig.add_trace(go.Histogram(
            x=subset[col],
            name=f"Class {int(cls)}",
            opacity=0.7,
            nbinsx=40,
        ))

    score_label = "P(positive)" if is_binary else f"P({prob_cols[0].replace('prob_', 'class ')})"
    fig.update_layout(
        title=f"Score distribution by class ({score_label})",
        xaxis_title=score_label,
        yaxis_title="Count",
        barmode="overlay",
        **_chart_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Regression charts
# ---------------------------------------------------------------------------


def scatter_predicted_vs_actual(folds: list[FoldData]) -> go.Figure:
    """Scatter of predicted vs actual values, color-coded by fold."""
    fig = go.Figure()
    all_vals: list[float] = []

    for fd in folds:
        if fd.predictions.empty or "predicted_value" not in fd.predictions.columns:
            continue
        y_true = fd.predictions["true_label"].values
        y_pred = fd.predictions["predicted_value"].values
        all_vals.extend(list(y_true) + list(y_pred))
        fig.add_trace(go.Scatter(
            x=y_true, y=y_pred,
            mode="markers",
            name=f"Fold {fd.fold}",
            marker=dict(color=_fold_color(fd.fold), size=6, opacity=0.7),
        ))

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines", name="Identity",
            line=dict(color="gray", dash="dash", width=1),
            showlegend=False,
        ))

    fig.update_layout(
        title="Predicted vs. actual",
        xaxis_title="Actual",
        yaxis_title="Predicted",
        **_chart_layout(),
    )
    return fig


def residual_plot(folds: list[FoldData]) -> go.Figure:
    """Residuals (actual − predicted) vs. predicted, color-coded by fold."""
    fig = go.Figure()

    for fd in folds:
        if fd.predictions.empty or "predicted_value" not in fd.predictions.columns:
            continue
        y_true = fd.predictions["true_label"].values
        y_pred = fd.predictions["predicted_value"].values
        residuals = y_true - y_pred
        fig.add_trace(go.Scatter(
            x=y_pred, y=residuals,
            mode="markers",
            name=f"Fold {fd.fold}",
            marker=dict(color=_fold_color(fd.fold), size=6, opacity=0.7),
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    fig.update_layout(
        title="Residuals vs. predicted",
        xaxis_title="Predicted",
        yaxis_title="Residual (actual − predicted)",
        **_chart_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_aggregate_predictions = aggregate_slice_predictions


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        annotations=[dict(text="No data available", showarrow=False, font_size=14)],
        **_chart_layout(),
    )
    return fig


# ---------------------------------------------------------------------------
# Cross-run comparison charts
# ---------------------------------------------------------------------------


def comparison_loss_curves(runs: list[RunData], labels: list[str]) -> go.Figure:
    """Train/tune loss curves overlaid across runs.

    Each run contributes one mean line (averaged across folds). When a run
    has multiple folds a ±1 std shaded band is drawn around the mean.
    """
    fig = go.Figure()
    for run, label, color in zip(runs, labels, _run_colors(len(runs))):
        _add_comparison_curve(fig, run, label, color, key="train_loss", dash="solid")
        _add_comparison_curve(fig, run, label, color, key="tune_loss", dash="dot", show_label=False)
    fig.update_layout(
        title="Loss curves (cross-run)",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        **_chart_layout(),
    )
    return fig


def comparison_metric_curves(
    runs: list[RunData],
    labels: list[str],
    metric_name: str,
) -> go.Figure:
    """Tune metric curves overlaid across runs.

    Each run contributes one mean line (averaged across folds). When a run
    has multiple folds a ±1 std shaded band is drawn around the mean.
    """
    fig = go.Figure()
    for run, label, color in zip(runs, labels, _run_colors(len(runs))):
        _add_comparison_curve(fig, run, label, color, metric_name=metric_name)
    fig.update_layout(
        title=f"Tune {metric_name} per epoch (cross-run)",
        xaxis_title="Epoch",
        yaxis_title=metric_name,
        **_chart_layout(),
    )
    return fig


def _add_comparison_curve(
    fig: go.Figure,
    run: RunData,
    label: str,
    color: str,
    *,
    key: str | None = None,
    metric_name: str | None = None,
    dash: str = "solid",
    show_label: bool = True,
) -> None:
    """Add one mean curve (+ optional ±1std band) for a single run."""
    folds_with_history = [fd for fd in run.folds if fd.training_history]
    if not folds_with_history:
        return

    n_epochs = max(len(fd.training_history) for fd in folds_with_history)
    epochs = list(range(n_epochs))

    per_fold_values = []
    for fd in folds_with_history:
        if key is not None:
            vals = [e[key] for e in fd.training_history]
        else:
            vals = [e["tune_metrics"].get(metric_name) for e in fd.training_history]
            if all(v is None for v in vals):
                continue
            vals = [v if v is not None else float("nan") for v in vals]
        # Pad to n_epochs if this fold stopped early
        if len(vals) < n_epochs:
            vals = vals + [float("nan")] * (n_epochs - len(vals))
        per_fold_values.append(vals)

    if not per_fold_values:
        return

    arr = np.array(per_fold_values, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0) if arr.shape[0] > 1 else None

    trace_label = f"{label} {key or metric_name}" if not show_label else label
    legend_label = f"{label} (train)" if key == "train_loss" else (
        f"{label} (tune)" if key == "tune_loss" else label
    )

    if std is not None:
        upper = mean + std
        lower = mean - std
        # Upper bound (invisible line, fills down to lower)
        fig.add_trace(go.Scatter(
            x=epochs, y=upper.tolist(),
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
            legendgroup=label,
        ))
        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=epochs, y=lower.tolist(),
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=_hex_to_rgba(color, alpha=0.15),
            showlegend=False, hoverinfo="skip",
            legendgroup=label,
        ))

    fig.add_trace(go.Scatter(
        x=epochs, y=mean.tolist(),
        mode="lines",
        name=legend_label,
        line=dict(color=color, width=2, dash=dash),
        legendgroup=label,
    ))


def _run_colors(n: int) -> list[str]:
    """Return n distinct colors for runs (different palette from fold colors)."""
    palette = [
        "#E63946", "#2A9D8F", "#E9C46A", "#264653",
        "#F4A261", "#A8DADC", "#457B9D", "#1D3557",
    ]
    return [palette[i % len(palette)] for i in range(n)]


def _hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    """Convert a hex color string to an rgba() CSS string."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def subgroup_metric_chart(
    subgroup_data: dict[str, dict[str, float | int]],
    metric_name: str,
    overall_value: float,
    column_name: str = "",
) -> go.Figure:
    """Bar chart comparing metric values across subgroups.

    Args:
        subgroup_data: {group_value → {metric: value, "n": count}}
        metric_name: Which metric to plot.
        overall_value: Overall (non-subgroup) metric value — shown as a reference line.
        column_name: Column name for the chart title.

    Returns:
        Plotly Figure with one bar per group.
    """
    groups = [g for g, d in subgroup_data.items() if metric_name in d]
    values = [subgroup_data[g][metric_name] for g in groups]
    ns = [subgroup_data[g].get("n", "") for g in groups]

    colors = [
        "#2A9D8F" if v >= overall_value else "#E63946"
        for v in values
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=groups,
        y=values,
        marker_color=colors,
        text=[f"n={n}" for n in ns],
        textposition="outside",
        name=metric_name,
    ))
    fig.add_hline(
        y=overall_value,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"overall={overall_value:.3f}",
        annotation_position="top right",
    )
    title = f"{metric_name} by {column_name}" if column_name else metric_name
    fig.update_layout(
        **_chart_layout(),
        title=title,
        xaxis_title=column_name or "Group",
        yaxis_title=metric_name,
        showlegend=False,
    )
    return fig


def subgroup_stats_heatmap(
    stats: dict[str, dict[str, dict[str, float]]],
    columns: list[str],
    metrics: list[str],
) -> go.Figure:
    """Heatmap of permutation-test p-values for subgroup analysis.

    Rows = column×group combinations, columns = metrics.
    Color scale: green (low p-value / significant) to white (high p-value).

    Args:
        stats: {column → {group → {metric: p_value}}}
        columns: Column names (controls row ordering).
        metrics: Metric names (controls column ordering).

    Returns:
        Plotly Figure heatmap.
    """
    row_labels: list[str] = []
    z: list[list[float]] = []

    for col in columns:
        if col not in stats:
            continue
        for group, metric_pvals in sorted(stats[col].items()):
            row_labels.append(f"{col}={group}")
            z.append([metric_pvals.get(m, float("nan")) for m in metrics])

    if not row_labels:
        fig = go.Figure()
        fig.update_layout(**_chart_layout(), title="No statistical test results")
        return fig

    fig = go.Figure(go.Heatmap(
        z=z,
        x=metrics,
        y=row_labels,
        colorscale=[[0, "#2A9D8F"], [0.05, "#E9C46A"], [1, "#f5f5f5"]],
        zmin=0,
        zmax=1,
        colorbar=dict(title="p-value"),
        text=[[f"{v:.3f}" if not np.isnan(v) else "—" for v in row] for row in z],
        texttemplate="%{text}",
    ))
    fig.add_shape(
        type="line", x0=-0.5, x1=len(metrics) - 0.5,
        y0=-0.5, y1=len(row_labels) - 0.5,
        line=dict(color="rgba(0,0,0,0)"),
    )
    fig.update_layout(
        **_chart_layout(),
        title="Permutation test p-values (p < 0.05 = significant)",
        height=max(400, 40 * len(row_labels)),
    )
    return fig


def _chart_layout() -> dict:
    return dict(
        template="plotly_white",
        height=400,
        margin=dict(l=50, r=30, t=50, b=50),
        font=dict(family="Inter, system-ui, sans-serif", size=13),
    )
