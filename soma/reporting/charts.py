"""Matplotlib chart builders for experiment reports.

All functions are pure: they take FoldData lists and return SVG strings.
No I/O, no side effects.
"""

from __future__ import annotations

import io
import re
import uuid

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

matplotlib.use("Agg")

from soma.reporting.data import FoldData, FoldSlice, RunData, aggregate_slice_predictions

SOMA_PALETTE = [
    "#E2558A",  # soma pink
    "#B52B65",  # deep rose
    "#F472B6",  # light pink
    "#FB7185",  # coral rose
    "#EC4899",  # magenta pink
    "#F9A8D4",  # blush
    "#DB2777",  # ruby rose
    "#C084FC",  # soft lavender accent
]

plt.rcParams.update({
    "figure.facecolor": "#FFF7FA",
    "axes.facecolor": "#FFFDFE",
    "axes.edgecolor": "#E9D5DF",
    "grid.color": "#F8E2EA",
    "grid.linewidth": 0.8,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


def _fold_color(fold: int) -> str:
    return SOMA_PALETTE[fold % len(SOMA_PALETTE)]


def _fig_to_svg(fig: plt.Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.index("<svg"):]
    prefix = f"s{uuid.uuid4().hex[:8]}"
    svg = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}-{m.group(1)}"', svg)
    svg = re.sub(r'(xlink:href|href)="#([^"]+)"', lambda m: f'{m.group(1)}="#{prefix}-{m.group(2)}"', svg)
    svg = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}-{m.group(1)})', svg)
    return svg


def _apply_soma_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E9D5DF")
    ax.spines["bottom"].set_color("#E9D5DF")
    ax.grid(True, color="#F8E2EA", linewidth=0.8, zorder=0)


def _legend_if_labeled(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()


# ---------------------------------------------------------------------------
# Training curves (all tasks)
# ---------------------------------------------------------------------------


def loss_curves(folds: list[FoldData]) -> str:
    fig, (ax_train, ax_tune) = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for fd in folds:
        if not fd.training_history:
            continue
        epochs = [e["epoch"] for e in fd.training_history]
        train_loss = [e["train_loss"] for e in fd.training_history]
        tune_loss = [e["tune_loss"] for e in fd.training_history]
        color = _fold_color(fd.fold)
        ax_train.plot(epochs, train_loss, color=color, linestyle="-", label=f"Fold {fd.fold}")
        ax_tune.plot(epochs, tune_loss, color=color, linestyle="-", label=f"Fold {fd.fold}")
    ax_train.set_title("Train loss")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_tune.set_title("Tune loss")
    ax_tune.set_xlabel("Epoch")
    _apply_soma_style(ax_train)
    _apply_soma_style(ax_tune)
    _legend_if_labeled(ax_train)
    _legend_if_labeled(ax_tune)
    fig.tight_layout()
    return _fig_to_svg(fig)


def metric_curves(folds: list[FoldData], metric_name: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for fd in folds:
        if not fd.training_history:
            continue
        epochs = [e["epoch"] for e in fd.training_history]
        values = [e["tune_metrics"].get(metric_name) for e in fd.training_history]
        if all(v is None for v in values):
            continue
        ax.plot(epochs, values, color=_fold_color(fd.fold), label=f"Fold {fd.fold}")
    ax.set_title(f"Tune {metric_name} per epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    _apply_soma_style(ax)
    _legend_if_labeled(ax)
    fig.tight_layout()
    return _fig_to_svg(fig)


def lr_curve(folds: list[FoldData]) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for fd in folds:
        if not fd.training_history:
            continue
        epochs = [e["epoch"] for e in fd.training_history]
        lrs = [e["lr"] for e in fd.training_history]
        ax.plot(epochs, lrs, color=_fold_color(fd.fold), label=f"Fold {fd.fold}")
    ax.set_title("Learning rate")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    _apply_soma_style(ax)
    _legend_if_labeled(ax)
    fig.tight_layout()
    return _fig_to_svg(fig)


# ---------------------------------------------------------------------------
# Classification charts (binary + multiclass)
# ---------------------------------------------------------------------------


def roc_curve_chart(folds: list[FoldData]) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    all_preds = _aggregate_predictions(folds)
    prob_cols = [c for c in all_preds.columns if c.startswith("prob_")]

    if not prob_cols:
        ax.text(0.5, 0.5, "No probability columns found", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("ROC curve")
        return _fig_to_svg(fig)

    is_binary = len(prob_cols) == 2

    for fd in folds:
        if fd.predictions.empty:
            continue
        color = _fold_color(fd.fold)
        if is_binary:
            fpr, tpr, _ = roc_curve(fd.predictions["true_label"], fd.predictions["prob_1"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=1.2, linestyle="--", alpha=0.6,
                    label=f"Fold {fd.fold} (AUC={roc_auc:.3f})")
        else:
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
                mean_fpr = np.linspace(0, 1, 200)
                mean_tpr = np.mean([np.interp(mean_fpr, f, t) for f, t in zip(fprs, tprs)], axis=0)
                mean_auc = float(np.mean(aucs))
                ax.plot(mean_fpr, mean_tpr, color=color, linewidth=1.2, linestyle="--", alpha=0.6,
                        label=f"Fold {fd.fold} macro (AUC={mean_auc:.3f})")

    if not all_preds.empty:
        if is_binary:
            fpr, tpr, _ = roc_curve(all_preds["true_label"], all_preds["prob_1"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=SOMA_PALETTE[0], linewidth=2.5, label=f"Pooled (AUC={roc_auc:.3f})")
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
                ax.plot(mean_fpr, mean_tpr, color=SOMA_PALETTE[0], linewidth=2.5,
                        label=f"Pooled macro (AUC={mean_auc:.3f})")

    ax.plot([0, 1], [0, 1], color="#C08497", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("ROC curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    _apply_soma_style(ax)
    _legend_if_labeled(ax)
    fig.tight_layout()
    return _fig_to_svg(fig)


def pr_curve_chart(folds: list[FoldData]) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    all_preds = _aggregate_predictions(folds)

    if all_preds.empty or "prob_1" not in all_preds.columns:
        ax.text(0.5, 0.5, "PR curve (binary only)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Precision-Recall curve")
        return _fig_to_svg(fig)

    for fd in folds:
        if fd.predictions.empty or "prob_1" not in fd.predictions.columns:
            continue
        precision, recall, _ = precision_recall_curve(
            fd.predictions["true_label"], fd.predictions["prob_1"]
        )
        ap = average_precision_score(fd.predictions["true_label"], fd.predictions["prob_1"])
        ax.plot(recall, precision, color=_fold_color(fd.fold), linewidth=1.2, linestyle="--", alpha=0.6,
                label=f"Fold {fd.fold} (AP={ap:.3f})")

    precision, recall, _ = precision_recall_curve(all_preds["true_label"], all_preds["prob_1"])
    ap = average_precision_score(all_preds["true_label"], all_preds["prob_1"])
    ax.plot(recall, precision, color=SOMA_PALETTE[0], linewidth=2.5, label=f"Pooled (AP={ap:.3f})")

    pos_rate = float(all_preds["true_label"].mean())
    ax.axhline(pos_rate, color="#C08497", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision-Recall curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    _apply_soma_style(ax)
    _legend_if_labeled(ax)
    fig.tight_layout()
    return _fig_to_svg(fig)


def confusion_matrix_chart(folds: list[FoldData]) -> str:
    all_preds = _aggregate_predictions(folds)
    if all_preds.empty or "predicted_label" not in all_preds.columns:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Confusion matrix")
        return _fig_to_svg(fig)

    y_true = all_preds["true_label"].astype(int).values
    y_pred = all_preds["predicted_label"].astype(int).values
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    soma_cmap = LinearSegmentedColormap.from_list("soma", ["#FFF7FA", SOMA_PALETTE[0]])

    n = len(labels)
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm_norm, cmap=soma_cmap, vmin=0, vmax=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    fig.colorbar(im, cax=cax, label="Recall")

    class_names = [f"Class {i}" for i in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > 0.5 else "#0F172A"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=10)

    n_folds = len([f for f in folds if not getattr(f, "predictions", pd.DataFrame()).empty])
    fold_note = f"all {n_folds} folds, " if n_folds > 1 else ""
    ax.set_title(f"Confusion matrix ({fold_note}row-normalized)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return _fig_to_svg(fig)



# ---------------------------------------------------------------------------
# Regression charts
# ---------------------------------------------------------------------------


def scatter_predicted_vs_actual(folds: list[FoldData]) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    all_vals: list[float] = []

    for fd in folds:
        if fd.predictions.empty or "predicted_value" not in fd.predictions.columns:
            continue
        y_true = fd.predictions["true_label"].values
        y_pred = fd.predictions["predicted_value"].values
        all_vals.extend(list(y_true) + list(y_pred))
        ax.scatter(y_true, y_pred, color=_fold_color(fd.fold), s=30, alpha=0.7, label=f"Fold {fd.fold}")

    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        ax.plot([lo, hi], [lo, hi], color="#C08497", linestyle="--", linewidth=1)

    ax.set_title("Predicted vs. actual")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    _apply_soma_style(ax)
    _legend_if_labeled(ax)
    fig.tight_layout()
    return _fig_to_svg(fig)


def residual_plot(folds: list[FoldData]) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    for fd in folds:
        if fd.predictions.empty or "predicted_value" not in fd.predictions.columns:
            continue
        y_true = fd.predictions["true_label"].values
        y_pred = fd.predictions["predicted_value"].values
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, color=_fold_color(fd.fold), s=30, alpha=0.7, label=f"Fold {fd.fold}")

    ax.axhline(0, color="#C08497", linestyle="--", linewidth=1)
    ax.set_title("Residuals vs. predicted")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual − predicted)")
    _apply_soma_style(ax)
    _legend_if_labeled(ax)
    fig.tight_layout()
    return _fig_to_svg(fig)


# ---------------------------------------------------------------------------
# Subgroup charts
# ---------------------------------------------------------------------------


def subgroup_metric_chart(
    subgroup_data: dict[str, dict[str, float | int]],
    metric_name: str,
    overall_value: float,
    column_name: str = "",
) -> str:
    groups = [g for g, d in subgroup_data.items() if metric_name in d]
    values = [subgroup_data[g][metric_name] for g in groups]
    ns = [subgroup_data[g].get("n", "") for g in groups]

    colors = [SOMA_PALETTE[0] if v >= overall_value else SOMA_PALETTE[1] for v in values]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.bar(groups, values, color=colors)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"n={n}", ha="center", va="bottom", fontsize=9)
    ax.axhline(overall_value, color="#C08497", linestyle="--", linewidth=1.2,
               label=f"overall={overall_value:.3f}")

    title = f"{metric_name} by {column_name}" if column_name else metric_name
    ax.set_title(title)
    ax.set_xlabel(column_name or "Group")
    ax.set_ylabel(metric_name)
    ax.legend(fontsize=9)
    _apply_soma_style(ax)
    fig.tight_layout()
    return _fig_to_svg(fig)


def subgroup_stats_heatmap(
    stats: dict[str, dict[str, dict[str, float]]],
    columns: list[str],
    metrics: list[str],
) -> str:
    row_labels: list[str] = []
    z: list[list[float]] = []

    for col in columns:
        if col not in stats:
            continue
        for group, metric_pvals in sorted(stats[col].items()):
            row_labels.append(f"{col}={group}")
            z.append([metric_pvals.get(m, float("nan")) for m in metrics])

    if not row_labels:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.set_title("No statistical test results")
        return _fig_to_svg(fig)

    z_arr = np.array(z, dtype=float)
    h = max(4, 0.4 * len(row_labels))
    fig, ax = plt.subplots(figsize=(max(4, len(metrics) * 1.2), h))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(z_arr, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, label="p-value")

    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)

    for i in range(len(row_labels)):
        for j in range(len(metrics)):
            val = z_arr[i, j]
            text = f"{val:.3f}" if not np.isnan(val) else "—"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if val < 0.3 else "#0F172A")

    ax.set_title("Permutation test p-values (p < 0.05 = significant)")
    fig.tight_layout()
    return _fig_to_svg(fig)


# ---------------------------------------------------------------------------
# Cross-run comparison charts
# ---------------------------------------------------------------------------


def comparison_loss_curves(runs: list[RunData], labels: list[str]) -> str:
    fig, (ax_train, ax_tune) = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
    for run, label, color in zip(runs, labels, _run_colors(len(runs))):
        _add_mean_band(ax_train, run, label, color, key="train_loss")
        _add_mean_band(ax_tune, run, label, color, key="tune_loss")
    ax_train.set_title("Train loss (cross-run)")
    ax_train.set_xlabel("Epoch")
    ax_train.set_ylabel("Loss")
    ax_tune.set_title("Tune loss (cross-run)")
    ax_tune.set_xlabel("Epoch")
    _apply_soma_style(ax_train)
    _apply_soma_style(ax_tune)
    _legend_if_labeled(ax_train)
    _legend_if_labeled(ax_tune)
    fig.tight_layout()
    return _fig_to_svg(fig)


def comparison_metric_curves(runs: list[RunData], labels: list[str], metric_name: str) -> str:
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for run, label, color in zip(runs, labels, _run_colors(len(runs))):
        _add_mean_band(ax, run, label, color, metric_name=metric_name)
    ax.set_title(f"Tune {metric_name} per epoch (cross-run)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    _apply_soma_style(ax)
    _legend_if_labeled(ax)
    fig.tight_layout()
    return _fig_to_svg(fig)


def _add_mean_band(
    ax: plt.Axes,
    run: RunData,
    label: str,
    color: str,
    *,
    key: str | None = None,
    metric_name: str | None = None,
    linestyle: str = "-",
    show_label: bool = True,
) -> None:
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
        if len(vals) < n_epochs:
            vals = vals + [float("nan")] * (n_epochs - len(vals))
        per_fold_values.append(vals)

    if not per_fold_values:
        return

    arr = np.array(per_fold_values, dtype=float)
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0) if arr.shape[0] > 1 else None

    plot_label = label if show_label else None
    ax.plot(epochs, mean, color=color, linestyle=linestyle, linewidth=2, label=plot_label)
    if std is not None:
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        ax.fill_between(epochs, mean - std, mean + std, color=f"#{r:02x}{g:02x}{b:02x}", alpha=0.15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_aggregate_predictions = aggregate_slice_predictions


def _run_colors(n: int) -> list[str]:
    return [SOMA_PALETTE[i % len(SOMA_PALETTE)] for i in range(n)]
