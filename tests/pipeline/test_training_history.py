"""A fold's ``training_history.json`` diagnostics, against hand-computed peaks."""

from __future__ import annotations

import math

import pandas as pd

from soma.pipeline.aggregate import peak_per_metric


def test_peak_per_metric_keys_tune_metrics_follows_their_direction_and_skips_the_loss():
    nan = math.nan
    history = pd.DataFrame(
        {
            "epoch": [0, 1, 2, 3],
            "train_loss": [0.9, 0.1, 0.5, 0.4],
            "lr": [1e-3, 1e-3, 1e-3, 1e-3],
            "elapsed_seconds": [1.0, 2.0, 3.0, 4.0],
            # Epoch 1 was not evaluated.
            "tune/loss": [0.8, nan, 0.2, 0.6],
            "tune/auroc": [0.6, nan, 0.7, 0.9],  # higher is better: peaks at epoch 4
            "tune/mae": [1.0, nan, 2.0, 3.0],  # lower is better: peaks at epoch 1
        }
    )
    assert peak_per_metric(history) == {
        "tune/auroc": {"epoch": 4, "value": 0.9},
        "tune/mae": {"epoch": 1, "value": 1.0},
    }


def test_peak_per_metric_keeps_the_mean_over_groups_and_skips_each_group_and_the_counts():
    nan = math.nan
    history = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train_loss": [0.9, 0.5, 0.4],
            "lr": [1e-3, 1e-3, 1e-3],
            "elapsed_seconds": [1.0, 2.0, 3.0],
            "tune/loss": [0.8, nan, 0.6],
            "tune/c_index": [0.6, nan, 0.7],
            "tune/centre=A/loss": [0.7, nan, 0.5],
            "tune/centre=A/c_index": [0.9, nan, 0.5],
            "tune/centre=B/c_index": [0.5, nan, 0.6],
            "tune/mean_over_centre/loss": [0.75, nan, 0.55],
            "tune/mean_over_centre/loss_groups": [2, nan, 2],
            "tune/mean_over_centre/c_index": [0.7, nan, 0.55],  # higher is better: peaks at epoch 1
            "tune/mean_over_centre/c_index_groups": [2, nan, 2],
        }
    )
    assert peak_per_metric(history) == {
        "tune/c_index": {"epoch": 3, "value": 0.7},
        "tune/mean_over_centre/c_index": {"epoch": 1, "value": 0.7},
    }


def test_peak_per_metric_skips_the_eval_mode_train_values():
    # train/* keys (trainer.eval_train) gauge the train-vs-tune gap; they are not a
    # selection proxy, so they get no peak.
    nan = math.nan
    history = pd.DataFrame(
        {
            "epoch": [0, 1, 2],
            "train_loss": [0.9, 0.5, 0.4],
            "lr": [1e-3, 1e-3, 1e-3],
            "elapsed_seconds": [1.0, 2.0, 3.0],
            "tune/loss": [0.8, nan, 0.6],
            "tune/auroc": [0.6, nan, 0.7],
            "train/loss": [0.7, nan, 0.3],
            "train/auroc": [0.8, nan, 0.99],
            "train/mean_over_centre/auroc": [0.8, nan, 0.98],
            "train/mean_over_centre/auroc_groups": [2, nan, 2],
        }
    )
    assert peak_per_metric(history) == {"tune/auroc": {"epoch": 3, "value": 0.7}}
