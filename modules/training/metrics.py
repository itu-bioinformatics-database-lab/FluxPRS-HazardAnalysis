"""Canonical aggregation helpers for classification training runs."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CLASSIFICATION_SUMMARY_KEYS = (
    "mean_accuracy",
    "std_accuracy",
    "mean_f1",
    "std_f1",
    "mean_roc_auc",
    "std_roc_auc",
    "fold_accuracy",
    "fold_f1",
    "fold_roc_auc",
    "fold_metrics",
)


def summarize_classification_folds(folds: Sequence[dict]) -> dict:
    """Return the canonical mean, sample-SD, and per-fold metric payload."""
    if not folds:
        raise ValueError("Cannot aggregate an empty fold list")
    rows = list(folds)
    accuracy = [float(row["accuracy"]) for row in rows]
    f1 = [float(row["f1"]) for row in rows]
    auc = [
        float(row["roc_auc"]) if row.get("roc_auc") is not None else np.nan
        for row in rows
    ]
    return {
        "mean_accuracy": float(np.mean(accuracy)),
        "std_accuracy": float(np.std(accuracy, ddof=1)) if len(accuracy) > 1 else 0.0,
        "mean_f1": float(np.mean(f1)),
        "std_f1": float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0,
        "mean_roc_auc": float(np.nanmean(auc)),
        "std_roc_auc": float(np.nanstd(auc, ddof=1)) if len(auc) > 1 else 0.0,
        "fold_accuracy": accuracy,
        "fold_f1": f1,
        "fold_roc_auc": auc,
        "fold_metrics": rows,
    }
