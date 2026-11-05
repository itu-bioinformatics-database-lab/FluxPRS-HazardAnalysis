from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from render_nested_classification_table import (  # noqa: E402
    heldout_table_fragment,
    table_fragment,
)
from run_notebook_classifiers import (  # noqa: E402
    aggregate_fold_scores,
    feature_count_grid,
    nested_select_k,
)

pytestmark = pytest.mark.filterwarnings("ignore:Only one class is present.*")


def test_feature_count_grid_matches_manuscript_protocol():
    assert feature_count_grid(6, "prs") == [1, 2, 3, 4, 5, 6]
    assert feature_count_grid(150, "all_data") == list(range(20, 101, 10))


def test_aggregate_fold_scores_uses_sample_standard_deviation():
    folds = [
        {"accuracy": 0.5, "f1": 0.4, "roc_auc": 0.6},
        {"accuracy": 0.7, "f1": 0.8, "roc_auc": 0.9},
    ]
    out = aggregate_fold_scores(folds)
    assert np.isclose(out["mean_accuracy"], 0.6)
    assert np.isclose(out["std_accuracy"], np.std([0.5, 0.7], ddof=1))
    assert out["fold_metrics"] == folds


def test_grouped_inner_selection_keeps_participants_in_one_fold():
    rng = np.random.default_rng(42)
    n_groups = 30
    groups = np.repeat([f"p{i}" for i in range(n_groups)], 2)
    y = np.repeat(np.arange(n_groups) % 3, 2)
    x = rng.normal(size=(len(y), 6))
    x[:, 0] += y
    ids = pd.DataFrame({"participant_id": groups})

    result = nested_select_k(
        x,
        y,
        ids,
        "prs",
        ("healthy", "Pre-diabetes", "Diabetes"),
        "multiclass",
        True,
        outer_fold=0,
    )

    folds = np.asarray(result["inner_val_fold"])
    assert result["best_k"] in range(1, 7)
    assert sorted(np.unique(folds).tolist()) == [0, 1, 2, 3, 4]
    for group in np.unique(groups):
        assert np.unique(folds[groups == group]).size == 1


def test_heldout_table_uses_existing_test_metrics():
    metrics_path = ROOT / "results" / "clf_notebook_metrics.csv"
    if not metrics_path.is_file():
        pytest.skip("held-out metrics are omitted from the public repository")
    metrics = pd.read_csv(metrics_path)
    cv = table_fragment(metrics)
    heldout = heldout_table_fragment(metrics)
    row = metrics.loc[
        (metrics["cohort"] == "ROSMAP-AD")
        & (metrics["feature_set"] == "prs_metabolites")
        & (metrics["model"] == "random_forest")
        & (metrics["selection"].str.startswith("nested_top_"))
    ].iloc[0]
    assert "tab:heldout_classification" in heldout
    assert f"{float(row['test_auc']):.4f}" in heldout
    assert f"{float(row['cv_auc_mean']):.4f}" in cv
    assert "mean $\\pm$ std" not in heldout
    assert "primary classification evidence" in heldout
