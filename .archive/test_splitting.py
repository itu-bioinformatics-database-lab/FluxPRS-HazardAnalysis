from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "Classification_SHAP_NMF"
sys.path.insert(0, str(MODULE_DIR))

from splitting import holdout_split, stratified_folds  # noqa: E402


def repeated_participant_data():
    groups = np.repeat([f"P{i:02d}" for i in range(30)], 2)
    y = np.repeat(np.arange(30) % 2, 2)
    return y, groups


def test_grouped_cv_never_splits_a_participant():
    y, groups = repeated_participant_data()
    folds = stratified_folds(y, n_splits=5, seed=42, groups=groups)
    assert len(folds) == 5
    for train, test in folds:
        assert not (set(groups[train]) & set(groups[test]))


def test_grouped_holdout_never_splits_a_participant():
    y, groups = repeated_participant_data()
    train, test = holdout_split(y, test_size=0.30, seed=42, groups=groups)
    assert not (set(groups[train]) & set(groups[test]))
    assert 0.20 <= len(test) / len(y) <= 0.40


def test_sample_level_split_remains_stratified_without_groups():
    y = np.array([0] * 30 + [1] * 30)
    train, test = holdout_split(y, test_size=0.30, seed=42)
    assert len(set(train) & set(test)) == 0
    assert y[train].mean() == y[test].mean() == 0.5
