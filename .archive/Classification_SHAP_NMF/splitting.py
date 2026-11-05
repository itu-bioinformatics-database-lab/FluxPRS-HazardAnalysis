"""Leakage-safe sample splitting utilities.

Participant identifiers are optional for cohorts with one row per person and
required for repeated-measure cohorts such as DiCAD.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


GROUP_COLUMN_CANDIDATES = (
    "participant_id",
    "Participant ID",
    "participant",
    "PatientID",
    "patient_id",
    "Subject",
    "subject_id",
    "individual_id",
)


def resolve_group_column(frame, requested=None):
    """Return the requested or first recognized participant column."""
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Group column {requested!r} is not present")
        return requested
    return next((column for column in GROUP_COLUMN_CANDIDATES if column in frame.columns), None)


def assert_no_group_overlap(train_indices, test_indices, groups):
    """Raise if a participant occurs on both sides of a split."""
    if groups is None:
        return
    values = np.asarray(groups).astype(str)
    overlap = set(values[train_indices]) & set(values[test_indices])
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise RuntimeError(f"Participant leakage detected ({len(overlap)} groups): {preview}")


def stratified_folds(y, n_splits=5, seed=42, groups=None):
    """Return stratified folds, grouped by participant when IDs are supplied."""
    y = np.asarray(y)
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(y)), y))
    else:
        groups = np.asarray(groups).astype(str)
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        splits = list(splitter.split(np.zeros(len(y)), y, groups))
        for train, test in splits:
            assert_no_group_overlap(train, test, groups)
    return splits


def holdout_split(y, test_size=0.30, seed=42, groups=None):
    """Create a stratified holdout, keeping each participant in one partition."""
    y = np.asarray(y)
    indices = np.arange(len(y))
    if groups is None:
        train, test = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y
        )
        return np.asarray(train), np.asarray(test)

    groups = np.asarray(groups).astype(str)
    n_splits = max(2, int(round(1.0 / test_size)))
    candidates = stratified_folds(y, n_splits=n_splits, seed=seed, groups=groups)
    prevalence = np.bincount(y) / len(y)

    def split_distance(split):
        _, test = split
        test_prevalence = np.bincount(y[test], minlength=len(prevalence)) / len(test)
        return abs(len(test) / len(y) - test_size) + np.abs(
            test_prevalence - prevalence
        ).sum()

    train, test = min(candidates, key=split_distance)
    assert_no_group_overlap(train, test, groups)
    return np.asarray(train), np.asarray(test)
