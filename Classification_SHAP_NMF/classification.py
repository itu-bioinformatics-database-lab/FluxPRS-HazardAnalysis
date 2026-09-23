#!/usr/bin/env python3
"""
Cross-validated classification of Alzheimer's disease (AD) vs. Control in the
ROSMAP cohort using different feature sets (manuscript Section 2.5,
Supplementary Section S2 and Supplementary Table 1).

Feature sets
------------
    PRS, Metabolites, Pathway diff scores, Reaction diff scores,
    PRS + Metabolites, PRS + Pathway, PRS + Reaction, All features

Models
------
    Random Forest and Logistic Regression   (RFE-selected features and all features)
    Gradient Boosting, AdaBoost, Stacking   (RFE-selected features)

Evaluation protocol (nested cross-validation, no information leakage)
----------------------------------------------------------------------
    * Outer loop : stratified k-fold CV (default 5 folds). Metrics (accuracy,
      F1, ROC AUC) are computed on the outer test folds and reported as
      mean +/- standard deviation.
    * Within each outer training portion:
        - features are standardised (StandardScaler fitted on training data only);
        - Recursive Feature Elimination (RFE) with a Random Forest estimator
          is refitted on the training portion only;
        - the number of RFE features is chosen by an inner stratified k-fold CV
          (default 5 folds), in which RFE is again refitted on each inner
          training fold.
    * The outer test fold is never used for scaling, feature ranking,
      feature-number selection or model fitting.

"""

import argparse
import logging
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier,
                              RandomForestClassifier, StackingClassifier)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

SEED = 42
METADATA_COLS = ["sample_id", "Gender", "Race", "Diagnosis"]

# Feature set name -> input file name
FEATURE_SETS = {
    "PRS": "merged_df_with_diagnosis.csv",
    "Metabolites": "metabolite_only_data.csv",
    "Pathway": "pathway_only_data.csv",
    "Reaction": "reaction_only_data.csv",
    "PRS+Metabolites": "metabolit_with_prs_data.csv",
    "PRS+Pathway": "pathway_with_prs_data.csv",
    "PRS+Reaction": "reaction_with_prs_data.csv",
    "All": "merged_data.csv",
}

# Candidate numbers of RFE-selected features, searched in the inner CV
FEATURE_GRID_SMALL = list(range(1, 21))          # single / paired feature sets
FEATURE_GRID_ALL = list(range(20, 101, 10))      # all features


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataset(path):
    """Load a feature table and return X (DataFrame) and binary y (1 = AD)."""
    df = pd.read_csv(path)
    df = df[df["Diagnosis"].isin(["AD", "Control"])].reset_index(drop=True)
    # Identifier, label and demographic columns are not used as features
    X = df.drop(columns=[c for c in METADATA_COLS if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    y = df["Diagnosis"].map({"AD": 1, "Control": 0}).to_numpy()
    return X, y


# --------------------------------------------------------------------------
# Feature ranking by RFE (fitted on training data only)
# --------------------------------------------------------------------------
def rf_estimator():
    return RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)


def rfe_ranking(X_train, y_train, k_max, coarse_step=0.1):
    """
    Return feature indices ordered from most to least important according to
    RFE with a Random Forest estimator.

    If the number of features exceeds k_max, a coarse RFE (removing a fraction
    `coarse_step` of the remaining features per iteration) first reduces the
    set to k_max features. A fine RFE (one feature per iteration) then ranks
    the remaining features, so that the top-k subset is uniquely defined for
    every k <= k_max.
    """
    idx = np.arange(X_train.shape[1])
    if X_train.shape[1] > k_max:
        coarse = RFE(rf_estimator(), n_features_to_select=k_max, step=coarse_step)
        coarse.fit(X_train, y_train)
        idx = idx[coarse.support_]
    fine = RFE(rf_estimator(), n_features_to_select=1, step=1)
    fine.fit(X_train[:, idx], y_train)
    return idx[np.argsort(fine.ranking_, kind="stable")]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def build_models():
    """Classifiers evaluated on the RFE-selected features."""
    return {
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "Gradient Boosting": lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, random_state=SEED),
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=100, learning_rate=1.0, random_state=SEED),
        "Stacking Ensemble": lambda: StackingClassifier(
            estimators=[
                ("lr", LogisticRegression(max_iter=1000)),
                ("rf", RandomForestClassifier(n_estimators=100, random_state=SEED)),
                ("knn", KNeighborsClassifier()),
            ],
            final_estimator=LogisticRegression(),
            cv=5,
        ),
    }


def build_all_feature_models():
    """Classifiers evaluated on the full feature set (no feature selection)."""
    return {
        "Random Forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
        "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    }


def evaluate(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    return {
        "accuracy": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred),
        "auc": roc_auc_score(y_te, prob),
    }


# --------------------------------------------------------------------------
# Inner CV: choose the number of RFE features
# --------------------------------------------------------------------------
def select_n_features(X_train, y_train, grid, n_inner, metric):
    """
    Choose the number of RFE-selected features by inner stratified CV using a
    Random Forest classifier. RFE is refitted on every inner training fold.
    """
    grid = [k for k in grid if k <= X_train.shape[1]] or [X_train.shape[1]]
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=SEED)
    scores = {k: [] for k in grid}
    for tr, va in inner.split(X_train, y_train):
        scaler = StandardScaler().fit(X_train[tr])
        Xtr, Xva = scaler.transform(X_train[tr]), scaler.transform(X_train[va])
        order = rfe_ranking(Xtr, y_train[tr], k_max=max(grid))
        for k in grid:
            sel = order[:k]
            res = evaluate(RandomForestClassifier(n_estimators=100, random_state=SEED),
                           Xtr[:, sel], y_train[tr], Xva[:, sel], y_train[va])
            scores[k].append(res[metric])
    mean_scores = {k: np.mean(v) for k, v in scores.items()}
    return max(mean_scores, key=mean_scores.get)


# --------------------------------------------------------------------------
# Outer CV
# --------------------------------------------------------------------------
def run_feature_set(name, X, y, n_outer, n_inner, n_repeats, metric):
    grid = FEATURE_GRID_ALL if name == "All" else FEATURE_GRID_SMALL
    Xv = X.to_numpy(dtype=float)
    outer = RepeatedStratifiedKFold(n_splits=n_outer, n_repeats=n_repeats, random_state=SEED)

    fold_rows, selected_rows = [], []
    for fold, (tr, te) in enumerate(outer.split(Xv, y), start=1):
        log.info("[%s] outer fold %d/%d", name, fold, n_outer * n_repeats)

        # 1) Scaling fitted on the outer training portion only
        scaler = StandardScaler().fit(Xv[tr])
        Xtr, Xte = scaler.transform(Xv[tr]), scaler.transform(Xv[te])

        # 2) Number of features chosen by inner CV on the outer training portion
        k = select_n_features(Xv[tr], y[tr], grid, n_inner, metric)

        # 3) RFE refitted on the outer training portion; top-k features
        order = rfe_ranking(Xtr, y[tr], k_max=max(grid))
        sel = order[:k]
        selected_rows += [{"feature_set": name, "fold": fold, "rank": r + 1,
                           "feature": X.columns[i]} for r, i in enumerate(sel)]

        # 4) Models on RFE-selected features
        for model_name, factory in build_models().items():
            res = evaluate(factory(), Xtr[:, sel], y[tr], Xte[:, sel], y[te])
            fold_rows.append({"feature_set": name, "model": model_name,
                              "features": "RFE", "n_features": k, "fold": fold, **res})

        # 5) Models on all features
        for model_name, factory in build_all_feature_models().items():
            res = evaluate(factory(), Xtr, y[tr], Xte, y[te])
            fold_rows.append({"feature_set": name, "model": model_name,
                              "features": "All", "n_features": Xv.shape[1],
                              "fold": fold, **res})

    return pd.DataFrame(fold_rows), pd.DataFrame(selected_rows)


def summarise(folds):
    """Mean +/- SD across outer folds (format of Supplementary Table 1)."""
    g = folds.groupby(["feature_set", "features", "model"], sort=False)
    out = g.agg(
        n_features_median=("n_features", "median"),
        accuracy_mean=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        f1_mean=("f1", "mean"), f1_sd=("f1", "std"),
        auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
    ).reset_index()
    return out


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="Folder with the input CSV files")
    ap.add_argument("--out-dir", default="results/classification")
    ap.add_argument("--feature-sets", nargs="+", default=list(FEATURE_SETS),
                    choices=list(FEATURE_SETS))
    ap.add_argument("--n-outer", type=int, default=5, help="Outer CV folds")
    ap.add_argument("--n-inner", type=int, default=5, help="Inner CV folds")
    ap.add_argument("--n-repeats", type=int, default=1, help="Repeats of the outer CV")
    ap.add_argument("--selection-metric", default="accuracy",
                    choices=["accuracy", "f1", "auc"],
                    help="Inner-CV metric used to choose the number of RFE features")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_folds, all_selected = [], []
    for name in args.feature_sets:
        path = os.path.join(args.data_dir, FEATURE_SETS[name])
        X, y = load_dataset(path)
        log.info("[%s] %d samples (%d AD, %d Control), %d features",
                 name, len(y), int(y.sum()), int((y == 0).sum()), X.shape[1])
        folds, selected = run_feature_set(name, X, y, args.n_outer, args.n_inner,
                                          args.n_repeats, args.selection_metric)
        all_folds.append(folds)
        all_selected.append(selected)

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(os.path.join(args.out_dir, "classification_fold_metrics.csv"), index=False)
    summarise(folds).to_csv(os.path.join(args.out_dir, "classification_summary.csv"), index=False)
    pd.concat(all_selected, ignore_index=True).to_csv(
        os.path.join(args.out_dir, "rfe_selected_features_per_fold.csv"), index=False)
    log.info("Results written to %s", args.out_dir)


if __name__ == "__main__":
    main()
