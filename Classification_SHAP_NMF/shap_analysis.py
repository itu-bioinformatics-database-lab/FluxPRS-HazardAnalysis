#!/usr/bin/env python3
"""
SHAP-based interpretation of the RFE-selected Random Forest classifier
(manuscript Section 2.6, Supplementary Section S3, Supplementary Figures 5-7).

Protocol
--------
    1. ROSMAP AD / Control samples are split into a training partition (70%)
       and a held-out test partition (30%) by stratified sampling
       (random_state = 42).
    2. On the training partition only:
         - missing values are imputed with training-partition medians;
         - features are standardised (StandardScaler);
         - Recursive Feature Elimination (RFE; Random Forest estimator,
           step = 0.1) selects the top 50 features;
         - a Random Forest classifier is fitted once on the selected features.
    3. On the held-out test partition only (same fitted model):
         - SHAP values are computed (TreeExplainer, positive class = AD);
         - the direction of each feature's effect is derived from the sign of
           the correlation between feature values and SHAP values;
         - Welch's t-tests compare AD and Control for the top-ranked features;
         - error symmetry is assessed with McNemar's test (continuity correction).
         
"""

import argparse
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

SEED = 42
DPI = 300
POS_COLOR, NEG_COLOR = "#e74c3c", "#3498db"

# Columns that are never used as features
NON_FEATURE_COLS = ["Diagnosis", "SampleID", "Sample_ID", "sample_id", "ID", "id",
                    "PatientID", "patient_id", "Subject", "subject_id"]
DEMOGRAPHIC_COLS = ["Gender", "Race"]


# --------------------------------------------------------------------------
# Data preparation (all fitted steps use the training partition only)
# --------------------------------------------------------------------------
def prepare_data(path, test_size, n_features, exclude_demographics):
    df = pd.read_csv(path)
    df = df[df["Diagnosis"].isin(["AD", "Control"])].reset_index(drop=True)
    y = (df["Diagnosis"] == "AD").astype(int).to_numpy()

    drop = [c for c in NON_FEATURE_COLS if c in df.columns]
    if exclude_demographics:
        drop += [c for c in DEMOGRAPHIC_COLS if c in df.columns]
    X = df.drop(columns=drop).select_dtypes(include=[np.number])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y)

    medians = X_tr.median(numeric_only=True)
    X_tr, X_te = X_tr.fillna(medians), X_te.fillna(medians)

    scaler = StandardScaler().fit(X_tr)
    X_tr = pd.DataFrame(scaler.transform(X_tr), columns=X.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=X.columns)

    rfe = RFE(RandomForestClassifier(n_estimators=100, random_state=SEED),
              n_features_to_select=n_features, step=0.1).fit(X_tr, y_tr)
    selected = X_tr.columns[rfe.support_].tolist()

    return X_tr[selected], X_te[selected], y_tr, y_te, selected


def positive_class_shap(values):
    if isinstance(values, list):
        return np.asarray(values[1])
    if values.ndim == 3:
        return values[:, :, 1]
    return np.asarray(values)


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------
def directional_importance(shap_values, X_df):
    rows = []
    for i, f in enumerate(X_df.columns):
        x, s = X_df[f].to_numpy(), shap_values[:, i]
        r = 0.0 if np.std(x) == 0 or np.std(s) == 0 else np.corrcoef(x, s)[0, 1]
        r = 0.0 if np.isnan(r) else r
        rows.append({"feature": f, "mean_abs_shap": np.abs(s).mean(), "correlation": r,
                     "group": "Positive (AD)" if r >= 0 else "Negative (Control)"})
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def plot_directional_bar(importance, top_n, path):
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top["feature"], top["mean_abs_shap"],
            color=[POS_COLOR if r >= 0 else NEG_COLOR for r in top["correlation"]])
    ax.invert_yaxis()
    ax.set_xlabel("Average SHAP Impact (Mean |SHAP|)")
    ax.set_ylabel("Feature Name")
    ax.legend(handles=[
        Line2D([0], [0], color=POS_COLOR, lw=6, label="High levels -> POSITIVE Group (1)"),
        Line2D([0], [0], color=NEG_COLOR, lw=6, label="High levels -> NEGATIVE Group (0)")],
        loc="lower right", frameon=True)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, path):
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="YlGnBu")
    ax.grid(False)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f}%)", ha="center", va="center",
                    fontsize=15, fontweight="bold",
                    color="black" if cm[i, j] < cm.max() / 2 else "white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Control (0)", "Positive (1)"], fontsize=12)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Control (0)", "Positive (1)"], fontsize=12)
    ax.set_xlabel("Predicted Group", fontsize=12)
    ax.set_ylabel("True Group", fontsize=12)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def welch_ttests(X_df, y, features):
    rows = []
    for f in features:
        a, b = X_df.loc[y == 1, f], X_df.loc[y == 0, f]
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"feature": f, "t_statistic": t, "p_value": p,
                     "direction_in_AD": "Up-regulated" if a.mean() > b.mean() else "Down-regulated",
                     "significant_0.05": p < 0.05})
    return pd.DataFrame(rows).sort_values("p_value")


def mcnemar_corrected(cm):
    """McNemar's test with continuity correction; cm = [[TN, FP], [FN, TP]]."""
    b, c = int(cm[0, 1]), int(cm[1, 0])
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return chi2, float(stats.chi2.sf(chi2, 1))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/merged_data.csv")
    ap.add_argument("--out-dir", default="results/shap")
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument("--n-features", type=int, default=50, help="Number of RFE-selected features")
    ap.add_argument("--top-n", type=int, default=15, help="Features in bar plot and t-tests")
    ap.add_argument("--top-n-summary", type=int, default=20, help="Features in the SHAP summary plot")
    ap.add_argument("--exclude-demographics", action="store_true",
                    help="Also exclude Gender and Race columns from the feature set")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X_tr, X_te, y_tr, y_te, features = prepare_data(
        args.input, args.test_size, args.n_features, args.exclude_demographics)
    log.info("Training: %d (%d AD, %d Control) | Test: %d (%d AD, %d Control)",
             len(y_tr), y_tr.sum(), (y_tr == 0).sum(), len(y_te), y_te.sum(), (y_te == 0).sum())

    # Random Forest fitted once, on the training partition only
    model = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(X_tr, y_tr)

    # SHAP on the held-out test partition (Supplementary Figures 5 and 6)
    sv = positive_class_shap(shap.TreeExplainer(model).shap_values(X_te))
    plt.figure(figsize=(12, 8))
    shap.summary_plot(sv, X_te, max_display=args.top_n_summary, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "supp_fig5_shap_summary_test.png"), dpi=DPI,
                bbox_inches="tight")
    plt.close("all")

    importance = directional_importance(sv, X_te)
    importance.to_csv(os.path.join(args.out_dir, "shap_feature_importance_test.csv"), index=False)
    plot_directional_bar(importance, args.top_n,
                         os.path.join(args.out_dir, "supp_fig6_shap_bar_test.png"))

    # Welch's t-tests on the test partition, top-ranked SHAP features
    welch_ttests(X_te, y_te, importance["feature"].head(args.top_n)).to_csv(
        os.path.join(args.out_dir, "ttests_top_features_test.csv"), index=False)

    # Confusion matrix and McNemar's test on the test partition (Supplementary Figure 7)
    cm = confusion_matrix(y_te, model.predict(X_te), labels=[0, 1])
    plot_confusion_matrix(cm, os.path.join(args.out_dir, "supp_fig7_confusion_matrix_test.png"))
    chi2, p = mcnemar_corrected(cm)
    tn, fp, fn, tp = cm.ravel()
    pd.DataFrame([{"n_test": len(y_te), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
                   "specificity": tn / (tn + fp), "sensitivity": tp / (tp + fn),
                   "mcnemar_chi2": chi2, "mcnemar_p": p}]).to_csv(
        os.path.join(args.out_dir, "error_symmetry_test.csv"), index=False)

    pd.Series(features, name="feature").to_csv(
        os.path.join(args.out_dir, "rfe_selected_features.csv"), index=False)
    log.info("Results written to %s", args.out_dir)


if __name__ == "__main__":
    main()
