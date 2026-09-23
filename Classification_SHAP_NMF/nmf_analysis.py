#!/usr/bin/env python3
"""
Exploratory Nonnegative Matrix Factorization (NMF) of the integrated ROSMAP
multi-omics feature matrix (manuscript Section 2.7, Supplementary Section S4,
Supplementary Figures 8-13).

Protocol
--------
    1. Initial NMF (k = 5) on the full MinMax-scaled feature matrix
       (reconstruction error reported descriptively).
    2. Two-stage supervised feature selection on the full cohort:
         - RFE with a Random Forest estimator (step = 500) -> 500 features;
         - RFECV (step = 50, 5-fold stratified CV, ROC AUC,
           min_features_to_select = 50) -> optimal feature set (450 in the
           manuscript).
       Because this selection uses the diagnosis labels of all samples, the
       associations between NMF factors and diagnosis are optimistic and are
       interpreted as descriptive.
    3. The selected features are MinMax-scaled and NMF is fitted for
       k = 2..10 (rank-selection curve); the final model uses k = 5.
    4. Factor scores are compared between AD and Control (Student's t-test),
       and factor loadings are summarised by feature category.

Reconstruction error is the Frobenius norm ||X - WH||_F (scikit-learn,
beta_loss = "frobenius"). Because it scales with the size of the matrix,
errors obtained on matrices of different dimensions are not directly comparable.

"""

import argparse
import logging
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

SEED = 42
METADATA_COLS = ["sample_id", "Gender", "Race", "Diagnosis"]
CATEGORY_ORDER = ["PRS", "Metabolite", "Reaction", "Pathway"]
CATEGORY_COLORS = {"PRS": "#4C72B0", "Metabolite": "#55A868",
                   "Reaction": "#C44E52", "Pathway": "#8172B3"}
PALETTE = {"Control": "#E41A1C", "AD": "#377EB8"}


def load_dataset(path):
    df = pd.read_csv(path)
    df = df[df["Diagnosis"].isin(["AD", "Control"])].reset_index(drop=True)
    X = df.drop(columns=[c for c in METADATA_COLS if c in df.columns])
    y = df["Diagnosis"].map({"Control": 0, "AD": 1}).to_numpy()
    return X, y, df["Diagnosis"].to_numpy()


def feature_categories(columns, n_prs, n_met, n_rxn):
    cats = {}
    for i, c in enumerate(columns):
        if i < n_prs:
            cats[c] = "PRS"
        elif i < n_prs + n_met:
            cats[c] = "Metabolite"
        elif i < n_prs + n_met + n_rxn:
            cats[c] = "Reaction"
        else:
            cats[c] = "Pathway"
    return pd.Series(cats)


def fit_nmf(X, k):
    """Fit NMF and return W, H, reconstruction error and convergence flag."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model = NMF(n_components=k, init="nndsvd", random_state=SEED, max_iter=1000)
        W = model.fit_transform(X)
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return W, model.components_, model.reconstruction_err_, converged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/merged_data.csv")
    ap.add_argument("--out-dir", default="results/nmf")
    ap.add_argument("--k", type=int, default=5, help="Final NMF rank")
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=10)
    ap.add_argument("--n-prs", type=int, default=29)
    ap.add_argument("--n-metabolites", type=int, default=98)
    ap.add_argument("--n-reactions", type=int, default=10604)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    X, y, diagnosis = load_dataset(args.input)
    cats = feature_categories(X.columns, args.n_prs, args.n_metabolites, args.n_reactions)
    log.info("%d samples (%d AD, %d Control), %d features",
             len(y), y.sum(), (y == 0).sum(), X.shape[1])

    # 1. Initial NMF on the full feature matrix (descriptive)
    _, _, err_full, _ = fit_nmf(MinMaxScaler().fit_transform(X), args.k)

    # 2. Two-stage feature selection on the full cohort
    rf = lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    coarse = RFE(rf(), n_features_to_select=500, step=500).fit(X, y)
    X_500 = X.loc[:, coarse.support_]
    rfecv = RFECV(rf(), step=50, cv=StratifiedKFold(5), scoring="roc_auc",
                  min_features_to_select=50, n_jobs=-1).fit(X_500, y)
    X_sel = X_500.loc[:, rfecv.support_]
    log.info("RFECV selected %d features", X_sel.shape[1])
    pd.DataFrame({"n_features": rfecv.cv_results_["n_features"],
                  "mean_cv_auc": rfecv.cv_results_["mean_test_score"]}).to_csv(
        os.path.join(args.out_dir, "rfecv_curve.csv"), index=False)

    # 3. MinMax scaling of the selected features and rank-selection curve
    X_scaled = MinMaxScaler().fit_transform(X_sel)
    total_ss = np.sum(X_scaled ** 2)                                   # uncentered
    total_ss_centered = np.sum((X_scaled - X_scaled.mean(axis=0)) ** 2)  # centered
    curve = []
    for k in range(args.k_min, args.k_max + 1):
        _, _, err, conv = fit_nmf(X_scaled, k)
        curve.append({"k": k, "reconstruction_error": err, "converged": conv,
                      "explained_variance_uncentered": 1 - err ** 2 / total_ss,
                      "explained_variance_centered": 1 - err ** 2 / total_ss_centered})
    curve = pd.DataFrame(curve)
    curve["relative_error_reduction"] = -curve["reconstruction_error"].pct_change()
    curve.to_csv(os.path.join(args.out_dir, "nmf_rank_selection.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve["k"], curve["reconstruction_error"], marker="o")
    ax.axvline(args.k, color="red", linestyle="--", label=f"Selected k={args.k}")
    ax.set_xlabel("Number of Latent Components (k)")
    ax.set_ylabel("Reconstruction Error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "nmf_rank_selection.png"), dpi=300)
    plt.close(fig)

    # 4. Final NMF
    W, H, err_sel, _ = fit_nmf(X_scaled, args.k)
    factors = [f"Factor_{i + 1}" for i in range(args.k)]
    scores = pd.DataFrame(W, columns=factors)
    scores["Diagnosis"] = diagnosis
    scores.to_csv(os.path.join(args.out_dir, "nmf_sample_scores.csv"), index=False)
    loadings = pd.DataFrame(H, index=factors, columns=X_sel.columns)
    loadings.to_csv(os.path.join(args.out_dir, "nmf_feature_loadings.csv"))

    pd.DataFrame([{"n_features": X.shape[1], "reconstruction_error": err_full},
                  {"n_features": X_sel.shape[1], "reconstruction_error": err_sel}]).to_csv(
        os.path.join(args.out_dir, "nmf_reconstruction_errors.csv"), index=False)

    # 5. Factor-diagnosis association (descriptive)
    tests = []
    for f in factors:
        t, p = stats.ttest_ind(scores.loc[scores.Diagnosis == "AD", f],
                               scores.loc[scores.Diagnosis == "Control", f])
        tests.append({"factor": f, "t_statistic": t, "p_value": p})
    tests = pd.DataFrame(tests).sort_values("p_value")
    tests.to_csv(os.path.join(args.out_dir, "nmf_factor_ttests.csv"), index=False)
    fx, fy = tests["factor"].iloc[0], tests["factor"].iloc[1]

    # 6. Figures: scatter, box plots, density
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(data=scores, x=fx, y=fy, hue="Diagnosis", palette=PALETTE,
                    s=120, alpha=0.7, edgecolor="white", ax=ax)
    ax.set_xlabel(f"NMF {fx.replace('_', ' ')}")
    ax.set_ylabel(f"NMF {fy.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "nmf_scatter.png"), dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, f in zip(axes, [fx, fy]):
        sns.boxplot(data=scores, x="Diagnosis", y=f, hue="Diagnosis", palette=PALETTE,
                    order=["Control", "AD"], legend=False, ax=ax)
        ax.set_ylabel(f"NMF {f.replace('_', ' ')} Weight")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "nmf_boxplots.png"), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=scores, x=fx, hue="Diagnosis", fill=True, common_norm=False,
                palette=PALETTE, alpha=0.4, ax=ax)
    ax.set_xlabel(f"NMF {fx.replace('_', ' ')}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "nmf_density.png"), dpi=300)
    plt.close(fig)

    # 7. Composition of the selected features by category (with selection rates)
    sel_cats = cats[X_sel.columns]
    comp = pd.DataFrame({
        "n_input": cats.value_counts().reindex(CATEGORY_ORDER, fill_value=0),
        "n_selected": sel_cats.value_counts().reindex(CATEGORY_ORDER, fill_value=0),
    })
    comp["selection_rate"] = comp["n_selected"] / comp["n_input"].replace(0, np.nan)
    comp["overall_selection_rate"] = X_sel.shape[1] / X.shape[1]
    comp.to_csv(os.path.join(args.out_dir, "selected_feature_composition.csv"))

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(comp.index, comp["n_selected"],
                  color=[CATEGORY_COLORS[c] for c in comp.index], edgecolor="black", alpha=0.85)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{int(bar.get_height())}", ha="center", va="bottom",
                fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Selected Features")
    ax.set_xlabel("Biological Layer / Category")
    ax.set_ylim(0, comp["n_selected"].max() + 50)
    ax.text(0.95, 0.95, f"Total Selected Features: {X_sel.shape[1]}", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "selected_feature_composition.png"), dpi=300)
    plt.close(fig)

    # 8. Total loading weight per category for the two disease-associated factors
    load_by_cat = loadings.loc[[fx, fy]].T.groupby(sel_cats).sum().reindex(CATEGORY_ORDER).fillna(0)
    load_by_cat.to_csv(os.path.join(args.out_dir, "loading_weight_by_category.csv"))
    fig, ax = plt.subplots(figsize=(9, 7))
    bottom = np.zeros(2)
    for c in CATEGORY_ORDER:
        vals = load_by_cat.loc[c, [fx, fy]].to_numpy()
        ax.bar([fx.replace("_", " "), fy.replace("_", " ")], vals, bottom=bottom,
               color=CATEGORY_COLORS[c], label=c)
        bottom += vals
    ax.set_ylabel(f"Total Loading Weight (Sum of {X_sel.shape[1]} Features)")
    ax.set_xlabel("NMF Factors")
    ax.legend(title="Feature Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loading_weight_by_category.png"), dpi=300)
    plt.close(fig)

    log.info("Results written to %s", args.out_dir)


if __name__ == "__main__":
    main()
