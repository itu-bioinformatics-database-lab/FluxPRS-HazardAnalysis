#!/usr/bin/env python3
"""
Exploratory Nonnegative Matrix Factorization (NMF) of the integrated ROSMAP
multi-omics feature matrix (manuscript Section 2.7, Supplementary Section S4,
Supplementary Figures 8-13).

Leakage-safe protocol
---------------------
    1. A stratified 70/30 train/test split is created. When participant IDs
       are supplied or detected, all rows from one participant remain in one
       partition.
    2. Supervised RFE/RFECV feature selection is fitted on training rows only.
       RFECV folds are participant-grouped when participant IDs are available.
    3. Imputation and MinMax scaling are fitted on training rows only.
    4. NMF rank diagnostics and the final k=5 model are fitted on training rows
       only. Held-out samples are projected with ``model.transform`` and NMF
       is never refitted on the test partition.
    5. Factor plots are descriptive; diagnosis labels are not used to select
       which factors to display.

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
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import RFE, RFECV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

try:
    from .splitting import holdout_split, resolve_group_column, stratified_folds
except ImportError:
    from splitting import holdout_split, resolve_group_column, stratified_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

SEED = 42
METADATA_COLS = [
    "sample_id", "Sample ID", "Gender", "Race", "Diagnosis",
    "participant_id", "Participant ID", "PatientID", "patient_id",
    "Subject", "subject_id", "individual_id",
]
CATEGORY_ORDER = ["PRS", "Metabolite", "Reaction", "Pathway"]
CATEGORY_COLORS = {"PRS": "#4C72B0", "Metabolite": "#55A868",
                   "Reaction": "#C44E52", "Pathway": "#8172B3"}
PALETTE = {"Control": "#E41A1C", "AD": "#377EB8"}


def load_dataset(path, group_column=None):
    df = pd.read_csv(path)
    df = df[df["Diagnosis"].isin(["AD", "Control"])].reset_index(drop=True)
    resolved_group = resolve_group_column(df, group_column)
    groups = (
        df[resolved_group].astype(str).to_numpy()
        if resolved_group is not None
        else None
    )
    excluded = set(METADATA_COLS)
    if resolved_group:
        excluded.add(resolved_group)
    X = df.drop(columns=[c for c in excluded if c in df.columns])
    X = X.select_dtypes(include=[np.number])
    y = df["Diagnosis"].map({"Control": 0, "AD": 1}).to_numpy()
    return X, y, df["Diagnosis"].to_numpy(), groups, resolved_group


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
    """Fit NMF and return the model, W, reconstruction error and convergence."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model = NMF(n_components=k, init="nndsvd", random_state=SEED, max_iter=1000)
        W = model.fit_transform(X)
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return model, W, model.reconstruction_err_, converged


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
    ap.add_argument("--test-size", type=float, default=0.30)
    ap.add_argument(
        "--group-column",
        default=None,
        help=(
            "Participant-ID column. Repeated observations are kept in one "
            "partition and one RFECV fold; common names are detected automatically."
        ),
    )
    ap.add_argument(
        "--plot-factors",
        nargs=2,
        type=int,
        default=[3, 2],
        metavar=("X", "Y"),
        help="One-based factor numbers to plot; selected without using diagnosis labels.",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    X, y, diagnosis, groups, resolved_group = load_dataset(
        args.input, args.group_column
    )
    cats = feature_categories(X.columns, args.n_prs, args.n_metabolites, args.n_reactions)
    log.info("%d samples (%d AD, %d Control), %d features",
             len(y), y.sum(), (y == 0).sum(), X.shape[1])

    # 1. Holdout split; participant grouping is enforced when IDs are available.
    tr, te = holdout_split(y, args.test_size, SEED, groups)
    X_train, X_test = X.iloc[tr].copy(), X.iloc[te].copy()
    y_train, y_test = y[tr], y[te]
    diagnosis_train, diagnosis_test = diagnosis[tr], diagnosis[te]
    groups_train = groups[tr] if groups is not None else None

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X.columns, index=X_train.index
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test), columns=X.columns, index=X_test.index
    )

    # 2. Two-stage supervised feature selection on training rows only.
    rf = lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    coarse_target = min(500, X_train.shape[1])
    coarse = RFE(
        rf(), n_features_to_select=coarse_target, step=min(500, X_train.shape[1])
    ).fit(X_train, y_train)
    X_train_500 = X_train.loc[:, coarse.support_]
    X_test_500 = X_test.loc[:, coarse.support_]
    inner_cv = stratified_folds(
        y_train, n_splits=5, seed=SEED, groups=groups_train
    )
    rfecv = RFECV(
        rf(),
        step=min(50, X_train_500.shape[1]),
        cv=inner_cv,
        scoring="roc_auc",
        min_features_to_select=min(50, X_train_500.shape[1]),
        n_jobs=-1,
    ).fit(X_train_500, y_train)
    X_train_sel = X_train_500.loc[:, rfecv.support_]
    X_test_sel = X_test_500.loc[:, rfecv.support_]
    log.info("RFECV selected %d training-derived features", X_train_sel.shape[1])
    pd.DataFrame({"n_features": rfecv.cv_results_["n_features"],
                  "mean_cv_auc": rfecv.cv_results_["mean_test_score"]}).to_csv(
        os.path.join(args.out_dir, "rfecv_curve.csv"), index=False)

    # 3. Train-fitted scaling and training-only rank-selection curve.
    scaler = MinMaxScaler().fit(X_train_sel)
    X_train_scaled = np.clip(scaler.transform(X_train_sel), 0, None)
    X_test_scaled = np.clip(scaler.transform(X_test_sel), 0, None)
    total_ss = np.sum(X_train_scaled ** 2)
    total_ss_centered = np.sum(
        (X_train_scaled - X_train_scaled.mean(axis=0)) ** 2
    )
    curve = []
    for k in range(args.k_min, args.k_max + 1):
        _, _, err, conv = fit_nmf(X_train_scaled, k)
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

    # 4. Final training NMF and held-out projection without refitting.
    model, W_train, err_train, converged = fit_nmf(X_train_scaled, args.k)
    W_test = model.transform(X_test_scaled)
    test_error = float(np.linalg.norm(X_test_scaled - W_test @ model.components_, ord="fro"))
    factors = [f"Factor_{i + 1}" for i in range(args.k)]
    scores_train = pd.DataFrame(W_train, columns=factors)
    scores_train["Diagnosis"] = diagnosis_train
    scores_train["split"] = "train"
    scores_test = pd.DataFrame(W_test, columns=factors)
    scores_test["Diagnosis"] = diagnosis_test
    scores_test["split"] = "test"
    scores = pd.concat([scores_train, scores_test], ignore_index=True)
    scores.to_csv(os.path.join(args.out_dir, "nmf_sample_scores.csv"), index=False)
    loadings = pd.DataFrame(
        model.components_, index=factors, columns=X_train_sel.columns
    )
    loadings.to_csv(os.path.join(args.out_dir, "nmf_feature_loadings.csv"))

    pd.DataFrame([
        {
            "split": "train",
            "n_features": X_train_sel.shape[1],
            "reconstruction_error": err_train,
            "converged": converged,
        },
        {
            "split": "test_projection",
            "n_features": X_test_sel.shape[1],
            "reconstruction_error": test_error,
            "converged": None,
        },
    ]).to_csv(
        os.path.join(args.out_dir, "nmf_reconstruction_errors.csv"), index=False)

    # 5. Factors are predeclared, not selected using diagnosis associations.
    if any(number < 1 or number > args.k for number in args.plot_factors):
        raise ValueError("--plot-factors must be between 1 and --k")
    fx, fy = [f"Factor_{number}" for number in args.plot_factors]

    train_participants = set(groups[tr]) if groups is not None else set()
    test_participants = set(groups[te]) if groups is not None else set()
    pd.DataFrame([{
        "group_column": resolved_group,
        "n_train_rows": len(tr),
        "n_test_rows": len(te),
        "n_train_participants": len(train_participants) if groups is not None else len(tr),
        "n_test_participants": len(test_participants) if groups is not None else len(te),
        "participant_overlap": len(train_participants & test_participants),
        "feature_selection_fit": "train_only",
        "scaler_fit": "train_only",
        "nmf_fit": "train_only",
        "test_handling": "transform_only",
    }]).to_csv(os.path.join(args.out_dir, "split_audit.csv"), index=False)

    # 6. Descriptive figures.
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=scores, x=fx, y=fy, hue="Diagnosis", style="split",
        palette=PALETTE, s=120, alpha=0.7, edgecolor="white", ax=ax
    )
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
    sel_cats = cats[X_train_sel.columns]
    comp = pd.DataFrame({
        "n_input": cats.value_counts().reindex(CATEGORY_ORDER, fill_value=0),
        "n_selected": sel_cats.value_counts().reindex(CATEGORY_ORDER, fill_value=0),
    })
    comp["selection_rate"] = comp["n_selected"] / comp["n_input"].replace(0, np.nan)
    comp["overall_selection_rate"] = X_train_sel.shape[1] / X.shape[1]
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
    ax.text(0.95, 0.95, f"Total Selected Features: {X_train_sel.shape[1]}", transform=ax.transAxes,
            ha="right", va="top", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "selected_feature_composition.png"), dpi=300)
    plt.close(fig)

    # 8. Total loading weight per category for the two predeclared factors
    load_by_cat = loadings.loc[[fx, fy]].T.groupby(sel_cats).sum().reindex(CATEGORY_ORDER).fillna(0)
    load_by_cat.to_csv(os.path.join(args.out_dir, "loading_weight_by_category.csv"))
    fig, ax = plt.subplots(figsize=(9, 7))
    bottom = np.zeros(2)
    for c in CATEGORY_ORDER:
        vals = load_by_cat.loc[c, [fx, fy]].to_numpy()
        ax.bar([fx.replace("_", " "), fy.replace("_", " ")], vals, bottom=bottom,
               color=CATEGORY_COLORS[c], label=c)
        bottom += vals
    ax.set_ylabel(f"Total Loading Weight (Sum of {X_train_sel.shape[1]} Features)")
    ax.set_xlabel("NMF Factors")
    ax.legend(title="Feature Category", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "loading_weight_by_category.png"), dpi=300)
    plt.close(fig)

    log.info("Results written to %s", args.out_dir)


if __name__ == "__main__":
    main()
