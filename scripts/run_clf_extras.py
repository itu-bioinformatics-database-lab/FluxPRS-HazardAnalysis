#!/usr/bin/env python3
"""RF+RFE / McNemar on ROSMAP+DiCAD; NMF on all four cohorts.

Classification (LogReg baseline, RF+RFE, McNemar, RF-on-NMF AUC) is
ROSMAP-AD (binary) and DIABETES (3-class) only. NMF factorization is
unsupervised and is written for every cohort.

Does not re-run FVA. Does not run Mayo classification.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.stats.contingency_tables import mcnemar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
for path in (str(PREPROCESSING_DIR), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from cohort_splits import N_CV_SPLITS, RANDOM_STATE, assign_cv_folds  # noqa: E402
from run_train_cv import (  # noqa: E402
    META_DROP,
    _align_proba,
    align_features,
    compute_metrics,
    encode_labels,
    load_cv_fold_features,
    load_cv_fold_ids,
    load_feature_matrix,
    load_split_table,
    numeric_x,
    to_builtin,
)

CLASSIFICATION_SPECS = {
    "ROSMAP-AD": {
        "task": "binary",
        "class_labels": ("healthy", "AD"),
        "grouped": False,
    },
    "DIABETES": {
        "task": "multiclass",
        "class_labels": ("healthy", "Pre-diabetes", "Diabetes"),
        "grouped": True,
    },
}
ALL_COHORTS = ("ROSMAP-AD", "MayoRNASeq-AD", "MayoRNASeq-PSP", "DIABETES")
NMF_K = 5
REACTION_VAR_K = 500


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")


def x_from(df: pd.DataFrame) -> pd.DataFrame:
    drop = META_DROP & set(df.columns)
    return df.drop(columns=list(drop)).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def rf_estimator(n_estimators: int = 100, n_jobs: int = 1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def logreg() -> LogisticRegression:
    return LogisticRegression(C=1.0, random_state=RANDOM_STATE, max_iter=1000)


def scale_xy(x_train: pd.DataFrame, x_other: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_tr = np.nan_to_num(scaler.fit_transform(x_train.to_numpy(dtype=float)), nan=0.0)
    x_ot = np.nan_to_num(scaler.transform(x_other.to_numpy(dtype=float)), nan=0.0)
    return x_tr, x_ot


def rfe_scoring(task: str) -> str:
    return "roc_auc" if task == "binary" else "roc_auc_ovr"


def make_rfe_cv(task_grouped: bool, y: np.ndarray, groups: Optional[np.ndarray], n: int):
    dummy = np.zeros((n, 1))
    if task_grouped:
        splitter = StratifiedGroupKFold(
            n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
        )
        return list(splitter.split(dummy, y, groups=groups))
    splitter = StratifiedKFold(
        n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    return list(splitter.split(dummy, y))


def cv_scores_on_train(
    estimator,
    x: np.ndarray,
    y: np.ndarray,
    val_fold: np.ndarray,
    class_labels: Sequence[str],
    task: str,
) -> dict:
    n_classes = len(class_labels)
    accs = []
    aucs = []
    oof_pred = np.full(len(y), -1, dtype=int)
    oof_proba = np.zeros((len(y), n_classes), dtype=float)
    for fold in range(int(val_fold.max()) + 1):
        tr = val_fold != fold
        va = val_fold == fold
        model = estimator.__class__(**estimator.get_params())
        x_tr, x_va = x[tr], x[va]
        if not isinstance(estimator, RandomForestClassifier):
            scaler = StandardScaler()
            x_tr = np.nan_to_num(scaler.fit_transform(x_tr), nan=0.0)
            x_va = np.nan_to_num(scaler.transform(x_va), nan=0.0)
        model.fit(x_tr, y[tr])
        pred = model.predict(x_va)
        proba = _align_proba(model, model.predict_proba(x_va), class_labels)
        oof_pred[va] = pred
        oof_proba[va] = proba
        accs.append(float(accuracy_score(y[va], pred)))
        try:
            if task == "binary":
                if np.unique(y[va]).size == 2:
                    aucs.append(float(roc_auc_score(y[va], proba[:, 1])))
            else:
                aucs.append(
                    float(
                        roc_auc_score(
                            y[va],
                            proba,
                            multi_class="ovr",
                            average="macro",
                            labels=list(range(n_classes)),
                        )
                    )
                )
        except ValueError:
            pass
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
        "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        "std_roc_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
        "fold_accuracy": accs,
        "fold_roc_auc": aucs,
    }


def mcnemar_from_preds(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n01 = int((a_correct & ~b_correct).sum())
    n10 = int((~a_correct & b_correct).sum())
    table = np.array(
        [[int((a_correct & b_correct).sum()), n01], [n10, int((~a_correct & ~b_correct).sum())]]
    )
    result = mcnemar(table, exact=True)
    chi2 = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10) if (n01 + n10) else None
    return {
        "table": table.tolist(),
        "n_a_correct_b_wrong": n01,
        "n_a_wrong_b_correct": n10,
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "chi2_continuity": chi2,
    }


def plot_rfecv(cv_results: dict, out_path: Path, title: str) -> None:
    scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(scores) + 1), scores)
    ax.set_xlabel("RFECV step")
    ax.set_ylabel("CV ROC AUC")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fit_nmf(x_train: pd.DataFrame, x_test: pd.DataFrame, k: int = NMF_K) -> dict:
    scaler = MinMaxScaler()
    x_tr = np.clip(scaler.fit_transform(x_train.to_numpy(dtype=float)), 0.0, None)
    x_te = np.clip(scaler.transform(x_test.to_numpy(dtype=float)), 0.0, None)
    nmf = NMF(n_components=k, init="nndsvd", random_state=RANDOM_STATE, max_iter=2000)
    w_train = nmf.fit_transform(x_tr)
    w_test = nmf.transform(x_te)
    return {
        "k": k,
        "reconstruction_err": float(nmf.reconstruction_err_),
        "n_iter": int(nmf.n_iter_),
        "W_train": w_train,
        "W_test": w_test,
        "H": nmf.components_,
        "feature_names": list(x_train.columns),
    }


def save_nmf_tables(
    out_dir: Path,
    prefix: str,
    nmf: dict,
    train_ids: pd.DataFrame,
    test_ids: pd.DataFrame,
    train_labels: Sequence[str],
    test_labels: Sequence[str],
) -> None:
    factor_cols = [f"Factor_{i+1}" for i in range(nmf["k"])]
    w_train = pd.DataFrame(nmf["W_train"], columns=factor_cols)
    w_train.insert(0, "unique_row_id", train_ids["unique_row_id"].to_numpy())
    w_train.insert(0, "sample_id", train_ids["sample_id"].to_numpy())
    w_train["Diagnosis"] = list(train_labels)
    w_test = pd.DataFrame(nmf["W_test"], columns=factor_cols)
    w_test.insert(0, "unique_row_id", test_ids["unique_row_id"].to_numpy())
    w_test.insert(0, "sample_id", test_ids["sample_id"].to_numpy())
    w_test["Diagnosis"] = list(test_labels)
    w_train.to_csv(out_dir / f"{prefix}_W_train.csv", index=False)
    w_test.to_csv(out_dir / f"{prefix}_W_test.csv", index=False)
    pd.DataFrame(nmf["H"], columns=nmf["feature_names"], index=factor_cols).to_csv(
        out_dir / f"{prefix}_H.csv"
    )


def load_reaction(cohort_dir: Path, split: str) -> pd.DataFrame:
    path = cohort_dir / "processed" / "metabolitics_output" / split / "X_reaction_diff.csv"
    df = pd.read_csv(path)
    df["Sample ID"] = df["Sample ID"].astype(str)
    df["Diagnosis"] = df["Diagnosis"].astype(str)
    return df.reset_index(drop=True)


def run_nmf_cohort(project_root: Path, cohort: str) -> dict:
    started = time.perf_counter()
    cohort_dir = project_root / cohort
    out_dir = cohort_dir / "processed" / "nmf"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = load_split_table(cohort_dir, "train")
    test_ids = load_split_table(cohort_dir, "test")
    train_path = load_feature_matrix(cohort_dir, "train", "X_pathway")
    test_path = load_feature_matrix(cohort_dir, "test", "X_pathway")
    align_features(train_path, train_ids, f"{cohort}/train/X_pathway")
    align_features(test_path, test_ids, f"{cohort}/test/X_pathway")
    x_tr_path = x_from(train_path)
    x_te_path = x_from(test_path)
    x_tr_path, x_te_path = x_tr_path.align(x_te_path, join="inner", axis=1)
    path_nmf = fit_nmf(x_tr_path, x_te_path)
    save_nmf_tables(
        out_dir,
        "pathway",
        path_nmf,
        train_ids,
        test_ids,
        train_path["Diagnosis"],
        test_path["Diagnosis"],
    )

    train_rx = load_reaction(cohort_dir, "train")
    test_rx = load_reaction(cohort_dir, "test")
    align_features(train_rx, train_ids, f"{cohort}/train/X_reaction")
    align_features(test_rx, test_ids, f"{cohort}/test/X_reaction")
    x_tr_rx = x_from(train_rx)
    x_te_rx = x_from(test_rx)
    x_tr_rx, x_te_rx = x_tr_rx.align(x_te_rx, join="inner", axis=1)
    variances = x_tr_rx.var(axis=0).sort_values(ascending=False)
    keep = variances.head(min(REACTION_VAR_K, x_tr_rx.shape[1])).index
    pd.Series(keep, name="feature").to_csv(out_dir / "reaction_variance_features.csv", index=False)
    rx_nmf = fit_nmf(x_tr_rx.loc[:, keep], x_te_rx.loc[:, keep])
    save_nmf_tables(
        out_dir,
        "reaction_var500",
        rx_nmf,
        train_ids,
        test_ids,
        train_rx["Diagnosis"],
        test_rx["Diagnosis"],
    )
    payload = {
        "cohort": cohort,
        "unsupervised": True,
        "k": NMF_K,
        "pathway": {
            "n_features": int(x_tr_path.shape[1]),
            "reconstruction_err": path_nmf["reconstruction_err"],
            "n_iter": path_nmf["n_iter"],
        },
        "reaction_var500": {
            "n_features": int(len(keep)),
            "reconstruction_err": rx_nmf["reconstruction_err"],
            "n_iter": rx_nmf["n_iter"],
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    _json_dump(out_dir / "run_summary.json", payload)
    return payload


def evaluate_selected(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    val_fold: np.ndarray,
    class_labels: Sequence[str],
    task: str,
) -> dict:
    rf = rf_estimator(n_jobs=-1)
    cv = cv_scores_on_train(
        rf, x_train.to_numpy(dtype=float), y_train, val_fold, class_labels, task
    )
    rf.fit(x_train.to_numpy(dtype=float), y_train)
    pred_test = rf.predict(x_test.to_numpy(dtype=float))
    proba_test = _align_proba(rf, rf.predict_proba(x_test.to_numpy(dtype=float)), class_labels)
    test_metrics = compute_metrics(y_test, pred_test, proba_test, class_labels, task)
    return {
        "n_features": int(x_train.shape[1]),
        "cv": cv,
        "test": test_metrics,
        "test_pred": pred_test,
        "selected_features": list(x_train.columns),
    }


def cv_scores_foldwise_features(
    cohort_dir: Path,
    feature_set: str,
    selected_features: Sequence[str],
    class_labels: Sequence[str],
    task: str,
    n_splits: int = N_CV_SPLITS,
) -> dict:
    accs = []
    aucs = []
    n_used = []
    for fold in range(n_splits):
        tr_ids = load_cv_fold_ids(cohort_dir, fold, "cv_train")
        va_ids = load_cv_fold_ids(cohort_dir, fold, "cv_val")
        tr_feat = load_cv_fold_features(cohort_dir, fold, "cv_train", feature_set)
        va_feat = load_cv_fold_features(cohort_dir, fold, "cv_val", feature_set)
        align_features(tr_feat, tr_ids, f"fold{fold}/cv_train/{feature_set}")
        align_features(va_feat, va_ids, f"fold{fold}/cv_val/{feature_set}")
        x_tr = numeric_x(tr_feat)
        x_va = numeric_x(va_feat)
        keep = [c for c in selected_features if c in x_tr.columns and c in x_va.columns]
        if len(keep) < 2:
            continue
        y_tr = encode_labels(tr_feat["Diagnosis"], class_labels)
        y_va = encode_labels(va_feat["Diagnosis"], class_labels)
        rf = rf_estimator(n_jobs=-1)
        rf.fit(x_tr.loc[:, keep].to_numpy(dtype=float), y_tr)
        x_va_np = x_va.loc[:, keep].to_numpy(dtype=float)
        pred = rf.predict(x_va_np)
        proba = _align_proba(rf, rf.predict_proba(x_va_np), class_labels)
        accs.append(float(accuracy_score(y_va, pred)))
        n_used.append(len(keep))
        try:
            if task == "binary":
                if np.unique(y_va).size == 2:
                    aucs.append(float(roc_auc_score(y_va, proba[:, 1])))
            else:
                aucs.append(
                    float(
                        roc_auc_score(
                            y_va,
                            proba,
                            multi_class="ovr",
                            average="macro",
                            labels=list(range(len(class_labels))),
                        )
                    )
                )
        except ValueError:
            pass
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else None,
        "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
        "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        "std_roc_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
        "fold_accuracy": accs,
        "fold_roc_auc": aucs,
        "n_features_used": n_used,
        "n_folds_scored": len(accs),
    }


def cv_scores_foldwise_nmf(
    cohort_dir: Path,
    selected_features: Sequence[str],
    class_labels: Sequence[str],
    task: str,
    n_splits: int = N_CV_SPLITS,
) -> dict:
    accs = []
    aucs = []
    recs = []
    n_used = []
    factor_cols = [f"Factor_{i+1}" for i in range(NMF_K)]
    for fold in range(n_splits):
        tr_ids = load_cv_fold_ids(cohort_dir, fold, "cv_train")
        va_ids = load_cv_fold_ids(cohort_dir, fold, "cv_val")
        tr_feat = load_cv_fold_features(cohort_dir, fold, "cv_train", "X_reaction_diff")
        va_feat = load_cv_fold_features(cohort_dir, fold, "cv_val", "X_reaction_diff")
        align_features(tr_feat, tr_ids, f"fold{fold}/cv_train/X_reaction_diff")
        align_features(va_feat, va_ids, f"fold{fold}/cv_val/X_reaction_diff")
        x_tr = numeric_x(tr_feat)
        x_va = numeric_x(va_feat)
        keep = [c for c in selected_features if c in x_tr.columns and c in x_va.columns]
        if len(keep) < NMF_K:
            continue
        y_tr = encode_labels(tr_feat["Diagnosis"], class_labels)
        y_va = encode_labels(va_feat["Diagnosis"], class_labels)
        nmf = fit_nmf(x_tr.loc[:, keep], x_va.loc[:, keep])
        recs.append(nmf["reconstruction_err"])
        n_used.append(len(keep))
        rf = rf_estimator(n_jobs=-1)
        w_tr = np.asarray(nmf["W_train"], dtype=float)
        w_va = np.asarray(nmf["W_test"], dtype=float)
        rf.fit(w_tr, y_tr)
        pred = rf.predict(w_va)
        proba = _align_proba(rf, rf.predict_proba(w_va), class_labels)
        accs.append(float(accuracy_score(y_va, pred)))
        try:
            if task == "binary":
                if np.unique(y_va).size == 2:
                    aucs.append(float(roc_auc_score(y_va, proba[:, 1])))
            else:
                aucs.append(
                    float(
                        roc_auc_score(
                            y_va,
                            proba,
                            multi_class="ovr",
                            average="macro",
                            labels=list(range(len(class_labels))),
                        )
                    )
                )
        except ValueError:
            pass
    return {
        "mean_accuracy": float(np.mean(accs)) if accs else None,
        "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
        "mean_roc_auc": float(np.mean(aucs)) if aucs else None,
        "std_roc_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
        "fold_accuracy": accs,
        "fold_roc_auc": aucs,
        "fold_reconstruction_err": recs,
        "n_features_used": n_used,
        "n_folds_scored": len(accs),
        "k": NMF_K,
        "factor_cols": factor_cols,
    }


def run_classification_extras(
    project_root: Path, cohort: str, skip_reaction_rfe: bool, foldwise: bool = False
) -> dict:
    started = time.perf_counter()
    spec = CLASSIFICATION_SPECS[cohort]
    class_labels = spec["class_labels"]
    task = spec["task"]
    cohort_dir = project_root / cohort
    out_dir = cohort_dir / "processed" / "clf_extras"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_table(cohort_dir, "train")
    test_ids = load_split_table(cohort_dir, "test")
    train_path = load_feature_matrix(cohort_dir, "train", "X_pathway")
    test_path = load_feature_matrix(cohort_dir, "test", "X_pathway")
    align_features(train_path, train_ids, f"{cohort}/train/X_pathway")
    align_features(test_path, test_ids, f"{cohort}/test/X_pathway")
    x_tr_path = x_from(train_path)
    x_te_path = x_from(test_path)
    x_tr_path, x_te_path = x_tr_path.align(x_te_path, join="inner", axis=1)
    y_train = encode_labels(train_path["Diagnosis"], class_labels)
    y_test = encode_labels(test_path["Diagnosis"], class_labels)
    groups = train_ids["participant_id"].to_numpy() if spec["grouped"] else None
    val_fold = assign_cv_folds(
        labels=y_train,
        groups=groups,
        n_splits=N_CV_SPLITS,
        random_state=RANDOM_STATE,
        grouped=spec["grouped"],
    )
    rfe_cv = make_rfe_cv(spec["grouped"], y_train, groups, len(y_train))

    print(f"  LogReg all pathway (McNemar baseline)")
    x_tr_s, x_te_s = scale_xy(x_tr_path, x_te_path)
    lr = logreg()
    lr.fit(x_tr_s, y_train)
    lr_pred = lr.predict(x_te_s)
    lr_proba = _align_proba(lr, lr.predict_proba(x_te_s), class_labels)
    lr_test = compute_metrics(y_test, lr_pred, lr_proba, class_labels, task)

    print(f"  RF+RFE on pathway")
    path_selector = RFECV(
        estimator=rf_estimator(),
        step=1,
        cv=rfe_cv,
        scoring=rfe_scoring(task),
        min_features_to_select=5,
        n_jobs=-1,
    )
    path_selector.fit(x_tr_path.to_numpy(dtype=float), y_train)
    path_mask = path_selector.support_
    pd.Series(x_tr_path.columns[path_mask], name="feature").to_csv(
        out_dir / "rfe_pathway_features.csv", index=False
    )
    plot_rfecv(
        path_selector.cv_results_,
        out_dir / "rfecv_pathway_curve.png",
        f"{cohort} pathway RFECV",
    )
    path_eval = evaluate_selected(
        x_tr_path.loc[:, path_mask],
        x_te_path.loc[:, path_mask],
        y_train,
        y_test,
        val_fold,
        class_labels,
        task,
    )
    mcnemar_path = mcnemar_from_preds(y_test, path_eval["test_pred"], lr_pred)
    path_cv_for_table = path_eval["cv"]
    if foldwise:
        path_eval["cv_foldwise"] = cv_scores_foldwise_features(
            cohort_dir,
            "X_pathway",
            path_eval["selected_features"],
            class_labels,
            task,
        )
        path_cv_for_table = path_eval["cv_foldwise"]
        print(
            f"    foldwise CV AUC {path_eval['cv_foldwise']['mean_roc_auc']}"
        )
    print(
        f"    {path_eval['n_features']} features; "
        f"CV AUC {path_cv_for_table['mean_roc_auc']}; "
        f"McNemar p={mcnemar_path['p_value']:.4f}"
    )

    reaction_block = None
    nmf_block = None
    if not skip_reaction_rfe:
        print(f"  RF+RFE on reaction diffs")
        train_rx = load_reaction(cohort_dir, "train")
        test_rx = load_reaction(cohort_dir, "test")
        align_features(train_rx, train_ids, f"{cohort}/train/X_reaction")
        align_features(test_rx, test_ids, f"{cohort}/test/X_reaction")
        x_tr_rx = x_from(train_rx)
        x_te_rx = x_from(test_rx)
        x_tr_rx, x_te_rx = x_tr_rx.align(x_te_rx, join="inner", axis=1)
        coarse = RFE(rf_estimator(), n_features_to_select=500, step=500)
        coarse.fit(x_tr_rx.to_numpy(dtype=float), y_train)
        x_500 = x_tr_rx.loc[:, coarse.support_]
        fine = RFECV(
            estimator=rf_estimator(),
            step=50,
            cv=rfe_cv,
            scoring=rfe_scoring(task),
            min_features_to_select=50,
            n_jobs=-1,
        )
        fine.fit(x_500.to_numpy(dtype=float), y_train)
        rx_mask = x_tr_rx.columns.isin(x_500.columns[fine.support_])
        pd.Series(x_tr_rx.columns[rx_mask], name="feature").to_csv(
            out_dir / "rfe_reaction_features.csv", index=False
        )
        plot_rfecv(fine.cv_results_, out_dir / "rfecv_reaction_curve.png", f"{cohort} reaction RFECV")
        rx_eval = evaluate_selected(
            x_tr_rx.loc[:, rx_mask],
            x_te_rx.loc[:, rx_mask],
            y_train,
            y_test,
            val_fold,
            class_labels,
            task,
        )
        mcnemar_rx = mcnemar_from_preds(y_test, rx_eval["test_pred"], lr_pred)
        rx_cv_for_table = rx_eval["cv"]
        if foldwise:
            rx_eval["cv_foldwise"] = cv_scores_foldwise_features(
                cohort_dir,
                "X_reaction_diff",
                rx_eval["selected_features"],
                class_labels,
                task,
            )
            rx_cv_for_table = rx_eval["cv_foldwise"]
            print(
                f"    foldwise CV AUC {rx_eval['cv_foldwise']['mean_roc_auc']}"
            )
        reaction_block = {
            "rfe": {"n_coarse": int(coarse.support_.sum()), "n_optimal": int(fine.n_features_)},
            "metrics": {
                k: v for k, v in rx_eval.items() if k not in {"test_pred", "selected_features"}
            },
            "mcnemar_vs_logreg_pathway": mcnemar_rx,
        }
        print(
            f"    {rx_eval['n_features']} features; "
            f"CV AUC {rx_cv_for_table['mean_roc_auc']}; "
            f"McNemar p={mcnemar_rx['p_value']:.4f}"
        )
        print(f"  NMF k=5 on RFE-selected reaction features")
        nmf = fit_nmf(x_tr_rx.loc[:, rx_mask], x_te_rx.loc[:, rx_mask])
        save_nmf_tables(
            out_dir,
            "nmf",
            nmf,
            train_ids,
            test_ids,
            train_path["Diagnosis"],
            test_path["Diagnosis"],
        )
        nmf_eval = evaluate_selected(
            pd.DataFrame(nmf["W_train"], columns=[f"Factor_{i+1}" for i in range(NMF_K)]),
            pd.DataFrame(nmf["W_test"], columns=[f"Factor_{i+1}" for i in range(NMF_K)]),
            y_train,
            y_test,
            val_fold,
            class_labels,
            task,
        )
        nmf_cv_for_table = nmf_eval["cv"]
        nmf_block = {
            "k": NMF_K,
            "n_features": int(rx_mask.sum()),
            "reconstruction_err": nmf["reconstruction_err"],
            "n_iter": nmf["n_iter"],
            "cv": nmf_eval["cv"],
            "test": nmf_eval["test"],
        }
        if foldwise:
            nmf_block["cv_outer_train_fva"] = nmf_eval["cv"]
            nmf_block["cv"] = cv_scores_foldwise_nmf(
                cohort_dir,
                rx_eval["selected_features"],
                class_labels,
                task,
            )
            nmf_cv_for_table = nmf_block["cv"]
            print(f"    foldwise NMF CV AUC {nmf_cv_for_table['mean_roc_auc']}")
        print(
            f"    NMF RF CV AUC {nmf_cv_for_table['mean_roc_auc']}; "
            f"test AUC {nmf_eval['test']['roc_auc']}"
        )

    payload = {
        "cohort": cohort,
        "task": task,
        "class_labels": list(class_labels),
        "grouped": spec["grouped"],
        "metabolitics_mode": (
            "fit_train_transform_holdout" if foldwise else "outer_train_fit_test_transform"
        ),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "elapsed_seconds": time.perf_counter() - started,
        "logreg_all_pathway_test": lr_test,
        "rf_rfe_pathway": {
            "n_features": path_eval["n_features"],
            "cv": path_cv_for_table,
            "cv_outer_train_fva": path_eval["cv"],
            "test": path_eval["test"],
            "mcnemar_vs_logreg_all_pathway": mcnemar_path,
        },
        "rf_rfe_reaction": reaction_block,
        "nmf": nmf_block,
    }
    _json_dump(out_dir / "run_summary.json", payload)
    auc_key = "roc_auc" if task == "binary" else "roc_auc_macro_ovr"
    metric_rows = [
        {
            "cohort": cohort,
            "model": "logreg_all_pathway",
            "n_features": int(x_tr_path.shape[1]),
            "cv_auc_mean": None,
            "cv_acc_mean": None,
            "test_auc": lr_test[auc_key],
            "test_acc": lr_test["accuracy"],
            "mcnemar_p_vs_logreg_pathway": None,
        },
        {
            "cohort": cohort,
            "model": "rf_rfe_pathway",
            "n_features": path_eval["n_features"],
            "cv_auc_mean": path_cv_for_table["mean_roc_auc"],
            "cv_acc_mean": path_cv_for_table["mean_accuracy"],
            "test_auc": path_eval["test"][auc_key],
            "test_acc": path_eval["test"]["accuracy"],
            "mcnemar_p_vs_logreg_pathway": mcnemar_path["p_value"],
        },
    ]
    if reaction_block is not None:
        metric_rows.append(
            {
                "cohort": cohort,
                "model": "rf_rfe_reaction",
                "n_features": reaction_block["metrics"]["n_features"],
                "cv_auc_mean": rx_cv_for_table["mean_roc_auc"],
                "cv_acc_mean": rx_cv_for_table["mean_accuracy"],
                "test_auc": reaction_block["metrics"]["test"][auc_key],
                "test_acc": reaction_block["metrics"]["test"]["accuracy"],
                "mcnemar_p_vs_logreg_pathway": reaction_block["mcnemar_vs_logreg_pathway"]["p_value"],
            }
        )
    if nmf_block is not None:
        metric_rows.append(
            {
                "cohort": cohort,
                "model": "nmf_k5_rf_on_rfe_reaction",
                "n_features": 5,
                "cv_auc_mean": nmf_cv_for_table["mean_roc_auc"],
                "cv_acc_mean": nmf_cv_for_table["mean_accuracy"],
                "test_auc": nmf_block["test"][auc_key],
                "test_acc": nmf_block["test"]["accuracy"],
                "mcnemar_p_vs_logreg_pathway": None,
            }
        )
    pd.DataFrame(metric_rows).to_csv(out_dir / "metrics.csv", index=False)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--classification-cohorts",
        nargs="+",
        default=list(CLASSIFICATION_SPECS),
        choices=list(CLASSIFICATION_SPECS),
    )
    parser.add_argument(
        "--nmf-cohorts",
        nargs="+",
        default=list(ALL_COHORTS),
        choices=list(ALL_COHORTS),
    )
    parser.add_argument("--skip-reaction-rfe", action="store_true")
    parser.add_argument("--skip-nmf", action="store_true")
    parser.add_argument("--skip-classification", action="store_true")
    parser.add_argument(
        "--foldwise",
        action="store_true",
        help="Score RF+RFE CV on per-fold train-fit / holdout-transform Metabolitics matrices",
    )
    args = parser.parse_args()
    results_dir = args.project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    nmf_rows = []
    if not args.skip_nmf:
        for cohort in args.nmf_cohorts:
            print(f"=== NMF {cohort} (unsupervised, all-cohort) ===")
            payload = run_nmf_cohort(args.project_root, cohort)
            nmf_rows.append(payload)
            print(
                f"  pathway recon {payload['pathway']['reconstruction_err']:.1f}; "
                f"reaction-var500 recon {payload['reaction_var500']['reconstruction_err']:.1f} "
                f"({payload['elapsed_seconds']:.1f}s)"
            )
        if nmf_rows:
            pd.DataFrame(
                [
                    {
                        "cohort": p["cohort"],
                        "k": p["k"],
                        "pathway_n_features": p["pathway"]["n_features"],
                        "pathway_reconstruction_err": p["pathway"]["reconstruction_err"],
                        "reaction_var500_n_features": p["reaction_var500"]["n_features"],
                        "reaction_var500_reconstruction_err": p["reaction_var500"]["reconstruction_err"],
                        "elapsed_seconds": p["elapsed_seconds"],
                    }
                    for p in nmf_rows
                ]
            ).to_csv(results_dir / "nmf_summaries.csv", index=False)

    clf_rows = []
    if not args.skip_classification:
        for cohort in args.classification_cohorts:
            print(f"=== Classification extras {cohort} ===")
            payload = run_classification_extras(
                args.project_root, cohort, args.skip_reaction_rfe, foldwise=args.foldwise
            )
            clf_rows.append(payload)
            print(f"  wrote {args.project_root / cohort / 'processed' / 'clf_extras'} ({payload['elapsed_seconds']:.1f}s)")

    if clf_rows:
        master = pd.concat(
            [
                pd.read_csv(args.project_root / p["cohort"] / "processed" / "clf_extras" / "metrics.csv")
                for p in clf_rows
            ],
            ignore_index=True,
        )
        master.to_csv(results_dir / "clf_extras_metrics.csv", index=False)


if __name__ == "__main__":
    main()
