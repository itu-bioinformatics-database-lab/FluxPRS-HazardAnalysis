#!/usr/bin/env python3
"""Nested-CV classifiers on saved 70/30 splits and Metabolitics outer folds.

Does not create new splits. Uses:
  {cohort}/processed/splits/{train,test}_ids.csv
  {cohort}/processed/cv/fold_ids.csv
  {cohort}/processed/metabolitics_output/cv_folds/fold*/{cv_train,cv_val}

Protocol:
  the five saved leakage-free Metabolitics folds are the outer CV folds;
  within each outer-training matrix, five inner folds choose the number of
  RF-ranked features (1..20, or 20..100 step 10 for all_data);
  inner folds reuse the outer-training Metabolitics transformation by design;
  outer validation matrices remain independently transformed;
  RF/LR are evaluated on all features and RF/LR/GB/Ada/Stacking/XGB on top-k;
  final k is selected across the five outer folds for the untouched 30% test.

ROSMAP-AD is binary; DIABETES (DiCAD) is 3-class on the same saved
participant-level 70/30 split. Both outer and inner DiCAD folds keep
participants grouped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
for path in (str(PREPROCESSING_DIR), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from cohort_splits import N_CV_SPLITS, RANDOM_STATE, assign_cv_folds  # noqa: E402
from modules.training.metrics import summarize_classification_folds  # noqa: E402
from collect_evaluation_results import collect_evaluation_results  # noqa: E402
from run_train_cv import (  # noqa: E402
    META_DROP,
    _align_proba,
    compute_metrics,
    encode_labels,
    load_cv_fold_features,
    load_cv_fold_ids,
    load_feature_matrix,
    load_split_table,
    to_builtin,
)

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    XGBClassifier = None  # type: ignore
    HAS_XGB = False

N_ESTIMATORS = 100

COHORT_SPECS = {
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

# Eight notebooks in Classifications/. "metabolite" = Metabolitics fold-changes.
FEATURE_SETS = [
    ("prs", ("prs",)),
    ("metabolites", ("fold_changes",)),
    ("prs_metabolites", ("prs", "fold_changes")),
    ("pathway", ("pathway",)),
    ("prs_pathway", ("prs", "pathway")),
    ("reaction", ("reaction",)),
    ("prs_reaction", ("prs", "reaction")),
    ("all_data", ("prs", "fold_changes", "pathway", "reaction")),
]


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")


def load_prs_matrix(cohort_dir: Path) -> pd.DataFrame:
    prs_root = cohort_dir / "Final_Prs_Scores"
    frames = []
    for pgs_dir in sorted(
        p.name for p in prs_root.iterdir() if p.is_dir() and p.name.startswith("PGS")
    ):
        path = prs_root / pgs_dir / "prs_scores.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path)
        id_col = part.columns[0]
        score_col = part.columns[1]
        part = part.rename(columns={id_col: "sample_id", score_col: pgs_dir})
        part["sample_id"] = part["sample_id"].astype(str)
        part = part.drop_duplicates("sample_id", keep="first")
        frames.append(part[["sample_id", pgs_dir]])
    if not frames:
        raise FileNotFoundError(f"No PRS files under {prs_root}")
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="sample_id", how="outer")
    return out


def load_layer(cohort_dir: Path, split: str, name: str) -> pd.DataFrame:
    if name == "fold_changes":
        return load_feature_matrix(cohort_dir, split, "fold_changes")
    if name == "pathway":
        return load_feature_matrix(cohort_dir, split, "X_pathway")
    if name == "reaction":
        return load_feature_matrix(cohort_dir, split, "X_reaction_diff")
    raise KeyError(name)


def attach_omics(ids: pd.DataFrame, feat: pd.DataFrame, source: str) -> tuple[pd.DataFrame, list[str]]:
    """Positional align: DiCAD repeats share sample_id, so a merge would explode."""
    if len(feat) != len(ids):
        raise ValueError(f"{source}: {len(feat)} feature rows vs {len(ids)} IDs")
    feat_ids = feat["Sample ID"].astype(str).to_numpy()
    split_ids = ids["sample_id"].astype(str).to_numpy()
    if not (feat_ids == split_ids).all():
        raise ValueError(f"{source}: Sample ID order does not match split sample_id")
    numeric = feat.drop(columns=list(META_DROP & set(feat.columns))).apply(
        pd.to_numeric, errors="coerce"
    )
    cols = list(numeric.columns)
    out = pd.concat([ids.reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)
    return out, cols


def assemble_xy(
    cohort_dir: Path,
    ids: pd.DataFrame,
    split: str,
    layers: tuple[str, ...],
    prs: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    merged = ids.reset_index(drop=True)
    feat_cols: list[str] = []
    for layer in layers:
        if layer == "prs":
            merged = merged.merge(prs, on="sample_id", how="left")
            if len(merged) != len(ids):
                raise ValueError(f"{split} PRS merge exploded {len(ids)} -> {len(merged)} rows")
            pgs_cols = [c for c in prs.columns if c != "sample_id"]
            feat_cols.extend(pgs_cols)
            continue
        feat = load_layer(cohort_dir, split, layer)
        block, new_cols = attach_omics(merged[ids.columns], feat, f"{cohort_dir.name}/{split}/{layer}")
        extra = block[new_cols]
        merged = pd.concat([merged.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        feat_cols.extend(new_cols)
    missing = [c for c in feat_cols if c not in merged.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing[:8]}")
    if len(merged) != len(ids):
        raise ValueError(f"{split} assemble: {len(merged)} rows vs {len(ids)} IDs")
    return merged.reset_index(drop=True), feat_cols


def load_outer_layer(cohort_dir: Path, fold: int, split: str, name: str) -> pd.DataFrame:
    feature_set = {
        "fold_changes": "fold_changes",
        "pathway": "X_pathway",
        "reaction": "X_reaction_diff",
    }.get(name)
    if feature_set is None:
        raise KeyError(name)
    manifest_path = (
        cohort_dir
        / "processed"
        / "metabolitics_output"
        / "cv_folds"
        / f"fold{fold}"
        / split
        / "run_manifest.json"
    )
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Metabolitics provenance: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_stage = "fit" if split == "cv_train" else "transform"
    if manifest.get("fit_scope") != "fit_train_transform_holdout":
        raise ValueError(f"{manifest_path}: unexpected fit_scope {manifest.get('fit_scope')}")
    if manifest.get("stage") != expected_stage:
        raise ValueError(
            f"{manifest_path}: stage={manifest.get('stage')} expected {expected_stage}"
        )
    if split == "cv_val" and manifest.get("healthy_reference_from") != "fit_split_only":
        raise ValueError(f"{manifest_path}: validation transformation was not train-fitted")
    return load_cv_fold_features(cohort_dir, fold, split, feature_set)


def assemble_outer_xy(
    cohort_dir: Path,
    fold: int,
    split: str,
    layers: tuple[str, ...],
    prs: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble one saved Metabolitics outer-fold matrix without changing rows."""
    ids = load_cv_fold_ids(cohort_dir, fold, split)
    merged = ids.reset_index(drop=True)
    feat_cols: list[str] = []
    for layer in layers:
        if layer == "prs":
            merged = merged.merge(prs, on="sample_id", how="left")
            if len(merged) != len(ids):
                raise ValueError(
                    f"fold{fold}/{split} PRS merge exploded {len(ids)} -> {len(merged)} rows"
                )
            feat_cols.extend([c for c in prs.columns if c != "sample_id"])
            continue
        feat = load_outer_layer(cohort_dir, fold, split, layer)
        block, new_cols = attach_omics(
            ids, feat, f"{cohort_dir.name}/fold{fold}/{split}/{layer}"
        )
        merged = pd.concat(
            [merged.reset_index(drop=True), block[new_cols].reset_index(drop=True)],
            axis=1,
        )
        feat_cols.extend(new_cols)
    if len(feat_cols) != len(set(feat_cols)):
        duplicates = pd.Series(feat_cols)[pd.Series(feat_cols).duplicated()].unique().tolist()
        raise ValueError(f"fold{fold}/{split}: duplicate feature names {duplicates[:8]}")
    return merged.reset_index(drop=True), feat_cols


def load_val_fold(cohort_dir: Path, ids: pd.DataFrame) -> np.ndarray:
    fold_ids = pd.read_csv(cohort_dir / "processed" / "cv" / "fold_ids.csv")
    fold_ids["unique_row_id"] = fold_ids["unique_row_id"].astype(str)
    one = fold_ids.drop_duplicates("unique_row_id")[["unique_row_id", "val_fold"]]
    merged = ids.merge(one, on="unique_row_id", how="left")
    if merged["val_fold"].isna().any():
        missing = merged.loc[merged["val_fold"].isna(), "unique_row_id"].tolist()[:8]
        raise ValueError(f"Samples missing val_fold: {missing}")
    val = merged["val_fold"].to_numpy(dtype=int)
    expected = set(range(N_CV_SPLITS))
    if set(val.tolist()) != expected:
        raise ValueError(f"val_fold values {sorted(set(val.tolist()))} != {sorted(expected)}")
    return val


def x_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)


def make_ada() -> AdaBoostClassifier:
    kwargs = dict(n_estimators=N_ESTIMATORS, learning_rate=1.0, random_state=RANDOM_STATE)
    try:
        return AdaBoostClassifier(algorithm="SAMME", **kwargs)
    except TypeError:
        return AdaBoostClassifier(**kwargs)


def make_stack() -> StackingClassifier:
    return StackingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ("rf", RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)),
            ("knn", KNeighborsClassifier()),
        ],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        passthrough=False,
    )


def model_factory(name: str, n_classes: int):
    factories = {
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "logistic_regression": lambda: LogisticRegression(
            random_state=RANDOM_STATE, max_iter=1000
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=N_ESTIMATORS, learning_rate=0.1, random_state=RANDOM_STATE
        ),
        "adaboost": make_ada,
        "stacking": make_stack,
    }
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost is not installed")

        def _xgb():
            kwargs = dict(
                n_estimators=N_ESTIMATORS,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
                verbosity=0,
            )
            if n_classes > 2:
                kwargs.update(objective="multi:softprob", eval_metric="mlogloss", num_class=n_classes)
            else:
                kwargs.update(eval_metric="logloss")
            return XGBClassifier(**kwargs)

        return _xgb
    return factories[name]


def feature_count_grid(n_features: int, set_name: str) -> list[int]:
    if n_features <= 1:
        return [n_features]
    if set_name == "all_data":
        counts = [k for k in range(20, min(n_features, 100) + 1, 10)]
        if not counts:
            counts = [n_features]
        return counts
    upper = min(n_features, 20)
    return list(range(1, upper + 1))


def score_split(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_labels: Sequence[str],
    task: str,
) -> dict:
    full = compute_metrics(y_true, y_pred, y_proba, class_labels, task)
    if task == "binary":
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        auc = full["roc_auc"]
    else:
        f1 = full["macro_f1"]
        auc = full["roc_auc_macro_ovr"]
    return {
        "accuracy": full["accuracy"],
        "f1": f1,
        "roc_auc": auc,
        "n": full["n"],
        "confusion_matrix": full["confusion_matrix"],
        "macro_f1": full["macro_f1"],
        "weighted_f1": full["weighted_f1"],
        "per_class": full["per_class"],
        "classification_report": full["classification_report"],
    }


def predict_fold(model, x_tr, y_tr, x_va, class_labels: Sequence[str]):
    model.fit(x_tr, y_tr)
    pred = model.predict(x_va)
    proba = _align_proba(model, model.predict_proba(x_va), class_labels)
    return pred, proba


def cv_scores(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    val_fold: np.ndarray,
    class_labels: Sequence[str],
    task: str,
) -> dict:
    accs, f1s, aucs = [], [], []
    factory = model_factory(model_name, len(class_labels))
    for fold in range(N_CV_SPLITS):
        tr, va = val_fold != fold, val_fold == fold
        pred, proba = predict_fold(factory(), x[tr], y[tr], x[va], class_labels)
        m = score_split(y[va], pred, proba, class_labels, task)
        accs.append(m["accuracy"])
        f1s.append(m["f1"])
        aucs.append(m["roc_auc"])
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "mean_roc_auc": float(np.nanmean(aucs)),
        "std_roc_auc": float(np.nanstd(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "fold_accuracy": accs,
        "fold_f1": f1s,
        "fold_roc_auc": aucs,
    }


def rf_importance(x: np.ndarray, y: np.ndarray, val_fold: np.ndarray, names: list[str]) -> pd.Series:
    imps = []
    for fold in range(N_CV_SPLITS):
        tr = val_fold != fold
        rf = RandomForestClassifier(n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(x[tr], y[tr])
        imps.append(rf.feature_importances_)
    return pd.Series(np.mean(imps, axis=0), index=names).sort_values(ascending=False)


def select_k(
    x: np.ndarray,
    y: np.ndarray,
    val_fold: np.ndarray,
    ranked: pd.Series,
    set_name: str,
    col_names: list[str],
    class_labels: Sequence[str],
    task: str,
) -> dict:
    names = list(ranked.index)
    col_index = {n: i for i, n in enumerate(col_names)}
    sweep = []
    for k in feature_count_grid(len(names), set_name):
        idx = [col_index[n] for n in names[:k]]
        cv = cv_scores("random_forest", x[:, idx], y, val_fold, class_labels, task)
        sweep.append(
            {"k": k, **{key: cv[key] for key in ("mean_accuracy", "mean_f1", "mean_roc_auc")}}
        )
    best = max(sweep, key=lambda r: (r["mean_accuracy"], r["mean_roc_auc"], -r["k"]))
    return {"best_k": int(best["k"]), "sweep": sweep, "ranked_features": names}


def rank_indices(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    return np.argsort(-model.feature_importances_, kind="stable")


def aggregate_fold_scores(folds: list[dict]) -> dict:
    return summarize_classification_folds(folds)


def fit_and_score(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    class_labels: Sequence[str],
    task: str,
) -> dict:
    pred, proba = predict_fold(
        model_factory(model_name, len(class_labels))(),
        x_train,
        y_train,
        x_valid,
        class_labels,
    )
    return score_split(y_valid, pred, proba, class_labels, task)


def choose_k_from_sweep(sweep: list[dict]) -> int:
    best = max(
        sweep,
        key=lambda row: (
            row["mean_accuracy"],
            row["mean_roc_auc"] if row["mean_roc_auc"] is not None else -np.inf,
            -row["k"],
        ),
    )
    return int(best["k"])


def nested_select_k(
    x_outer_train: np.ndarray,
    y_outer_train: np.ndarray,
    outer_train_ids: pd.DataFrame,
    set_name: str,
    class_labels: Sequence[str],
    task: str,
    grouped: bool,
    outer_fold: int,
) -> dict:
    """Choose k inside one outer fold.

    The matrix is already transformed by the corresponding outer-training
    Metabolitics job. Scaling, RF ranking, and classification are refitted in
    every inner fold. Reusing the outer Metabolitics transformation is the
    explicitly accepted approximation for inner tuning.
    """
    groups = (
        outer_train_ids["participant_id"].astype(str).to_numpy() if grouped else None
    )
    inner_fold = assign_cv_folds(
        labels=y_outer_train,
        groups=groups,
        n_splits=N_CV_SPLITS,
        random_state=RANDOM_STATE + outer_fold + 1,
        grouped=grouped,
    )
    grid = feature_count_grid(x_outer_train.shape[1], set_name)
    by_k: dict[int, list[dict]] = {k: [] for k in grid}
    for fold in range(N_CV_SPLITS):
        tr, va = inner_fold != fold, inner_fold == fold
        scaler = StandardScaler().fit(x_outer_train[tr])
        x_tr = np.nan_to_num(scaler.transform(x_outer_train[tr]))
        x_va = np.nan_to_num(scaler.transform(x_outer_train[va]))
        order = rank_indices(x_tr, y_outer_train[tr])
        for k in grid:
            by_k[k].append(
                fit_and_score(
                    "random_forest",
                    x_tr[:, order[:k]],
                    y_outer_train[tr],
                    x_va[:, order[:k]],
                    y_outer_train[va],
                    class_labels,
                    task,
                )
            )
    sweep = []
    for k in grid:
        summary = aggregate_fold_scores(by_k[k])
        sweep.append(
            {
                "k": int(k),
                "mean_accuracy": summary["mean_accuracy"],
                "std_accuracy": summary["std_accuracy"],
                "mean_f1": summary["mean_f1"],
                "std_f1": summary["std_f1"],
                "mean_roc_auc": summary["mean_roc_auc"],
                "std_roc_auc": summary["std_roc_auc"],
            }
        )
    return {
        "best_k": choose_k_from_sweep(sweep),
        "sweep": sweep,
        "inner_val_fold": inner_fold.tolist(),
        "grouped": grouped,
        "metabolitics_scope": "outer_train_reused_within_inner_cv",
    }


def test_metrics(model, x_tr, y_tr, x_te, y_te, class_labels: Sequence[str], task: str) -> dict:
    pred, proba = predict_fold(model, x_tr, y_tr, x_te, class_labels)
    return score_split(y_te, pred, proba, class_labels, task)


def flat_row(
    cohort: str, feature_set: str, model: str, selection: str, n_features: int, cv: dict, te: dict
) -> dict:
    return {
        "cohort": cohort,
        "feature_set": feature_set,
        "model": model,
        "selection": selection,
        "n_features": n_features,
        "cv_accuracy_mean": cv["mean_accuracy"],
        "cv_accuracy_std": cv["std_accuracy"],
        "cv_f1_mean": cv["mean_f1"],
        "cv_f1_std": cv["std_f1"],
        "cv_auc_mean": cv["mean_roc_auc"],
        "cv_auc_std": cv["std_roc_auc"],
        "test_accuracy": te["accuracy"],
        "test_f1": te["f1"],
        "test_auc": te["roc_auc"],
        "test_n": te["n"],
        "split": "saved_70_30_and_nested_5x5fold",
        "new_splits_created": False,
        "outer_metabolitics_leakage_free": True,
        "inner_metabolitics_reused_from_outer_train": True,
    }


def run_feature_set(
    cohort: str,
    spec: dict,
    cohort_dir: Path,
    train_ids: pd.DataFrame,
    test_ids: pd.DataFrame,
    prs: pd.DataFrame,
    set_name: str,
    layers: tuple[str, ...],
) -> dict:
    t0 = time.perf_counter()
    class_labels = spec["class_labels"]
    task = spec["task"]
    n_classes = len(class_labels)
    models_all = ("random_forest", "logistic_regression")
    models_k = [
        "random_forest",
        "logistic_regression",
        "gradient_boosting",
        "adaboost",
        "stacking",
    ]
    if HAS_XGB:
        models_k.append("xgboost")

    # Outer CV uses independently generated Metabolitics train/validation matrices.
    outer_model_folds: dict[str, list[dict]] = {
        **{f"{name}_all": [] for name in models_all},
        **{f"{name}_nested": [] for name in models_k},
    }
    outer_selections = []
    outer_candidate_scores: dict[int, list[dict]] = {}
    sweep_rows = []
    n_features_outer = []
    for outer_fold in range(N_CV_SPLITS):
        outer_t0 = time.perf_counter()
        tr_df, tr_cols = assemble_outer_xy(
            cohort_dir, outer_fold, "cv_train", layers, prs
        )
        va_df, va_cols = assemble_outer_xy(
            cohort_dir, outer_fold, "cv_val", layers, prs
        )
        common_outer = [c for c in tr_cols if c in set(va_cols)]
        if not common_outer:
            raise ValueError(f"{set_name}/outer{outer_fold}: no shared features")
        n_features_outer.append(len(common_outer))
        x_tr_raw = x_matrix(tr_df, common_outer)
        x_va_raw = x_matrix(va_df, common_outer)
        y_tr = encode_labels(tr_df["metabolitics_label"], class_labels)
        y_va = encode_labels(va_df["metabolitics_label"], class_labels)

        inner = nested_select_k(
            x_tr_raw,
            y_tr,
            tr_df,
            set_name,
            class_labels,
            task,
            bool(spec["grouped"]),
            outer_fold,
        )
        selected_k = int(inner["best_k"])
        for row in inner["sweep"]:
            sweep_rows.append(
                {
                    "cohort": cohort,
                    "feature_set": set_name,
                    "scope": "inner_selection",
                    "outer_fold": outer_fold,
                    **row,
                }
            )

        scaler = StandardScaler().fit(x_tr_raw)
        x_tr = np.nan_to_num(scaler.transform(x_tr_raw))
        x_va = np.nan_to_num(scaler.transform(x_va_raw))
        order = rank_indices(x_tr, y_tr)
        selected_names = [common_outer[i] for i in order[:selected_k]]

        # Candidate scores across independent outer folds select final k for
        # the untouched 30% test model. They are not used for nested CV scores.
        for k in feature_count_grid(len(common_outer), set_name):
            metric = fit_and_score(
                "random_forest",
                x_tr[:, order[:k]],
                y_tr,
                x_va[:, order[:k]],
                y_va,
                class_labels,
                task,
            )
            metric["outer_fold"] = outer_fold
            outer_candidate_scores.setdefault(k, []).append(metric)

        for name in models_all:
            metric = fit_and_score(
                name, x_tr, y_tr, x_va, y_va, class_labels, task
            )
            metric.update(
                {
                    "outer_fold": outer_fold,
                    "n_features": len(common_outer),
                    "selected_k": None,
                }
            )
            outer_model_folds[f"{name}_all"].append(metric)
        for name in models_k:
            metric = fit_and_score(
                name,
                x_tr[:, order[:selected_k]],
                y_tr,
                x_va[:, order[:selected_k]],
                y_va,
                class_labels,
                task,
            )
            metric.update(
                {
                    "outer_fold": outer_fold,
                    "n_features": selected_k,
                    "selected_k": selected_k,
                }
            )
            outer_model_folds[f"{name}_nested"].append(metric)

        outer_selections.append(
            {
                "outer_fold": outer_fold,
                "selected_k": selected_k,
                "selected_features": selected_names,
                "inner_selection": inner,
                "n_outer_train": int(len(tr_df)),
                "n_outer_validation": int(len(va_df)),
                "n_features_available": int(len(common_outer)),
                "elapsed_seconds": float(time.perf_counter() - outer_t0),
            }
        )
        print(
            f"      outer fold {outer_fold + 1}/{N_CV_SPLITS}: "
            f"train={len(tr_df)} val={len(va_df)} p={len(common_outer)} "
            f"inner-selected k={selected_k}",
            flush=True,
        )

    final_sweep = []
    for k, fold_metrics in sorted(outer_candidate_scores.items()):
        summary_k = aggregate_fold_scores(fold_metrics)
        row = {
            "k": int(k),
            "mean_accuracy": summary_k["mean_accuracy"],
            "std_accuracy": summary_k["std_accuracy"],
            "mean_f1": summary_k["mean_f1"],
            "std_f1": summary_k["std_f1"],
            "mean_roc_auc": summary_k["mean_roc_auc"],
            "std_roc_auc": summary_k["std_roc_auc"],
        }
        final_sweep.append(row)
        sweep_rows.append(
            {
                "cohort": cohort,
                "feature_set": set_name,
                "scope": "final_k_outer_folds",
                "outer_fold": None,
                **row,
            }
        )
    final_k = choose_k_from_sweep(final_sweep)

    # Final model uses full train-fit features and evaluates the held-out test once.
    train_df, feat_cols = assemble_xy(cohort_dir, train_ids, "train", layers, prs)
    test_df, test_cols = assemble_xy(cohort_dir, test_ids, "test", layers, prs)
    common = [c for c in feat_cols if c in set(test_cols)]
    if not common:
        raise ValueError(f"{set_name}: no shared full-train/test features")
    scaler = StandardScaler().fit(x_matrix(train_df, common))
    x_train = np.nan_to_num(scaler.transform(x_matrix(train_df, common)))
    x_test = np.nan_to_num(scaler.transform(x_matrix(test_df, common)))
    y_train = encode_labels(train_df["metabolitics_label"], class_labels)
    y_test = encode_labels(test_df["metabolitics_label"], class_labels)
    final_order = rank_indices(x_train, y_train)
    final_idx = final_order[:final_k]
    final_features = [common[i] for i in final_idx]

    rows = []
    models_out = {}
    for name in models_all:
        cv = aggregate_fold_scores(outer_model_folds[f"{name}_all"])
        te = test_metrics(
            model_factory(name, n_classes)(),
            x_train,
            y_train,
            x_test,
            y_test,
            class_labels,
            task,
        )
        models_out[f"{name}_all"] = {
            "n_features": len(common),
            "selection": "all",
            "cv": cv,
            "test": te,
        }
        rows.append(flat_row(cohort, set_name, name, "all", len(common), cv, te))
    for name in models_k:
        cv = aggregate_fold_scores(outer_model_folds[f"{name}_nested"])
        te = test_metrics(
            model_factory(name, n_classes)(),
            x_train[:, final_idx],
            y_train,
            x_test[:, final_idx],
            y_test,
            class_labels,
            task,
        )
        models_out[f"{name}_nested_top{final_k}"] = {
            "n_features": final_k,
            "outer_selected_k": [row["selected_k"] for row in outer_selections],
            "selection": f"nested_top_{final_k}",
            "cv": cv,
            "test": te,
        }
        rows.append(
            flat_row(
                cohort,
                set_name,
                name,
                f"nested_top_{final_k}",
                final_k,
                cv,
                te,
            )
        )

    elapsed_seconds = float(time.perf_counter() - t0)
    for row in sweep_rows:
        row["reconstruction_error"] = None
        row["error"] = ""
        row["run_elapsed_seconds"] = elapsed_seconds

    return {
        "feature_set": set_name,
        "layers": list(layers),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_features_all": len(common),
        "n_features_outer_folds": n_features_outer,
        "best_k": final_k,
        "k_selection": {
            "protocol": "nested_inner_selection_with_final_k_selected_across_outer_folds",
            "inner_metabolitics_approximation": (
                "Inner folds reuse the corresponding outer-training Metabolitics "
                "transformation; scaling, RF ranking, and classifiers are inner-fold fitted."
            ),
            "outer_selections": outer_selections,
            "final_k": final_k,
            "final_sweep": final_sweep,
            "final_selected_features": final_features,
        },
        "models": models_out,
        "timing": {"elapsed_seconds": elapsed_seconds},
        "errors": [],
        "reconstruction_error": None,
        "table_rows": rows,
        "sweep_rows": sweep_rows,
    }


def run_cohort(root: Path, cohort: str, wanted: set[str] | None) -> pd.DataFrame:
    spec = COHORT_SPECS[cohort]
    cohort_dir = root / cohort
    out_dir = cohort_dir / "processed" / "clf_notebook"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_table(cohort_dir, "train")
    test_ids = load_split_table(cohort_dir, "test")
    prs = load_prs_matrix(cohort_dir)
    n_people_train = int(train_ids["participant_id"].nunique())
    n_people_test = int(test_ids["participant_id"].nunique())
    print(
        f"{cohort} saved splits train={len(train_ids)} ({n_people_train} people) "
        f"test={len(test_ids)} ({n_people_test} people) "
        f"PRS={prs.shape[1] - 1} task={spec['task']} xgboost={HAS_XGB} new_splits=False",
        flush=True,
    )

    all_rows = []
    all_sweep_rows = []
    summaries = []
    t0 = time.perf_counter()
    for set_name, layers in FEATURE_SETS:
        if wanted is not None and set_name not in wanted:
            continue
        print(f"  feature set {set_name} layers={layers}", flush=True)
        block = run_feature_set(
            cohort, spec, cohort_dir, train_ids, test_ids, prs, set_name, layers
        )
        summaries.append(
            {k: v for k, v in block.items() if k not in {"table_rows", "sweep_rows"}}
        )
        all_rows.extend(block["table_rows"])
        all_sweep_rows.extend(block["sweep_rows"])
        pd.DataFrame(all_rows).to_csv(out_dir / "metrics.csv", index=False)
        pd.DataFrame(all_sweep_rows).to_csv(out_dir / "sweep_results.csv", index=False)
        print(
            f"    n_train={block['n_train']} n_test={block['n_test']} "
            f"p={block['n_features_all']} k={block['best_k']} "
            f"{block['timing']['elapsed_seconds']:.1f}s",
            flush=True,
        )

    table = pd.DataFrame(all_rows)
    table.to_csv(out_dir / "metrics.csv", index=False)
    pd.DataFrame(all_sweep_rows).to_csv(out_dir / "sweep_results.csv", index=False)
    summary = {
        "cohort": cohort,
        "analysis": "nested_notebook_classifiers",
        "protocol": "saved_70_30_with_metabolitics_outer_5fold_and_inner_5fold_k_selection",
        "metabolitics_mode": "outer_fit_train_transform_holdout_inner_reuses_outer_train",
        "task": spec["task"],
        "class_labels": list(spec["class_labels"]),
        "n_outer_splits": N_CV_SPLITS,
        "n_inner_splits": N_CV_SPLITS,
        "outer_metabolitics_leakage_free": True,
        "inner_metabolitics_reused_from_outer_train": True,
        "random_state": RANDOM_STATE,
        "new_splits_created": False,
        "xgboost_available": HAS_XGB,
        "train_n": int(len(train_ids)),
        "test_n": int(len(test_ids)),
        "train_n_participants": n_people_train,
        "test_n_participants": n_people_test,
        "feature_sets": summaries,
        "classification_metrics_file": str(out_dir / "metrics.csv"),
        "sweep_results_file": str(out_dir / "sweep_results.csv"),
        "reconstruction_metrics": None,
        "errors": [],
        "timing": {"elapsed_seconds": float(time.perf_counter() - t0)},
    }
    _json_dump(out_dir / "run_summary.json", summary)
    print(
        f"Wrote {out_dir / 'metrics.csv'}  {len(table)} rows  "
        f"{summary['timing']['elapsed_seconds']:.1f}s"
    )
    return table


def write_combined(root: Path) -> None:
    frames = []
    for cohort in COHORT_SPECS:
        path = root / cohort / "processed" / "clf_notebook" / "metrics.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    table = pd.concat(frames, ignore_index=True)
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    table.to_csv(results_dir / "clf_notebook_metrics.csv", index=False)
    print(f"Wrote {results_dir / 'clf_notebook_metrics.csv'}  {len(table)} rows")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cohorts", nargs="*", default=list(COHORT_SPECS))
    parser.add_argument("--feature-sets", nargs="*", default=None)
    args = parser.parse_args()
    wanted = set(args.feature_sets) if args.feature_sets else None
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    for cohort in args.cohorts:
        if cohort not in COHORT_SPECS:
            raise ValueError(f"Unknown cohort {cohort}; expected {list(COHORT_SPECS)}")
        run_cohort(args.project_root, cohort, wanted)
    write_combined(args.project_root)
    collect_evaluation_results(args.project_root)


if __name__ == "__main__":
    main()
