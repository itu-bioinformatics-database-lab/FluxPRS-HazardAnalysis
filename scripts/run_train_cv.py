#!/usr/bin/env python3
"""5-fold CV plus held-out test metrics and SHAP for ROSMAP (binary) and DiCAD (3-class).

Uses already-computed Metabolitics train/test matrices. Does not re-run FVA.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
for path in (str(PREPROCESSING_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cohort_splits import N_CV_SPLITS, RANDOM_STATE, assign_cv_folds  # noqa: E402

META_DROP = {"Sample ID", "Diagnosis"}

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

FEATURE_FILES = {
    "X_pathway": "X_pathway.csv",
    "fold_changes": "fold_changes.csv",
    "X_reaction_diff": "X_reaction_diff.csv",
}
DEFAULT_FEATURE_SETS = ("X_pathway", "fold_changes")


def to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return obj


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")


def load_split_table(cohort_dir: Path, split: str) -> pd.DataFrame:
    path = cohort_dir / "processed" / "splits" / f"{split}_ids.csv"
    ids = pd.read_csv(path)
    ids["sample_id"] = ids["sample_id"].astype(str)
    ids["participant_id"] = ids["participant_id"].astype(str)
    ids["unique_row_id"] = ids["unique_row_id"].astype(str)
    ids["metabolitics_label"] = ids["metabolitics_label"].astype(str)
    return ids.reset_index(drop=True)


def load_feature_matrix(cohort_dir: Path, split: str, feature_set: str) -> pd.DataFrame:
    filename = FEATURE_FILES[feature_set]
    path = cohort_dir / "processed" / "metabolitics_output" / split / filename
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "Sample ID" not in df.columns or "Diagnosis" not in df.columns:
        raise ValueError(f"{path} must contain Sample ID and Diagnosis")
    df = df.copy()
    df["Sample ID"] = df["Sample ID"].astype(str)
    df["Diagnosis"] = df["Diagnosis"].astype(str)
    return df.reset_index(drop=True)


def load_cv_fold_ids(cohort_dir: Path, fold: int, split: str) -> pd.DataFrame:
    path = cohort_dir / "processed" / "cv_folds" / "ids" / f"fold{fold}_{split}_ids.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    ids = pd.read_csv(path)
    ids["sample_id"] = ids["sample_id"].astype(str)
    ids["participant_id"] = ids["participant_id"].astype(str)
    ids["unique_row_id"] = ids["unique_row_id"].astype(str)
    ids["metabolitics_label"] = ids["metabolitics_label"].astype(str)
    return ids.reset_index(drop=True)


def load_cv_fold_features(
    cohort_dir: Path, fold: int, split: str, feature_set: str
) -> pd.DataFrame:
    filename = FEATURE_FILES[feature_set]
    path = (
        cohort_dir
        / "processed"
        / "metabolitics_output"
        / "cv_folds"
        / f"fold{fold}"
        / split
        / filename
    )
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "Sample ID" not in df.columns or "Diagnosis" not in df.columns:
        raise ValueError(f"{path} must contain Sample ID and Diagnosis")
    df = df.copy()
    df["Sample ID"] = df["Sample ID"].astype(str)
    df["Diagnosis"] = df["Diagnosis"].astype(str)
    return df.reset_index(drop=True)


def numeric_x(df: pd.DataFrame) -> pd.DataFrame:
    drop = META_DROP & set(df.columns)
    return df.drop(columns=list(drop)).apply(pd.to_numeric, errors="coerce").fillna(0.0)


def missing_foldwise_outputs(
    cohort_dir: Path, feature_sets: Sequence[str], n_splits: int
) -> list[str]:
    missing = []
    for fold in range(n_splits):
        for split in ("cv_train", "cv_val"):
            for feature_set in feature_sets:
                filename = FEATURE_FILES[feature_set]
                path = (
                    cohort_dir
                    / "processed"
                    / "metabolitics_output"
                    / "cv_folds"
                    / f"fold{fold}"
                    / split
                    / filename
                )
                if not path.exists():
                    missing.append(str(path))
    return missing


def align_features(feat: pd.DataFrame, ids: pd.DataFrame, source: str) -> None:
    if len(feat) != len(ids):
        raise ValueError(
            f"{source}: {len(feat)} feature rows vs {len(ids)} split IDs"
        )
    feat_ids = feat["Sample ID"].astype(str)
    split_ids = ids["sample_id"].astype(str)
    if not (feat_ids.to_numpy() == split_ids.to_numpy()).all():
        if feat_ids.nunique() != len(feat) or split_ids.nunique() != len(ids):
            raise ValueError(
                f"{source}: Sample ID order does not match split IDs, "
                "and IDs are not unique so a merge is unsafe"
            )
        raise ValueError(f"{source}: Sample ID order does not match split IDs")
    feat_labels = feat["Diagnosis"].astype(str)
    id_labels = ids["metabolitics_label"].astype(str)
    if not (feat_labels.to_numpy() == id_labels.to_numpy()).all():
        raise ValueError(f"{source}: Diagnosis does not match metabolitics_label")


def encode_labels(raw: Sequence[str], class_labels: Sequence[str]) -> np.ndarray:
    mapping = {label: i for i, label in enumerate(class_labels)}
    encoded = np.array([mapping.get(str(v)) for v in raw], dtype=object)
    if np.any(encoded == None):  # noqa: E711
        unknown = sorted({str(v) for v, e in zip(raw, encoded) if e is None})
        raise ValueError(f"Unexpected labels {unknown}; expected {list(class_labels)}")
    return encoded.astype(int)


def make_model() -> LogisticRegression:
    return LogisticRegression(C=1.0, random_state=RANDOM_STATE, max_iter=1000)


def scale_split(
    x_train: pd.DataFrame, x_other: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train.to_numpy(dtype=float))
    x_other_s = scaler.transform(x_other.to_numpy(dtype=float))
    x_train_s = np.nan_to_num(x_train_s, nan=0.0, posinf=0.0, neginf=0.0)
    x_other_s = np.nan_to_num(x_other_s, nan=0.0, posinf=0.0, neginf=0.0)
    return x_train_s, x_other_s, scaler


def _safe_auc(y_true: np.ndarray, y_proba: np.ndarray, class_labels: Sequence[str], task: str):
    labels = list(range(len(class_labels)))
    present = np.unique(y_true)
    out = {
        "roc_auc": None,
        "roc_auc_macro_ovr": None,
        "roc_auc_per_class": {lab: None for lab in class_labels},
    }
    try:
        if task == "binary":
            if len(present) < 2:
                return out
            auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            out["roc_auc"] = auc
            out["roc_auc_macro_ovr"] = auc
            out["roc_auc_per_class"][class_labels[1]] = auc
        else:
            if len(present) < 2:
                return out
            out["roc_auc_macro_ovr"] = float(
                roc_auc_score(
                    y_true,
                    y_proba,
                    multi_class="ovr",
                    average="macro",
                    labels=labels,
                )
            )
            per = roc_auc_score(
                y_true,
                y_proba,
                multi_class="ovr",
                average=None,
                labels=labels,
            )
            out["roc_auc"] = out["roc_auc_macro_ovr"]
            for i, lab in enumerate(class_labels):
                out["roc_auc_per_class"][lab] = float(per[i]) if np.isfinite(per[i]) else None
    except ValueError:
        return out
    return out


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_labels: Sequence[str],
    task: str,
) -> dict:
    labels = list(range(len(class_labels)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_labels),
        output_dict=True,
        zero_division=0,
    )
    aucs = _safe_auc(y_true, y_proba, class_labels, task)
    per_class = {}
    for i, lab in enumerate(class_labels):
        y_bin = (y_true == i).astype(int)
        pred_bin = (y_pred == i).astype(int)
        per_class[lab] = {
            "n": int((y_true == i).sum()),
            "precision": float(precision_score(y_bin, pred_bin, zero_division=0)),
            "recall": float(recall_score(y_bin, pred_bin, zero_division=0)),
            "f1": float(f1_score(y_bin, pred_bin, zero_division=0)),
            "roc_auc_ovr": aucs["roc_auc_per_class"][lab],
        }
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": aucs["roc_auc"],
        "roc_auc_macro_ovr": aucs["roc_auc_macro_ovr"],
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "label_counts": {lab: int((y_true == i).sum()) for i, lab in enumerate(class_labels)},
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(class_labels),
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "classification_report": report,
    }


def prediction_frame(
    ids: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_labels: Sequence[str],
    extra: Optional[dict] = None,
) -> pd.DataFrame:
    rows = {
        "sample_id": ids["sample_id"].to_numpy(),
        "unique_row_id": ids["unique_row_id"].to_numpy(),
        "participant_id": ids["participant_id"].to_numpy(),
        "y_true": [class_labels[i] for i in y_true],
        "y_pred": [class_labels[i] for i in y_pred],
    }
    if extra:
        rows.update(extra)
    for i, lab in enumerate(class_labels):
        rows[f"proba_{lab}"] = y_proba[:, i]
    return pd.DataFrame(rows)


def fold_id_frame(ids: pd.DataFrame, val_fold: np.ndarray, n_splits: int) -> pd.DataFrame:
    rows = []
    for fold in range(n_splits):
        split = np.where(val_fold == fold, "cv_val", "cv_train")
        part = pd.DataFrame(
            {
                "sample_id": ids["sample_id"].to_numpy(),
                "unique_row_id": ids["unique_row_id"].to_numpy(),
                "participant_id": ids["participant_id"].to_numpy(),
                "diagnosis": ids["metabolitics_label"].to_numpy(),
                "fold": fold,
                "split": split,
                "val_fold": val_fold,
            }
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def compute_shap_tables(
    model: LogisticRegression,
    x_train: np.ndarray,
    x_test: np.ndarray,
    feature_names: Sequence[str],
    class_labels: Sequence[str],
    sample_ids: pd.DataFrame,
):
    import shap

    masker = shap.maskers.Independent(x_train, max_samples=int(x_train.shape[0]))
    explainer = shap.LinearExplainer(model, masker)
    try:
        explanation = explainer(x_test)
        values = np.asarray(explanation.values)
    except Exception:
        values = explainer.shap_values(x_test)
    if isinstance(values, list):
        values = np.stack([np.asarray(v) for v in values], axis=-1)
    values = np.asarray(values, dtype=float)
    n, p = x_test.shape
    n_classes = len(class_labels)
    if values.ndim == 2:
        if values.shape != (n, p):
            raise ValueError(f"Unexpected binary SHAP shape {values.shape}")
        class_values = {class_labels[-1]: values}
        overall = np.abs(values).mean(axis=0)
    elif values.ndim == 3:
        if values.shape == (n, p, n_classes):
            stacked = values
        elif values.shape == (n, n_classes, p):
            stacked = np.transpose(values, (0, 2, 1))
        elif values.shape == (n_classes, n, p):
            stacked = np.transpose(values, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected multiclass SHAP shape {values.shape}")
        class_values = {lab: stacked[:, :, i] for i, lab in enumerate(class_labels)}
        overall = np.abs(stacked).mean(axis=(0, 2))
    else:
        raise ValueError(f"Unexpected SHAP ndim {values.ndim}")

    summary = pd.DataFrame(
        {
            "feature": list(feature_names),
            "mean_abs_shap": overall,
        }
    )
    per_class_tables = {}
    value_tables = {}
    for lab, arr in class_values.items():
        mean_abs = np.abs(arr).mean(axis=0)
        summary[f"mean_abs_shap_{lab}"] = mean_abs
        per_class_tables[lab] = (
            pd.DataFrame({"feature": list(feature_names), "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        value_df = pd.DataFrame(arr, columns=list(feature_names))
        value_df.insert(0, "participant_id", sample_ids["participant_id"].to_numpy())
        value_df.insert(0, "unique_row_id", sample_ids["unique_row_id"].to_numpy())
        value_df.insert(0, "sample_id", sample_ids["sample_id"].to_numpy())
        value_tables[lab] = value_df
    summary = summary.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    return summary, per_class_tables, value_tables


def plot_shap_bar(summary: pd.DataFrame, title: str, out_path: Path, top_n: int = 20) -> None:
    plot_df = summary.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df["feature"], plot_df["mean_abs_shap"])
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_feature_set(
    cohort: str,
    spec: dict,
    feature_set: str,
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    train_ids: pd.DataFrame,
    test_ids: pd.DataFrame,
    val_fold: np.ndarray,
    out_dir: Path,
    n_splits: int,
) -> tuple[list[dict], dict, pd.DataFrame, pd.DataFrame]:
    class_labels = spec["class_labels"]
    task = spec["task"]
    x_train = train_feat.drop(columns=list(META_DROP & set(train_feat.columns)))
    x_test = test_feat.drop(columns=list(META_DROP & set(test_feat.columns)))
    x_train, x_test = x_train.align(x_test, join="inner", axis=1)
    if x_train.shape[1] == 0:
        raise ValueError(f"{cohort}/{feature_set}: no shared numeric features")

    y_train = encode_labels(train_feat["Diagnosis"], class_labels)
    y_test = encode_labels(test_feat["Diagnosis"], class_labels)

    oof_pred = np.full(len(y_train), -1, dtype=int)
    oof_proba = np.zeros((len(y_train), len(class_labels)), dtype=float)
    fold_rows = []
    for fold in range(n_splits):
        tr_mask = val_fold != fold
        va_mask = val_fold == fold
        x_tr, x_va, _ = scale_split(x_train.loc[tr_mask], x_train.loc[va_mask])
        y_tr = y_train[tr_mask]
        y_va = y_train[va_mask]
        model = make_model()
        model.fit(x_tr, y_tr)
        proba_va = model.predict_proba(x_va)
        proba_va = _align_proba(model, proba_va, class_labels)
        pred_va = np.argmax(proba_va, axis=1)
        oof_pred[va_mask] = pred_va
        oof_proba[va_mask] = proba_va
        metrics = compute_metrics(y_va, pred_va, proba_va, class_labels, task)
        fold_rows.append(
            {
                "cohort": cohort,
                "feature_set": feature_set,
                "fold": fold,
                "n_train": int(tr_mask.sum()),
                "n_val": int(va_mask.sum()),
                "n_train_participants": int(train_ids.loc[tr_mask, "participant_id"].nunique()),
                "n_val_participants": int(train_ids.loc[va_mask, "participant_id"].nunique()),
                **{
                    f"n_val_{lab}": int((y_va == i).sum())
                    for i, lab in enumerate(class_labels)
                },
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "roc_auc_macro_ovr": metrics["roc_auc_macro_ovr"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )

    oof_metrics = compute_metrics(y_train, oof_pred, oof_proba, class_labels, task)
    cv_pred = prediction_frame(
        train_ids,
        y_train,
        oof_pred,
        oof_proba,
        class_labels,
        extra={"feature_set": np.repeat(feature_set, len(y_train)), "val_fold": val_fold},
    )

    x_tr_all, x_te_all, _ = scale_split(x_train, x_test)
    final_model = make_model()
    final_model.fit(x_tr_all, y_train)
    proba_test = _align_proba(final_model, final_model.predict_proba(x_te_all), class_labels)
    pred_test = np.argmax(proba_test, axis=1)
    test_metrics = compute_metrics(y_test, pred_test, proba_test, class_labels, task)
    test_pred = prediction_frame(
        test_ids,
        y_test,
        pred_test,
        proba_test,
        class_labels,
        extra={"feature_set": np.repeat(feature_set, len(y_test))},
    )

    shap_dir = out_dir / feature_set
    shap_dir.mkdir(parents=True, exist_ok=True)
    summary, per_class_tables, value_tables = compute_shap_tables(
        final_model,
        x_tr_all,
        x_te_all,
        x_train.columns,
        class_labels,
        test_ids,
    )
    summary.to_csv(shap_dir / "test_shap.csv", index=False)
    plot_shap_bar(
        summary,
        f"{cohort} {feature_set} held-out mean |SHAP|",
        shap_dir / "test_shap_bar.png",
    )
    if task == "multiclass":
        for lab, table in per_class_tables.items():
            safe = lab.replace(" ", "_")
            table.to_csv(shap_dir / f"test_shap_class_{safe}.csv", index=False)
            value_tables[lab].to_csv(shap_dir / f"test_shap_values_{safe}.csv", index=False)
            plot_shap_bar(
                table,
                f"{cohort} {feature_set} held-out mean |SHAP| ({lab})",
                shap_dir / f"test_shap_bar_{safe}.png",
            )
    else:
        pos = class_labels[-1]
        value_tables[pos].to_csv(shap_dir / "test_shap_values.csv", index=False)

    auc_values = [row["roc_auc_macro_ovr"] for row in fold_rows if row["roc_auc_macro_ovr"] is not None]
    acc_values = [row["accuracy"] for row in fold_rows]
    summary_block = {
        "cohort": cohort,
        "feature_set": feature_set,
        "task": task,
        "n_features": int(x_train.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_labels": list(class_labels),
        "cv": {
            "n_splits": n_splits,
            "mean_roc_auc_macro_ovr": float(np.mean(auc_values)) if auc_values else None,
            "std_roc_auc_macro_ovr": float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else None,
            "mean_accuracy": float(np.mean(acc_values)),
            "std_accuracy": float(np.std(acc_values, ddof=1)) if len(acc_values) > 1 else None,
            "oof": oof_metrics,
        },
        "test": test_metrics,
        "shap_dir": str(shap_dir),
    }
    return fold_rows, summary_block, cv_pred, test_pred


def _align_proba(model: LogisticRegression, proba: np.ndarray, class_labels: Sequence[str]) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(class_labels)), dtype=float)
    class_to_col = {int(cls): i for i, cls in enumerate(model.classes_)}
    for i in range(len(class_labels)):
        if i in class_to_col:
            aligned[:, i] = proba[:, class_to_col[i]]
    return aligned


def run_cohort(
    project_root: Path,
    cohort: str,
    feature_sets: Sequence[str],
    n_splits: int,
    random_state: int,
) -> dict:
    spec = COHORT_SPECS[cohort]
    cohort_dir = project_root / cohort
    out_dir = cohort_dir / "processed" / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_table(cohort_dir, "train")
    test_ids = load_split_table(cohort_dir, "test")
    train_ref = load_feature_matrix(cohort_dir, "train", feature_sets[0])
    align_features(train_ref, train_ids, f"{cohort}/train/{feature_sets[0]}")
    y_for_folds = train_ref["Diagnosis"].astype(str)
    groups = train_ids["participant_id"].to_numpy() if spec["grouped"] else None
    val_fold = assign_cv_folds(
        labels=y_for_folds,
        groups=groups,
        n_splits=n_splits,
        random_state=random_state,
        grouped=spec["grouped"],
    )
    fold_ids = fold_id_frame(train_ids, val_fold, n_splits)
    fold_ids.to_csv(out_dir / "fold_ids.csv", index=False)

    fold_metric_rows = []
    summaries = []
    cv_preds = []
    test_preds = []
    started = time.perf_counter()
    for feature_set in feature_sets:
        train_feat = load_feature_matrix(cohort_dir, "train", feature_set)
        test_feat = load_feature_matrix(cohort_dir, "test", feature_set)
        align_features(train_feat, train_ids, f"{cohort}/train/{feature_set}")
        align_features(test_feat, test_ids, f"{cohort}/test/{feature_set}")
        fold_rows, summary_block, cv_pred, test_pred = run_feature_set(
            cohort,
            spec,
            feature_set,
            train_feat,
            test_feat,
            train_ids,
            test_ids,
            val_fold,
            out_dir,
            n_splits,
        )
        fold_metric_rows.extend(fold_rows)
        summaries.append(summary_block)
        cv_preds.append(cv_pred)
        test_preds.append(test_pred)

    elapsed = time.perf_counter() - started
    fold_metrics = pd.DataFrame(fold_metric_rows)
    fold_metrics.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    pd.concat(cv_preds, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
    pd.concat(test_preds, ignore_index=True).to_csv(out_dir / "test_predictions.csv", index=False)

    payload = {
        "cohort": cohort,
        "task": spec["task"],
        "class_labels": list(spec["class_labels"]),
        "grouped": spec["grouped"],
        "n_splits": n_splits,
        "random_state": random_state,
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_train_participants": int(train_ids["participant_id"].nunique()),
        "n_test_participants": int(test_ids["participant_id"].nunique()),
        "elapsed_seconds": elapsed,
        "feature_sets": summaries,
    }
    _json_dump(out_dir / "cv_summary.json", payload)
    test_only = {
        "cohort": cohort,
        "task": spec["task"],
        "elapsed_seconds": elapsed,
        "feature_sets": {block["feature_set"]: block["test"] for block in summaries},
    }
    _json_dump(out_dir / "test_metrics.json", test_only)
    return payload


def run_feature_set_foldwise(
    cohort: str,
    spec: dict,
    feature_set: str,
    cohort_dir: Path,
    train_ids: pd.DataFrame,
    test_ids: pd.DataFrame,
    train_feat: pd.DataFrame,
    test_feat: pd.DataFrame,
    out_dir: Path,
    n_splits: int,
) -> tuple[list[dict], dict, pd.DataFrame, pd.DataFrame]:
    class_labels = spec["class_labels"]
    task = spec["task"]
    y_train_full = encode_labels(train_feat["Diagnosis"], class_labels)
    y_test = encode_labels(test_feat["Diagnosis"], class_labels)
    n_classes = len(class_labels)
    oof_pred = np.full(len(train_ids), -1, dtype=int)
    oof_proba = np.zeros((len(train_ids), n_classes), dtype=float)
    fold_rows = []
    row_index = {uid: i for i, uid in enumerate(train_ids["unique_row_id"].astype(str))}

    for fold in range(n_splits):
        tr_ids = load_cv_fold_ids(cohort_dir, fold, "cv_train")
        va_ids = load_cv_fold_ids(cohort_dir, fold, "cv_val")
        tr_feat = load_cv_fold_features(cohort_dir, fold, "cv_train", feature_set)
        va_feat = load_cv_fold_features(cohort_dir, fold, "cv_val", feature_set)
        align_features(tr_feat, tr_ids, f"{cohort}/fold{fold}/cv_train/{feature_set}")
        align_features(va_feat, va_ids, f"{cohort}/fold{fold}/cv_val/{feature_set}")
        x_tr = numeric_x(tr_feat)
        x_va = numeric_x(va_feat)
        x_tr, x_va = x_tr.align(x_va, join="inner", axis=1)
        y_tr = encode_labels(tr_feat["Diagnosis"], class_labels)
        y_va = encode_labels(va_feat["Diagnosis"], class_labels)
        x_tr_s, x_va_s, _ = scale_split(x_tr, x_va)
        model = make_model()
        model.fit(x_tr_s, y_tr)
        proba_va = _align_proba(model, model.predict_proba(x_va_s), class_labels)
        pred_va = np.argmax(proba_va, axis=1)
        va_pos = [row_index[uid] for uid in va_ids["unique_row_id"].astype(str)]
        oof_pred[va_pos] = pred_va
        oof_proba[va_pos] = proba_va
        metrics = compute_metrics(y_va, pred_va, proba_va, class_labels, task)
        fold_rows.append(
            {
                "cohort": cohort,
                "feature_set": feature_set,
                "fold": fold,
                "n_train": int(len(y_tr)),
                "n_val": int(len(y_va)),
                "n_train_participants": int(tr_ids["participant_id"].nunique()),
                "n_val_participants": int(va_ids["participant_id"].nunique()),
                **{
                    f"n_val_{lab}": int((y_va == i).sum())
                    for i, lab in enumerate(class_labels)
                },
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "roc_auc_macro_ovr": metrics["roc_auc_macro_ovr"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "n_features_fold": int(x_tr.shape[1]),
            }
        )

    if (oof_pred < 0).any():
        raise RuntimeError(f"{cohort}/{feature_set}: some training rows missing OOF predictions")

    oof_metrics = compute_metrics(y_train_full, oof_pred, oof_proba, class_labels, task)
    val_fold = np.full(len(train_ids), -1, dtype=int)
    for fold in range(n_splits):
        va_ids = load_cv_fold_ids(cohort_dir, fold, "cv_val")
        for uid in va_ids["unique_row_id"].astype(str):
            val_fold[row_index[uid]] = fold
    cv_pred = prediction_frame(
        train_ids,
        y_train_full,
        oof_pred,
        oof_proba,
        class_labels,
        extra={"feature_set": np.repeat(feature_set, len(y_train_full)), "val_fold": val_fold},
    )

    x_train = numeric_x(train_feat)
    x_test = numeric_x(test_feat)
    x_train, x_test = x_train.align(x_test, join="inner", axis=1)
    x_tr_all, x_te_all, _ = scale_split(x_train, x_test)
    final_model = make_model()
    final_model.fit(x_tr_all, y_train_full)
    proba_test = _align_proba(final_model, final_model.predict_proba(x_te_all), class_labels)
    pred_test = np.argmax(proba_test, axis=1)
    test_metrics = compute_metrics(y_test, pred_test, proba_test, class_labels, task)
    test_pred = prediction_frame(
        test_ids,
        y_test,
        pred_test,
        proba_test,
        class_labels,
        extra={"feature_set": np.repeat(feature_set, len(y_test))},
    )

    shap_dir = out_dir / feature_set
    shap_dir.mkdir(parents=True, exist_ok=True)
    summary, per_class_tables, value_tables = compute_shap_tables(
        final_model,
        x_tr_all,
        x_te_all,
        x_train.columns,
        class_labels,
        test_ids,
    )
    summary.to_csv(shap_dir / "test_shap.csv", index=False)
    plot_shap_bar(
        summary,
        f"{cohort} {feature_set} held-out mean |SHAP|",
        shap_dir / "test_shap_bar.png",
    )
    if task == "multiclass":
        for lab, table in per_class_tables.items():
            safe = lab.replace(" ", "_")
            table.to_csv(shap_dir / f"test_shap_class_{safe}.csv", index=False)
            value_tables[lab].to_csv(shap_dir / f"test_shap_values_{safe}.csv", index=False)
            plot_shap_bar(
                table,
                f"{cohort} {feature_set} held-out mean |SHAP| ({lab})",
                shap_dir / f"test_shap_bar_{safe}.png",
            )
    else:
        pos = class_labels[-1]
        value_tables[pos].to_csv(shap_dir / "test_shap_values.csv", index=False)

    auc_values = [row["roc_auc_macro_ovr"] for row in fold_rows if row["roc_auc_macro_ovr"] is not None]
    acc_values = [row["accuracy"] for row in fold_rows]
    summary_block = {
        "cohort": cohort,
        "feature_set": feature_set,
        "task": task,
        "metabolitics_mode": "fit_train_transform_holdout",
        "n_features_outer_train": int(x_train.shape[1]),
        "n_train": int(len(y_train_full)),
        "n_test": int(len(y_test)),
        "class_labels": list(class_labels),
        "cv": {
            "n_splits": n_splits,
            "mean_roc_auc_macro_ovr": float(np.mean(auc_values)) if auc_values else None,
            "std_roc_auc_macro_ovr": float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else None,
            "mean_accuracy": float(np.mean(acc_values)),
            "std_accuracy": float(np.std(acc_values, ddof=1)) if len(acc_values) > 1 else None,
            "oof": oof_metrics,
        },
        "test": test_metrics,
        "shap_dir": str(shap_dir),
    }
    return fold_rows, summary_block, cv_pred, test_pred


def run_cohort_foldwise(
    project_root: Path,
    cohort: str,
    feature_sets: Sequence[str],
    n_splits: int,
    random_state: int,
) -> dict:
    spec = COHORT_SPECS[cohort]
    cohort_dir = project_root / cohort
    missing = missing_foldwise_outputs(cohort_dir, feature_sets, n_splits)
    if missing:
        preview = "\n  ".join(missing[:8])
        raise FileNotFoundError(
            f"{cohort}: {len(missing)} foldwise Metabolitics outputs missing. "
            f"Import TRUBA cv-fold jobs first. Missing include:\n  {preview}"
        )
    out_dir = cohort_dir / "processed" / "cv"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids = load_split_table(cohort_dir, "train")
    test_ids = load_split_table(cohort_dir, "test")
    fold_metric_rows = []
    summaries = []
    cv_preds = []
    test_preds = []
    started = time.perf_counter()
    for feature_set in feature_sets:
        train_feat = load_feature_matrix(cohort_dir, "train", feature_set)
        test_feat = load_feature_matrix(cohort_dir, "test", feature_set)
        align_features(train_feat, train_ids, f"{cohort}/train/{feature_set}")
        align_features(test_feat, test_ids, f"{cohort}/test/{feature_set}")
        fold_rows, summary_block, cv_pred, test_pred = run_feature_set_foldwise(
            cohort,
            spec,
            feature_set,
            cohort_dir,
            train_ids,
            test_ids,
            train_feat,
            test_feat,
            out_dir,
            n_splits,
        )
        fold_metric_rows.extend(fold_rows)
        summaries.append(summary_block)
        cv_preds.append(cv_pred)
        test_preds.append(test_pred)

    elapsed = time.perf_counter() - started
    pd.DataFrame(fold_metric_rows).to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    pd.concat(cv_preds, ignore_index=True).to_csv(out_dir / "cv_predictions.csv", index=False)
    pd.concat(test_preds, ignore_index=True).to_csv(out_dir / "test_predictions.csv", index=False)
    payload = {
        "cohort": cohort,
        "task": spec["task"],
        "class_labels": list(spec["class_labels"]),
        "grouped": spec["grouped"],
        "metabolitics_mode": "fit_train_transform_holdout",
        "n_splits": n_splits,
        "random_state": random_state,
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_train_participants": int(train_ids["participant_id"].nunique()),
        "n_test_participants": int(test_ids["participant_id"].nunique()),
        "elapsed_seconds": elapsed,
        "feature_sets": summaries,
    }
    _json_dump(out_dir / "cv_summary.json", payload)
    test_only = {
        "cohort": cohort,
        "task": spec["task"],
        "metabolitics_mode": "fit_train_transform_holdout",
        "elapsed_seconds": elapsed,
        "feature_sets": {block["feature_set"]: block["test"] for block in summaries},
    }
    _json_dump(out_dir / "test_metrics.json", test_only)
    return payload


def print_summary(payloads: Sequence[dict]) -> None:
    rows = []
    for payload in payloads:
        for block in payload["feature_sets"]:
            cv = block["cv"]
            test = block["test"]
            rows.append(
                {
                    "cohort": payload["cohort"],
                    "feature_set": block["feature_set"],
                    "cv_auc_mean": cv["mean_roc_auc_macro_ovr"],
                    "cv_auc_std": cv["std_roc_auc_macro_ovr"],
                    "cv_acc_mean": cv["mean_accuracy"],
                    "test_auc": test["roc_auc_macro_ovr"],
                    "test_acc": test["accuracy"],
                    "test_macro_f1": test["macro_f1"],
                    "n_train": block["n_train"],
                    "n_test": block["n_test"],
                }
            )
    table = pd.DataFrame(rows)
    print("\n=== CV + held-out summary ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=list(COHORT_SPECS),
        choices=list(COHORT_SPECS),
    )
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=list(DEFAULT_FEATURE_SETS),
        choices=list(FEATURE_FILES),
    )
    parser.add_argument("--n-splits", type=int, default=N_CV_SPLITS)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--foldwise",
        action="store_true",
        help=(
            "Use per-fold Metabolitics matrices (FC+FVA fit on cv_train healthy, "
            "applied to cv_val). Held-out test uses outer train-fit / test-transform."
        ),
    )
    args = parser.parse_args()

    payloads = []
    for cohort in args.cohorts:
        mode = "foldwise FVA" if args.foldwise else "outer-train FVA"
        print(f"=== {cohort} ({mode}) ===")
        runner = run_cohort_foldwise if args.foldwise else run_cohort
        payload = runner(
            args.project_root,
            cohort,
            args.feature_sets,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        payloads.append(payload)
        print(
            f"  wrote {args.project_root / cohort / 'processed' / 'cv'} "
            f"({payload['elapsed_seconds']:.1f}s)"
        )
    print_summary(payloads)


if __name__ == "__main__":
    main()
