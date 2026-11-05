#!/usr/bin/env python3
"""Hazard ratios on saved 70/30 splits and train-fit / holdout-transform Metabolitics.

Fits logistic risk scores on train only (no second split), scores train+test,
then Cox on the concatenated scores. DiCAD time-to-event is sampling age
(AgeDeath). Control/healthy = 0, all other diagnoses = 1.

Three Cox encodings are written:
  top10_high_risk           — primary: binary high-risk = top 10% of scores
  logit_z_per_sd            — train-only mean/SD, then apply to test; HR per 1 train-SD
  logit_z_per_sd_coxsample  — z-score on all samples entering that Cox model (Kaan);
                              HR per 1 SD in the Cox sample. Label-free scaling.
                     (PRS-only scores skip logit and are z-scored as linear)

Does not re-run FVA. Does not call analyse_with_logistic_regression.
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
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
for path in (str(PREPROCESSING_DIR), str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

from cohort_splits import RANDOM_STATE  # noqa: E402
from run_train_cv import (  # noqa: E402
    META_DROP,
    align_features,
    load_split_table,
    to_builtin,
)

CONTROL_LABELS = {"control", "healthy", "normal", "cn"}
META_EXTRA = {"Gender", "Race", "PMI", "Braak", "sex"}
FEATURE_DISPLAY = {
    "metabolite_levels": "Metabolite Levels",
    "fold_changes": "Fold Changes",
    "pathway": "Pathway Diff Scores",
    "reaction": "Reaction Diff Scores",
    "metabolite_pathway": "Metabolite Levels Pathway Combined",
    "metabolite_reaction": "Metabolite Levels Reaction Combined",
}
COHORTS = ("ROSMAP-AD", "MayoRNASeq-AD", "MayoRNASeq-PSP", "DIABETES")
COHORT_PLOT_LABELS = {
    "ROSMAP-AD": "ROSMAP",
    "MayoRNASeq-AD": "Mayo-AD",
    "MayoRNASeq-PSP": "Mayo-PSP",
    "DIABETES": "DiCAD",
}
DISTANCE_PLOT_FEATURES = [
    "Metabolite Levels",
    "Metabolite Levels + PRS",
    "Fold Changes",
    "Fold Changes + PRS",
    "Pathway Diff Scores",
    "Pathway Diff Scores + PRS",
    "Reaction Diff Scores",
    "Reaction Diff Scores + PRS",
    "Metabolite Levels Pathway Combined",
    "Metabolite Levels Pathway Combined + PRS",
    "Metabolite Levels Reaction Combined",
    "Metabolite Levels Reaction Combined + PRS",
    "Prs Score",
]
# Notebook / paper wide table: one row per PRS, paired omics and omics+PRS columns.
WIDE_FEATURE_ORDER = (
    "metabolite_levels",
    "pathway",
    "reaction",
    "metabolite_pathway",
    "metabolite_reaction",
)
STANDARD_LONG_COLUMNS = [
    "cohort",
    "feature_set",
    "display_name",
    "pgs_id",
    "with_prs",
    "n",
    "n_event",
    "n_censored",
    "n_high_risk",
    "n_high_risk_events",
    "hr",
    "log_hr",
    "p",
    "hr_ci_low",
    "hr_ci_high",
    "error",
]
DURATION_SOURCE = {
    "ROSMAP-AD": "age_first_ad_dx filled with age_at_visit_max; 90+ recoded to 90",
    "MayoRNASeq-AD": "AgeDeath; 90+ recoded to 90",
    "MayoRNASeq-PSP": "AgeDeath; 90+ recoded to 90",
    "DIABETES": "AgeDeath = sampling age at data collection; 90+ recoded to 90",
}
HR_COPY_TAG = {
    "ROSMAP-AD": "ROSMAPAD",
    "MayoRNASeq-AD": "MayoRNASeqAD",
    "MayoRNASeq-PSP": "MayoRNASeqPSP",
    "DIABETES": "DIABETES",
}
LOGIT_EPS = 1e-12


def _json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(to_builtin(payload), indent=2) + "\n")


def binary_event(labels: Sequence[str]) -> np.ndarray:
    return np.array(
        [0 if str(x).strip().lower() in CONTROL_LABELS else 1 for x in labels],
        dtype=int,
    )


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    drop = (META_DROP | META_EXTRA) & set(df.columns)
    out = df.drop(columns=list(drop)).apply(pd.to_numeric, errors="coerce")
    return out.reset_index(drop=True)


def load_csv_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Sample ID" not in df.columns or "Diagnosis" not in df.columns:
        raise ValueError(f"{path} must contain Sample ID and Diagnosis")
    df = df.copy()
    df["Sample ID"] = df["Sample ID"].astype(str)
    df["Diagnosis"] = df["Diagnosis"].astype(str)
    return df.reset_index(drop=True)


def load_metabolitics(cohort_dir: Path, split: str, filename: str) -> pd.DataFrame:
    path = cohort_dir / "processed" / "metabolitics_output" / split / filename
    if not path.exists():
        raise FileNotFoundError(path)
    return load_csv_matrix(path)


def load_metabolite_input(cohort_dir: Path, split: str) -> pd.DataFrame:
    path = cohort_dir / "processed" / "metabolitics_input" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_csv_matrix(path)


def recode_90plus(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.replace("90+", "90"), errors="coerce")


def build_duration(cohort: str, ids: pd.DataFrame, clinical: pd.DataFrame) -> pd.DataFrame:
    clin = clinical.copy()
    clin["Sample ID"] = clin["Sample ID"].astype(str)
    clin = clin.drop_duplicates(subset=["Sample ID"], keep="first")
    merged = ids.merge(clin, left_on="sample_id", right_on="Sample ID", how="left")
    if cohort == "ROSMAP-AD":
        onset = recode_90plus(merged["age_first_ad_dx"])
        visit = recode_90plus(merged["age_at_visit_max"])
        duration = onset.fillna(visit)
    else:
        duration = recode_90plus(merged["AgeDeath"])
    event = binary_event(ids["metabolitics_label"])
    out = ids.copy()
    out["duration"] = duration.to_numpy()
    out["event"] = event
    out["duration_source"] = DURATION_SOURCE[cohort]
    return out


def event_censor_table(cohort: str, scored: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, part in [("all", scored)] + [
        (name, scored.loc[scored["split"] == name]) for name in ("train", "test")
    ]:
        usable = part.dropna(subset=["duration"]).copy()
        usable = usable.loc[usable["duration"] > 0]
        n_event = int(usable["event"].sum())
        rows.append(
            {
                "cohort": cohort,
                "split": split,
                "n_rows": int(len(part)),
                "n_participants": int(part["participant_id"].nunique()),
                "n_with_duration": int(len(usable)),
                "n_event": n_event,
                "n_censored": int(len(usable) - n_event),
                "duration_min": float(usable["duration"].min()) if len(usable) else None,
                "duration_median": float(usable["duration"].median()) if len(usable) else None,
                "duration_max": float(usable["duration"].max()) if len(usable) else None,
                "duration_source": DURATION_SOURCE[cohort],
            }
        )
    return pd.DataFrame(rows)


def scale_train_test(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train.to_numpy(dtype=float))
    x_te = scaler.transform(x_test.to_numpy(dtype=float))
    x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0)
    x_te = np.nan_to_num(x_te, nan=0.0, posinf=0.0, neginf=0.0)
    return x_tr, x_te


def fit_risk_scores(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    x_tr, x_te = scale_train_test(x_train, x_test)
    model = LogisticRegression(C=1.0, random_state=random_state, max_iter=1000)
    model.fit(x_tr, y_train)
    train_score = model.predict_proba(x_tr)[:, 1]
    test_score = model.predict_proba(x_te)[:, 1]
    return train_score, test_score


def logit_clip(prob: np.ndarray, eps: float = LOGIT_EPS) -> np.ndarray:
    p = np.asarray(prob, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def zscore_1d(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    mean = np.nanmean(x)
    sd = np.nanstd(x, ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return np.full(x.shape, np.nan, dtype=float)
    return (x - mean) / sd


def _as_z_input(risk_score: np.ndarray, score_kind: str) -> np.ndarray:
    x = np.asarray(risk_score, dtype=float)
    if score_kind == "probability":
        return logit_clip(x)
    if score_kind != "linear":
        raise ValueError(f"Unknown score_kind {score_kind!r}")
    return x


def to_per_sd_score(
    risk_score: np.ndarray,
    score_kind: str,
    train_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Logit (if probability) then z-score.

    If train_mask is given, mean/SD are fit on train rows only and applied to all.
    If train_mask is omitted, mean/SD are fit on the whole vector (Cox-sample z).
    """
    x = _as_z_input(risk_score, score_kind)
    if train_mask is None:
        return zscore_1d(x)
    fit = x[np.asarray(train_mask, dtype=bool)]
    fit = fit[np.isfinite(fit)]
    if fit.size < 2:
        return np.full(x.shape, np.nan, dtype=float)
    mean = float(np.mean(fit))
    sd = float(np.std(fit, ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return np.full(x.shape, np.nan, dtype=float)
    return (x - mean) / sd


def _cox_base_frame(duration: np.ndarray, event: np.ndarray, risk_score: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {"duration": duration, "event": event, "risk_score": risk_score}
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df.loc[df["duration"] > 0].copy()


def cox_empty_result(n: int, n_event: int, n_censored: int, error: str, high_risk: bool) -> dict:
    out = {
        "n": n,
        "n_event": n_event,
        "n_censored": n_censored,
        "hr": None,
        "log_hr": None,
        "p": None,
        "hr_ci_low": None,
        "hr_ci_high": None,
        "error": error,
    }
    if high_risk:
        out["n_high_risk"] = 0
        out["n_high_risk_events"] = 0
    else:
        out["n_high_risk"] = None
        out["n_high_risk_events"] = None
        out["score_mean"] = None
        out["score_sd"] = None
    return out


def cox_high_risk_hr(
    duration: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
    high_risk_frac: float = 0.10,
) -> dict:
    df = _cox_base_frame(duration, event, risk_score)
    n = int(len(df))
    n_event = int(df["event"].sum())
    n_censored = n - n_event
    if n < 8 or n_event == 0 or df["risk_score"].nunique() < 2:
        return {
            "n": n,
            "n_event": n_event,
            "n_censored": n_censored,
            "n_high_risk": 0,
            "n_high_risk_events": 0,
            "hr": None,
            "log_hr": None,
            "p": None,
            "hr_ci_low": None,
            "hr_ci_high": None,
            "error": "too_few_samples_or_events",
        }
    top_n = max(1, int(n * high_risk_frac))
    df["high_risk"] = 0
    df.loc[df["risk_score"].nlargest(top_n).index, "high_risk"] = 1
    n_high = int(df["high_risk"].sum())
    n_high_events = int(df.loc[df["high_risk"] == 1, "event"].sum())
    if df["high_risk"].nunique() < 2:
        return {
            "n": n,
            "n_event": n_event,
            "n_censored": n_censored,
            "n_high_risk": n_high,
            "n_high_risk_events": n_high_events,
            "hr": None,
            "log_hr": None,
            "p": None,
            "hr_ci_low": None,
            "hr_ci_high": None,
            "error": "high_risk_not_contrasted",
        }
    try:
        cph = CoxPHFitter()
        cph.fit(df[["duration", "event", "high_risk"]], duration_col="duration", event_col="event")
        row = cph.summary.loc["high_risk"]
        return {
            "n": n,
            "n_event": n_event,
            "n_censored": n_censored,
            "n_high_risk": n_high,
            "n_high_risk_events": n_high_events,
            "hr": float(row["exp(coef)"]),
            "log_hr": float(row["coef"]),
            "p": float(row["p"]),
            "hr_ci_low": float(row["exp(coef) lower 95%"]),
            "hr_ci_high": float(row["exp(coef) upper 95%"]),
            "error": None,
        }
    except Exception as exc:
        return {
            "n": n,
            "n_event": n_event,
            "n_censored": n_censored,
            "n_high_risk": n_high,
            "n_high_risk_events": n_high_events,
            "hr": None,
            "log_hr": None,
            "p": None,
            "hr_ci_low": None,
            "hr_ci_high": None,
            "error": str(exc),
        }


def cox_per_sd_hr(duration: np.ndarray, event: np.ndarray, score_z: np.ndarray) -> dict:
    """Continuous Cox: HR for a one-SD increase in the z-scored (logit) score."""
    df = _cox_base_frame(duration, event, score_z)
    n = int(len(df))
    n_event = int(df["event"].sum())
    n_censored = n - n_event
    if n < 8 or n_event == 0 or df["risk_score"].nunique() < 2:
        return cox_empty_result(n, n_event, n_censored, "too_few_samples_or_events", high_risk=False)
    try:
        cph = CoxPHFitter()
        cph.fit(df[["duration", "event", "risk_score"]], duration_col="duration", event_col="event")
        row = cph.summary.loc["risk_score"]
        return {
            "n": n,
            "n_event": n_event,
            "n_censored": n_censored,
            "n_high_risk": None,
            "n_high_risk_events": None,
            "score_mean": float(df["risk_score"].mean()),
            "score_sd": float(df["risk_score"].std(ddof=0)),
            "hr": float(row["exp(coef)"]),
            "log_hr": float(row["coef"]),
            "p": float(row["p"]),
            "hr_ci_low": float(row["exp(coef) lower 95%"]),
            "hr_ci_high": float(row["exp(coef) upper 95%"]),
            "error": None,
        }
    except Exception as exc:
        return {
            "n": n,
            "n_event": n_event,
            "n_censored": n_censored,
            "n_high_risk": None,
            "n_high_risk_events": None,
            "score_mean": float(df["risk_score"].mean()) if n else None,
            "score_sd": float(df["risk_score"].std(ddof=0)) if n else None,
            "hr": None,
            "log_hr": None,
            "p": None,
            "hr_ci_low": None,
            "hr_ci_high": None,
            "error": str(exc),
        }


def combine_with_prs(
    risk_score: np.ndarray,
    event: np.ndarray,
    prs: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Fit omics+PRS logistic on train only, then score train+test."""
    risk = np.asarray(risk_score, dtype=float)
    prs_arr = np.asarray(prs, dtype=float)
    y = np.asarray(event)
    tr = np.asarray(train_mask, dtype=bool)
    out = np.full(len(y), np.nan, dtype=float)
    if tr.sum() < 8 or np.unique(y[tr]).size < 2:
        return out
    prs_scaler = MinMaxScaler()
    z_prs_tr = prs_scaler.fit_transform(prs_arr[tr].reshape(-1, 1)).ravel()
    z_prs_all = prs_scaler.transform(prs_arr.reshape(-1, 1)).ravel()
    x_tr = np.column_stack([risk[tr], z_prs_tr])
    x_all = np.column_stack([risk, z_prs_all])
    feat_scaler = StandardScaler()
    x_tr_s = feat_scaler.fit_transform(x_tr)
    x_all_s = feat_scaler.transform(x_all)
    model = LogisticRegression(C=1.0, random_state=RANDOM_STATE, max_iter=1000)
    model.fit(x_tr_s, y[tr])
    return model.predict_proba(x_all_s)[:, 1]


def list_pgs(cohort_dir: Path) -> list[str]:
    root = cohort_dir / "Final_Prs_Scores"
    if not root.exists():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and (p / "prs_scores.csv").exists()
    )


def load_prs(cohort_dir: Path, pgs_id: str) -> pd.DataFrame:
    path = cohort_dir / "Final_Prs_Scores" / pgs_id / "prs_scores.csv"
    df = pd.read_csv(path)
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.drop_duplicates(subset=["sample_id"], keep="first")
    return df[["sample_id", "score"]].rename(columns={"score": "prs_score"})


def hstack_aligned(left: pd.DataFrame, right: pd.DataFrame, name: str) -> pd.DataFrame:
    if len(left) != len(right):
        raise ValueError(f"{name}: {len(left)} vs {len(right)} rows")
    if not (left["Sample ID"].to_numpy() == right["Sample ID"].to_numpy()).all():
        raise ValueError(f"{name}: Sample ID order mismatch")
    return pd.concat(
        [left[["Sample ID", "Diagnosis"]].reset_index(drop=True), numeric_features(left), numeric_features(right)],
        axis=1,
    )


def assemble_feature_sets(cohort_dir: Path, split: str, ids: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metabolite = load_metabolite_input(cohort_dir, split)
    fold_changes = load_metabolitics(cohort_dir, split, "fold_changes.csv")
    pathway = load_metabolitics(cohort_dir, split, "X_pathway.csv")
    reaction = load_metabolitics(cohort_dir, split, "X_reaction_diff.csv")
    for name, df in (
        ("metabolite_levels", metabolite),
        ("fold_changes", fold_changes),
        ("pathway", pathway),
        ("reaction", reaction),
    ):
        align_features(df, ids, f"{cohort_dir.name}/{split}/{name}")
    return {
        "metabolite_levels": metabolite,
        "fold_changes": fold_changes,
        "pathway": pathway,
        "reaction": reaction,
        "metabolite_pathway": hstack_aligned(metabolite, pathway, f"{split}/metabolite_pathway"),
        "metabolite_reaction": hstack_aligned(metabolite, reaction, f"{split}/metabolite_reaction"),
    }


def _lookup_hr(long_df: pd.DataFrame, feature_set: str, pgs_id, display_endswith: Optional[str] = None) -> Optional[float]:
    part = long_df.loc[long_df["feature_set"] == feature_set]
    if pgs_id is None:
        part = part.loc[part["pgs_id"].isna()]
    else:
        part = part.loc[part["pgs_id"] == pgs_id]
    if display_endswith:
        part = part.loc[part["display_name"].astype(str).str.endswith(display_endswith)]
    if part.empty:
        return None
    val = pd.to_numeric(part["hr"].iloc[0], errors="coerce")
    return float(val) if np.isfinite(val) else None


def wide_hr_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """One row per PRS, paired omics / omics+PRS columns (notebook format)."""
    pgs_ids = [p for p in long_df["pgs_id"].dropna().unique()]
    columns = ["PRS name"]
    for key in WIDE_FEATURE_ORDER:
        display = FEATURE_DISPLAY[key]
        columns.extend([display, f"{display} PRS"])
    columns.append("Prs Score")
    rows = []
    for pgs_id in pgs_ids:
        row = {"PRS name": pgs_id}
        for key in WIDE_FEATURE_ORDER:
            display = FEATURE_DISPLAY[key]
            row[display] = _lookup_hr(long_df, key, None)
            row[f"{display} PRS"] = _lookup_hr(long_df, key, pgs_id, " PRS")
        row["Prs Score"] = _lookup_hr(long_df, "prs_only", pgs_id)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def standardize_long_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in STANDARD_LONG_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[STANDARD_LONG_COLUMNS]


def group_median_hr_stats(
    z: np.ndarray,
    event: np.ndarray,
    duration: np.ndarray,
    log_hr: Optional[float],
) -> dict:
    """Separability of healthy vs patients on the Cox-sample z scale.

    Individual relative hazard is exp(beta * z_i). Median HR distance is
    median(r | patient) - median(r | healthy); the ratio equals
    exp(beta * (median_z_patient - median_z_healthy)).
    """
    z = np.asarray(z, dtype=float)
    event = np.asarray(event, dtype=float)
    duration = np.asarray(duration, dtype=float)
    ok = np.isfinite(z) & np.isfinite(event) & np.isfinite(duration) & (duration > 0)
    z = z[ok]
    event = event[ok]
    healthy = event == 0
    patient = event == 1
    n_h = int(healthy.sum())
    n_p = int(patient.sum())
    med_z_h = float(np.median(z[healthy])) if n_h else float("nan")
    med_z_p = float(np.median(z[patient])) if n_p else float("nan")
    delta_z = med_z_p - med_z_h
    out = {
        "n_healthy": n_h,
        "n_patient": n_p,
        "median_z_healthy": med_z_h,
        "median_z_patient": med_z_p,
        "delta_median_z": delta_z,
        "median_rel_hr_healthy": None,
        "median_rel_hr_patient": None,
        "median_hr_distance": None,
        "median_hr_ratio": None,
    }
    if log_hr is None or not np.isfinite(float(log_hr)) or n_h == 0 or n_p == 0:
        return out
    rel = np.exp(float(log_hr) * z)
    med_h = float(np.median(rel[healthy]))
    med_p = float(np.median(rel[patient]))
    out["median_rel_hr_healthy"] = med_h
    out["median_rel_hr_patient"] = med_p
    out["median_hr_distance"] = med_p - med_h
    out["median_hr_ratio"] = (med_p / med_h) if med_h != 0 else None
    return out


def build_cohort_distance_table(
    score_frame: pd.DataFrame,
    cox_df: pd.DataFrame,
    cohort_dir: Path,
    out_dir: Path,
) -> pd.DataFrame:
    dist_df = median_hr_distance_table(score_frame, cox_df)
    prs_dist = prs_only_median_hr_distance(score_frame, cox_df, cohort_dir)
    if not prs_dist.empty:
        prs_dist.to_csv(out_dir / "median_hr_distance_coxsample_prs.csv", index=False)
        dist_df = pd.concat([dist_df, aggregate_prs_only_distance(prs_dist)], ignore_index=True)
    integ = integrated_prs_median_hr_distance(score_frame, cox_df, cohort_dir)
    if not integ.empty:
        integ.to_csv(out_dir / "median_hr_distance_coxsample_integrated_prs.csv", index=False)
        dist_df = pd.concat([dist_df, aggregate_distance_across_pgs(integ)], ignore_index=True)
    dist_df.to_csv(out_dir / "median_hr_distance_coxsample.csv", index=False)
    return dist_df


def median_hr_distance_table(
    score_frame: pd.DataFrame,
    cox_long: pd.DataFrame,
) -> pd.DataFrame:
    """Biomarker-only median HR distance using Cox-sample logit-z scores."""
    bio = cox_long.loc[cox_long["pgs_id"].isna()].copy()
    rows = []
    for _, rec in bio.iterrows():
        key = rec["feature_set"]
        z_col = f"risk_{key}_logit_z_coxsample"
        if z_col not in score_frame.columns:
            continue
        stats = group_median_hr_stats(
            score_frame[z_col].to_numpy(),
            score_frame["event"].to_numpy(),
            score_frame["duration"].to_numpy(),
            rec["log_hr"],
        )
        rows.append(
            {
                "cohort": rec["cohort"],
                "feature_set": key,
                "display_name": rec["display_name"],
                "hr_per_sd": rec["hr"],
                "log_hr": rec["log_hr"],
                "p": rec["p"],
                **stats,
            }
        )
    return pd.DataFrame(rows)


def prs_only_median_hr_distance(
    score_frame: pd.DataFrame,
    cox_long: pd.DataFrame,
    cohort_dir: Path,
) -> pd.DataFrame:
    """Per-PGS PRS-only median HR distance (Cox-sample z of the raw PRS)."""
    prs_long = cox_long.loc[cox_long["feature_set"] == "prs_only"].copy()
    rows = []
    for _, rec in prs_long.iterrows():
        pgs_id = rec["pgs_id"]
        if pd.isna(pgs_id):
            continue
        prs = load_prs(cohort_dir, str(pgs_id))
        frame = score_frame.copy()
        frame["sample_id"] = frame["sample_id"].astype(str)
        merged = frame.merge(prs, on="sample_id", how="left")
        mask = merged["prs_score"].notna()
        if int(mask.sum()) < 8:
            continue
        part = merged.loc[mask]
        z = to_per_sd_score(part["prs_score"].to_numpy(dtype=float), "linear", train_mask=None)
        stats = group_median_hr_stats(
            z,
            part["event"].to_numpy(),
            part["duration"].to_numpy(),
            rec["log_hr"],
        )
        rows.append(
            {
                "cohort": rec["cohort"],
                "feature_set": "prs_only",
                "display_name": "Prs Score",
                "pgs_id": str(pgs_id),
                "hr_per_sd": rec["hr"],
                "log_hr": rec["log_hr"],
                "p": rec["p"],
                **stats,
            }
        )
    return pd.DataFrame(rows)


def aggregate_distance_across_pgs(dist_df: pd.DataFrame) -> pd.DataFrame:
    """Median across PGS of per-PGS median-HR distances, one row per cohort/config."""
    if dist_df.empty:
        return dist_df
    rows = []
    grouped = dist_df.groupby(["cohort", "feature_set", "display_name"], sort=False)
    for (cohort, feature_set, display_name), part in grouped:
        part = part.dropna(subset=["median_hr_distance", "median_hr_ratio"])
        if part.empty:
            continue
        rows.append(
            {
                "cohort": cohort,
                "feature_set": feature_set,
                "display_name": display_name,
                "hr_per_sd": float(part["hr_per_sd"].median()),
                "log_hr": float(part["log_hr"].median()),
                "p": float(part["p"].median()),
                "n_healthy": int(round(float(part["n_healthy"].median()))),
                "n_patient": int(round(float(part["n_patient"].median()))),
                "median_z_healthy": float(part["median_z_healthy"].median()),
                "median_z_patient": float(part["median_z_patient"].median()),
                "delta_median_z": float(part["delta_median_z"].median()),
                "median_rel_hr_healthy": float(part["median_rel_hr_healthy"].median()),
                "median_rel_hr_patient": float(part["median_rel_hr_patient"].median()),
                "median_hr_distance": float(part["median_hr_distance"].median()),
                "median_hr_ratio": float(part["median_hr_ratio"].median()),
                "n_pgs": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def aggregate_prs_only_distance(prs_df: pd.DataFrame) -> pd.DataFrame:
    """One PRS-only row per cohort: median across PGS of the per-PGS distances."""
    if prs_df.empty:
        return prs_df
    return aggregate_distance_across_pgs(prs_df)


def integrated_prs_median_hr_distance(
    score_frame: pd.DataFrame,
    cox_long: pd.DataFrame,
    cohort_dir: Path,
) -> pd.DataFrame:
    """Per-PGS omics+PRS median HR distance using Cox-sample logit-z scores."""
    combo = cox_long.loc[
        cox_long["pgs_id"].notna()
        & cox_long["feature_set"].ne("prs_only")
        & cox_long["display_name"].astype(str).str.endswith(" PRS")
    ].copy()
    rows = []
    for _, rec in combo.iterrows():
        key = rec["feature_set"]
        pgs_id = str(rec["pgs_id"])
        risk_col = f"risk_{key}"
        if risk_col not in score_frame.columns:
            continue
        prs = load_prs(cohort_dir, pgs_id)
        frame = score_frame.copy()
        frame["sample_id"] = frame["sample_id"].astype(str)
        merged = frame.merge(prs, on="sample_id", how="left")
        mask = merged["prs_score"].notna()
        if int(mask.sum()) < 8:
            continue
        part = merged.loc[mask]
        combo_score = combine_with_prs(
            part[risk_col].to_numpy(dtype=float),
            part["event"].to_numpy(),
            part["prs_score"].to_numpy(dtype=float),
            train_mask=part["split"].eq("train").to_numpy(),
        )
        z = to_per_sd_score(combo_score, "probability", train_mask=None)
        stats = group_median_hr_stats(
            z,
            part["event"].to_numpy(),
            part["duration"].to_numpy(),
            rec["log_hr"],
        )
        rows.append(
            {
                "cohort": rec["cohort"],
                "feature_set": key,
                "display_name": f"{FEATURE_DISPLAY.get(key, key)} + PRS",
                "pgs_id": pgs_id,
                "hr_per_sd": rec["hr"],
                "log_hr": rec["log_hr"],
                "p": rec["p"],
                **stats,
            }
        )
    return pd.DataFrame(rows)


def summarize_median_hr_distance(dist_master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, part in dist_master.groupby("cohort", sort=False):
        part = part.dropna(subset=["median_hr_distance", "median_hr_ratio"])
        if part.empty:
            continue
        best_d = part.loc[part["median_hr_distance"].idxmax()]
        best_r = part.loc[part["median_hr_ratio"].idxmax()]
        rows.append(
            {
                "cohort": cohort,
                "n_healthy": int(part["n_healthy"].iloc[0]),
                "n_patient": int(part["n_patient"].iloc[0]),
                "median_distance_across_features": float(part["median_hr_distance"].median()),
                "median_ratio_across_features": float(part["median_hr_ratio"].median()),
                "max_distance": float(best_d["median_hr_distance"]),
                "max_distance_feature": best_d["display_name"],
                "ratio_at_max_distance": float(best_d["median_hr_ratio"]),
                "max_ratio": float(best_r["median_hr_ratio"]),
                "max_ratio_feature": best_r["display_name"],
                "distance_at_max_ratio": float(best_r["median_hr_distance"]),
                "max_delta_median_z": float(part["delta_median_z"].max()),
            }
        )
    order = {c: i for i, c in enumerate(COHORTS)}
    out = pd.DataFrame(rows)
    return out.sort_values("cohort", key=lambda s: s.map(order)).reset_index(drop=True)


def plot_median_hr_distance(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if df.empty:
        return
    part = df.dropna(subset=["median_hr_distance"]).copy()
    if part.empty:
        return
    order = {name: i for i, name in enumerate(DISTANCE_PLOT_FEATURES)}
    part = part.sort_values("display_name", key=lambda s: s.map(lambda x: order.get(x, 99)))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    labels = part["display_name"].tolist()
    y = np.arange(len(labels))
    axes[0].barh(y, part["median_hr_distance"].to_numpy(dtype=float), color="#4C78A8")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Median relative-HR distance (patient − healthy)")
    axes[0].axvline(0, color="#888", lw=1)
    axes[1].barh(y, part["median_hr_ratio"].to_numpy(dtype=float), color="#F58518")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Median relative-HR ratio (patient / healthy)")
    axes[1].axvline(1, color="#888", lw=1)
    fig.suptitle(title)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_median_hr_distance_by_cohort(df: pd.DataFrame, out_path: Path) -> None:
    part = df.dropna(subset=["median_hr_distance"]).copy()
    if part.empty:
        return
    cohorts = [c for c in COHORTS if c in set(part["cohort"])]
    features = [f for f in DISTANCE_PLOT_FEATURES if f in set(part["display_name"])]
    x = np.arange(len(cohorts))
    width = 0.8 / max(len(features), 1)
    colors = plt.cm.tab20.colors
    fig, axes = plt.subplots(2, 1, figsize=(14.8, 9.4), sharex=True)
    for ax, value_col, ylabel, ref in (
        (axes[0], "median_hr_distance", "Median relative-HR distance (patient − healthy)", 0.0),
        (axes[1], "median_hr_ratio", "Median relative-HR ratio (patient / healthy)", 1.0),
    ):
        for i, feat in enumerate(features):
            sub = part.loc[part["display_name"] == feat].drop_duplicates("cohort").set_index("cohort")
            vals = [float(sub.loc[c, value_col]) if c in sub.index else np.nan for c in cohorts]
            ax.bar(
                x + (i - (len(features) - 1) / 2) * width,
                vals,
                width=width,
                label=feat,
                color=colors[i % 10],
            )
        ax.axhline(ref, color="#888", lw=1)
        ax.set_ylabel(ylabel)
        if value_col == "median_hr_ratio":
            ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([COHORT_PLOT_LABELS.get(c, c) for c in cohorts])
    fig.suptitle("Healthy vs patient median HR separability (Cox-sample z)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def plot_forest(
    labels: Sequence[str],
    hrs: Sequence[Optional[float]],
    title: str,
    out_path: Path,
    xlabel: str = "Hazard Ratio",
) -> None:
    values = []
    kept = []
    for lab, hr in zip(labels, hrs):
        if hr is None or not np.isfinite(hr):
            continue
        kept.append(lab)
        values.append(float(hr))
    if not values:
        return
    fig, ax = plt.subplots(figsize=(10, max(4.0, 0.38 * len(kept))))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y_pos = np.arange(len(kept))
    colors = plt.cm.Set2.colors
    for i, (hr, lab) in enumerate(zip(values, kept)):
        color = colors[(i // 2) % len(colors)]
        ax.plot(hr, y_pos[i], "o", color=color, markersize=8)
        ax.text(hr + 0.15, y_pos[i], f"{hr:.2f}", va="center", ha="left", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(kept)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0, max(values) + 3)
    ax.grid(True, axis="x", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def run_cohort(project_root: Path, cohort: str, random_state: int) -> dict:
    started = time.perf_counter()
    cohort_dir = project_root / cohort
    out_dir = cohort_dir / "processed" / "cox"
    plot_dir = project_root / "plots" / cohort / "hazard_ratios"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_ids = load_split_table(cohort_dir, "train")
    test_ids = load_split_table(cohort_dir, "test")
    all_ids = pd.concat(
        [train_ids.assign(split="train"), test_ids.assign(split="test")],
        ignore_index=True,
    )
    clinical = pd.read_csv(cohort_dir / "raw" / "clinical.csv")
    scored = build_duration(cohort, all_ids, clinical)
    event_table = event_censor_table(cohort, scored)
    event_table.to_csv(out_dir / "event_censor_table.csv", index=False)

    train_sets = assemble_feature_sets(cohort_dir, "train", train_ids)
    test_sets = assemble_feature_sets(cohort_dir, "test", test_ids)
    y_train = binary_event(train_ids["metabolitics_label"])
    y_test = binary_event(test_ids["metabolitics_label"])

    score_frame = scored[
        ["unique_row_id", "sample_id", "participant_id", "split", "metabolitics_label", "duration", "event"]
    ].copy()
    hr_rows = []
    alt_rows = []
    coxsample_rows = []

    def add_hr_row(
        feature_set: str,
        display: str,
        pgs_id: Optional[str],
        scores: np.ndarray,
        split_mask=None,
        score_kind: str = "probability",
    ) -> dict:
        is_train = scored["split"].eq("train").to_numpy()
        if split_mask is None:
            part = scored
            part_scores = scores
            part_train = is_train
        else:
            mask = split_mask.to_numpy() if hasattr(split_mask, "to_numpy") else np.asarray(split_mask, dtype=bool)
            part = scored.loc[split_mask]
            part_scores = scores[mask]
            part_train = is_train[mask]
        duration = part["duration"].to_numpy()
        event = part["event"].to_numpy()
        std_stats = cox_high_risk_hr(duration, event, part_scores)
        shared = {
            "cohort": cohort,
            "feature_set": feature_set,
            "display_name": display,
            "pgs_id": pgs_id,
            "with_prs": pgs_id is not None and not str(display).startswith("Prs Score"),
            "score_kind": score_kind,
        }
        hr_rows.append({**shared, "hr_method": "top10_high_risk", **std_stats})
        z_train = to_per_sd_score(part_scores, score_kind, train_mask=part_train)
        alt_method = "logit_z_per_sd" if score_kind == "probability" else "z_per_sd"
        alt_stats = cox_per_sd_hr(duration, event, z_train)
        alt_rows.append({**shared, "hr_method": alt_method, **alt_stats})
        z_cox = to_per_sd_score(part_scores, score_kind, train_mask=None)
        cox_method = "logit_z_per_sd_coxsample" if score_kind == "probability" else "z_per_sd_coxsample"
        cox_stats = cox_per_sd_hr(duration, event, z_cox)
        coxsample_rows.append({**shared, "hr_method": cox_method, **cox_stats})
        return std_stats

    for key, display in FEATURE_DISPLAY.items():
        x_tr = numeric_features(train_sets[key])
        x_te = numeric_features(test_sets[key])
        x_tr, x_te = x_tr.align(x_te, join="inner", axis=1)
        train_score, test_score = fit_risk_scores(x_tr, x_te, y_train, random_state=random_state)
        combined = np.concatenate([train_score, test_score])
        score_frame[f"risk_{key}"] = combined
        score_frame[f"risk_{key}_logit_z"] = to_per_sd_score(
            combined, "probability", train_mask=score_frame["split"].eq("train").to_numpy()
        )
        score_frame[f"risk_{key}_logit_z_coxsample"] = to_per_sd_score(
            combined, "probability", train_mask=None
        )
        add_hr_row(key, display, None, combined, score_kind="probability")
        print(
            f"  {display}: HR_top10={hr_rows[-1]['hr']}  "
            f"HR_perSD_train={alt_rows[-1]['hr']}  "
            f"HR_perSD_coxsample={coxsample_rows[-1]['hr']}"
        )

    pgs_ids = list_pgs(cohort_dir)
    for pgs_id in pgs_ids:
        prs = load_prs(cohort_dir, pgs_id)
        merged = score_frame.merge(prs, on="sample_id", how="left")
        prs_mask = merged["prs_score"].notna()
        if int(prs_mask.sum()) < 8:
            continue
        raw_prs_full = np.full(len(merged), np.nan)
        raw_prs_full[prs_mask.to_numpy()] = merged.loc[prs_mask, "prs_score"].to_numpy(dtype=float)
        add_hr_row(
            "prs_only",
            f"Prs Score {pgs_id}",
            pgs_id,
            raw_prs_full,
            split_mask=prs_mask,
            score_kind="linear",
        )
        forest_labels = []
        forest_hrs = []
        forest_hrs_alt = []
        forest_hrs_cox = []
        for key, display in FEATURE_DISPLAY.items():
            risk = merged[f"risk_{key}"].to_numpy()
            combo = np.full(len(merged), np.nan)
            combo[prs_mask.to_numpy()] = combine_with_prs(
                risk[prs_mask.to_numpy()],
                merged.loc[prs_mask, "event"].to_numpy(),
                merged.loc[prs_mask, "prs_score"].to_numpy(),
                train_mask=merged.loc[prs_mask, "split"].eq("train").to_numpy(),
            )
            stats = add_hr_row(
                key, f"{display} PRS", pgs_id, combo, split_mask=prs_mask, score_kind="probability"
            )
            forest_labels.extend([display, f"{display} PRS"])
            bio = next(r for r in hr_rows if r["feature_set"] == key and r["pgs_id"] is None)
            bio_alt = next(r for r in alt_rows if r["feature_set"] == key and r["pgs_id"] is None)
            combo_alt = next(
                r
                for r in alt_rows
                if r["feature_set"] == key
                and r["pgs_id"] == pgs_id
                and str(r["display_name"]).endswith(" PRS")
            )
            forest_hrs.extend([bio["hr"], stats["hr"]])
            forest_hrs_alt.extend([bio_alt["hr"], combo_alt["hr"]])
            bio_cox = next(r for r in coxsample_rows if r["feature_set"] == key and r["pgs_id"] is None)
            combo_cox = next(
                r
                for r in coxsample_rows
                if r["feature_set"] == key
                and r["pgs_id"] == pgs_id
                and str(r["display_name"]).endswith(" PRS")
            )
            forest_hrs_cox.extend([bio_cox["hr"], combo_cox["hr"]])
        prs_row = next(
            r for r in hr_rows if r["feature_set"] == "prs_only" and r["pgs_id"] == pgs_id
        )
        prs_alt = next(
            r for r in alt_rows if r["feature_set"] == "prs_only" and r["pgs_id"] == pgs_id
        )
        prs_cox = next(
            r for r in coxsample_rows if r["feature_set"] == "prs_only" and r["pgs_id"] == pgs_id
        )
        forest_labels.append(f"Prs Score {pgs_id}")
        forest_hrs.append(prs_row["hr"])
        forest_hrs_alt.append(prs_alt["hr"])
        forest_hrs_cox.append(prs_cox["hr"])
        plot_forest(
            forest_labels,
            forest_hrs,
            f"{cohort} forest plot with PRS {pgs_id}",
            plot_dir / f"forest_plot_{pgs_id}.png",
        )
        plot_forest(
            forest_labels,
            forest_hrs_alt,
            f"{cohort} per-1SD (train z) forest plot with PRS {pgs_id}",
            plot_dir / f"forest_plot_{pgs_id}_per_sd.png",
            xlabel="Hazard Ratio per 1 train-SD",
        )
        plot_forest(
            forest_labels,
            forest_hrs_cox,
            f"{cohort} per-1SD (Cox-sample z) forest plot with PRS {pgs_id}",
            plot_dir / f"forest_plot_{pgs_id}_per_sd_coxsample.png",
            xlabel="Hazard Ratio per 1 SD (Cox sample)",
        )

    long_df = standardize_long_table(pd.DataFrame(hr_rows))
    long_df.to_csv(out_dir / "hazard_ratios_long.csv", index=False)
    score_frame.to_csv(out_dir / "risk_scores.csv", index=False)

    alt_df = standardize_long_table(pd.DataFrame(alt_rows))
    alt_df.to_csv(out_dir / "hazard_ratios_long_per_sd.csv", index=False)
    cox_df = standardize_long_table(pd.DataFrame(coxsample_rows))
    cox_df.to_csv(out_dir / "hazard_ratios_long_per_sd_coxsample.csv", index=False)

    comparison = build_hr_comparison(pd.DataFrame(hr_rows), pd.DataFrame(alt_rows))
    comparison.to_csv(out_dir / "hazard_ratios_comparison.csv", index=False)
    comparison_cox = build_hr_comparison(pd.DataFrame(hr_rows), pd.DataFrame(coxsample_rows))
    comparison_cox.to_csv(out_dir / "hazard_ratios_comparison_coxsample.csv", index=False)

    bio_only = long_df.loc[long_df["pgs_id"].isna()].copy()
    plot_forest(
        bio_only["display_name"].tolist(),
        bio_only["hr"].tolist(),
        f"{cohort} biomarker hazard ratios (no PRS)",
        plot_dir / "forest_plot_biomarkers.png",
    )
    bio_alt = alt_df.loc[alt_df["pgs_id"].isna()].copy()
    plot_forest(
        bio_alt["display_name"].tolist(),
        bio_alt["hr"].tolist(),
        f"{cohort} biomarker hazard ratios per 1 train-SD (no PRS)",
        plot_dir / "forest_plot_biomarkers_per_sd.png",
        xlabel="Hazard Ratio per 1 train-SD",
    )
    bio_cox = cox_df.loc[cox_df["pgs_id"].isna()].copy()
    plot_forest(
        bio_cox["display_name"].tolist(),
        bio_cox["hr"].tolist(),
        f"{cohort} biomarker hazard ratios per 1 SD in Cox sample (no PRS)",
        plot_dir / "forest_plot_biomarkers_per_sd_coxsample.png",
        xlabel="Hazard Ratio per 1 SD (Cox sample)",
    )
    dist_df = build_cohort_distance_table(score_frame, cox_df, cohort_dir, out_dir)
    plot_median_hr_distance(
        dist_df,
        plot_dir / "median_hr_distance_coxsample.png",
        f"{cohort} median HR distance (Cox-sample z)",
    )

    wide_hr_table(long_df).to_csv(out_dir / "hazard_ratios.csv", index=False)
    wide_hr_table(alt_df).to_csv(out_dir / "hazard_ratios_per_sd.csv", index=False)
    wide_hr_table(cox_df).to_csv(out_dir / "hazard_ratios_per_sd_coxsample.csv", index=False)

    payload = {
        "cohort": cohort,
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "n_train_participants": int(train_ids["participant_id"].nunique()),
        "n_test_participants": int(test_ids["participant_id"].nunique()),
        "n_pgs": int(len(pgs_ids)),
        "pgs_ids": pgs_ids,
        "duration_source": DURATION_SOURCE[cohort],
        "event_definition": "Control/healthy/normal/CN = 0; all other diagnoses = 1",
        "risk_model": "StandardScaler + LogisticRegression(C=1, max_iter=1000) fit on train only",
        "cox_rule": "high_risk = top 10% of risk scores; CoxPH on high_risk",
        "cox_rule_alternative_train": (
            "logistic probabilities -> logit; mean/SD fit on train only then applied to test; "
            "PRS-only z-scored as linear with train-only mean/SD; CoxPH HR per 1 train-SD"
        ),
        "cox_rule_alternative_coxsample": (
            "logistic probabilities -> logit; z-score on all samples entering that Cox model; "
            "PRS-only z-scored on the same Cox sample; HR per 1 SD in the Cox sample (Kaan)"
        ),
        "no_second_split": True,
        "metabolitics_mode": "fit_train_transform_holdout",
        "elapsed_seconds": time.perf_counter() - started,
        "event_censor": event_table.to_dict(orient="records"),
        "biomarker_hrs": bio_only.to_dict(orient="records"),
        "biomarker_hrs_per_sd": bio_alt.to_dict(orient="records"),
        "biomarker_hrs_per_sd_coxsample": bio_cox.to_dict(orient="records"),
        "median_hr_distance_coxsample": dist_df.to_dict(orient="records"),
    }
    _json_dump(out_dir / "run_summary.json", payload)
    return payload


def print_summary(payloads: Sequence[dict]) -> None:
    print("\n=== Event / censor ===")
    rows = []
    for payload in payloads:
        for rec in payload["event_censor"]:
            if rec["split"] == "all":
                rows.append(rec)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== Biomarker HRs ===")
    hr_rows = []
    for payload in payloads:
        for rec in payload["biomarker_hrs"]:
            hr_rows.append(
                {
                    "cohort": payload["cohort"],
                    "feature_set": rec["display_name"],
                    "hr": rec["hr"],
                    "p": rec["p"],
                    "n_event": rec["n_event"],
                    "n_censored": rec["n_censored"],
                }
            )
    print(pd.DataFrame(hr_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Biomarker HRs per 1 SD (logit-z / z) ===")
    alt_print = []
    for payload in payloads:
        for rec in payload.get("biomarker_hrs_per_sd", []):
            alt_print.append(
                {
                    "cohort": payload["cohort"],
                    "feature_set": rec["display_name"],
                    "hr_per_sd": rec["hr"],
                    "p": rec["p"],
                    "n_event": rec["n_event"],
                    "n_censored": rec["n_censored"],
                }
            )
    if alt_print:
        print(pd.DataFrame(alt_print).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Biomarker HRs per 1 SD (Cox-sample z) ===")
    cox_print = []
    for payload in payloads:
        for rec in payload.get("biomarker_hrs_per_sd_coxsample", []):
            cox_print.append(
                {
                    "cohort": payload["cohort"],
                    "feature_set": rec["display_name"],
                    "hr_per_sd_coxsample": rec["hr"],
                    "p": rec["p"],
                    "n_event": rec["n_event"],
                    "n_censored": rec["n_censored"],
                }
            )
    if cox_print:
        print(pd.DataFrame(cox_print).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Median HR distance (Cox-sample z; patient vs healthy) ===")
    dist_print = []
    for payload in payloads:
        for rec in payload.get("median_hr_distance_coxsample", []):
            dist_print.append(
                {
                    "cohort": rec["cohort"],
                    "feature_set": rec["display_name"],
                    "n_healthy": rec["n_healthy"],
                    "n_patient": rec["n_patient"],
                    "delta_median_z": rec["delta_median_z"],
                    "median_hr_healthy": rec["median_rel_hr_healthy"],
                    "median_hr_patient": rec["median_rel_hr_patient"],
                    "distance": rec["median_hr_distance"],
                    "ratio": rec["median_hr_ratio"],
                }
            )
    if dist_print:
        print(pd.DataFrame(dist_print).to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def build_hr_comparison(standard: pd.DataFrame, alternative: pd.DataFrame) -> pd.DataFrame:
    std = standard.copy()
    alt = alternative.copy()
    std["pgs_id"] = std["pgs_id"].astype(object)
    alt["pgs_id"] = alt["pgs_id"].astype(object)
    std["_pgs"] = std["pgs_id"].where(std["pgs_id"].notna(), "")
    alt["_pgs"] = alt["pgs_id"].where(alt["pgs_id"].notna(), "")
    merge_on = ["cohort", "feature_set", "display_name", "_pgs"]
    keep_std = merge_on + ["hr", "p", "hr_ci_low", "hr_ci_high", "n", "n_event", "error"]
    keep_alt = merge_on + [
        "hr", "p", "hr_ci_low", "hr_ci_high", "hr_method", "score_kind", "error"
    ]
    left = std[keep_std].rename(
        columns={
            "hr": "hr_top10",
            "p": "p_top10",
            "hr_ci_low": "hr_ci_low_top10",
            "hr_ci_high": "hr_ci_high_top10",
            "error": "error_top10",
        }
    )
    right = alt[keep_alt].rename(
        columns={
            "hr": "hr_per_sd",
            "p": "p_per_sd",
            "hr_ci_low": "hr_ci_low_per_sd",
            "hr_ci_high": "hr_ci_high_per_sd",
            "error": "error_per_sd",
        }
    )
    out = left.merge(right, on=merge_on, how="outer")
    out["pgs_id"] = out["_pgs"].replace("", np.nan)
    out = out.drop(columns=["_pgs"])
    return out


def plot_hr_method_scatter(comparison: pd.DataFrame, out_path: Path) -> None:
    df = comparison.dropna(subset=["hr_top10", "hr_per_sd"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for cohort, part in df.groupby("cohort"):
        ax.scatter(part["hr_top10"], part["hr_per_sd"], s=22, alpha=0.7, label=cohort)
    finite = df[["hr_top10", "hr_per_sd"]].to_numpy(dtype=float)
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    pad = 0.05 * (hi - lo) if hi > lo else 0.5
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888", ls="--", lw=1)
    ax.set_xlabel("Standard HR (top 10% high-risk)")
    ax.set_ylabel("Alternative HR (per 1 SD of logit-z score)")
    ax.set_title("Hazard-ratio encoding comparison")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def copy_hazard_ration_files(project_root: Path, cohorts: Sequence[str]) -> None:
    dest_dir = project_root / "hazard_rations"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for cohort in cohorts:
        tag = HR_COPY_TAG[cohort]
        src_dir = project_root / cohort / "processed" / "cox"
        mapping = {
            "hazard_ratios.csv": dest_dir / f"hazard_rations_{tag}.csv",
            "hazard_ratios_per_sd.csv": dest_dir / f"hazard_rations_{tag}_per_sd.csv",
            "hazard_ratios_per_sd_coxsample.csv": dest_dir / f"hazard_rations_{tag}_per_sd_coxsample.csv",
        }
        for src_name, dest in mapping.items():
            src = src_dir / src_name
            if src.exists():
                dest.write_bytes(src.read_bytes())


def write_distance_masters(project_root: Path, cohorts: Sequence[str]) -> None:
    master_dir = project_root / "results"
    master_dir.mkdir(parents=True, exist_ok=True)
    dist_master = pd.concat(
        [
            pd.read_csv(project_root / c / "processed" / "cox" / "median_hr_distance_coxsample.csv")
            for c in cohorts
        ],
        ignore_index=True,
    )
    dist_master.to_csv(master_dir / "median_hr_distance_coxsample.csv", index=False)
    prs_paths = [
        project_root / c / "processed" / "cox" / "median_hr_distance_coxsample_prs.csv"
        for c in cohorts
    ]
    prs_existing = [p for p in prs_paths if p.exists()]
    if prs_existing:
        pd.concat([pd.read_csv(p) for p in prs_existing], ignore_index=True).to_csv(
            master_dir / "median_hr_distance_coxsample_prs.csv", index=False
        )
    integ_paths = [
        project_root / c / "processed" / "cox" / "median_hr_distance_coxsample_integrated_prs.csv"
        for c in cohorts
    ]
    integ_existing = [p for p in integ_paths if p.exists()]
    if integ_existing:
        pd.concat([pd.read_csv(p) for p in integ_existing], ignore_index=True).to_csv(
            master_dir / "median_hr_distance_coxsample_integrated_prs.csv", index=False
        )
    dist_wide = dist_master.pivot_table(
        index="cohort",
        columns="display_name",
        values="median_hr_distance",
        aggfunc="first",
    ).reset_index()
    dist_wide.to_csv(master_dir / "median_hr_distance_coxsample_wide.csv", index=False)
    dist_ratio_wide = dist_master.pivot_table(
        index="cohort",
        columns="display_name",
        values="median_hr_ratio",
        aggfunc="first",
    ).reset_index()
    dist_ratio_wide.to_csv(master_dir / "median_hr_ratio_coxsample_wide.csv", index=False)
    summary = summarize_median_hr_distance(
        dist_master.loc[~dist_master["display_name"].astype(str).str.contains("PRS", case=False)].copy()
    )
    summary.to_csv(master_dir / "median_hr_distance_coxsample_cohort_summary.csv", index=False)
    plot_median_hr_distance_by_cohort(
        dist_master,
        master_dir / "median_hr_distance_coxsample.png",
    )
    supplement = project_root / "manuscript" / "supplementary" / "median_hr_distance_coxsample.png"
    if (master_dir / "median_hr_distance_coxsample.png").exists():
        supplement.write_bytes((master_dir / "median_hr_distance_coxsample.png").read_bytes())


def rebuild_distance_only(project_root: Path, cohorts: Sequence[str]) -> None:
    for cohort in cohorts:
        cohort_dir = project_root / cohort
        out_dir = cohort_dir / "processed" / "cox"
        plot_dir = project_root / "plots" / cohort / "hazard_ratios"
        score_frame = pd.read_csv(out_dir / "risk_scores.csv")
        cox_df = pd.read_csv(out_dir / "hazard_ratios_long_per_sd_coxsample.csv")
        print(f"=== {cohort} distance-only ===")
        dist_df = build_cohort_distance_table(score_frame, cox_df, cohort_dir, out_dir)
        plot_median_hr_distance(
            dist_df,
            plot_dir / "median_hr_distance_coxsample.png",
            f"{cohort} median HR distance (Cox-sample z)",
        )
        print(f"  configs={len(dist_df)} -> {out_dir / 'median_hr_distance_coxsample.csv'}")
    write_distance_masters(project_root, cohorts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument(
        "--distance-only",
        action="store_true",
        help="Rebuild median-HR-distance tables/figures from saved risk scores.",
    )
    args = parser.parse_args()
    if args.distance_only:
        rebuild_distance_only(args.project_root, args.cohorts)
        return

    payloads = []
    for cohort in args.cohorts:
        print(f"=== {cohort} ===")
        payload = run_cohort(args.project_root, cohort, args.random_state)
        payloads.append(payload)
        print(f"  wrote {args.project_root / cohort / 'processed' / 'cox'} ({payload['elapsed_seconds']:.1f}s)")
    master = pd.concat(
        [
            pd.read_csv(args.project_root / c / "processed" / "cox" / "hazard_ratios_long.csv")
            for c in args.cohorts
        ],
        ignore_index=True,
    )
    master_dir = args.project_root / "results"
    master_dir.mkdir(parents=True, exist_ok=True)
    master.to_csv(master_dir / "cox_hazard_ratios_long.csv", index=False)
    master_alt = pd.concat(
        [
            pd.read_csv(args.project_root / c / "processed" / "cox" / "hazard_ratios_long_per_sd.csv")
            for c in args.cohorts
        ],
        ignore_index=True,
    )
    master_alt.to_csv(master_dir / "cox_hazard_ratios_long_per_sd.csv", index=False)
    master_cox = pd.concat(
        [
            pd.read_csv(
                args.project_root / c / "processed" / "cox" / "hazard_ratios_long_per_sd_coxsample.csv"
            )
            for c in args.cohorts
        ],
        ignore_index=True,
    )
    master_cox.to_csv(master_dir / "cox_hazard_ratios_long_per_sd_coxsample.csv", index=False)
    master_cmp = pd.concat(
        [
            pd.read_csv(args.project_root / c / "processed" / "cox" / "hazard_ratios_comparison.csv")
            for c in args.cohorts
        ],
        ignore_index=True,
    )
    master_cmp.to_csv(master_dir / "cox_hazard_ratios_comparison.csv", index=False)
    plot_hr_method_scatter(master_cmp, master_dir / "cox_hr_top10_vs_per_sd.png")
    master_cmp_cox = pd.concat(
        [
            pd.read_csv(
                args.project_root / c / "processed" / "cox" / "hazard_ratios_comparison_coxsample.csv"
            )
            for c in args.cohorts
        ],
        ignore_index=True,
    )
    master_cmp_cox.to_csv(master_dir / "cox_hazard_ratios_comparison_coxsample.csv", index=False)
    plot_hr_method_scatter(master_cmp_cox, master_dir / "cox_hr_top10_vs_per_sd_coxsample.png")
    write_distance_masters(args.project_root, args.cohorts)
    copy_hazard_ration_files(args.project_root, args.cohorts)
    print_summary(payloads)


if __name__ == "__main__":
    main()
