#!/usr/bin/env python3
"""Assemble BMC Bioinformatics tables from leakage-safe FluxPRS outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "BMC_tables"


def fmt_pm(mean, std) -> str:
    if mean is None or pd.isna(mean):
        return ""
    if std is None or pd.isna(std):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} ± {float(std):.4f}"


def fmt4(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return f"{float(val):.4f}"


def save_csv(df: pd.DataFrame, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {caption}\n")
        df.to_csv(fh, index=False)


def save_table_png(df: pd.DataFrame, title: str, caption: str, path: Path, col_width: float = 0.18) -> None:
    n_rows, n_cols = df.shape
    fig_w = max(11.0, 1.15 * n_cols + 4.0)
    fig_h = max(2.8, 0.38 * (n_rows + 4) + 1.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12, loc="left", fontweight="bold")
    table = ax.table(
        cellText=df.astype(str).values,
        colLabels=list(df.columns),
        loc="upper center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#2f4a6d")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f4f7fb")
        else:
            cell.set_facecolor("white")
    fig.text(0.01, 0.02, caption, fontsize=8, wrap=True, va="bottom")
    fig.tight_layout(rect=(0, 0.08, 1, 0.98))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def diagnosis_counts(ids: pd.DataFrame) -> pd.DataFrame:
    rows = (
        ids.groupby("diagnosis", dropna=False)
        .agg(n_rows=("unique_row_id", "nunique"), n_participants=("participant_id", "nunique"))
        .reset_index()
    )
    return rows


def table1() -> pd.DataFrame:
    records = []
    specs = [
        (
            "ROSMAP-AD",
            "ROSMAP-AD [13, 14]",
            "sample",
            "70/30 sample split (rs=42); binary AD vs control; HR/PRS on concatenated train+test scores",
        ),
        (
            "MayoRNASeq-AD",
            "MayoRNASeq-AD [13, 15]",
            "sample",
            "70/30 sample split (rs=42); Mayo controls reused with PSP; HR/PRS on concatenated scores",
        ),
        (
            "MayoRNASeq-PSP",
            "MayoRNASeq-PSP [13, 15]",
            "sample",
            "70/30 sample split (rs=42); same 20 Mayo control Sample IDs as Mayo-AD",
        ),
        (
            "DIABETES",
            "DIABETES (DiCAD) [13]",
            "participant",
            "70/30 participant-level split (rs=42); 3-class classification; Cox TTE = sampling age stored as AgeDeath",
        ),
    ]
    for cohort, dataset, split_unit, context in specs:
        train = pd.read_csv(ROOT / cohort / "processed" / "splits" / "train_ids.csv")
        test = pd.read_csv(ROOT / cohort / "processed" / "splits" / "test_ids.csv")
        all_ids = pd.concat([train, test], ignore_index=True)
        for split_name, part in (("Total", all_ids), ("Train (70%)", train), ("Test (30%)", test)):
            counts = diagnosis_counts(part)
            for _, row in counts.iterrows():
                records.append(
                    {
                        "Dataset": dataset,
                        "Diagnosis": row["diagnosis"],
                        "Partition": split_name,
                        "N rows": int(row["n_rows"]),
                        "N participants": int(row["n_participants"]),
                        "Split unit": split_unit,
                        "Analysis context": context,
                    }
                )
    df = pd.DataFrame(records)
    dataset_order = {
        "MayoRNASeq-AD [13, 15]": 0,
        "MayoRNASeq-PSP [13, 15]": 1,
        "ROSMAP-AD [13, 14]": 2,
        "DIABETES (DiCAD) [13]": 3,
    }
    dx_order = {"AD": 0, "PSP": 1, "Diabetes": 2, "Pre-diabetes": 3, "Control": 9}
    df["_ds"] = df["Dataset"].map(dataset_order)
    df["_dx"] = df["Diagnosis"].map(lambda x: dx_order.get(str(x), 5))
    part_order = {"Total": 0, "Train (70%)": 1, "Test (30%)": 2}
    df["_ord"] = df["Partition"].map(part_order)
    df = df.sort_values(["_ds", "_dx", "_ord"]).drop(columns=["_ds", "_dx", "_ord"]).reset_index(drop=True)
    return df


def table1_wide() -> pd.DataFrame:
    """Paper-style one row per diagnosis with train/test columns."""
    long = table1()
    rows = []
    for (dataset, diagnosis, split_unit, context), part in long.groupby(
        ["Dataset", "Diagnosis", "Split unit", "Analysis context"], sort=False
    ):
        rec = {
            "Dataset": dataset,
            "Diagnosis": diagnosis,
            "Total N (rows)": int(part.loc[part["Partition"] == "Total", "N rows"].iloc[0]),
            "Total N (participants)": int(part.loc[part["Partition"] == "Total", "N participants"].iloc[0]),
            "Train rows": int(part.loc[part["Partition"] == "Train (70%)", "N rows"].iloc[0]),
            "Train participants": int(part.loc[part["Partition"] == "Train (70%)", "N participants"].iloc[0]),
            "Test rows": int(part.loc[part["Partition"] == "Test (30%)", "N rows"].iloc[0]),
            "Test participants": int(part.loc[part["Partition"] == "Test (30%)", "N participants"].iloc[0]),
            "Split unit": split_unit,
            "Analysis context": context,
        }
        rows.append(rec)
    out = pd.DataFrame(rows)
    dataset_order = {
        "MayoRNASeq-AD [13, 15]": 0,
        "MayoRNASeq-PSP [13, 15]": 1,
        "ROSMAP-AD [13, 14]": 2,
        "DIABETES (DiCAD) [13]": 3,
    }
    dx_order = {"AD": 0, "PSP": 1, "Diabetes": 2, "Pre-diabetes": 3, "Control": 9}
    out["_ds"] = out["Dataset"].map(dataset_order)
    out["_dx"] = out["Diagnosis"].map(lambda x: dx_order.get(str(x), 5))
    return out.sort_values(["_ds", "_dx"]).drop(columns=["_ds", "_dx"]).reset_index(drop=True)


def extras_block(summary: dict, key: str) -> dict:
    block = summary.get(key, {})
    if not block:
        return {}
    metrics = block.get("metrics") if isinstance(block.get("metrics"), dict) else {}
    cv = (
        block.get("cv_foldwise")
        or metrics.get("cv_foldwise")
        or block.get("cv")
        or metrics.get("cv")
        or {}
    )
    test = block.get("test") or metrics.get("test") or {}
    n_features = block.get("n_features") or metrics.get("n_features")
    if key == "nmf" and block.get("k") is not None:
        n_features = block["k"]
    return {"cv": cv, "test": test, "n_features": n_features}


def cv_logreg_row(cv_summary: dict, feature_set: str, display: str) -> dict:
    for item in cv_summary.get("feature_sets", []):
        if item.get("feature_set") == feature_set:
            cv = item.get("cv", {})
            test = item.get("test", {})
            return {
                "Cohort": cv_summary["cohort"],
                "Feature set": display,
                "Model": "Logistic Regression (all features)",
                "n features": item.get("n_features_outer_train"),
                "CV Accuracy (mean ± std)": fmt_pm(cv.get("mean_accuracy"), cv.get("std_accuracy")),
                "CV AUC (mean ± std)": fmt_pm(
                    cv.get("mean_roc_auc_macro_ovr"), cv.get("std_roc_auc_macro_ovr")
                ),
                "CV F1 (OOF macro)": fmt4((cv.get("oof") or {}).get("macro_f1")),
                "Test Accuracy": fmt4(test.get("accuracy")),
                "Test F1 (macro)": fmt4(test.get("macro_f1")),
                "Test AUC": fmt4(test.get("roc_auc_macro_ovr") or test.get("roc_auc")),
                "McNemar p vs LogReg-pathway": "",
                "Protocol": "5-fold CV on train using FC+FVA fit on cv_train healthy and applied to cv_val; held-out 30% test uses train-fit/test-transform",
            }
    return {}


def extras_row(cohort: str, summary: dict, key: str, feature_set: str, model: str) -> dict:
    block = extras_block(summary, key)
    if not block:
        return {}
    cv, test = block["cv"], block["test"]
    nested = summary.get(key, {})
    metrics = nested.get("metrics") if isinstance(nested.get("metrics"), dict) else {}
    mcnemar_block = (
        nested.get("mcnemar_vs_logreg_all_pathway")
        or nested.get("mcnemar_vs_logreg_pathway")
        or metrics.get("mcnemar_vs_logreg_all_pathway")
        or metrics.get("mcnemar_vs_logreg_pathway")
        or {}
    )
    mcnemar = fmt4(mcnemar_block.get("p_value")) if mcnemar_block else ""
    return {
        "Cohort": cohort,
        "Feature set": feature_set,
        "Model": model,
        "n features": block["n_features"],
        "CV Accuracy (mean ± std)": fmt_pm(cv.get("mean_accuracy"), cv.get("std_accuracy")),
        "CV AUC (mean ± std)": fmt_pm(cv.get("mean_roc_auc"), cv.get("std_roc_auc")),
        "CV F1 (OOF macro)": "",
        "Test Accuracy": fmt4(test.get("accuracy")),
        "Test F1 (macro)": fmt4(test.get("macro_f1")),
        "Test AUC": fmt4(test.get("roc_auc_macro_ovr") or test.get("roc_auc")),
        "McNemar p vs LogReg-pathway": mcnemar,
        "Protocol": "RFECV on outer train-fit matrices; reported CV scores use fold-wise train-fit/holdout-transform FC+FVA; held-out 30% test",
    }


def supplementary_table1() -> pd.DataFrame:
    metrics = pd.read_csv(ROOT / "results" / "clf_notebook_metrics.csv")
    feature_labels = {
        "all_data": "All Features",
        "prs": "PRS",
        "metabolites": "Metabolite Fold-Changes",
        "pathway": "Pathway Diff Scores",
        "reaction": "Reaction Diff Scores",
        "prs_metabolites": "PRS + Metabolite Fold-Changes",
        "prs_pathway": "PRS + Pathway Diff Scores",
        "prs_reaction": "PRS + Reaction Diff Scores",
    }
    model_labels = {
        "random_forest": "Random Forest",
        "logistic_regression": "Logistic Regression",
        "gradient_boosting": "Gradient Boosting",
        "adaboost": "AdaBoost",
        "xgboost": "XGBoost",
        "stacking": "Stacking Ensemble",
    }
    rows = []
    for _, row in metrics.iterrows():
        rows.append(
            {
                "Cohort": "DiCAD" if row["cohort"] == "DIABETES" else row["cohort"],
                "Feature set": feature_labels[row["feature_set"]],
                "Model": model_labels[row["model"]],
                "Selection": row["selection"],
                "n features": int(row["n_features"]),
                "CV Accuracy (mean ± std)": fmt_pm(
                    row["cv_accuracy_mean"], row["cv_accuracy_std"]
                ),
                "CV F1 (mean ± std)": fmt_pm(row["cv_f1_mean"], row["cv_f1_std"]),
                "CV AUC (mean ± std)": fmt_pm(row["cv_auc_mean"], row["cv_auc_std"]),
                "Test Accuracy": fmt4(row["test_accuracy"]),
                "Test F1": fmt4(row["test_f1"]),
                "Test AUC": fmt4(row["test_auc"]),
                "Protocol": (
                    "Leakage-free Metabolitics outer folds; inner 5-fold feature-count "
                    "selection reuses outer-training Metabolitics matrix"
                ),
            }
        )
    return pd.DataFrame(rows)


def event_censor_table() -> pd.DataFrame:
    frames = []
    for cohort in ("ROSMAP-AD", "MayoRNASeq-AD", "MayoRNASeq-PSP", "DIABETES"):
        frames.append(pd.read_csv(ROOT / cohort / "processed" / "cox" / "event_censor_table.csv"))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    t1 = table1_wide()
    cap1 = (
        "Table 1: Summary of cohort sample sizes, diagnosis breakdown, and train/test partition splits. "
        "DiCAD is split by participant; Mayo AD and Mayo PSP reuse the same 20 control Sample IDs."
    )
    save_csv(t1, OUT / "Table_1_cohort_sample_sizes_train_test_splits.csv", cap1)
    save_table_png(
        t1.drop(columns=["Analysis context"]),
        "Table 1",
        cap1,
        OUT / "Table_1_cohort_sample_sizes_train_test_splits.png",
    )

    s1 = supplementary_table1()
    cap_s1 = (
        "Supplementary Table 2: Nested classification performance on ROSMAP "
        "(binary AD versus control) and DiCAD (three-class). Metabolitics "
        "transformations are fitted on each outer-training fold and applied to "
        "its validation fold; inner feature-count tuning reuses the corresponding "
        "outer-training Metabolitics matrix."
    )
    save_csv(s1, OUT / "Supplementary_Table_1_classification_performance.csv", cap_s1)
    png_s1 = s1.drop(columns=["Protocol"])
    save_table_png(
        png_s1,
        "Supplementary Table 2",
        cap_s1,
        OUT / "Supplementary_Table_1_classification_performance.png",
    )

    heldout = s1[
        [
            "Cohort",
            "Feature set",
            "Model",
            "Selection",
            "n features",
            "Test Accuracy",
            "Test F1",
            "Test AUC",
            "Protocol",
        ]
    ].copy()
    cap_heldout = (
        "Supplementary Table 3: Held-out 30% test performance of the same classifiers "
        "as Supplementary Table 2. Nested cross-validation remains the primary "
        "classification evidence. Flux-only pathway and reaction feature sets did not "
        "outperform metabolite fold-changes on this split."
    )
    save_csv(
        heldout,
        OUT / "Supplementary_Table_3_heldout_classification.csv",
        cap_heldout,
    )

    ev = event_censor_table()
    cap_ev = (
        "Cox event and censoring summary used for hazard-ratio analyses (Figures 4–14). "
        "DiCAD duration is sampling age stored as AgeDeath, not death age."
    )
    save_csv(ev, OUT / "Cox_event_censor_summary.csv", cap_ev)
    save_table_png(ev, "Cox event / censoring summary", cap_ev, OUT / "Cox_event_censor_summary.png")

    print("Wrote", len(list(OUT.iterdir())), "files to", OUT)


if __name__ == "__main__":
    main()
