#!/usr/bin/env python3
"""Table 2: change in hazard ratio relative to the PRS-only baseline.

For each cohort, analysis (primary top-decile and per-SD on the Cox sample) and
predefined integrated configuration, the HR of the integrated model is compared
with the HR of the PRS-only model for the same PRS:

    delta_HR = HR(integrated configuration) - HR(PRS only)

Only PRSs with an estimable HR for both models are used. The script reports the
median delta_HR and the numbers of PRSs with a positive, exactly zero and negative
delta_HR. PRS-specific comparisons share the same individuals and omics-derived
scores, so no significance test is computed.

Inputs are the wide tables written by run_cox_hrs.py:
    hazard_rations/hazard_rations_{TAG}.csv
    hazard_rations/hazard_rations_{TAG}_per_sd_coxsample.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORTS = ("ROSMAP-AD", "MayoRNASeq-AD", "DIABETES", "MayoRNASeq-PSP")
HR_COPY_TAG = {
    "ROSMAP-AD": "ROSMAPAD",
    "MayoRNASeq-AD": "MayoRNASeqAD",
    "MayoRNASeq-PSP": "MayoRNASeqPSP",
    "DIABETES": "DIABETES",
}
COHORT_LABELS = {
    "ROSMAP-AD": "ROSMAP AD",
    "MayoRNASeq-AD": "Mayo RNAseq AD",
    "DIABETES": "DiCAD Diabetes",
    "MayoRNASeq-PSP": "Mayo RNAseq PSP",
}
ANALYSES = {
    "primary": "",
    "per_sd": "_per_sd_coxsample",
}
BASELINE_COLUMN = "Prs Score"
# (label used in Table 2, column in the wide HR table)
CONFIGURATIONS = (
    ("Metabolomics + PRS", "Metabolite Levels PRS"),
    ("Fluxomics (Pathway-based) + PRS", "Pathway Diff Scores PRS"),
    ("Fluxomics (Reaction-based) + PRS", "Reaction Diff Scores PRS"),
    ("Fluxomics (Pathway-based) + Metabolomics + PRS", "Metabolite Levels Pathway Combined PRS"),
    ("Fluxomics (Reaction-based) + Metabolomics + PRS", "Metabolite Levels Reaction Combined PRS"),
)


def hr_table_path(hr_dir: Path, cohort: str, analysis: str) -> Path:
    return hr_dir / f"hazard_rations_{HR_COPY_TAG[cohort]}{ANALYSES[analysis]}.csv"


def delta_hr(table: pd.DataFrame, column: str) -> pd.Series:
    pairs = table[["PRS name", column, BASELINE_COLUMN]].dropna()
    return pd.Series(
        pairs[column].to_numpy() - pairs[BASELINE_COLUMN].to_numpy(),
        index=pairs["PRS name"].to_numpy(),
        name="delta_hr",
    )


def summarize(delta: pd.Series) -> dict:
    values = delta.to_numpy(dtype=float)
    return {
        "n_prs": int(values.size),
        "median_delta_hr": float(np.median(values)) if values.size else np.nan,
        "n_positive": int((values > 0).sum()),
        "n_zero": int((values == 0).sum()),
        "n_negative": int((values < 0).sum()),
    }


def collect(hr_dir: Path, cohorts: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows, per_prs_rows = [], []
    for cohort in cohorts:
        for analysis in ANALYSES:
            path = hr_table_path(hr_dir, cohort, analysis)
            if not path.exists():
                print(f"[skip] {path} not found")
                continue
            table = pd.read_csv(path)
            for label, column in CONFIGURATIONS:
                delta = delta_hr(table, column)
                summary_rows.append(
                    {"cohort": cohort, "analysis": analysis, "configuration": label, **summarize(delta)}
                )
                per_prs_rows.extend(
                    {"cohort": cohort, "analysis": analysis, "configuration": label,
                     "pgs_id": pgs_id, "delta_hr": value}
                    for pgs_id, value in delta.items()
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(per_prs_rows)


def format_median(value: float) -> str:
    if pd.isna(value):
        return ""
    rounded = round(float(value), 3)
    if rounded == 0:
        return "0.000"
    return f"{rounded:+.3f}".replace("-", "\u2212")


def table2(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort in summary["cohort"].unique():
        block = summary[summary["cohort"] == cohort]
        n_prs = int(block["n_prs"].max())
        for label, _ in CONFIGURATIONS:
            row = {"Cohort": f"{COHORT_LABELS[cohort]} ({n_prs} PRSs)", "Configuration": label}
            for analysis, name in (("primary", "Primary"), ("per_sd", "Per-SD")):
                hit = block[(block["analysis"] == analysis) & (block["configuration"] == label)]
                if hit.empty:
                    continue
                rec = hit.iloc[0]
                row[f"{name} median dHR"] = format_median(rec["median_delta_hr"])
                row[f"{name} dHR >0 / =0 / <0"] = f"{rec['n_positive']} / {rec['n_zero']} / {rec['n_negative']}"
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--hr-dir", type=Path, default=None, help="default: <project-root>/hazard_rations")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <project-root>/results")
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS), choices=list(COHORTS))
    args = parser.parse_args()

    hr_dir = args.hr_dir or args.project_root / "hazard_rations"
    out_dir = args.out_dir or args.project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, per_prs = collect(hr_dir, args.cohorts)
    if summary.empty:
        raise SystemExit(f"No hazard-ratio tables found in {hr_dir}")

    summary.to_csv(out_dir / "delta_hr_summary.csv", index=False)
    per_prs.to_csv(out_dir / "delta_hr_per_prs.csv", index=False)
    table = table2(summary)
    table.to_csv(out_dir / "table2_delta_hr.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table.to_string(index=False))
    print(f"\nWritten to {out_dir}")


if __name__ == "__main__":
    main()
