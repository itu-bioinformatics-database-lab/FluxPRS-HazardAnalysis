#!/usr/bin/env python3
"""Write supplementary-figure PNGs next to manuscript/supplementary/*.tex."""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUP = ROOT / "manuscript" / "supplementary"
BMC = ROOT / "BMC_figures"
PRIMARY = ROOT / "results" / "cox_hazard_ratios_long.csv"
PERSD = ROOT / "results" / "cox_hazard_ratios_long_per_sd_coxsample.csv"

_SPEC = importlib.util.spec_from_file_location("assemble_bmc_figures", ROOT / "scripts" / "assemble_bmc_figures.py")
_ABF = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_ABF)

S1 = {
    "ROSMAP-AD": "rosmap_sup.png",
    "DIABETES": "dicad_sup.png",
    "MayoRNASeq-AD": "mayoad_sup.png",
    "MayoRNASeq-PSP": "mayopsp_sup.png",
}
S5_PREFIX = {
    "ROSMAP-AD": "ROSMAP",
    "DIABETES": "DICAD",
    "MayoRNASeq-AD": "MAYOAD",
    "MayoRNASeq-PSP": "MAYOPSP",
}
SHAP_NMF = [
    ("Figure_17_SHAP_summary_pathway_test.png", "Figure_17.png"),
    ("Figure_18_SHAP_bar_pathway_test.png", "Figure_18.png"),
    ("Figure_19_confusion_matrix_RF_RFE_test.png", "Figure_19.png"),
    ("Figure_20_NMF_reconstruction_error.png", "Figure_20.png"),
    ("Figure_21_NMF_Factor3_vs_Factor2.png", "Figure_21.png"),
    ("Figure_22_NMF_factor_weight_boxplots.png", "Figure_22.png"),
    ("Figure_23_NMF_Factor3_density.png", "Figure_23.png"),
    ("Figure_24_RFECV_feature_type_distribution.png", "Figure_24.png"),
    ("Figure_25_top50_RFECV_features_by_NMF_factor.png", "Figure_25.png"),
]


def main() -> None:
    SUP.mkdir(parents=True, exist_ok=True)
    hr = pd.read_csv(PRIMARY)
    psd = pd.read_csv(PERSD)

    for cohort, dest in S1.items():
        spec = _ABF.COHORT_FIG[cohort]
        df = _ABF.cohort_model_table(hr, cohort)
        _ABF.plot_forest(df, spec["title"], SUP / dest, spec["order"])
        print(f"S1 {cohort}: {len(df)} PRS -> {dest}")

    for cohort, prefix in S5_PREFIX.items():
        spec = _ABF.COHORT_FIG[cohort]
        df = _ABF.cohort_model_table(psd, cohort)
        title = f"{spec['title']} (per-SD)"
        log_x = cohort == "MayoRNASeq-AD"
        _ABF.plot_summary(df, title, SUP / f"{prefix}_BARPIE_PERSD.png")
        _ABF.plot_dumbbell(df, title, SUP / f"{prefix}_DUMBBELL_PERSD.png", spec["order"], log_x=log_x)
        _ABF.plot_forest(df, title, SUP / f"{prefix}_FOREST_PERSD.png", spec["order"], log_x=log_x)
        print(f"S5 {cohort}: {len(df)} PRS -> {prefix}_*_PERSD.png")

    for src_name, dest_name in SHAP_NMF:
        src = BMC / src_name
        if not src.exists():
            print(f"warning: missing {src}")
            continue
        shutil.copy(src, SUP / dest_name)
        print(f"copied {src_name} -> {dest_name}")

    print("Wrote supplementary figures to", SUP)


if __name__ == "__main__":
    main()
