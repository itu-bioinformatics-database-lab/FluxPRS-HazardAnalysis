#!/usr/bin/env python3
"""Assemble BMC Bioinformatics figures from leakage-safe FluxPRS outputs."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "BMC_figures"
ORIG = Path("/tmp/bmc_figs")
HR_PATH = ROOT / "results" / "cox_hazard_ratios_long.csv"

# Paper Fig. 4–14 use 11 model classes (fold-changes are extra in our Cox run).
MODEL_KEYS = [
    ("metabolite_levels", False, "Metabolomics", "#2ec4d6", "o"),
    ("metabolite_levels", True, "Metabolomics + PRS (No Fluxomics)", "#2d6a4f", "o"),
    ("pathway", False, "Fluxomics (Pathway-based)", "#f4d35e", "s"),
    ("pathway", True, "Fluxomics (Pathway-based) + PRS", "#f4d35e", "s"),
    ("reaction", False, "Fluxomics (Reaction-based)", "#6c757d", "s"),
    ("reaction", True, "Fluxomics (Reaction-based) + PRS", "#9b5de5", "s"),
    ("metabolite_pathway", False, "Fluxomics (Pathway-based) + Metabolomics (No PRS)", "#e07a3d", "s"),
    ("metabolite_pathway", True, "Fluxomics (Pathway-based) + Metabolomics + PRS", "#3d5a80", "s"),
    ("metabolite_reaction", False, "Fluxomics (Reaction-based) + Metabolomics", "#e87bb0", "s"),
    ("metabolite_reaction", True, "Fluxomics (Reaction-based) + Metabolomics + PRS", "#8b5a2b", "s"),
    ("prs_only", True, "PRS", "#e63946", "o"),
]
FLUX_LABELS = {lab for _, _, lab, _, _ in MODEL_KEYS if lab.startswith("Fluxomics")}
NO_FLUX = {lab for _, _, lab, _, _ in MODEL_KEYS if lab not in FLUX_LABELS}

COHORT_FIG = {
    "ROSMAP-AD": {
        "forest": "Figure_4_ROSMAP_hazard_ratios_forest_plot.png",
        "summary": "Figure_5_ROSMAP_top_performing_models.png",
        "dumbbell": "Figure_6_ROSMAP_HR_improvement_dumbbell.png",
        "title": "ROSMAP Alzheimer’s disease",
        "order": [
            "PGS004227", "PGS000779", "PGS000026", "PGS001349", "PGS004600",
            "PGS004286", "PGS003953", "PGS003956", "PGS004285", "PGS002731",
            "PGS004228", "PGS000025", "PGS004288", "PGS003954", "PGS003955",
            "PGS004287", "PGS004008", "PGS001775", "PGS004116", "PGS001828",
            "PGS004289", "PGS003958", "PGS001348", "PGS003957", "PGS004062",
            "PGS002280", "PGS004590", "PGS004229", "PGS000945", "PGS004863",
        ],
    },
    "DIABETES": {
        "forest": "Figure_7_DiCAD_hazard_ratios_forest_plot.png",
        "summary": "Figure_8_DiCAD_top_performing_models.png",
        "dumbbell": "Figure_9_DiCAD_HR_improvement_dumbbell.png",
        "title": "DiCAD Diabetes",
        "order": None,
    },
    "MayoRNASeq-AD": {
        "forest": "Figure_10_Mayo_AD_hazard_ratios_forest_plot.png",
        "summary": "Figure_11_Mayo_AD_top_performing_models.png",
        "dumbbell": "Figure_12_Mayo_AD_HR_improvement_dumbbell.png",
        "title": "Mayo RNAseq Alzheimer’s Disease",
        "order": None,
    },
    "MayoRNASeq-PSP": {
        "forest": "Figure_13_Mayo_PSP_hazard_ratios_forest_plot.png",
        "summary": None,
        "dumbbell": "Figure_14_Mayo_PSP_HR_improvement_dumbbell.png",
        "title": "Mayo RNAseq Progressive Supranuclear Palsy",
        "order": None,
    },
}


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)


def model_lookup(part: pd.DataFrame) -> dict[str, float]:
    out = {}
    for feat, with_prs, label, _, _ in MODEL_KEYS:
        if feat == "prs_only":
            row = part.loc[part["feature_set"] == "prs_only"]
        elif with_prs:
            row = part.loc[(part["feature_set"] == feat) & (part["pgs_id"].notna())]
            row = row.loc[row["display_name"].astype(str).str.endswith(" PRS")]
        else:
            row = part.loc[(part["feature_set"] == feat) & (part["pgs_id"].isna())]
            if row.empty:
                # biomarker-only rows are stored once with pgs_id NA at cohort level
                row = part
        if row.empty:
            continue
        hr = pd.to_numeric(row["hr"].iloc[0], errors="coerce")
        if np.isfinite(hr):
            out[label] = float(hr)
    return out


def cohort_model_table(hr: pd.DataFrame, cohort: str) -> pd.DataFrame:
    sub = hr.loc[hr["cohort"] == cohort].copy()
    bio = sub.loc[sub["pgs_id"].isna()].set_index("feature_set")
    rows = []
    pgs_ids = [p for p in sub["pgs_id"].dropna().unique()]
    for pgs in pgs_ids:
        part = sub.loc[sub["pgs_id"] == pgs]
        rec = {"pgs_id": pgs}
        for feat, with_prs, label, _, _ in MODEL_KEYS:
            if feat == "prs_only":
                vals = part.loc[part["feature_set"] == "prs_only", "hr"]
            elif with_prs:
                vals = part.loc[part["feature_set"] == feat, "hr"]
            else:
                vals = bio.loc[[feat], "hr"] if feat in bio.index else pd.Series(dtype=float)
            hr_val = pd.to_numeric(vals.iloc[0], errors="coerce") if len(vals) else np.nan
            rec[label] = float(hr_val) if np.isfinite(hr_val) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def ordered_pgs(df: pd.DataFrame, preferred: list[str] | None) -> list[str]:
    have = list(df["pgs_id"])
    if not preferred:
        return have
    ordered = [p for p in preferred if p in have]
    ordered += [p for p in have if p not in ordered]
    return ordered


def page_columns(n: int) -> int:
    return 2 if n > 24 else 1


def split_columns(items: list[str], n_cols: int) -> list[list[str]]:
    if n_cols <= 1:
        return [items]
    size = int(np.ceil(len(items) / n_cols))
    return [items[i * size : (i + 1) * size] for i in range(n_cols) if items[i * size : (i + 1) * size]]


def finite_xlim(values, log_x: bool) -> tuple[float, float]:
    vals = [float(v) for v in values if np.isfinite(v) and (not log_x or v > 0)]
    if not vals:
        return (0.5, 2.0)
    lo, hi = min(vals), max(vals)
    if log_x:
        return (max(lo / 1.2, 1e-4), hi * 1.2)
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    return (lo - pad, hi + pad)


def plot_forest(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    preferred: list[str] | None,
    log_x: bool = False,
) -> None:
    pgs = ordered_pgs(df, preferred)
    n_cols = page_columns(len(pgs))
    chunks = split_columns(pgs, n_cols)
    n_rows = max(len(c) for c in chunks)
    row_h = 0.155 if n_cols > 1 else 0.32
    fig_w = 12.8 if n_cols > 1 else 12.5
    fig_h = min(9.4, max(5.4, row_h * n_rows + 2.1))
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), sharex=True)
    if n_cols == 1:
        axes = [axes]
    labels = [lab for _, _, lab, _, _ in MODEL_KEYS]
    xlim = finite_xlim(df[labels].to_numpy(dtype=float).ravel(), log_x)
    xlabel = "Hazard Ratio (log scale)" if log_x else "Hazard Ratio"
    marker_s = 22 if n_cols > 1 else 42
    tick_fs = 5.5 if n_cols > 1 else 9
    for ax, chunk in zip(axes, chunks):
        ymap = {p: i for i, p in enumerate(chunk)}
        for feat, with_prs, label, color, marker in MODEL_KEYS:
            xs, ys = [], []
            for pgs_id in chunk:
                val = df.loc[df["pgs_id"] == pgs_id, label]
                if val.empty or not np.isfinite(val.iloc[0]):
                    continue
                xs.append(val.iloc[0])
                ys.append(ymap[pgs_id])
            if not xs:
                continue
            ax.scatter(
                xs, ys, c=color, marker=marker, s=marker_s, zorder=3,
                edgecolors="white", linewidths=0.25, label=label,
            )
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(chunk, fontsize=tick_fs)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=8)
        if log_x:
            ax.set_xscale("log")
            ax.axvline(1.0, color="0.4", ls=":", lw=1.0)
        style_ax(ax)
    handles, handle_labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"Hazard Ratios Across Different Models per PRS ({title})", fontsize=11, y=0.995)
    fig.legend(
        handles, handle_labels, loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=3, fontsize=7, frameon=False, title="Model configuration",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_summary(df: pd.DataFrame, title: str, out_path: Path) -> None:
    winners = []
    flux_flags = []
    labels = [lab for _, _, lab, _, _ in MODEL_KEYS]
    for _, row in df.iterrows():
        vals = {lab: row[lab] for lab in labels if np.isfinite(row[lab])}
        if not vals:
            continue
        winner = max(vals, key=vals.get)
        winners.append(winner)
        flux_flags.append(winner in FLUX_LABELS)  # "No Fluxomics" labels are not in FLUX_LABELS
    if not winners:
        return
    counts = pd.Series(winners).value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    color_map = {lab: col for _, _, lab, col, _ in MODEL_KEYS}
    y = np.arange(len(counts))
    axes[0].barh(y, counts.values, color=[color_map.get(i, "#888") for i in counts.index])
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(counts.index, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Frequency (Count)")
    axes[0].set_title("a) Top-Performing Model Count")
    style_ax(axes[0])
    n_flux = int(np.sum(flux_flags))
    n_non = int(len(flux_flags) - n_flux)
    pie_vals = [n_flux, n_non] if n_non else [n_flux]
    pie_labs = ["Fluxomics-Involving", "Non-Fluxomics"] if n_non else ["Fluxomics-Involving"]
    pie_cols = ["#d9a8d4", "#5b7c99"] if n_non else ["#d9a8d4"]
    wedges, texts, autotexts = axes[1].pie(
        pie_vals, colors=pie_cols, autopct="%1.1f%%", startangle=90,
        pctdistance=0.62,
    )
    axes[1].legend(wedges, pie_labs, loc="lower center", frameon=False)
    axes[1].set_title("b) Proportion of Top-Performing Models:\nFluxomics vs. Non-Fluxomics")
    fig.suptitle(f"Highest-hazard-ratio model configurations ({title})", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dumbbell(
    df: pd.DataFrame,
    title: str,
    out_path: Path,
    preferred: list[str] | None,
    log_x: bool = False,
) -> None:
    pgs = ordered_pgs(df, preferred)
    color_map = {lab: col for _, _, lab, col, _ in MODEL_KEYS}
    marker_map = {lab: m for _, _, lab, _, m in MODEL_KEYS}
    n_cols = page_columns(len(pgs))
    chunks = split_columns(pgs, n_cols)
    n_rows = max(len(c) for c in chunks)
    row_h = 0.155 if n_cols > 1 else 0.32
    fig_w = 12.8 if n_cols > 1 else 12.2
    fig_h = min(9.4, max(5.4, row_h * n_rows + 2.0))
    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), sharex=True)
    if n_cols == 1:
        axes = [axes]
    labels = [lab for lab in color_map]
    xlim = finite_xlim(df[labels].to_numpy(dtype=float).ravel(), log_x)
    xlabel = "Hazard Ratio (HR, log scale)" if log_x else "Hazard Ratio (HR)"
    used = set()
    marker_s = 18 if n_cols > 1 else 36
    best_s = 22 if n_cols > 1 else 46
    tick_fs = 5.5 if n_cols > 1 else 9
    for ax, chunk in zip(axes, chunks):
        ymap = {p: i for i, p in enumerate(chunk)}
        for pgs_id in chunk:
            row = df.loc[df["pgs_id"] == pgs_id].iloc[0]
            base = row["PRS"]
            if not np.isfinite(base):
                continue
            others = {
                lab: row[lab]
                for lab in color_map
                if lab != "PRS" and np.isfinite(row[lab])
            }
            if not others:
                ax.scatter(base, ymap[pgs_id], c="#e63946", marker="o", s=marker_s, zorder=4)
                used.add("PRS")
                continue
            best_lab = max(others, key=others.get)
            best = others[best_lab]
            ax.plot([base, best], [ymap[pgs_id], ymap[pgs_id]], color="#b0b0b0", lw=1.0, zorder=1)
            ax.scatter(base, ymap[pgs_id], c="#e63946", marker="o", s=marker_s, zorder=4)
            ax.scatter(
                best, ymap[pgs_id], c=color_map[best_lab], marker=marker_map[best_lab],
                s=best_s, zorder=5, edgecolors="white", linewidths=0.3,
            )
            used.add("PRS")
            used.add(best_lab)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels(chunk, fontsize=tick_fs)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel, fontsize=8)
        if log_x:
            ax.set_xscale("log")
            ax.axvline(1.0, color="0.4", ls=":", lw=1.0)
        style_ax(ax)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor="#e63946",
                      markersize=6, label="PRS Only (Baseline)")]
    for feat, with_prs, lab, col, mk in MODEL_KEYS:
        if lab in used and lab != "PRS":
            handles.append(Line2D([0], [0], marker=mk, color="none", markerfacecolor=col,
                                  markersize=6, label=lab))
    fig.suptitle(f"PRS-only and highest alternative hazard ratios ({title})", fontsize=11, y=0.995)
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02),
        ncol=3, fontsize=7, frameon=False,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def beeswarm(ax, shap_vals: pd.DataFrame, feat_vals: pd.DataFrame, features: list[str]) -> None:
    rng = np.random.default_rng(42)
    for i, feat in enumerate(features):
        y = np.full(len(shap_vals), i, dtype=float)
        x = shap_vals[feat].to_numpy(dtype=float)
        fv = feat_vals[feat].to_numpy(dtype=float)
        # jitter within SHAP bins
        order = np.argsort(x)
        y_j = y.copy()
        # local density jitter
        y_j += rng.normal(0, 0.12, size=len(y))
        finite = np.isfinite(fv)
        if finite.any():
            ranks = np.zeros_like(fv)
            ranks[finite] = pd.Series(fv[finite]).rank(pct=True).to_numpy()
            ranks[~finite] = 0.5
        else:
            ranks = np.full_like(fv, 0.5, dtype=float)
        ax.scatter(x, y_j, c=ranks, cmap="coolwarm", s=12, alpha=0.85, linewidths=0, vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels(features)
    ax.axvline(0, color="0.3", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP value (impact on model output)")
    style_ax(ax)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)


def shap_direction_bar(ax, shap_vals: pd.DataFrame, feat_vals: pd.DataFrame, features: list[str]) -> None:
    mag = shap_vals[features].abs().mean()
    colors = []
    for feat in features:
        corr = np.corrcoef(feat_vals[feat].to_numpy(dtype=float), shap_vals[feat].to_numpy(dtype=float))[0, 1]
        colors.append("#e74c3c" if np.isfinite(corr) and corr >= 0 else "#4c78a8")
    y = np.arange(len(features))
    ax.barh(y, mag.loc[features].to_numpy(), color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Average SHAP Impact (Mean |SHAP|)")
    style_ax(ax)
    handles = [
        Line2D([0], [0], color="#e74c3c", lw=6, label="High levels → POSITIVE Group (1)"),
        Line2D([0], [0], color="#4c78a8", lw=6, label="High levels → NEGATIVE Group (0)"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)


def load_aligned_shap(shap_path: Path, feat_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    shap_df = pd.read_csv(shap_path)
    feat_df = pd.read_csv(feat_path)
    shap_df = shap_df.rename(columns={"sample_id": "Sample ID"})
    feat_df = feat_df.rename(columns={"sample_id": "Sample ID"})
    merged = shap_df.merge(feat_df, on="Sample ID", suffixes=("_shap", "_feat"))
    shap_cols = [c for c in shap_df.columns if c not in {"Sample ID", "unique_row_id", "participant_id"}]
    feat_cols = [c[:-5] if c.endswith("_feat") else c for c in merged.columns]
    shap_aligned = pd.DataFrame({c: merged[f"{c}_shap"] if f"{c}_shap" in merged.columns else merged[c] for c in shap_cols if c in shap_df.columns})
    feat_aligned = pd.DataFrame()
    for c in shap_cols:
        if f"{c}_feat" in merged.columns:
            feat_aligned[c] = pd.to_numeric(merged[f"{c}_feat"], errors="coerce")
        elif c in feat_df.columns:
            # already merged without suffix if unique
            feat_aligned[c] = pd.to_numeric(merged[c], errors="coerce")
    # If merge suffixes didn't apply (no overlapping names besides ID), use feat_df reindex
    if feat_aligned.empty or feat_aligned.shape[1] < 5:
        feat_idx = feat_df.set_index("Sample ID")
        feat_aligned = feat_idx.reindex(shap_df["Sample ID"])[shap_cols].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
        shap_aligned = shap_df[shap_cols].apply(pd.to_numeric, errors="coerce")
    return shap_aligned, feat_aligned


def confusion_plot(matrix, labels, out_path: Path, title: str) -> None:
    mat = np.asarray(matrix, dtype=float)
    row_pct = mat / mat.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(mat, cmap="YlGnBu")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"{labels[0]} (0)", f"{labels[1]} (1)"])
    ax.set_yticklabels([f"{labels[0]} (0)", f"{labels[1]} (1)"])
    ax.set_xlabel("Predicted Group")
    ax.set_ylabel("True Group")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            txt = f"{int(mat[i, j])}\n({row_pct[i, j]:.1f}%)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Figures 1–3 are methods diagrams; copy BMC originals when present.
    orig_copies = [
        ("image1.jpg", "Figure_1_PRS_Base_Data_Inclusion_Exclusion_Flowchart.jpg"),
        ("image2.jpg", "Figure_2_Classification_modeling_workflow.jpg"),
        ("image3.jpg", "Figure_3_Systems_Level_Workflow_PRS_GEM_FVA.jpg"),
    ]
    for src_name, dest_name in orig_copies:
        src = ORIG / src_name
        dest = OUT / dest_name
        if src.exists():
            shutil.copy(src, dest)
        elif not dest.exists():
            print(f"warning: missing methods figure {src} and {dest}")

    hr = pd.read_csv(HR_PATH)
    for cohort, spec in COHORT_FIG.items():
        df = cohort_model_table(hr, cohort)
        plot_forest(df, spec["title"], OUT / spec["forest"], spec["order"])
        if spec["summary"]:
            plot_summary(df, spec["title"], OUT / spec["summary"])
        plot_dumbbell(df, spec["title"], OUT / spec["dumbbell"], spec["order"])
        print(f"wrote HR figures for {cohort}: {len(df)} PRS")

    # Figures 15–18: leakage-safe held-out SHAP (fold-changes ≈ metabolites; pathway).
    fc_shap, fc_feat = load_aligned_shap(
        ROOT / "ROSMAP-AD/processed/cv/fold_changes/test_shap_values.csv",
        ROOT / "ROSMAP-AD/processed/metabolitics_output/test/fold_changes.csv",
    )
    pw_shap, pw_feat = load_aligned_shap(
        ROOT / "ROSMAP-AD/processed/cv/X_pathway/test_shap_values.csv",
        ROOT / "ROSMAP-AD/processed/metabolitics_output/test/X_pathway.csv",
    )
    fc_top20 = fc_shap.abs().mean().sort_values(ascending=False).head(20).index.tolist()
    fc_top15 = fc_top20[:15]
    pw_top20 = pw_shap.abs().mean().sort_values(ascending=False).head(20).index.tolist()
    pw_top15 = pw_top20[:15]

    fig, ax = plt.subplots(figsize=(8.6, 10.2))
    beeswarm(ax, fc_shap, fc_feat, fc_top20)
    ax.set_title("Test-set SHAP summary (metabolite fold-changes)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Feature value")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_15_SHAP_summary_metabolite_fold_changes.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    shap_direction_bar(ax, fc_shap, fc_feat, fc_top15)
    ax.set_title("Metabolic biomarkers: group association and importance (test set, fold-changes)")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_16_SHAP_bar_metabolite_fold_changes.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 10.2))
    beeswarm(ax, pw_shap, pw_feat, pw_top20)
    ax.set_title("Test-set SHAP summary (pathway flux features)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Feature value")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_17_SHAP_summary_pathway_test.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    shap_direction_bar(ax, pw_shap, pw_feat, pw_top15)
    ax.set_title("Test-set global feature importance (pathway flux features)")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_18_SHAP_bar_pathway_test.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    summary = json.loads((ROOT / "ROSMAP-AD/processed/clf_extras/run_summary.json").read_text())
    cm = summary["rf_rfe_pathway"]["test"]["confusion_matrix"]
    confusion_plot(
        cm["matrix"],
        ["Control", "Positive"],
        OUT / "Figure_19_confusion_matrix_RF_RFE_test.png",
        "Confusion matrix of the pathway RFECV Random Forest on the independent test set",
    )

    # Figure 20: NMF reconstruction error on RFE reaction features (train).
    rx = pd.read_csv(ROOT / "ROSMAP-AD/processed/metabolitics_output/train/X_reaction_diff.csv")
    rfe_rx = pd.read_csv(ROOT / "ROSMAP-AD/processed/clf_extras/rfe_reaction_features.csv")["feature"].tolist()
    drop = {"Sample ID", "Diagnosis", "sample_id", "unique_row_id", "participant_id"}
    rx_num = rx[[c for c in rfe_rx if c in rx.columns]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = MinMaxScaler().fit_transform(rx_num.to_numpy(dtype=float))
    x = np.clip(x, 0, None)
    ks = list(range(2, 11))
    errors = []
    for k in ks:
        model = NMF(n_components=k, init="nndsvd", random_state=42, max_iter=2000)
        w = model.fit_transform(x)
        recon = w @ model.components_
        errors.append(float(np.linalg.norm(x - recon, ord="fro")))
        print(f"  NMF k={k} error={errors[-1]:.3f}")
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.plot(ks, errors, "-o", color="#3d5a80")
    ax.axvline(5, color="red", ls="--", label="k=5 used in analysis")
    ax.set_xlabel("Number of Latent Components (k)")
    ax.set_ylabel("Reconstruction Error")
    ax.legend(frameon=False)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_20_NMF_reconstruction_error.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    w_tr = pd.read_csv(ROOT / "ROSMAP-AD/processed/clf_extras/nmf_W_train.csv")
    w_te = pd.read_csv(ROOT / "ROSMAP-AD/processed/clf_extras/nmf_W_test.csv")
    w = pd.concat([w_tr, w_te], ignore_index=True)
    w["group"] = w["Diagnosis"].map(lambda x: "Control" if str(x).lower() in {"healthy", "control"} else "AD")
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    for grp, color in [("Control", "#e74c3c"), ("AD", "#4c78a8")]:
        part = w.loc[w["group"] == grp]
        ax.scatter(part["Factor_3"], part["Factor_2"], c=color, s=28, alpha=0.75, label=grp)
    ax.set_xlabel("NMF Factor 3")
    ax.set_ylabel("NMF Factor 2")
    ax.set_title("Spatial distribution of patient embeddings across NMF Factor 3 vs Factor 2")
    ax.legend(title="Diagnosis", frameon=True)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_21_NMF_Factor3_vs_Factor2.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, factor, color_ad in zip(axes, ["Factor_3", "Factor_2"], ["#4c78a8", "#4c78a8"]):
        data = [w.loc[w["group"] == "Control", factor], w.loc[w["group"] == "AD", factor]]
        bp = ax.boxplot(data, labels=["Control", "AD"], patch_artist=True, widths=0.55)
        bp["boxes"][0].set_facecolor("#e74c3c")
        bp["boxes"][1].set_facecolor("#4c78a8")
        for box in bp["boxes"]:
            box.set_alpha(0.75)
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel(f"NMF {factor.replace('_', ' ')} Weight")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    fig.suptitle("Comparative analysis of latent factor weight distributions")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_22_NMF_factor_weight_boxplots.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for grp, color in [("Control", "#e74c3c"), ("AD", "#4c78a8")]:
        vals = w.loc[w["group"] == grp, "Factor_3"].to_numpy(dtype=float)
        ax.hist(vals, bins=30, density=True, alpha=0.25, color=color)
        if len(vals) > 2:
            from scipy.stats import gaussian_kde
            grid = np.linspace(min(-0.3, vals.min()), max(1.5, vals.max()), 200)
            kde = gaussian_kde(vals)
            ax.plot(grid, kde(grid), color=color, lw=2, label=grp)
            ax.fill_between(grid, kde(grid), color=color, alpha=0.18)
    ax.set_xlabel("NMF Factor 3")
    ax.set_ylabel("Density")
    ax.legend(title="Diagnosis", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_23_NMF_Factor3_density.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 24: the implemented reaction-feature selection path feeding NMF.
    rx_feats = pd.read_csv(ROOT / "ROSMAP-AD/processed/clf_extras/rfe_reaction_features.csv")["feature"]
    n_initial = len([c for c in rx.columns if c not in drop])
    n_coarse = int(summary["rf_rfe_reaction"]["rfe"]["n_coarse"])
    selection_counts = pd.Series(
        {
            "Initial reaction\ndiff-score features": n_initial,
            "Coarse RFE\nretained": n_coarse,
            "RFECV-selected\nNMF inputs": int(len(rx_feats)),
        }
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    bars = ax.barh(selection_counts.index, selection_counts.values, color=["#9aa0a6", "#d08c60", "#c75b5b"])
    for bar, val in zip(bars, selection_counts.values):
        ax.text(val * 1.04, bar.get_y() + bar.get_height() / 2, f"{val:,}", va="center")
    ax.set_xscale("log")
    ax.set_xlabel("Number of reaction features (log scale)")
    ax.set_title("Reaction-feature selection pipeline used before NMF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "Figure_24_RFECV_feature_type_distribution.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 25: actual reaction features with the largest NMF loadings.
    h = pd.read_csv(ROOT / "ROSMAP-AD/processed/clf_extras/nmf_H.csv", index_col=0)
    factors = ["Factor_3", "Factor_2"]
    loading = h.loc[factors].T
    top_features = loading.abs().max(axis=1).sort_values(ascending=False).head(50).index
    plot_loading = loading.loc[top_features, factors].iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 12.0))
    im = ax.imshow(plot_loading.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Factor 3", "Factor 2"])
    ax.set_yticks(np.arange(len(plot_loading)))
    ax.set_yticklabels(plot_loading.index, fontsize=7)
    ax.set_xlabel("NMF factor")
    ax.set_ylabel("Reaction feature")
    ax.set_title("Top 50 reaction features by NMF loading")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="NMF loading weight")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_25_top50_RFECV_features_by_NMF_factor.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Wrote", len(list(OUT.iterdir())), "files to", OUT)


if __name__ == "__main__":
    main()
