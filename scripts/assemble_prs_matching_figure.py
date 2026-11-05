#!/usr/bin/env python3
"""Build Supplementary Figure S1 from Kaan/TRUBA matching_summary.tsv files."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "manuscript" / "supplementary" / "Fig_PGS_variant_matching_panels.png"

SOURCES = [
    ("ROSMAP AD", ROOT / "incoming/truba_prs_rosmap/matching_summary.tsv", False),
    ("DiCAD Diabetes", ROOT / "incoming/kaan_pgs_scores/PRS_results_DiCAD/matching_summary.tsv", True),
    ("Mayo RNAseq AD", ROOT / "incoming/kaan_pgs_scores/PRS_results_MayoRNAseq/matching_summary.tsv", False),
    ("Mayo RNAseq PSP", ROOT / "incoming/kaan_pgs_scores/PRS_results_MayoRNAseq/matching_summary.tsv", True),
]
COLORS = {
    "ROSMAP AD": "#4C78A8",
    "DiCAD Diabetes": "#F58518",
    "Mayo RNAseq AD": "#54A24B",
    "Mayo RNAseq PSP": "#E45756",
}


def load_cohort(path: Path, keep_psp: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    is_psp = df["MODEL_ID"].astype(str).str.contains("PMC7116626", na=False)
    df = df.loc[is_psp if keep_psp else ~is_psp].copy()
    df["pct_matched"] = pd.to_numeric(df["PCT_MATCHED_OF_FILE"], errors="coerce")
    df["pct_weight"] = pd.to_numeric(df["PCT_ABS_WEIGHT_MATCHED"], errors="coerce")
    df["n_file"] = pd.to_numeric(df["N_VARIANTS_FILE"], errors="coerce")
    return df


def jitter(n: int, rng: np.random.Generator, scale: float = 0.12) -> np.ndarray:
    return rng.normal(0.0, scale, size=n)


def box_points(ax, values_by_cohort: dict[str, np.ndarray], ylabel: str) -> None:
    labels = list(values_by_cohort)
    data = [values_by_cohort[k] for k in labels]
    bp = ax.boxplot(
        data,
        positions=np.arange(1, len(labels) + 1),
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "0.35"},
        capprops={"color": "0.35"},
        boxprops={"color": "0.35"},
    )
    rng = np.random.default_rng(42)
    for i, (lab, vals) in enumerate(values_by_cohort.items(), start=1):
        bp["boxes"][i - 1].set_facecolor(COLORS[lab])
        bp["boxes"][i - 1].set_alpha(0.35)
        ax.scatter(
            np.full(len(vals), i) + jitter(len(vals), rng),
            vals,
            s=18,
            c=COLORS[lab],
            alpha=0.85,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )
    ax.axhline(50, color="0.35", ls="--", lw=0.9)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(-2, 104)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)


def main() -> None:
    frames = []
    for label, path, keep_psp in SOURCES:
        part = load_cohort(path, keep_psp)
        part["cohort"] = label
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.6))
    cohorts = [lab for lab, _, _ in SOURCES]
    box_points(axes[0], {c: df.loc[df.cohort == c, "pct_matched"].to_numpy() for c in cohorts}, "Matched variants (%)")
    axes[0].set_title("a")
    box_points(axes[1], {c: df.loc[df.cohort == c, "pct_weight"].to_numpy() for c in cohorts}, r"Matched $|\beta|$ (%)")
    axes[1].set_title("b")

    ax = axes[2]
    for cohort in cohorts:
        part = df.loc[df.cohort == cohort]
        ax.scatter(
            part["n_file"],
            part["pct_matched"],
            s=22,
            c=COLORS[cohort],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
            label=cohort,
        )
    ax.set_xscale("log")
    ax.axhline(50, color="0.35", ls="--", lw=0.9)
    ax.set_xlabel("Variants in score")
    ax.set_ylabel("Matched variants (%)")
    ax.set_ylim(-2, 104)
    ax.set_title("c")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", OUT, "n=", len(df))


if __name__ == "__main__":
    main()
