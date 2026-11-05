#!/usr/bin/env python3
"""Write 5-fold Metabolitics inputs from the ROSMAP and DiCAD training sets.

20 jobs = 2 cohorts x 5 folds x {cv_train, cv_val}. Each CSV is a raw
metabolite-level matrix. TRUBA fits FoldChangeScaler and reaction-diff
on cv_train healthy only, then transforms cv_val (fit_train_transform_holdout).

Fold assignment is the saved StratifiedKFold / StratifiedGroupKFold table
in processed/cv/fold_ids.csv (random_state=42). Held-out test is untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
for path in (str(PREPROCESSING_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cohort_splits import N_CV_SPLITS, RANDOM_STATE  # noqa: E402
from fold_change import DEFAULT_REFERENCE_LABEL  # noqa: E402

COHORT_PLATFORMS = {
    "ROSMAP-AD": "MyRIAD Metabolomics",
    "DIABETES": "Diabetes Metabolomics",
}


def _json_dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _counts(df: pd.DataFrame, ids: pd.DataFrame) -> dict:
    labels = df["Diagnosis"].astype(str)
    return {
        "n_rows": int(len(df)),
        "n_participants": int(ids["participant_id"].nunique()),
        "labels": {str(k): int(v) for k, v in labels.value_counts().items()},
        "healthy_rows": int((labels == DEFAULT_REFERENCE_LABEL).sum()),
        "healthy_participants": int(
            ids.loc[ids["metabolitics_label"] == DEFAULT_REFERENCE_LABEL, "participant_id"].nunique()
        ),
    }


def prepare_cohort(project_root: Path, cohort: str, n_splits: int) -> dict:
    cohort_dir = project_root / cohort
    train_csv = pd.read_csv(cohort_dir / "processed" / "metabolitics_input" / "train.csv")
    train_ids = pd.read_csv(cohort_dir / "processed" / "splits" / "train_ids.csv")
    fold_ids = pd.read_csv(cohort_dir / "processed" / "cv" / "fold_ids.csv")
    train_csv["Sample ID"] = train_csv["Sample ID"].astype(str)
    train_ids["sample_id"] = train_ids["sample_id"].astype(str)
    train_ids["unique_row_id"] = train_ids["unique_row_id"].astype(str)
    train_ids["participant_id"] = train_ids["participant_id"].astype(str)
    train_ids["metabolitics_label"] = train_ids["metabolitics_label"].astype(str)
    fold0 = fold_ids.loc[fold_ids["fold"] == 0].reset_index(drop=True)
    fold0["unique_row_id"] = fold0["unique_row_id"].astype(str)
    if list(train_csv["Sample ID"]) != list(train_ids["sample_id"]):
        raise ValueError(f"{cohort}: train.csv Sample ID order != train_ids")
    if list(fold0["unique_row_id"]) != list(train_ids["unique_row_id"]):
        raise ValueError(f"{cohort}: fold_ids unique_row_id order != train_ids")
    val_fold = fold0["val_fold"].to_numpy()
    if sorted(set(val_fold.tolist())) != list(range(n_splits)):
        raise ValueError(f"{cohort}: expected val_fold 0..{n_splits-1}, got {sorted(set(val_fold.tolist()))}")

    out_dir = cohort_dir / "processed" / "cv_folds"
    input_dir = out_dir / "inputs"
    id_dir = out_dir / "ids"
    input_dir.mkdir(parents=True, exist_ok=True)
    id_dir.mkdir(parents=True, exist_ok=True)
    staging_dir = project_root / "scripts" / "metabolitics" / "cv_fold_inputs" / cohort
    staging_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    folds = []
    for fold in range(n_splits):
        parts = {
            "cv_train": val_fold != fold,
            "cv_val": val_fold == fold,
        }
        fold_block = {"fold": fold}
        for split_name, mask in parts.items():
            matrix = train_csv.loc[mask].reset_index(drop=True)
            ids = train_ids.loc[mask].reset_index(drop=True)
            counts = _counts(matrix, ids)
            if counts["healthy_rows"] < 1:
                raise ValueError(
                    f"{cohort} fold{fold} {split_name} has no healthy samples for FoldChangeScaler"
                )
            job_split = f"fold{fold}_{split_name}"
            matrix.to_csv(input_dir / f"{job_split}.csv", index=False)
            matrix.to_csv(staging_dir / f"{job_split}.csv", index=False)
            ids.to_csv(id_dir / f"{job_split}_ids.csv", index=False)
            jobs.append(
                {
                    "cohort": cohort,
                    "split": job_split,
                    "platform": COHORT_PLATFORMS[cohort],
                    "n_rows": counts["n_rows"],
                    "healthy_rows": counts["healthy_rows"],
                }
            )
            fold_block[split_name] = counts
            fold_block[f"{split_name}_file"] = str(input_dir / f"{job_split}.csv")
        folds.append(fold_block)

    manifest = {
        "cohort": cohort,
        "n_splits": n_splits,
        "random_state": RANDOM_STATE,
        "source_train_csv": str(cohort_dir / "processed" / "metabolitics_input" / "train.csv"),
        "source_fold_ids": str(cohort_dir / "processed" / "cv" / "fold_ids.csv"),
        "fold_change": {
            "fit_scope": "independent_per_cv_split",
            "reference_label": DEFAULT_REFERENCE_LABEL,
            "note": (
                "Each cv_train and cv_val CSV is its own Metabolitics job. "
                "Fold-change and FVA are fit on that file only."
            ),
        },
        "n_train_rows": int(len(train_csv)),
        "folds": folds,
        "jobs": jobs,
    }
    _json_dump(out_dir / "fold_manifest.json", manifest)
    return manifest


def write_jobs_tsv(project_root: Path, manifests: list[dict]) -> Path:
    jobs_path = project_root / "scripts" / "metabolitics" / "jobs.tsv"
    backup = project_root / "scripts" / "metabolitics" / "jobs_train_test.tsv"
    if jobs_path.exists() and not backup.exists():
        backup.write_text(jobs_path.read_text())
    lines = []
    for manifest in manifests:
        for job in manifest["jobs"]:
            lines.append(f"{job['cohort']}\t{job['split']}\t{job['platform']}")
    jobs_path.write_text("\n".join(lines) + "\n")
    return jobs_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cohorts", nargs="+", default=list(COHORT_PLATFORMS))
    parser.add_argument("--n-splits", type=int, default=N_CV_SPLITS)
    args = parser.parse_args()

    manifests = []
    for cohort in args.cohorts:
        print(f"=== {cohort} ===")
        manifest = prepare_cohort(args.project_root, cohort, args.n_splits)
        manifests.append(manifest)
        for fold in manifest["folds"]:
            tr = fold["cv_train"]
            va = fold["cv_val"]
            print(
                f"  fold{fold['fold']}: cv_train {tr['n_rows']} rows "
                f"({tr['healthy_rows']} healthy) | "
                f"cv_val {va['n_rows']} rows ({va['healthy_rows']} healthy)"
            )

    jobs_path = write_jobs_tsv(args.project_root, manifests)
    n_jobs = sum(len(m["jobs"]) for m in manifests)
    print(f"Wrote {n_jobs} jobs -> {jobs_path}")
    summary_path = args.project_root / "processed" / "cv_fold_jobs_summary.json"
    _json_dump(
        summary_path,
        {
            "n_jobs": n_jobs,
            "cohorts": [m["cohort"] for m in manifests],
            "jobs": [job for m in manifests for job in m["jobs"]],
        },
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
