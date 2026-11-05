#!/usr/bin/env python3
"""Copy TRUBA cv-fold Metabolitics outputs into the local layout.

TRUBA:
  fluxprs_splits/outputs/{cohort}/fold{k}_{cv_train|cv_val}/
Local:
  {cohort}/processed/metabolitics_output/cv_folds/fold{k}/{cv_train|cv_val}/
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("ROSMAP-AD", "DIABETES")
N_SPLITS = 5
SPLITS = ("cv_train", "cv_val")


def import_outputs(src_root: Path, project_root: Path) -> list[str]:
    copied = []
    missing = []
    for cohort in COHORTS:
        for fold in range(N_SPLITS):
            for split in SPLITS:
                job_split = f"fold{fold}_{split}"
                src = src_root / cohort / job_split
                dest = (
                    project_root
                    / cohort
                    / "processed"
                    / "metabolitics_output"
                    / "cv_folds"
                    / f"fold{fold}"
                    / split
                )
                if not (src / "X_pathway.csv").exists():
                    missing.append(str(src))
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(src, dest)
                copied.append(str(dest))
    if missing and not copied:
        raise FileNotFoundError(
            "No complete fold outputs found. Missing examples:\n  " + "\n  ".join(missing[:8])
        )
    print(f"Copied {len(copied)} fold directories")
    if missing:
        print(f"Still missing {len(missing)}: {missing[:8]}")
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src-root",
        type=Path,
        required=True,
        help="Directory containing {cohort}/fold0_cv_train/ ... (TRUBA outputs tree)",
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    import_outputs(args.src_root, args.project_root)


if __name__ == "__main__":
    main()
