#!/usr/bin/env python3
"""Write 70/30 train/test IDs and per-split Metabolitics fold-change for every cohort."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DIR = PROJECT_ROOT / "preprocessing"
for path in (str(PREPROCESSING_DIR), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from cohort_splits import COHORTS, prepare_all_cohorts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument("--cohorts", nargs="+", default=list(COHORTS))
    args = parser.parse_args()

    results = prepare_all_cohorts(args.project_root, cohorts=args.cohorts)
    summary = []
    for result in results:
        m = result.manifest
        print(f"=== {m['cohort']} ({m['split_unit']}-level) ===")
        print(
            f"  rows {m['n_rows']} people {m['n_participants']} | "
            f"train {m['train']['n_rows']} rows / {m['train']['n_participants']} people "
            f"(healthy {m['train']['healthy_rows']}) | "
            f"test {m['test']['n_rows']} rows / {m['test']['n_participants']} people "
            f"(healthy {m['test']['healthy_rows']})"
        )
        print(f"  train labels {m['train']['labels']}")
        print(f"  test labels  {m['test']['labels']}")
        print(f"  ids -> {m['files']['all_ids']}")
        summary.append(
            {
                "cohort": m["cohort"],
                "split_unit": m["split_unit"],
                "train": m["train"],
                "test": m["test"],
            }
        )

    out = args.project_root / "processed" / "splits_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
