#!/usr/bin/env python3
"""Collect analysis run summaries into results/evaluation_manifest.csv."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.evaluation.results_schema import (  # noqa: E402
    EVALUATION_MANIFEST_COLUMNS,
    flatten_run_summary,
)


def collect_evaluation_results(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    rows = []
    summary_paths = list(project_root.glob("*/processed/*/run_summary.json"))
    summary_paths += list(project_root.glob("*/processed/cv/cv_summary.json"))
    for path in sorted(set(summary_paths)):
        try:
            summary = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            rows.append(
                {
                    "analysis": path.parent.name,
                    "cohort": path.parents[2].name,
                    "task": None,
                    "metabolitics_mode": None,
                    "n_train": None,
                    "n_test": None,
                    "elapsed_seconds": None,
                    "extra": f"collector_error={exc}",
                    "output": str(path.resolve()),
                }
            )
            continue
        rows.append(flatten_run_summary(summary, path))

    table = pd.DataFrame(rows, columns=EVALUATION_MANIFEST_COLUMNS)
    if not table.empty:
        table = table.sort_values(["analysis", "cohort", "output"]).reset_index(drop=True)
    out = project_root / "results" / "evaluation_manifest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    table = collect_evaluation_results(args.project_root)
    print(f"Wrote {len(table)} rows to {args.project_root / 'results/evaluation_manifest.csv'}")


if __name__ == "__main__":
    main()
