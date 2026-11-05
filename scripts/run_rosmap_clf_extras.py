#!/usr/bin/env python3
"""Backward-compatible wrapper: ROSMAP classification extras only.

NMF for all cohorts and DiCAD classification extras live in run_clf_extras.py.
This wrapper keeps the old ROSMAP-only command working.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_clf_extras import main as extras_main  # noqa: E402


if __name__ == "__main__":
    sys.argv = [
        sys.argv[0],
        "--classification-cohorts",
        "ROSMAP-AD",
        "--skip-nmf",
        *sys.argv[1:],
    ]
    extras_main()
