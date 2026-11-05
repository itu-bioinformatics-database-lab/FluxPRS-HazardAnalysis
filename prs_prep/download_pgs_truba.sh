#!/bin/bash
# Run on a TRUBA login node (internet). Do not submit as a compute job.
set -euo pipefail
ROOT=/arf/scratch/cnjume/prs_compute
REFDIR="$ROOT/prs_ref"
CONDA_ROOT=/arf/home/cnjume/miniconda3
mkdir -p "$REFDIR" "$ROOT/logs"
cd "$ROOT"
if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate metaboliticsenv || true
fi
echo "download start $(date) on $(hostname)"
nohup python3 "$ROOT/prs_pipeline.py" download --refdir "$REFDIR" \
  > "$ROOT/logs/pgs_download.out" 2> "$ROOT/logs/pgs_download.err" &
echo "PID $!  log $ROOT/logs/pgs_download.out"
