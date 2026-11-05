#!/bin/bash
# Login-node driver: download T2D PGS, then submit the DiCAD scoring job.
set -euo pipefail
ROOT=/arf/scratch/cnjume/prs_compute
cd "$ROOT"
mkdir -p logs

# Keep the Alzheimer DiCAD run so it cannot be confused with the T2D scores.
if [[ -d PRS_results_DiCAD && ! -d PRS_results_DiCAD_Alzheimer ]]; then
  mv PRS_results_DiCAD PRS_results_DiCAD_Alzheimer
  echo "archived Alzheimer DiCAD scores -> PRS_results_DiCAD_Alzheimer"
fi

bash "$ROOT/download_pgs_t2d_truba.sh"
sbatch --chdir="$ROOT" run_prs_dicad.sbatch
echo "submitted DiCAD T2D PRS $(date)"
squeue -u cnjume
