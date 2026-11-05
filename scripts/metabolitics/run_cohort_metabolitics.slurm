#!/bin/bash
#SBATCH -p barbun
#SBATCH -A cnjume
#SBATCH -J fluxprs_ft
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=3-00:00:00
#SBATCH --array=0-13

set -euo pipefail

ROOT=/arf/scratch/cnjume/metabolitics-dev
cd "${ROOT}"
mkdir -p logs

JOBS_FILE="${ROOT}/fluxprs_splits/jobs.tsv"
mapfile -t JOBS < "${JOBS_FILE}"
LINE="${JOBS[${SLURM_ARRAY_TASK_ID}]}"
cohort="$(echo "${LINE}" | cut -f1)"
fit_split="$(echo "${LINE}" | cut -f2)"
tf_split="$(echo "${LINE}" | cut -f3)"
platform="$(echo "${LINE}" | cut -f4)"

INPUT_FIT="${ROOT}/fluxprs_splits/inputs/${cohort}/${fit_split}.csv"
INPUT_TF="${ROOT}/fluxprs_splits/inputs/${cohort}/${tf_split}.csv"
OUTPUT_FIT="${ROOT}/fluxprs_splits/outputs_trainfit/${cohort}/${fit_split}"
OUTPUT_TF="${ROOT}/fluxprs_splits/outputs_trainfit/${cohort}/${tf_split}"

echo "========================================================"
echo "Start $(date)"
echo "Node $(hostname)"
echo "Job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Cohort ${cohort}"
echo "FIT ${fit_split} -> ${OUTPUT_FIT}"
echo "TRANSFORM ${tf_split} -> ${OUTPUT_TF}"
echo "Platform ${platform}"
echo "========================================================"

if [[ ! -f "${INPUT_FIT}" ]]; then
  echo "Missing fit input: ${INPUT_FIT}" >&2
  exit 1
fi
if [[ ! -f "${INPUT_TF}" ]]; then
  echo "Missing transform input: ${INPUT_TF}" >&2
  exit 1
fi

CONDA_ROOT=/arf/home/cnjume/miniconda3
# Compute nodes do not provide `module`; activate conda directly.
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate metaboliticsenv
echo "Python $(which python)"

export CPLEX_STUDIO_DIR2212=/arf/home/cnjume/cplex_v22
export CPLEX_STUDIO_BINARIES=/arf/home/cnjume/cplex_v22/cplex/bin/x86-64_linux
export LD_LIBRARY_PATH="${CPLEX_STUDIO_BINARIES}:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD=/arf/home/cnjume/cplex_v22/cplex/bin/x86-64_linux/libcplex2212.so
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JOBLIB_START_METHOD=loky

python run_cohort_metabolitics.py \
  --input-fit "${INPUT_FIT}" \
  --output-fit "${OUTPUT_FIT}" \
  --input-transform "${INPUT_TF}" \
  --output-transform "${OUTPUT_TF}" \
  --label-column Diagnosis \
  --mapping-kind new-synonym \
  --mapping-platform "${platform}"

echo "Finished $(date)"
