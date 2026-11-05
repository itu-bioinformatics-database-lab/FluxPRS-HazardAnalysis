#!/bin/bash
#SBATCH -p barbun
#SBATCH -A cnjume
#SBATCH -J prs_rosmap
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=1-00:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# ROSMAP Affy imputed dosages (PLINK .dosage.gz + .fam), Control + AD (cogdx 1 / 4+5).
set -euo pipefail

WORKDIR="${WORKDIR:-/arf/scratch/cnjume/prs_compute}"
GENO_DIR="${GENO_DIR:-/arf/scratch/cnjume/snp_data_download/ROSMAP_syn3157325/Imputed}"
GENO_GLOB='AMP-AD_ROSMAP_Rush-Broad_AffymetrixGenechip6_Imputed_chr*.dosage.gz'
SAMPLE_FILE="${SAMPLE_FILE:-$GENO_DIR/AMP-AD_ROSMAP_Rush-Broad_AffymetrixGenechip6_Imputed.fam}"
REFDIR="${REFDIR:-/arf/scratch/cnjume/prs_compute/prs_ref}"
OUTDIR="${OUTDIR:-/arf/scratch/cnjume/prs_compute/PRS_results_ROSMAP}"
PHENO="${PHENO:-$WORKDIR/phenotypes/rosmap_pheno.csv}"
PHENO_ID_COL="individualID"
PHENO_DX_COL="diagnosis"
AD_GROUPS="AD,CONTROL"
PSP_GROUPS="PSP,CONTROL"
BUILD="GRCh37"
PALINDROMIC="keep"
EXTRA_ARGS="--max-variants 0"
CONDA_SH="/arf/home/cnjume/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="metaboliticsenv"
VENV="${VENV:-}"

if [[ -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  set +u; conda activate "$CONDA_ENV"; set -u
fi
if [[ -f "$VENV/bin/activate" ]]; then
  set +u; source "$VENV/bin/activate"; set -u
fi

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$WORKDIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname) | cpus=${SLURM_CPUS_PER_TASK:-?} | $(date)"
python3 -c "import numpy, sys; print('python', sys.version.split()[0], 'numpy', numpy.__version__)"

[[ -f "$REFDIR/manifest.json" ]] || { echo "ERROR: $REFDIR/manifest.json not found (run download first)."; exit 1; }
[[ -f "$PHENO" ]] || { echo "ERROR: phenotype file not found: $PHENO"; exit 1; }
[[ -f "$SAMPLE_FILE" ]] || { echo "ERROR: FAM not found: $SAMPLE_FILE"; exit 1; }
mkdir -p "$OUTDIR"

python3 prs_pipeline.py run \
  --geno-dir "$GENO_DIR" \
  --geno-glob "$GENO_GLOB" \
  --sample-file "$SAMPLE_FILE" \
  --refdir "$REFDIR" \
  --outdir "$OUTDIR" \
  --threads "${SLURM_CPUS_PER_TASK:-128}" \
  --build "$BUILD" \
  --pheno "$PHENO" --pheno-id-col "$PHENO_ID_COL" --pheno-dx-col "$PHENO_DX_COL" \
  --ad-groups "$AD_GROUPS" \
  --psp-groups "$PSP_GROUPS" \
  --palindromic "$PALINDROMIC" \
  $EXTRA_ARGS

echo; echo "================ BUILD DETECTION ================"
cat "$OUTDIR/build_detection.txt"
echo; echo "================ MATCHING SUMMARY ================"
python3 - "$OUTDIR/matching_summary.tsv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
rows.sort(key=lambda r: (r["GROUP"], -float(r["PCT_MATCHED_OF_FILE"] if r["PCT_MATCHED_OF_FILE"] != "NA" else -1)))
fmt = "%-30s %-4s %10s %10s %8s %9s %7s %6s %6s %s"
print(fmt % ("MODEL", "GRP", "N_FILE", "N_MATCHED", "%MATCH", "%|BETA|", "FLIP", "N_OUT", "INVAR", "APOE(rs429358)"))
for r in rows:
    print(fmt % (r["MODEL_ID"][:30], r["GROUP"], r["N_VARIANTS_FILE"], r["N_MATCHED"], r["PCT_MATCHED_OF_FILE"],
                 r["PCT_ABS_WEIGHT_MATCHED"], r["N_STRAND_FLIPPED"], r["N_SAMPLES_OUTPUT"], r["INVARIANT"],
                 r["APOE_rs429358"]))
PYEOF
echo "Finished: $(date)"
