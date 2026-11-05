#!/bin/bash
#SBATCH -p barbun
#SBATCH -A cnjume
#SBATCH -J prs_dicad_t2d
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=12:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# DiCAD GSA calls converted to PLINK .bed (see convert_dicad_gwas_to_plink.py).
# Type 2 diabetes PGS Catalog scores (prs_ref_t2d), not the Alzheimer catalog.
# Same T2D weights for CONTROL, OTHER(DIABETES), OTHER(PRE-DIABETES).
set -euo pipefail

WORKDIR="${WORKDIR:-/arf/scratch/cnjume/prs_compute}"
GENO_DIR="${GENO_DIR:-$WORKDIR/dicad_geno}"
GENO_GLOB='dicad_gsa.bed'
REFDIR="${REFDIR:-/arf/scratch/cnjume/prs_compute/prs_ref_t2d}"
OUTDIR="${OUTDIR:-/arf/scratch/cnjume/prs_compute/PRS_results_DiCAD}"
PHENO="${PHENO:-$WORKDIR/phenotypes/dicad_pheno.csv}"
ID_MAP="${ID_MAP:-$WORKDIR/phenotypes/dicad_id_map.tsv}"
PHENO_ID_COL="individualID"
PHENO_DX_COL="diagnosis"
AD_GROUPS="CONTROL,OTHER(DIABETES),OTHER(PRE-DIABETES)"
PSP_GROUPS="CONTROL"
BUILD="auto"
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
python3 - "$REFDIR/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
terms = " ".join(m.get("search_terms") or []).lower()
if "type 2 diabetes" not in terms and "EFO_0001360" not in (m.get("efo_ids") or []):
    raise SystemExit("ERROR: %s is not a type 2 diabetes catalog (search_terms=%s efo=%s)"
                     % (sys.argv[1], m.get("search_terms"), m.get("efo_ids")))
print("refdir OK:", m.get("search_terms"), m.get("efo_ids"),
      "n_models", len(m.get("models") or []))
PY
[[ -f "$PHENO" ]] || { echo "ERROR: phenotype file not found: $PHENO"; exit 1; }
mkdir -p "$OUTDIR"

python3 prs_pipeline.py inspect \
  --geno-dir "$GENO_DIR" --geno-glob "$GENO_GLOB" | tee "$OUTDIR/inspect.txt"

python3 prs_pipeline.py run \
  --geno-dir "$GENO_DIR" \
  --geno-glob "$GENO_GLOB" \
  --refdir "$REFDIR" \
  --outdir "$OUTDIR" \
  --threads "${SLURM_CPUS_PER_TASK:-32}" \
  --build "$BUILD" \
  --pheno "$PHENO" --pheno-id-col "$PHENO_ID_COL" --pheno-dx-col "$PHENO_DX_COL" \
  --id-map "$ID_MAP" \
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
