#!/bin/bash
#SBATCH -p barbun
#SBATCH -A cnjume
#SBATCH -J dicad_csv2bed
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --time=12:00:00

# Convert DiCAD long-format SNP-array CSV (already on TRUBA) to PLINK .bed
set -euo pipefail

ROOT=/arf/scratch/cnjume/prs_compute
CSV=/arf/scratch/cnjume/snp_data_download/DiCAD_syn22024511/DiCAD_GWAS_JUNE2020.csv
OUT="$ROOT/dicad_geno/dicad_gsa"
CONDA_ROOT=/arf/home/cnjume/miniconda3

mkdir -p "$(dirname "$OUT")" "$ROOT/logs"
cd "$ROOT"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate metaboliticsenv || true
fi

echo "Start $(date) on $(hostname)"
python3 -c "import sys; print(sys.executable, sys.version.split()[0])"
[[ -f "$CSV" ]] || { echo "ERROR missing $CSV"; exit 1; }
python3 "$ROOT/convert_dicad_gwas_to_plink.py" --input "$CSV" --out-prefix "$OUT"
ls -lh "$OUT".bed "$OUT".bim "$OUT".fam "$OUT".conversion_summary.json
echo "Finished $(date)"
