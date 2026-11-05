#!/bin/bash
# Convert DiCAD_GWAS_JUNE2020.csv on TRUBA (file already at snp_data_download).
set -euo pipefail
CSV="${1:-/arf/scratch/cnjume/snp_data_download/DiCAD_syn22024511/DiCAD_GWAS_JUNE2020.csv}"
OUT="${2:-$HOME/prs_dicad/geno/dicad_gsa}"
mkdir -p "$(dirname "$OUT")"
python3 convert_dicad_gwas_to_plink.py --input "$CSV" --out-prefix "$OUT"
ls -lh "$OUT".bed "$OUT".bim "$OUT".fam "$OUT".conversion_summary.json
