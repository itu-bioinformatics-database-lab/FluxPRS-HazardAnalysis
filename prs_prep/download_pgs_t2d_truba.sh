#!/bin/bash
# Run on a TRUBA login node (internet). Do not submit as a compute job.
# Downloads type 2 diabetes PGS Catalog models into a separate refdir
# so the Alzheimer/PSP catalog in prs_ref is left untouched.
set -euo pipefail
ROOT=/arf/scratch/cnjume/prs_compute
REFDIR="$ROOT/prs_ref_t2d"
AD_REFDIR="$ROOT/prs_ref"
CONDA_ROOT=/arf/home/cnjume/miniconda3
mkdir -p "$REFDIR/dbsnp" "$ROOT/logs"
cd "$ROOT"

# Reuse the already-downloaded dbSNP merge table (do not fetch it again).
DBSNP_SRC="$AD_REFDIR/dbsnp/refsnp-merged.json.bz2"
DBSNP_DST="$REFDIR/dbsnp/refsnp-merged.json.bz2"
if [[ -f "$DBSNP_SRC" && ! -e "$DBSNP_DST" ]]; then
  ln -s "$DBSNP_SRC" "$DBSNP_DST"
  echo "linked dbSNP merge from $DBSNP_SRC"
fi

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate metaboliticsenv || true
fi

echo "T2D download start $(date) on $(hostname)"
python3 "$ROOT/prs_pipeline.py" download \
  --refdir "$REFDIR" \
  --terms "type 2 diabetes" \
  --efo EFO_0001360 MONDO_0005148 \
  --skip-psp
echo "T2D download done $(date)"
test -f "$REFDIR/manifest.json"
python3 - "$REFDIR/manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
n = sum(1 for x in m.get("models", []) if x.get("files"))
print("manifest models with files:", n, "search_terms:", m.get("search_terms"), "efo:", m.get("efo_ids"))
PY
