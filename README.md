# FluxPRS-HazardAnalysis

A reproducible framework for survival analysis and disease classification
using metabolomics, polygenic risk scores (PRS), and Metabolitics-derived
reaction/pathway perturbation scores.

## Analyses

- Cox proportional-hazards models and hazard-ratio comparisons
- PRS-only, omics-only, and integrated multi-omics configurations
- Nested cross-validated classification
- Held-out SHAP interpretation
- Train-fitted NMF with held-out projection

## Data-leakage controls

The public workflow is designed so evaluation rows cannot influence model
training or preprocessing.

1. **Participant grouping.** Repeated observations from the same participant
   must remain in one partition. Common participant-ID columns are detected
   automatically, or can be supplied with `--group-column`. This is required
   for DiCAD, where multiple samples can belong to one participant.
2. **Training-only preprocessing.** Missing-value imputation, scaling,
   feature ranking, RFE/RFECV, and model fitting are fitted only on the
   relevant training partition. Validation/test rows are transformed with
   those fitted objects.
3. **Nested model selection.** Classification feature counts are selected in
   inner folds and evaluated in untouched outer folds. Cross-validation folds
   are not treated as independent biological replicates.
4. **Held-out interpretation.** SHAP values are computed only after fitting
   the model and feature selection on the training partition. Features ranked
   on held-out SHAP values receive descriptive group summaries only; the same
   test rows are not reused for post-selection significance tests.
5. **Train-fitted NMF.** Supervised feature selection, scaling, rank
   diagnostics, and NMF fitting use training rows only. Test samples are
   projected with `NMF.transform`; the factorization is not refitted.
6. **Split audits.** Classification, SHAP, and NMF commands write
   `split_audit.csv`, including participant counts and overlap. Any detected
   participant overlap raises an error.

For cohorts with one row per participant, stratified sample-level splitting
is used when no participant column exists. For repeated-measure cohorts,
provide a participant identifier; do not substitute the sample identifier.

## Input modalities

1. Metabolite levels
2. Polygenic risk scores
3. Reaction differential scores
4. Pathway differential scores
5. Combined multi-omics matrices

## Installation

```bash
git clone https://github.com/itu-bioinformatics-database-lab/FluxPRS-HazardAnalysis.git
cd FluxPRS-HazardAnalysis
python -m pip install -r requirements.txt
```

## Leakage-safe classification

Input files are expected under `data/` by default. Raw and processed cohort
data are intentionally excluded from this public repository.

```bash
python Classification_SHAP_NMF/classification.py \
  --data-dir data \
  --out-dir results/classification \
  --n-outer 5 \
  --n-inner 5
```

For a repeated-measure cohort:

```bash
python Classification_SHAP_NMF/classification.py \
  --data-dir data \
  --group-column participant_id
```

The same `--group-column` option is available in `shap_analysis.py` and
`nmf_analysis.py`.

## Public-data policy

This repository contains analysis code only. Cohort-level source data,
sample-level predictions, participant identifiers, private manuscript files,
and generated result directories must not be committed.
