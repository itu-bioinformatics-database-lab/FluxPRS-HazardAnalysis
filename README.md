# FluxPRS-HazardAnalysis

Code for integrating polygenic risk scores (PRS) with metabolomics and
Metabolitics-derived reaction and pathway perturbation scores, as reported in
the BMC Bioinformatics manuscript.

Public repository: https://github.com/itu-bioinformatics-database-lab/FluxPRS-HazardAnalysis

This working copy may include access-controlled cohort files. Those files are
not part of the public repository.

## Reported analyses

The reported analyses live in `scripts/`, `modules/`, `preprocessing/`, and
`tests/`. Unused legacy notebooks and earlier analysis modules are in `.archive/`.

| Reported analysis | Script |
| --- | --- |
| 70/30 splits (DiCAD grouped by participant) | `scripts/prepare_cohort_splits.py`, `preprocessing/cohort_splits.py` |
| Metabolitics train-fit / holdout-transform | `scripts/metabolitics/run_cohort_metabolitics.py` |
| Nested classification and held-out test metrics | `scripts/run_notebook_classifiers.py` |
| Nested CV table (Supplementary Table 2) and held-out table (Supplementary Table 3) | `scripts/render_nested_classification_table.py` |
| Cox hazard ratios on concatenated train+test scores | `scripts/run_cox_hrs.py` |
| Table 2 ΔHR summaries (median ΔHR; positive / zero / negative counts) | `scripts/summarize_delta_hr.py` |
| SHAP on the held-out partition | `scripts/run_train_cv.py` |
| Pathway RFECV, McNemar, and train-fitted NMF | `scripts/run_clf_extras.py` |
| PRS variant-matching figure | `scripts/assemble_prs_matching_figure.py` |
| Main-text and supplementary figures | `scripts/assemble_bmc_figures.py`, `scripts/assemble_supplementary_figures.py` |
| Evaluation manifest | `scripts/collect_evaluation_results.py` |

Do not copy cohort `raw/` or `processed/` directories, `incoming/`, or any
participant-level files into the public repository.

## Protocol (matches Methods)

1. Use the saved 70/30 splits (`random_state=42`). DiCAD is split by participant.
2. Fit Metabolitics on training rows only and transform held-out or validation
   rows (`fit_train_transform_holdout`). Nested-classification outer folds use
   independently generated Metabolitics matrices.
3. Nested 5×5 classification: inner folds choose the number of Random-Forest-ranked
   features; inner folds reuse the outer-training Metabolitics matrix.
4. Cox models use scores from the concatenated training and test samples after
   the logistic risk models are fitted on training diagnoses only. PRS-only
   scores are defined externally. These comparisons are associations with
   disease status and do not establish incremental value over PRS alone.
5. Nested cross-validation is the primary classification evidence. Held-out
   metrics are the already-computed 30% test evaluation of the same models.

## Installation

```bash
git clone https://github.com/itu-bioinformatics-database-lab/FluxPRS-HazardAnalysis.git
cd FluxPRS-HazardAnalysis
python -m pip install -r requirements.txt
```

## Reproducing reported analyses

Place access-controlled cohort files in the internal layout (`ROSMAP-AD/`,
`MayoRNASeq-AD/`, `MayoRNASeq-PSP/`, `DIABETES/`), then:

```bash
python scripts/prepare_cohort_splits.py
# Metabolitics FVA: scripts/metabolitics/run_cohort_metabolitics.py
python scripts/run_cox_hrs.py
python scripts/summarize_delta_hr.py
python scripts/run_notebook_classifiers.py
python scripts/run_train_cv.py
python scripts/run_clf_extras.py
python scripts/render_nested_classification_table.py
python scripts/collect_evaluation_results.py
python -m pytest tests
```
