# FluxPRS-HazardAnalysis
A reproducible framework for computing Cox proportional hazard ratios from metabolomics, polygenic risk scores (PRS), and Metabolitics-derived reaction/pathway perturbation scores.

---

## 🧬 Overview

This repository provides a reproducible pipeline for survival analysis using:

* Metabolomics measurements
* Polygenic Risk Scores (PRS)
* Metabolitics-derived Reaction Differential Scores
* Metabolitics-derived Pathway Differential Scores
* Combined multi-modal models

The framework computes hazard ratios using Cox Proportional Hazards modeling and evaluates risk stratification across disease cohorts.

This tool was developed to support integrative biomarker modeling in complex diseases such as:

* Alzheimer’s Disease
* Diabetes
* Cancer
* Neurodegenerative disorders

---

## 🎯 Objectives

* Quantify risk contribution of metabolite levels
* Evaluate PRS as independent and combined predictors
* Assess added value of reaction-level perturbation
* Assess added value of pathway-level perturbation
* Compare multi-modal risk models

---

## 🧠 Modeling Framework

We implement:

### 🔹 Survival Modeling

* Cox Proportional Hazards Model (lifelines / scikit-survival)
* Hazard ratio computation
* Confidence intervals
* Log-rank testing
* Kaplan-Meier stratification

### 🔹 Input Modalities

1. Metabolite Levels
2. PRS Scores
3. Reaction Differential Scores (Metabolitics)
4. Pathway Differential Scores (Metabolitics)
5. Combined multi-omics matrices

---

## 📊 Example Model Comparisons

* Metabolite Only
* PRS Only
* Reaction Scores Only
* Pathway Scores Only
* Metabolite + PRS
* Reaction + PRS
* Pathway + PRS
* Full Integrated Model

---

## 🛠 Installation

```bash
git clone https://github.com/itu-bioinformatics-database-lab/FluxPRS-HazardAnalysis.git
cd FluxPRS-HazardAnalysis
pip install -r requirements.txt
```
