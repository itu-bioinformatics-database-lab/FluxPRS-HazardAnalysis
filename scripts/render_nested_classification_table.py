#!/usr/bin/env python3
"""Render Supplementary Table 2 (nested classifier metrics) into the supplementary TeX."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

COHORT_LABELS = {
    "ROSMAP-AD": "ROSMAP-AD (binary AD versus control)",
    "DIABETES": "DiCAD (three-class classification)",
}
FEATURE_LABELS = {
    "all_data": "All Features (PRS + Metabolite Fold-Changes + Reaction Diff Scores + Pathway Diff Scores)",
    "prs": "PRS",
    "metabolites": "Metabolite Fold-Changes",
    "pathway": "Pathway Diff Scores",
    "reaction": "Reaction Diff Scores",
    "prs_metabolites": "PRS + Metabolite Fold-Changes",
    "prs_pathway": "PRS + Pathway Diff Scores",
    "prs_reaction": "PRS + Reaction Diff Scores",
}
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "AdaBoost",
    "xgboost": "XGBoost",
    "stacking": "Stacking Ensemble",
}
ROW_ORDER = [
    ("random_forest", "nested"),
    ("logistic_regression", "nested"),
    ("random_forest", "all"),
    ("logistic_regression", "all"),
    ("gradient_boosting", "nested"),
    ("adaboost", "nested"),
    ("xgboost", "nested"),
    ("stacking", "nested"),
]


def pm(mean, std) -> str:
    return f"{float(mean):.4f} $\\pm$ {float(std):.4f}"


def fmt4(val) -> str:
    return f"{float(val):.4f}"


def iter_table_rows(metrics: pd.DataFrame):
    for cohort, cohort_label in COHORT_LABELS.items():
        cohort_rows = metrics.loc[metrics["cohort"] == cohort]
        for feature_set, feature_label in FEATURE_LABELS.items():
            part = cohort_rows.loc[cohort_rows["feature_set"] == feature_set]
            if part.empty:
                continue
            yield cohort, cohort_label, feature_set, feature_label, part


def model_label(item: pd.Series, selection_kind: str) -> str:
    if selection_kind == "all":
        return f"{MODEL_LABELS[item['model']]} (All features)"
    return (
        f"{MODEL_LABELS[item['model']]} "
        f"(nested selection; final top {int(item['n_features'])})"
    )


def select_row(part: pd.DataFrame, model: str, selection_kind: str):
    row = part.loc[part["model"] == model]
    if selection_kind == "all":
        row = row.loc[row["selection"] == "all"]
    else:
        row = row.loc[row["selection"].str.startswith("nested_top_")]
    if row.empty:
        return None
    return row.iloc[0]


def longtable_header(caption: str, label: str, col_headers: tuple[str, str, str, str]) -> list[str]:
    h1, h2, h3, h4 = col_headers
    return [
        r"\begin{center}",
        r"\scriptsize",
        r"\begin{longtable}{>{\raggedright\arraybackslash}p{5.5cm} >{\centering\arraybackslash}p{3cm} >{\centering\arraybackslash}p{3cm} >{\centering\arraybackslash}p{3cm}}",
        rf"\caption{{{caption}}}\label{{{label}}} \\",
        r"\toprule",
        rf"\textbf{{{h1}}} & \textbf{{{h2}}} & \textbf{{{h3}}} & \textbf{{{h4}}} \\",
        r"\midrule",
        r"\endfirsthead",
        "",
        r"\multicolumn{4}{c}%",
        r"{{\tablename\ \thetable{} -- continued from previous page}} \\",
        r"\toprule",
        rf"\textbf{{{h1}}} & \textbf{{{h2}}} & \textbf{{{h3}}} & \textbf{{{h4}}} \\",
        r"\midrule",
        r"\endhead",
        "",
        r"\midrule",
        r"\multicolumn{4}{r}{{Continued on next page}} \\",
        r"\endfoot",
        "",
        r"\bottomrule",
        r"\endlastfoot",
        "",
    ]


def table_fragment(metrics: pd.DataFrame) -> str:
    caption = (
        r"Nested cross-validated performance of machine-learning classifiers trained on "
        r"genomic and omics-derived feature sets. Outer folds use independently fitted "
        r"Metabolitics transformations. For selected-feature models, feature count and "
        r"Random-Forest ranking are refitted within each outer-training partition; "
        r"``final top X'' denotes the feature count selected across the outer folds for "
        r"fitting the final model evaluated on the independent held-out test partition."
    )
    lines = longtable_header(
        caption,
        "tab:classification",
        (
            "Model",
            r"Accuracy (mean $\pm$ std)",
            r"F1 Score (mean $\pm$ std)",
            r"AUC (mean $\pm$ std)",
        ),
    )
    current_cohort = None
    for cohort, cohort_label, _feature_set, feature_label, part in iter_table_rows(metrics):
        if cohort != current_cohort:
            lines += [rf"\multicolumn{{4}}{{c}}{{\textbf{{{cohort_label}}}}} \\", r"\midrule"]
            current_cohort = cohort
        lines += [
            rf"\multicolumn{{4}}{{c}}{{\textbf{{{feature_label}}}}} \\",
            r"\midrule",
        ]
        for model, selection_kind in ROW_ORDER:
            item = select_row(part, model, selection_kind)
            if item is None:
                continue
            lines.append(
                f"{model_label(item, selection_kind)} & "
                f"{pm(item['cv_accuracy_mean'], item['cv_accuracy_std'])} & "
                f"{pm(item['cv_f1_mean'], item['cv_f1_std'])} & "
                f"{pm(item['cv_auc_mean'], item['cv_auc_std'])} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [
        r"\end{longtable}",
        r"\end{center}",
        "",
        r"\noindent\footnotesize",
        (
            r"\textit{Notes:} Values are mean $\pm$ sample standard deviation across "
            r"the five predefined outer folds within the 70\% training partition. "
            r"Metabolitics transformations were fitted on each outer-training fold and "
            r"applied unchanged to its outer validation fold. Inner five-fold tuning "
            r"reused the corresponding outer-training Metabolitics matrix; scaling, "
            r"Random-Forest feature ranking, and classification were refitted in every "
            r"inner fold. DiCAD outer and inner folds were grouped by participant. "
            r"The independent 30\% held-out test partition was not used for feature "
            r"ranking, feature-count selection, or cross-validation performance. "
            r"Held-out point estimates for the same models are in Supplementary Table~\ref{tab:heldout_classification}."
        ),
        "",
    ]
    return "\n".join(lines)


def heldout_table_fragment(metrics: pd.DataFrame) -> str:
    caption = (
        r"Held-out 30\% test performance of the same classifiers reported in "
        r"Supplementary Table~\ref{tab:classification}. Final models were fitted on the "
        r"70\% training partition after nested feature-count selection; Metabolitics "
        r"transformations were fitted on training samples only. Values are single-split "
        r"point estimates (ROSMAP-AD $N=81$; DiCAD $N=85$ samples on the saved "
        r"participant-grouped split). Nested cross-validation remains the primary "
        r"classification evidence."
    )
    lines = longtable_header(
        caption,
        "tab:heldout_classification",
        ("Model", "Accuracy", "F1 Score", "AUC"),
    )
    current_cohort = None
    for cohort, cohort_label, _feature_set, feature_label, part in iter_table_rows(metrics):
        if cohort != current_cohort:
            lines += [rf"\multicolumn{{4}}{{c}}{{\textbf{{{cohort_label}}}}} \\", r"\midrule"]
            current_cohort = cohort
        lines += [
            rf"\multicolumn{{4}}{{c}}{{\textbf{{{feature_label}}}}} \\",
            r"\midrule",
        ]
        for model, selection_kind in ROW_ORDER:
            item = select_row(part, model, selection_kind)
            if item is None:
                continue
            lines.append(
                f"{model_label(item, selection_kind)} & "
                f"{fmt4(item['test_accuracy'])} & "
                f"{fmt4(item['test_f1'])} & "
                f"{fmt4(item['test_auc'])} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [
        r"\end{longtable}",
        r"\end{center}",
        "",
        r"\noindent\footnotesize",
        (
            r"\textit{Notes:} These metrics were already computed with the saved 70/30 "
            r"split and are reported here without new model fitting. The test partition "
            r"was not used for Metabolitics fitting, scaling, feature ranking, or "
            r"feature-count selection. DiCAD F1 and AUC are macro-averaged. Flux-only "
            r"pathway and reaction feature sets did not outperform metabolite "
            r"fold-changes on this split in either cohort."
        ),
        "",
    ]
    return "\n".join(lines)


def heldout_section(metrics: pd.DataFrame) -> str:
    intro = (
        r"\section*{Held-out classification performance of nested models (Supplementary Table 3)}"
        "\n\n"
        r"The already-computed held-out results are reported here in addition to nested "
        r"cross-validation. The table below uses the same saved 70/30 split, feature sets, "
        r"and classifiers as Supplementary Table~\ref{tab:classification}. Each final model "
        r"was fitted on the 70\% training partition; the 30\% test partition was used only "
        r"for this evaluation. Nested cross-validation remains the primary classification "
        r"evidence."
        "\n\n"
        r"On this split, ROSMAP-AD models that combined PRS with metabolite fold-changes "
        r"remained among the strongest. Flux-only pathway and reaction feature sets did not "
        r"outperform metabolite fold-changes in ROSMAP-AD or DiCAD. Because these are "
        r"single-split point estimates, they should not be read as a second primary ranking "
        r"of feature sets."
        "\n\n"
        r"\vspace{1em}"
        "\n"
    )
    return intro + heldout_table_fragment(metrics)


def standalone_document(*fragments: str) -> str:
    return "\n".join(
        [
            r"\documentclass[10pt,a4paper]{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[left=2cm,right=2cm,top=2.5cm,bottom=2.5cm]{geometry}",
            r"\usepackage{booktabs}",
            r"\usepackage{longtable}",
            r"\usepackage{array}",
            r"\usepackage{caption}",
            r"\renewcommand{\tablename}{Supplementary Table}",
            r"\setcounter{table}{0}",
            r"\begin{document}",
            *fragments,
            r"\end{document}",
            "",
        ]
    )


def update_inventory(text: str) -> str:
    old = (
        "Supplementary Table~2 (nested classification); "
        "Supplementary Figures~18--20 (held-out confusion matrix and SHAP)"
    )
    new = (
        "Supplementary Table~2 (nested classification); "
        "Supplementary Table~3 (held-out classification); "
        "Supplementary Figures~18--20 (held-out confusion matrix and SHAP)"
    )
    if old not in text:
        return text
    return text.replace(old, new, 1)


def render(project_root: Path = PROJECT_ROOT) -> None:
    metrics_path = project_root / "results" / "clf_notebook_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    fragment = table_fragment(metrics)
    heldout_fragment = heldout_table_fragment(metrics)
    heldout = heldout_section(metrics)

    supplement_dir = project_root / "manuscript" / "supplementary"
    supplement_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "manuscript" / "Supplementary_Table_1_classification_performance.tex").write_text(
        standalone_document(fragment)
    )
    (project_root / "manuscript" / "Supplementary_Table_3_heldout_classification.tex").write_text(
        standalone_document(heldout_fragment)
    )
    (supplement_dir / "Supplementary_Table_1_classification_performance_fragment.tex").write_text(
        fragment
    )
    (supplement_dir / "Supplementary_Table_3_heldout_classification_fragment.tex").write_text(
        heldout_fragment
    )
    main_tex = supplement_dir / "main.tex"
    if main_tex.exists():
        text = embed_classification_table(main_tex.read_text(), fragment, heldout)
        text = update_inventory(text)
        text = point_to_heldout_table(text)
        main_tex.write_text(text)
        (supplement_dir / "Supplementary_Material_updated.tex").write_text(text)


def embed_classification_table(text: str, fragment: str, heldout: str | None = None) -> str:
    section_starts = [
        "\\section*{Classification performance of integrated genomic",
        "\\section*{S4. Classification",
        "\\section*{S2. Classification",
    ]
    end_tokens = [
        "\n\\vspace{2em}\n%%",
        "\n\\vspace{2em}\n\\section*{Held-out evaluation",
        "\n\\vspace{2em}\n\\section*{S5.",
        "\n\\vspace{2em}\n\\section*{S3.",
    ]
    start = -1
    for marker in section_starts:
        start = text.find(marker)
        if start >= 0:
            break
    if start < 0:
        raise ValueError("Could not locate the classification section in supplementary/main.tex")
    table_at = text.find("\\vspace{1em}", start)
    if table_at < 0:
        raise ValueError("Could not locate the classification table block")
    end = -1
    for token in end_tokens:
        pos = text.find(token, table_at + 1)
        if pos >= 0 and (end < 0 or pos < end):
            end = pos
    if end < 0:
        raise ValueError("Could not locate the end of the classification table block")
    block = "\\vspace{1em}\n" + fragment
    if heldout:
        block += "\n\\vspace{2em}\n" + heldout
    return text[:table_at] + block + "\n" + text[end:]


def point_to_heldout_table(text: str) -> str:
    old = (
        "Results for all classifiers are given in Supplementary Table~\\ref{tab:classification}."
    )
    new = (
        "Results for all classifiers are given in Supplementary Table~\\ref{tab:classification}. "
        "Held-out test metrics for the same models are given in Supplementary "
        "Table~\\ref{tab:heldout_classification}; nested cross-validation remains the "
        "primary classification evidence."
    )
    if new in text:
        return text
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    render(args.project_root)
    print("Rendered nested Supplementary Table 2 and held-out Supplementary Table 3")


if __name__ == "__main__":
    main()
