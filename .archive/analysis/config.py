from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

PathLike = Union[str, Path]


@dataclass(frozen=True)
class AnalysisPaths:
    # root dir for this dataset (contains raw/, interim/, processed/)
    data_dir: Path

    # raw
    metabolite_levels_file_path: Path
    x_pathway_path: Path
    x_reaction_path: Path
    transcriptomics_path: Optional[Path] = None
    clinical_ages_path: Optional[Path] = None
    best_prs_path: Optional[Path] = None

    # interim / processed
    pathway_file_path: Optional[Path] = None
    reaction_file_path: Optional[Path] = None
    metabolite_level_pathway_combined_file_path: Optional[Path] = None
    metabolite_level_reaction_combined_file_path: Optional[Path] = None
    transcriptomics_metabolites_combined_path: Optional[Path] = None
    transcriptomics_pathway_combined_path: Optional[Path] = None
    transcriptomics_reaction_combined_path: Optional[Path] = None
    prs_combined_path: Optional[Path] = None


@dataclass(frozen=True)
class DatasetSpec:
    # Outcome/labels
    outcome_col: str = "Diagnosis"
    control_labels: Sequence[str] = ("Control", "Normal", "CN", "0")
    case_labels: Sequence[str] = ("AD", "Alzheimer", "Case", "1", "PSP", "Diabetes","Pre-diabetes")

    # IDs
    sample_id_col: str = "Sample ID"   # column name in biomarker tables

    # Optional: PRS linking
    prs_id_col: str = "individualID"   # column used to connect PRS↔clinical↔biomarker


def make_default_paths(data_dir: PathLike) -> AnalysisPaths:
    """
    Creates the same default file layout as the student's original config,
    but rooted at a user-provided data_dir (so you can swap datasets easily).
    """
    data_dir = Path(data_dir)

    raw = data_dir / "raw"
    interim = data_dir / "interim"
    processed = data_dir / "processed"

    return AnalysisPaths(
        data_dir=data_dir,

        metabolite_levels_file_path=raw / "metabolite_levels.csv",
        x_pathway_path=raw / "X_pathway.csv",
        x_reaction_path=raw / "X_reaction_diff.csv",

        pathway_file_path=interim / "pathway.csv",
        reaction_file_path=interim / "reaction.csv",

        metabolite_level_pathway_combined_file_path=interim / "m_level_pathway_comb.csv",
        metabolite_level_reaction_combined_file_path=interim / "m_level_reaction_comb.csv",

        transcriptomics_path=raw / "transcriptomics.csv",
        clinical_ages_path=raw / "clinical.csv",

        best_prs_path=raw / "prs_scores.csv",
        prs_combined_path=processed / "prs_combined_scores.csv",

        transcriptomics_metabolites_combined_path=interim / "transcriptomics_metabolites.csv",
        transcriptomics_pathway_combined_path=interim / "transcriptomics_pathway.csv",
        transcriptomics_reaction_combined_path=interim / "transcriptomics_reaction.csv",
    )
