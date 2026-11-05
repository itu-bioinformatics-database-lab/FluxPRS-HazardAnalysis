"""70/30 cohort splits and per-split Metabolitics fold-change.

Split rules
-----------
* ROSMAP-AD, MayoRNASeq-AD, MayoRNASeq-PSP: one row = one person.
  Stratified 70/30 on Diagnosis, random_state=42.
* DIABETES (DiCAD): 285 rows / 177 people. Split on participant_id
  (dicad_id, 1:1 with Sample ID). Every repeat biospecimen of a person
  stays in the same split.

Fold-change
-----------
Train and test are scaled independently. Each split uses its own
healthy/Control samples as the FoldChangeScaler reference. Test is
never transformed with the train reference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

try:
    from fold_change import (
        DEFAULT_REFERENCE_LABEL,
        FoldChangeScaler,
        metabolite_columns,
        to_metabolitics_labels,
    )
except ImportError:
    from preprocessing.fold_change import (
        DEFAULT_REFERENCE_LABEL,
        FoldChangeScaler,
        metabolite_columns,
        to_metabolitics_labels,
    )

PathLike = Union[str, Path]

COHORTS: tuple[str, ...] = (
    "ROSMAP-AD",
    "MayoRNASeq-AD",
    "MayoRNASeq-PSP",
    "DIABETES",
)

TEST_SIZE = 0.3
RANDOM_STATE = 42
ID_COLUMNS = [
    "row_index",
    "unique_row_id",
    "sample_id",
    "participant_id",
    "diagnosis",
    "metabolitics_label",
    "split",
    "n_rows_for_participant",
]


@dataclass(frozen=True)
class SplitPaths:
    data_dir: Path
    metabolite_levels_file_path: Path
    splits_dir: Path
    train_ids_path: Path
    test_ids_path: Path
    splits_all_ids_path: Path
    split_manifest_path: Path
    metabolitics_input_dir: Path


def make_split_paths(data_dir: PathLike) -> SplitPaths:
    data_dir = Path(data_dir)
    processed = data_dir / "processed"
    splits = processed / "splits"
    meta = processed / "metabolitics_input"
    return SplitPaths(
        data_dir=data_dir,
        metabolite_levels_file_path=data_dir / "raw" / "metabolite_levels.csv",
        splits_dir=splits,
        train_ids_path=splits / "train_ids.csv",
        test_ids_path=splits / "test_ids.csv",
        splits_all_ids_path=splits / "all_ids.csv",
        split_manifest_path=splits / "split_manifest.json",
        metabolitics_input_dir=meta,
    )


@dataclass(frozen=True)
class CohortSplitResult:
    cohort: str
    ids: pd.DataFrame
    train_input: pd.DataFrame
    test_input: pd.DataFrame
    train_foldchange: pd.DataFrame
    test_foldchange: pd.DataFrame
    train_reference_means: pd.Series
    test_reference_means: pd.Series
    manifest: dict
    paths: SplitPaths


def _read_metabolite_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Sample ID" not in df.columns or "Diagnosis" not in df.columns:
        raise ValueError(f"{path} must contain Sample ID and Diagnosis")
    df = df.copy()
    df["Sample ID"] = df["Sample ID"].astype(str)
    df["row_index"] = np.arange(len(df), dtype=int)
    return df


def _attach_participant_ids(df: pd.DataFrame, data_dir: Path, cohort: str) -> pd.DataFrame:
    out = df.copy()
    if cohort == "DIABETES":
        source_path = data_dir / "raw" / "Diabetes_source.csv"
        if not source_path.exists():
            raise FileNotFoundError(
                f"DiCAD person split needs {source_path} (dicad_id / Sample Bar Code)"
            )
        source = pd.read_csv(source_path)
        if len(source) != len(out):
            raise ValueError(
                f"Diabetes_source.csv has {len(source)} rows, metabolite_levels has {len(out)}"
            )
        out["participant_id"] = source["dicad_id"].astype(str).values
        out["unique_row_id"] = source["Sample Bar Code"].astype(str).values
        mixed = out.groupby("participant_id")["Diagnosis"].nunique()
        if int((mixed > 1).sum()) != 0:
            raise ValueError("DiCAD participants with mixed Diagnosis; refusing to split")
        sample_vs_person = out.groupby("Sample ID")["participant_id"].nunique()
        if int(sample_vs_person.max()) != 1:
            raise ValueError("DiCAD Sample ID maps to more than one dicad_id")
    else:
        out["participant_id"] = out["Sample ID"].astype(str)
        out["unique_row_id"] = out["Sample ID"].astype(str)
    return out


def _person_level_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> pd.DataFrame:
    people = (
        df.groupby("participant_id", sort=False)
        .agg(diagnosis=("Diagnosis", "first"), n_rows=("row_index", "size"))
        .reset_index()
    )
    counts = people["diagnosis"].value_counts()
    if counts.min() < 2:
        raise ValueError(
            f"Cannot stratify 70/30: a Diagnosis class has <2 people ({counts.to_dict()})"
        )
    train_people, test_people = train_test_split(
        people["participant_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=people["diagnosis"],
    )
    train_set = set(train_people.astype(str))
    test_set = set(test_people.astype(str))
    overlap = train_set & test_set
    if overlap:
        raise RuntimeError(f"Participant leak between train and test: {sorted(overlap)[:5]}")

    n_rows = df.groupby("participant_id")["row_index"].transform("size")
    split = np.where(df["participant_id"].astype(str).isin(train_set), "train", "test")
    ids = pd.DataFrame(
        {
            "row_index": df["row_index"].astype(int),
            "unique_row_id": df["unique_row_id"].astype(str),
            "sample_id": df["Sample ID"].astype(str),
            "participant_id": df["participant_id"].astype(str),
            "diagnosis": df["Diagnosis"].astype(str),
            "metabolitics_label": to_metabolitics_labels(df["Diagnosis"]).astype(str),
            "split": split,
            "n_rows_for_participant": n_rows.astype(int),
        }
    )
    return ids[ID_COLUMNS]


def _metabolitics_matrix(df: pd.DataFrame, ids: pd.DataFrame, split: str) -> pd.DataFrame:
    keep = ids.loc[ids["split"] == split, "row_index"]
    part = df.loc[df["row_index"].isin(keep)].copy()
    part = part.merge(
        ids[["row_index", "metabolitics_label"]],
        on="row_index",
        how="left",
    )
    part["Diagnosis"] = part["metabolitics_label"]
    cols = ["Sample ID", "Diagnosis"] + metabolite_columns(part)
    return part[cols].reset_index(drop=True)


def _foldchange_split(matrix: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.Series]:
    y = matrix["Diagnosis"]
    n_healthy = int((y == DEFAULT_REFERENCE_LABEL).sum())
    if n_healthy == 0:
        raise ValueError(f"{split_name} has no '{DEFAULT_REFERENCE_LABEL}' samples for fold-change")
    scaler = FoldChangeScaler(reference_label=DEFAULT_REFERENCE_LABEL)
    scaled = scaler.fit_transform(matrix, y)
    return scaled, scaler.reference_means


def _counts(ids: pd.DataFrame, split: str, col: str = "diagnosis") -> dict:
    part = ids.loc[ids["split"] == split]
    return {
        "n_rows": int(len(part)),
        "n_participants": int(part["participant_id"].nunique()),
        "labels": {str(k): int(v) for k, v in part[col].value_counts().items()},
        "healthy_rows": int((part["metabolitics_label"] == DEFAULT_REFERENCE_LABEL).sum()),
        "healthy_participants": int(
            part.loc[part["metabolitics_label"] == DEFAULT_REFERENCE_LABEL, "participant_id"].nunique()
        ),
    }


def prepare_cohort_split(
    project_root: PathLike,
    cohort: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    write: bool = True,
) -> CohortSplitResult:
    project_root = Path(project_root)
    data_dir = project_root / cohort
    paths = make_split_paths(data_dir)
    df = _read_metabolite_table(paths.metabolite_levels_file_path)
    df = _attach_participant_ids(df, data_dir, cohort)
    ids = _person_level_split(df, test_size=test_size, random_state=random_state)

    train_input = _metabolitics_matrix(df, ids, "train")
    test_input = _metabolitics_matrix(df, ids, "test")
    train_fc, train_means = _foldchange_split(train_input, f"{cohort}/train")
    test_fc, test_means = _foldchange_split(test_input, f"{cohort}/test")

    split_unit = "participant" if cohort == "DIABETES" else "sample"
    manifest = {
        "cohort": cohort,
        "test_size": test_size,
        "random_state": random_state,
        "split_unit": split_unit,
        "fold_change": {
            "method": "sklearn_utils.FoldChangeScaler",
            "reference_label": DEFAULT_REFERENCE_LABEL,
            "bounds": [-10, 10],
            "fit_scope": "independent_per_split",
            "note": (
                "Train fold-change uses train healthy means only. "
                "Test fold-change uses test healthy means only. "
                "Control is remapped to healthy for Metabolitics."
            ),
        },
        "n_rows": int(len(ids)),
        "n_participants": int(ids["participant_id"].nunique()),
        "train": _counts(ids, "train"),
        "test": _counts(ids, "test"),
        "files": {
            "all_ids": str(paths.splits_all_ids_path),
            "train_ids": str(paths.train_ids_path),
            "test_ids": str(paths.test_ids_path),
            "manifest": str(paths.split_manifest_path),
            "train_metabolitics_input": str(paths.metabolitics_input_dir / "train.csv"),
            "test_metabolitics_input": str(paths.metabolitics_input_dir / "test.csv"),
            "train_foldchange": str(paths.metabolitics_input_dir / "train_foldchange.csv"),
            "test_foldchange": str(paths.metabolitics_input_dir / "test_foldchange.csv"),
        },
    }

    if write:
        _write_outputs(
            paths,
            ids,
            train_input,
            test_input,
            train_fc,
            test_fc,
            train_means,
            test_means,
            manifest,
        )

    return CohortSplitResult(
        cohort=cohort,
        ids=ids,
        train_input=train_input,
        test_input=test_input,
        train_foldchange=train_fc,
        test_foldchange=test_fc,
        train_reference_means=train_means,
        test_reference_means=test_means,
        manifest=manifest,
        paths=paths,
    )


def _write_outputs(
    paths: SplitPaths,
    ids: pd.DataFrame,
    train_input: pd.DataFrame,
    test_input: pd.DataFrame,
    train_fc: pd.DataFrame,
    test_fc: pd.DataFrame,
    train_means: pd.Series,
    test_means: pd.Series,
    manifest: dict,
) -> None:
    splits_dir = paths.splits_dir
    meta_dir = paths.metabolitics_input_dir
    splits_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    ids.to_csv(paths.splits_all_ids_path, index=False)
    ids.loc[ids["split"] == "train"].to_csv(paths.train_ids_path, index=False)
    ids.loc[ids["split"] == "test"].to_csv(paths.test_ids_path, index=False)
    train_input.to_csv(meta_dir / "train.csv", index=False)
    test_input.to_csv(meta_dir / "test.csv", index=False)
    train_fc.to_csv(meta_dir / "train_foldchange.csv", index=False)
    test_fc.to_csv(meta_dir / "test_foldchange.csv", index=False)
    train_means.rename("reference_mean").to_csv(meta_dir / "train_reference_means.csv", header=True)
    test_means.rename("reference_mean").to_csv(meta_dir / "test_reference_means.csv", header=True)
    paths.split_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def load_split_ids(data_dir: PathLike) -> pd.DataFrame:
    """Load the saved all_ids table for a cohort data_dir."""
    paths = make_split_paths(data_dir)
    if not paths.splits_all_ids_path.exists():
        raise FileNotFoundError(f"No saved split at {paths.splits_all_ids_path}")
    return pd.read_csv(paths.splits_all_ids_path)


N_CV_SPLITS = 5


def assign_cv_folds(
    labels: Sequence,
    groups: Optional[Sequence] = None,
    n_splits: int = N_CV_SPLITS,
    random_state: int = RANDOM_STATE,
    grouped: bool = False,
) -> np.ndarray:
    """Return the validation-fold index (0 .. n_splits-1) for each row.

    ROSMAP uses StratifiedKFold. DiCAD uses StratifiedGroupKFold on
    participant_id so repeat biospecimens stay in the same fold.
    """
    y = np.asarray(labels)
    n = len(y)
    if n == 0:
        raise ValueError("Cannot assign CV folds to an empty label vector")
    val_fold = np.full(n, -1, dtype=int)
    dummy = np.zeros((n, 1))
    if grouped:
        if groups is None:
            raise ValueError("grouped CV requires groups (participant_id)")
        group_arr = np.asarray(groups)
        if len(group_arr) != n:
            raise ValueError("groups length does not match labels")
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = splitter.split(dummy, y, groups=group_arr)
    else:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
        splits = splitter.split(dummy, y)
    for fold, (_, val_idx) in enumerate(splits):
        val_fold[val_idx] = fold
    if (val_fold < 0).any():
        raise RuntimeError("Some rows were not assigned a CV fold")
    if grouped:
        group_arr = np.asarray(groups)
        for group in np.unique(group_arr):
            folds = np.unique(val_fold[group_arr == group])
            if len(folds) != 1:
                raise RuntimeError(
                    f"Participant {group} was split across CV folds {folds.tolist()}"
                )
    return val_fold


def apply_saved_split(
    df: pd.DataFrame,
    ids: pd.DataFrame,
    id_col: str = "Sample ID",
    unique_row_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a table into train/test using saved IDs.

    Prefer unique_row_id when the table can have repeat Sample IDs (DiCAD).
    """
    ids = ids.copy()
    work = df.copy()
    if unique_row_col and unique_row_col in work.columns:
        key = unique_row_col
        id_map = ids.set_index("unique_row_id")["split"]
        work["_split"] = work[key].astype(str).map(id_map)
    elif work[id_col].astype(str).nunique() == len(work):
        id_map = ids.drop_duplicates("sample_id").set_index("sample_id")["split"]
        work["_split"] = work[id_col].astype(str).map(id_map)
    else:
        id_map = ids.set_index("row_index")["split"]
        if "row_index" not in work.columns:
            if len(work) != ids["row_index"].nunique() and len(work) != len(ids):
                raise ValueError(
                    "Repeat Sample IDs and no unique_row_id/row_index; "
                    "cannot apply the DiCAD split safely"
                )
            work = work.copy()
            work["row_index"] = np.arange(len(work), dtype=int)
        work["_split"] = work["row_index"].map(id_map)

    if work["_split"].isna().any():
        missing = int(work["_split"].isna().sum())
        raise ValueError(f"{missing} rows could not be matched to saved split IDs")
    train = work.loc[work["_split"] == "train"].drop(columns=["_split"])
    test = work.loc[work["_split"] == "test"].drop(columns=["_split"])
    return train, test


def prepare_all_cohorts(
    project_root: PathLike,
    cohorts: Sequence[str] = COHORTS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> list[CohortSplitResult]:
    return [
        prepare_cohort_split(
            project_root,
            cohort,
            test_size=test_size,
            random_state=random_state,
            write=True,
        )
        for cohort in cohorts
    ]
