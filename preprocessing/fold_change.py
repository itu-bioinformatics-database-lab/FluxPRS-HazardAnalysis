"""Fold-change scaler matching TRUBA metabolitics-dev / sklearn_utils.

Source of truth on TRUBA:
  /arf/home/cnjume/miniconda3/envs/metaboliticsenv/lib/python3.8/site-packages/
    sklearn_utils/preprocessing/fold_change_preprocessing.py

MetaboliticsPipeline registers:
  'fold-change-scaler': FoldChangeScaler('healthy')

Formula (per metabolite k, value v, reference mean avg):
  e = v / avg
  if avg > v:  clip(1 - 1/e, min_bound)
  else:        clip(e - 1, max_bound)

The reference mean is the column-wise mean of samples whose label equals
reference_label (default 'healthy'). Fit and transform must be run on the
same split: train fold-change uses train healthy only; test fold-change
uses test healthy only.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_REFERENCE_LABEL = "healthy"
DEFAULT_BOUNDS = (-10.0, 10.0)

NON_METABOLITE_COLUMNS = (
    "Sample ID",
    "unique_row_id",
    "participant_id",
    "Diagnosis",
    "Gender",
    "Race",
    "PMI",
    "Braak",
    "sex",
    "row_index",
    "split",
)


class FoldChangeScaler:
    """Pandas replica of sklearn_utils.preprocessing.FoldChangeScaler."""

    def __init__(
        self,
        reference_label: str = DEFAULT_REFERENCE_LABEL,
        bounds: Tuple[float, float] = DEFAULT_BOUNDS,
    ):
        self.reference_label = reference_label
        self._min = bounds[0]
        self._max = bounds[1]
        self._avgs: Optional[pd.Series] = None
        self.metabolite_columns: Optional[list[str]] = None

    def fit(self, X: pd.DataFrame, y: Sequence[str]) -> "FoldChangeScaler":
        y = pd.Series(y, index=X.index)
        n_ref = int((y == self.reference_label).sum())
        if n_ref == 0:
            raise ValueError(
                f"No samples with reference_label={self.reference_label!r}. "
                "Remap Control -> healthy before fold-change."
            )
        numeric_names = metabolite_columns(X)
        if not numeric_names:
            raise ValueError("No metabolite columns found for fold-change.")
        self.metabolite_columns = numeric_names
        self._avgs = X.loc[y == self.reference_label, self.metabolite_columns].mean()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._avgs is None or self.metabolite_columns is None:
            raise RuntimeError("FoldChangeScaler must be fit before transform.")
        out = X.copy()
        scaled = _scale_matrix(
            X[self.metabolite_columns].astype(float),
            self._avgs,
            min_bound=self._min,
            max_bound=self._max,
        )
        out[self.metabolite_columns] = scaled
        return out

    def fit_transform(self, X: pd.DataFrame, y: Sequence[str]) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def reference_means(self) -> pd.Series:
        if self._avgs is None:
            raise RuntimeError("FoldChangeScaler has not been fit.")
        return self._avgs.copy()


def _scale_matrix(
    X: pd.DataFrame,
    avgs: pd.Series,
    min_bound: float,
    max_bound: float,
) -> pd.DataFrame:
    """Vectorized FoldChangeScaler._scale, including the (-10, 10) clip."""
    avg = avgs.reindex(X.columns).astype(float)
    values = X.to_numpy(dtype=float)
    avg_vals = avg.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        e = values / avg_vals
        inv_e = np.where(e == 0.0, np.inf, 1.0 / e)
        low = 1.0 - inv_e
        high = e - 1.0
        scaled = np.where(avg_vals > values, low, high)

    # inf from v=0 clips to bounds; 0/0 stays NaN
    scaled = np.clip(scaled, min_bound, max_bound)
    return pd.DataFrame(scaled, index=X.index, columns=X.columns)


def metabolite_columns(df: pd.DataFrame, extra_drop: Iterable[str] = ()) -> list[str]:
    drop = set(NON_METABOLITE_COLUMNS) | set(extra_drop)
    cols = []
    for col in df.columns:
        if col in drop:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def to_metabolitics_labels(diagnosis: pd.Series) -> pd.Series:
    """FoldChangeScaler and ReactionDiffTransformer look up label 'healthy'."""
    mapping = {
        "Control": "healthy",
        "control": "healthy",
        "CN": "healthy",
        "Normal": "healthy",
        "NGT": "healthy",
        "Healthy": "healthy",
    }
    return diagnosis.replace(mapping)
