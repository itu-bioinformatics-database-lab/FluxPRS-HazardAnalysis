#!/usr/bin/env python
"""Run Metabolitics: fit on train (or cv_train), transform the hold-out.

Adapted from the pairwise Burcu runner:
  metabolite-name-mapping -> fold-change-scaler -> metabolitics-transformer
  -> reaction-diff -> pathway-transformer -> transport-pathway-elimination

Fit FoldChangeScaler and ReactionDiffTransformer on the fit split (healthy
reference from that split only). Transform the hold-out with those fitted
objects. FVA itself is per-sample and does not use labels.

Input CSV columns: Sample ID, Diagnosis, <metabolites>
Diagnosis must use Metabolitics labels: healthy (not Control), plus case labels.

SkUtilsIO.from_csv treats every non-label column as a numeric metabolite and
calls math.isfinite on each value. Sample ID (and other metadata) must be
stripped first; string IDs crash, numeric IDs would be scored as metabolites.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

META_COLUMNS = {
    "Sample ID",
    "unique_row_id",
    "participant_id",
    "Gender",
    "Race",
    "PMI",
    "Braak",
    "sex",
    "row_index",
    "split",
}

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.2g}".format)


def prepare_skutils_frame(
    raw: pd.DataFrame,
    label_column: str,
    source: Path,
    require_healthy: bool = True,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Keep Sample ID aside; give SkUtilsIO only Diagnosis + numeric metabolites."""
    if "Sample ID" not in raw.columns or label_column not in raw.columns:
        raise ValueError(f"{source} must have Sample ID and {label_column}")

    sample_ids = raw["Sample ID"].astype(str).reset_index(drop=True)
    labels = raw[label_column].astype(str).reset_index(drop=True)
    if require_healthy and int((labels == "healthy").sum()) == 0:
        raise ValueError(f"{source} has no 'healthy' labels for FoldChangeScaler")

    drop = [c for c in raw.columns if c in META_COLUMNS or c == label_column]
    mets = raw.drop(columns=drop)
    if mets.shape[1] == 0:
        raise ValueError(f"{source} has no metabolite columns after dropping metadata")

    coerced = mets.apply(pd.to_numeric, errors="coerce")
    bad_cols: list[str] = []
    for col in mets.columns:
        original = mets[col]
        numeric = coerced[col]
        leftover = original.notna() & numeric.isna()
        if leftover.any():
            nonempty = original.astype(str).str.strip() != ""
            if (leftover & nonempty).any():
                bad_cols.append(col)
    if bad_cols:
        raise ValueError(
            f"{source} has non-numeric metabolite values in: {bad_cols[:12]}"
        )

    coerced = coerced.replace([np.inf, -np.inf], np.nan)
    sk_df = coerced.copy()
    sk_df[label_column] = labels.to_numpy()
    return sample_ids, labels, sk_df


def mapping_to_recon_ids(mapping: dict, recon_ids: set[str]) -> tuple[dict, int]:
    """Keep only synonym targets that are metabolite IDs in the COBRA model."""
    kept: dict = {}
    n_drop = 0
    for key, value in mapping.items():
        if isinstance(value, str) and value in recon_ids:
            kept[key] = value
        else:
            n_drop += 1
    return kept, n_drop


def keep_recon_keys(rows: list, recon_ids: set[str]) -> tuple[list, int]:
    """Drop measured metabolites whose names are not in Recon3D."""
    kept: list = []
    n_drop = 0
    for row in rows:
        filtered = {k: v for k, v in row.items() if k in recon_ids}
        n_drop += len(row) - len(filtered)
        if not filtered:
            raise ValueError("No metabolites remain after excluding names absent from Recon3D")
        kept.append(filtered)
    return kept, n_drop


def _skutils_isfinite_ok(sk_df: pd.DataFrame, label_column: str) -> None:
    """Same check SkUtilsIO.from_csv runs: math.isfinite on every feature value."""
    features = sk_df.drop(columns=[label_column])
    for row in features.to_dict(orient="records"):
        for key, value in row.items():
            if not value:
                continue
            if isinstance(value, str):
                raise TypeError(f"string feature {key!r}={value!r}")
            if not math.isfinite(float(value)):
                continue


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if obj is None:
        return None
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def save_json(file_path: Path, obj: Any) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(_jsonable(obj), indent=2))


def save_matrix(file_path: Path, obj: Any, sample_ids: pd.Series, labels: pd.Series) -> None:
    """Write list-of-dicts (or DataFrame) with Sample ID and Diagnosis restored."""
    df = pd.DataFrame(obj)
    out = pd.concat(
        [
            pd.DataFrame({"Sample ID": sample_ids.reset_index(drop=True), "Diagnosis": labels.reset_index(drop=True)}),
            df.reset_index(drop=True),
        ],
        axis=1,
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(file_path, index=False)


class AnalysisPipeline:
    def __init__(
        self,
        fit_path: Path,
        fit_output_dir: Path,
        mapping_kind: str | None,
        mapping_platform: str,
        label_column: str,
    ) -> None:
        self.fit_path = Path(fit_path)
        self.fit_output_dir = Path(fit_output_dir)
        self.fit_output_dir.mkdir(parents=True, exist_ok=True)
        self.mapping_kind = mapping_kind
        self.mapping_platform = mapping_platform
        self.label_column = label_column

        from preprocessing.metabolitics_pipeline import MetaboliticsPipeline
        from sklearn_utils.preprocessing import FeatureRenaming
        from utils import load_metabolite_mapping, load_network_model

        self.sample_ids, self.y, self.X = self._load_split(
            self.fit_path, self.fit_output_dir, require_healthy=True
        )
        self.recon_ids = {m.id for m in load_network_model("recon3D").metabolites}
        self.n_mapping_targets_dropped = 0
        initial_steps: list[str] = []
        if mapping_kind:
            mapping = load_metabolite_mapping(mapping_kind, self.X, mapping_platform)
            mapping, self.n_mapping_targets_dropped = mapping_to_recon_ids(
                mapping, self.recon_ids
            )
            print(
                f"Mapping targets in Recon3D: {len(mapping)}; "
                f"dropped non-model targets: {self.n_mapping_targets_dropped}"
            )
            MetaboliticsPipeline.steps["metabolite-name-mapping"] = FeatureRenaming(mapping)
            initial_steps.append("metabolite-name-mapping")

        self.initial_steps = initial_steps
        self.transformer_pipe = MetaboliticsPipeline(
            initial_steps
            + [
                "fold-change-scaler",
                "metabolitics-transformer",
                "reaction-diff",
                "pathway-transformer",
                "transport-pathway-elimination",
            ]
        )

    def _load_split(
        self, path: Path, output_dir: Path, require_healthy: bool
    ) -> tuple[pd.Series, pd.Series, list]:
        from sklearn_utils.utils import SkUtilsIO

        raw = pd.read_csv(path)
        sample_ids, labels, sk_df = prepare_skutils_frame(
            raw, self.label_column, path, require_healthy=require_healthy
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        sk_path = output_dir / "skutils_input.csv"
        sk_df.to_csv(sk_path, index=False)
        X, y = SkUtilsIO(str(sk_path)).from_csv(label_column=self.label_column)
        y = pd.Series(y).astype(str).reset_index(drop=True)
        if len(y) != len(sample_ids):
            raise ValueError(f"SkUtilsIO row count {len(y)} != CSV rows {len(sample_ids)}")
        return sample_ids, y, X

    def _run_chain(
        self,
        X,
        y: pd.Series,
        sample_ids: pd.Series,
        output_dir: Path,
        *,
        fit: bool,
        extra_manifest: dict,
    ) -> None:
        from sklearn_utils.utils import feature_importance_report

        output_dir.mkdir(parents=True, exist_ok=True)
        n_healthy = int((y == "healthy").sum())
        if fit and n_healthy == 0:
            raise ValueError(f"{extra_manifest.get('input')} has no 'healthy' labels for FoldChangeScaler")

        manifest = {
            "n_rows": int(len(y)),
            "label_counts": {str(k): int(v) for k, v in y.value_counts().items()},
            "n_healthy": n_healthy,
            "mapping_kind": self.mapping_kind,
            "mapping_platform": self.mapping_platform,
            "n_mapping_targets_dropped": int(self.n_mapping_targets_dropped),
            "fit_scope": "fit_train_transform_holdout",
            "stage": "fit" if fit else "transform",
            "healthy_reference_from": "this_split" if fit else "fit_split_only",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "hostname": os.uname().nodename,
        }
        manifest.update(extra_manifest)
        save_json(output_dir / "run_manifest.json", manifest)
        pd.DataFrame(
            {"Sample ID": sample_ids.reset_index(drop=True), "Diagnosis": y}
        ).to_csv(output_dir / "sample_ids.csv", index=False)

        if self.initial_steps:
            mapper = self.transformer_pipe["metabolite-name-mapping"]
            x1 = mapper.fit_transform(X=X, y=y) if fit else mapper.transform(X=X, y=y)
        else:
            x1 = X
        x1, n_unmapped_model = keep_recon_keys(x1, self.recon_ids)
        print(
            f"{'FIT' if fit else 'TRANSFORM'} metabolites in Recon3D after mapping: "
            f"{len(x1[0])}; excluded (not in model): {n_unmapped_model}"
        )
        manifest["n_metabolites_in_model"] = int(len(x1[0]))
        manifest["n_names_excluded_not_in_model"] = int(n_unmapped_model)
        save_json(output_dir / "mapped-metabolites.json", x1)
        save_json(output_dir / "run_manifest.json", manifest)

        fc = self.transformer_pipe["fold-change-scaler"]
        x2 = fc.fit_transform(X=x1, y=y) if fit else fc.transform(x1)
        save_json(output_dir / "metabolite-fold-changes.json", x2)
        save_matrix(output_dir / "fold_changes.csv", x2, sample_ids, y)

        x3 = self.transformer_pipe["metabolitics-transformer"].transform(X=x2)
        save_matrix(output_dir / "X_transformed.csv", x3, sample_ids, y)

        rd = self.transformer_pipe["reaction-diff"]
        x4 = rd.fit_transform(X=x3, y=y) if fit else rd.transform(X=x3)
        save_matrix(output_dir / "X_reaction_diff.csv", x4, sample_ids, y)

        pw = self.transformer_pipe["pathway-transformer"]
        x6 = pw.fit_transform(X=x4, y=y) if fit else pw.transform(X=x4, y=y)
        save_matrix(output_dir / "X_pathway_with_transport.csv", x6, sample_ids, y)
        save_json(output_dir / "pathway-diff-scores.json", x6)

        tp = self.transformer_pipe["transport-pathway-elimination"]
        x7 = tp.fit_transform(X=x6, y=y) if fit else tp.transform(X=x6, y=y)
        save_matrix(output_dir / "X_pathway.csv", x7, sample_ids, y)
        save_json(output_dir / "pathway-diff-scores-no-transport-pathways.json", x7)

        x8 = feature_importance_report(pd.DataFrame(x6).fillna(0), y)
        x = x8.copy()
        x.index.name = "Pathways"
        x.to_csv(output_dir / "pathway_feature_importances.csv", index=True)

        x9 = feature_importance_report(pd.DataFrame(x7).fillna(0), y)
        x = x9.copy()
        x.index.name = "Pathways"
        x.to_csv(
            output_dir / "pathway_feature_importances-no-transport-pathways.csv",
            index=True,
        )

        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_json(output_dir / "run_manifest.json", manifest)

    def fit_transform(self) -> None:
        self._run_chain(
            self.X,
            self.y,
            self.sample_ids,
            self.fit_output_dir,
            fit=True,
            extra_manifest={"input": str(self.fit_path)},
        )

    def transform_holdout(self, holdout_path: Path, holdout_output_dir: Path) -> None:
        holdout_path = Path(holdout_path)
        holdout_output_dir = Path(holdout_output_dir)
        sample_ids, y, X = self._load_split(
            holdout_path, holdout_output_dir, require_healthy=False
        )
        self._run_chain(
            X,
            y,
            sample_ids,
            holdout_output_dir,
            fit=False,
            extra_manifest={
                "input": str(holdout_path),
                "fit_input": str(self.fit_path),
                "fit_output": str(self.fit_output_dir),
            },
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit Metabolitics on one split and transform a hold-out split."
    )
    parser.add_argument("--input-fit", help="CSV to fit (train or cv_train)")
    parser.add_argument("--output-fit", help="Output directory for the fit split")
    parser.add_argument("--input-transform", help="CSV to transform (test or cv_val)")
    parser.add_argument("--output-transform", help="Output directory for the hold-out split")
    parser.add_argument(
        "--input",
        help="Deprecated single-file mode (fit_transform on this file only)",
    )
    parser.add_argument("--output", help="Deprecated single-file output directory")
    parser.add_argument("--label-column", default="Diagnosis")
    parser.add_argument("--mapping-kind", default="new-synonym")
    parser.add_argument("--mapping-platform")
    parser.add_argument("--skip-mapping", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check Sample ID stripping and numeric metabolites; do not run Metabolitics.",
    )
    return parser.parse_args()


def validate_input(input_path: Path, label_column: str, require_healthy: bool) -> int:
    raw = pd.read_csv(input_path)
    sample_ids, labels, sk_df = prepare_skutils_frame(
        raw, label_column, input_path, require_healthy=require_healthy
    )
    _skutils_isfinite_ok(sk_df, label_column)
    n_feat = sk_df.shape[1] - 1
    print(
        f"OK {input_path}: n={len(sample_ids)} metabolites={n_feat} "
        f"labels={labels.value_counts().to_dict()} healthy={int((labels == 'healthy').sum())}"
    )
    return 0


def main() -> int:
    args = parse_args()
    mapping_kind = None if args.skip_mapping else args.mapping_kind

    if args.validate_only:
        paths = [p for p in (args.input_fit, args.input_transform, args.input) if p]
        if not paths:
            print("--validate-only needs --input-fit/--input-transform or --input", file=sys.stderr)
            return 2
        rc = 0
        for i, p in enumerate(paths):
            rc = validate_input(Path(p), args.label_column, require_healthy=(i == 0)) or rc
        return rc

    fit_path = Path(args.input_fit) if args.input_fit else (Path(args.input) if args.input else None)
    fit_out = Path(args.output_fit) if args.output_fit else (Path(args.output) if args.output else None)
    if fit_path is None or fit_out is None:
        print("--input-fit and --output-fit are required", file=sys.stderr)
        return 2
    if not fit_path.is_file():
        print(f"Fit input not found: {fit_path}", file=sys.stderr)
        return 1
    if not args.mapping_platform and not args.skip_mapping:
        print("--mapping-platform is required unless --skip-mapping", file=sys.stderr)
        return 2

    holdout_path = Path(args.input_transform) if args.input_transform else None
    holdout_out = Path(args.output_transform) if args.output_transform else None
    if (holdout_path is None) != (holdout_out is None):
        print("Pass both --input-transform and --output-transform, or neither", file=sys.stderr)
        return 2
    if holdout_path is not None and not holdout_path.is_file():
        print(f"Transform input not found: {holdout_path}", file=sys.stderr)
        return 1

    print(f"FIT Metabolitics: {fit_path}")
    print(f"FIT outputs: {fit_out}")
    if holdout_path is not None:
        print(f"TRANSFORM: {holdout_path}")
        print(f"TRANSFORM outputs: {holdout_out}")
    print(f"Mapping: {mapping_kind} / {args.mapping_platform}")
    pipe = AnalysisPipeline(
        fit_path=fit_path,
        fit_output_dir=fit_out,
        mapping_kind=mapping_kind,
        mapping_platform=args.mapping_platform,
        label_column=args.label_column,
    )
    pipe.fit_transform()
    print(f"Finished fit: {fit_out}")
    if holdout_path is not None:
        pipe.transform_holdout(holdout_path, holdout_out)
        print(f"Finished transform: {holdout_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
