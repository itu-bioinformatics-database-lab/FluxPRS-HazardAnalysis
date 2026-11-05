"""Canonical schema for the master evaluation manifest."""
from __future__ import annotations

from pathlib import Path


EVALUATION_MANIFEST_COLUMNS = (
    "analysis",
    "cohort",
    "task",
    "metabolitics_mode",
    "n_train",
    "n_test",
    "elapsed_seconds",
    "cv_accuracy_mean",
    "cv_f1_mean",
    "cv_auc_mean",
    "test_accuracy",
    "test_f1",
    "test_auc",
    "reconstruction_error",
    "error_count",
    "extra",
    "output",
)


def _elapsed(summary: dict):
    timing = summary.get("timing")
    if isinstance(timing, dict) and timing.get("elapsed_seconds") is not None:
        return timing["elapsed_seconds"]
    return summary.get("elapsed_seconds", summary.get("elapsed_sec"))


def flatten_run_summary(summary: dict, output: str | Path) -> dict:
    """Flatten one run summary into the master manifest schema."""
    output = Path(output)
    analysis = summary.get("analysis")
    if not analysis:
        analysis = {
            "cv": "logreg_cv_test",
            "clf_extras": "clf_extras",
            "clf_notebook": "notebook_classifiers",
            "cox": "cox_hrs",
            "nmf": "nmf",
        }.get(output.parent.name, output.parent.name)

    feature_sets = summary.get("feature_sets")
    extras = []
    if isinstance(feature_sets, list):
        extras.append(f"feature_sets={len(feature_sets)}")
    if summary.get("xgboost_available") is not None:
        extras.append(f"xgboost={int(bool(summary['xgboost_available']))}")
    if summary.get("n_outer_splits") is not None:
        extras.append(f"outer_folds={summary['n_outer_splits']}")
    if summary.get("n_inner_splits") is not None:
        extras.append(f"inner_folds={summary['n_inner_splits']}")

    primary = {}
    if isinstance(feature_sets, list):
        target = next(
            (item for item in feature_sets if item.get("feature_set") == "all_data"),
            feature_sets[0] if feature_sets else {},
        )
        models = target.get("models", {}) if isinstance(target, dict) else {}
        model = next(
            (
                block
                for name, block in models.items()
                if name.startswith("random_forest_nested")
            ),
            {},
        )
        primary = model if isinstance(model, dict) else {}
    cv = primary.get("cv", {}) if isinstance(primary.get("cv"), dict) else {}
    test = primary.get("test", {}) if isinstance(primary.get("test"), dict) else {}
    reconstruction = summary.get("reconstruction_error")
    if reconstruction is None:
        reconstruction = summary.get("reconstruction_metrics")
    if isinstance(reconstruction, dict):
        reconstruction = reconstruction.get("reconstruction_error")
    errors = summary.get("errors", [])

    return {
        "analysis": analysis,
        "cohort": summary.get("cohort"),
        "task": summary.get("task"),
        "metabolitics_mode": summary.get("metabolitics_mode"),
        "n_train": summary.get("train_n", summary.get("n_train")),
        "n_test": summary.get("test_n", summary.get("n_test")),
        "elapsed_seconds": _elapsed(summary),
        "cv_accuracy_mean": cv.get("mean_accuracy"),
        "cv_f1_mean": cv.get("mean_f1"),
        "cv_auc_mean": cv.get("mean_roc_auc"),
        "test_accuracy": test.get("accuracy"),
        "test_f1": test.get("f1", test.get("macro_f1")),
        "test_auc": test.get("roc_auc", test.get("roc_auc_macro_ovr")),
        "reconstruction_error": reconstruction,
        "error_count": len(errors) if isinstance(errors, list) else None,
        "extra": "; ".join(extras),
        "output": str(output.resolve()),
    }
