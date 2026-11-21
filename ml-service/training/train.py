"""Home Credit: compare on Training CV, refit separately, evaluate Test separately.

Training exports a versioned candidate. Activation requires separate reload/API checks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import sys
from time import perf_counter
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from app.config import get_settings
from app.home_credit_features import (
    CATEGORICAL, CONTRACT_VERSION, DATASET, FEATURES, FEATURE_SPEC, ID_COLUMN, NUMERIC,
    SPECIAL_CATEGORIES, TARGET, TARGET_MEANING, build_preprocessor, validate_features,
)

MODEL_NAMES = ("DecisionTree", "LinearSVM", "ANN")
CONFIGURATIONS = {
    "DecisionTree": [
        {"max_depth": 5, "min_samples_split": 800, "min_samples_leaf": 400, "class_weight": "balanced"},
        {"max_depth": 8, "min_samples_split": 400, "min_samples_leaf": 200, "class_weight": "balanced"},
    ],
    "LinearSVM": [
        {"C": 0.01, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
    ],
    "ANN": [
        {"hidden_layer_sizes": (16,), "alpha": 0.001},
        {"hidden_layer_sizes": (24, 12), "alpha": 0.01},
    ],
}
CLASSIFICATION_THRESHOLD = 0.5  # Fixed before any Test evaluation; no Test tuning.
SELECTION_POLICY = "max CV mean AP; exact ties use candidate order (DecisionTree, LinearSVM, ANN)"
CALIBRATION = {"method": "sigmoid", "n_splits": 3, "random_state": 42, "ensemble": False}


@dataclass(frozen=True)
class TrainingData:
    x: pd.DataFrame
    y: pd.Series
    record_ids: np.ndarray


@dataclass(frozen=True)
class HeldOutTest:
    x: pd.DataFrame
    y: pd.Series
    record_ids: np.ndarray


@dataclass(frozen=True)
class RefitModel:
    pipeline: CalibratedClassifierCV
    model_name: str
    training_record_ids: np.ndarray
    profile: str
    fit_seconds: float
    convergence_warnings: list[str]


def split_training_test(frame: pd.DataFrame) -> tuple[TrainingData, HeldOutTest]:
    """One fixed internal 80/20 split from labeled application_train only.

    Kaggle application_test.csv is never used. Return separate types so a Test
    partition cannot accidentally be passed into comparison/refitting.
    """
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("Split requires nonmissing, unique SK_ID_CURR")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("Split requires complete binary TARGET with both classes")
    positions = np.arange(len(frame))
    train_pos, test_pos = train_test_split(
        positions, test_size=0.2, stratify=frame[TARGET], random_state=42,
    )

    def partition(kind, rows):
        return kind(
            x=frame.loc[:, FEATURES].iloc[rows],
            y=frame[TARGET].iloc[rows].astype(int),
            record_ids=frame[ID_COLUMN].iloc[rows].to_numpy(copy=True),
        )

    return partition(TrainingData, train_pos), partition(HeldOutTest, test_pos)


def _validate_training(training: TrainingData, min_per_class: int = 1) -> None:
    if not isinstance(training, TrainingData):
        raise TypeError("Only TrainingData is accepted; HeldOutTest must stay outside model selection")
    validate_features(training.x)
    if not (len(training.x) == len(training.y) == len(training.record_ids)):
        raise ValueError("Training features, labels and IDs must have the same length")
    if not training.x.index.equals(training.y.index):
        raise ValueError("Training features and labels must be aligned")
    if len(np.unique(training.record_ids)) != len(training.record_ids):
        raise ValueError("Training IDs must be unique")
    counts = training.y.value_counts()
    if training.y.isna().any() or set(counts.index) != {0, 1} or counts.min() < min_per_class:
        raise ValueError(f"Training needs both labels and at least {min_per_class} rows per class")


def make_model(name: str, *, profile: str = "standard", candidate: int = 0):
    if profile not in {"standard", "smoke"}:
        raise ValueError(f"Unknown model profile: {profile}")
    params = CONFIGURATIONS[name][candidate].copy()
    if name == "DecisionTree":
        if profile == "smoke":
            params.update(min_samples_split=10, min_samples_leaf=5)
        return DecisionTreeClassifier(random_state=42, **params)
    if name == "LinearSVM":
        return LinearSVC(dual=False, tol=1e-5, max_iter=2000, random_state=42, **params)
    if name == "ANN":
        return MLPClassifier(
            activation="relu", solver="adam", batch_size=32 if profile == "smoke" else 256, random_state=42,
            max_iter=8 if profile == "smoke" else 60, tol=1e-4, n_iter_no_change=7, **params,
        )
    raise ValueError(f"Unknown model: {name}")


def _new_pipeline(name: str, *, profile: str, candidate: int) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=name != "DecisionTree")),
        ("model", make_model(name, profile=profile, candidate=candidate)),
    ])


def _fit(estimator, x, y) -> tuple[float, list[str]]:
    started = perf_counter()
    with threadpool_limits(limits=2), warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(x, y)
    convergence = []
    for warning in recorded:
        if issubclass(warning.category, ConvergenceWarning):
            convergence.append(str(warning.message))
        else:
            warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return perf_counter() - started, convergence


def _partition_digest(record_ids) -> str:
    return hashlib.sha256(json.dumps(np.asarray(record_ids).tolist()).encode("utf-8")).hexdigest()


def probability_metrics(y_true, probabilities) -> dict:
    """Threshold-free AP (average precision, not trapezoidal PR-AUC) and AUC."""
    return {
        "ap": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def compare_on_training(training: TrainingData, *, profile: str = "standard", progress=None) -> dict:
    """Compare candidate configurations on the same five Training folds."""
    _validate_training(training, min_per_class=5)
    started = perf_counter()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = [
        (fit, valid, _partition_digest(training.record_ids[fit]), _partition_digest(training.record_ids[valid]))
        for fit, valid in splitter.split(training.x, training.y)
    ]
    results = {}
    for name in MODEL_NAMES:
        candidates = []
        count = len(CONFIGURATIONS[name]) if profile == "standard" else 1
        for candidate in range(count):
            folds = []
            for fold, (fit_pos, valid_pos, fit_digest, valid_digest) in enumerate(splits, start=1):
                x_fit, y_fit = training.x.iloc[fit_pos], training.y.iloc[fit_pos]
                x_valid, y_valid = training.x.iloc[valid_pos], training.y.iloc[valid_pos]
                pipe = _new_pipeline(name, profile=profile, candidate=candidate)
                seconds, convergence = _fit(pipe, x_fit, y_fit)
                scores = pipe.decision_function(x_valid) if name == "LinearSVM" else pipe.predict_proba(x_valid)[:, 1]
                folds.append({
                    "fold": fold, "fit_rows": len(fit_pos), "validation_rows": len(valid_pos),
                    "fit_ids_sha256": fit_digest, "validation_ids_sha256": valid_digest,
                    "fit_seconds": seconds, "convergence_warnings": convergence,
                    **probability_metrics(y_valid, scores),
                })
                message = f"CV {name} config {candidate + 1}/{count} fold {fold}/5 AP={folds[-1]['ap']:.6f}"
                print(message, file=sys.stderr, flush=True)
                if progress:
                    progress(message)
            candidates.append({
                "candidate": candidate, "parameters": pipe.named_steps["model"].get_params(),
                "score_type": "decision_function" if name == "LinearSVM" else "predict_proba",
                "folds": folds,
                "fit_seconds": sum(fold["fit_seconds"] for fold in folds),
                "summary": {
                    f"{metric}_{stat}": float(operation([fold[metric] for fold in folds]))
                    for metric in ("ap", "roc_auc")
                    for stat, operation in (("mean", np.mean), ("std", np.std))
                },
            })
        best = max(candidates, key=lambda result: result["summary"]["ap_mean"])
        results[name] = {
            **best, "configurations": candidates,
            "total_fit_seconds": sum(result["fit_seconds"] for result in candidates),
        }
    return {
        "dataset": DATASET, "contract_version": CONTRACT_VERSION, "profile": profile,
        "training_rows": len(training.x), "training_ids_sha256": _partition_digest(training.record_ids),
        "cv": {"n_splits": 5, "shuffle": True, "random_state": 42, "std_ddof": 0},
        "models": results, "elapsed_seconds": perf_counter() - started,
        "selection_policy": SELECTION_POLICY, "selected_model": select_model(results),
        "test_evaluated": False, "model_exported": False,
    }


def select_model(results: dict) -> str:
    if set(results) != set(MODEL_NAMES):
        raise ValueError("Selection requires CV results for all three algorithms")
    scores = {name: results[name]["summary"]["ap_mean"] for name in MODEL_NAMES}
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Mean AP must be finite")
    return max(scores, key=scores.__getitem__)


def refit_selected_on_training(
    training: TrainingData, selected_name: str, *, profile: str = "standard", candidate: int = 0,
) -> RefitModel:
    """Fit out-of-fold sigmoid calibration, then a fresh base pipeline on all Training."""
    _validate_training(training, min_per_class=CALIBRATION["n_splits"])
    pipe = _new_pipeline(selected_name, profile=profile, candidate=candidate)
    cv = StratifiedKFold(n_splits=CALIBRATION["n_splits"], shuffle=True, random_state=42)
    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=cv, ensemble=False, n_jobs=1)
    seconds, convergence = _fit(calibrated, training.x, training.y)
    return RefitModel(calibrated, selected_name, training.record_ids.copy(), profile, seconds, convergence)


def evaluate_on_test(fitted: RefitModel, test: HeldOutTest, *, threshold=CLASSIFICATION_THRESHOLD, curves=None) -> dict:
    """Evaluate the frozen classifier on its independent holdout."""
    if not isinstance(test, HeldOutTest):
        raise TypeError("Evaluation requires a separate HeldOutTest partition")
    if np.intersect1d(fitted.training_record_ids, test.record_ids).size:
        raise ValueError("Test IDs overlap the refit Training partition")
    probabilities = fitted.pipeline.predict_proba(test.x)[:, 1]
    labels = probabilities >= threshold
    if curves is not None:
        precision, recall, _ = precision_recall_curve(test.y, probabilities)
        fpr, tpr, _ = roc_curve(test.y, probabilities)
        curves.update(pr={"recall": recall.tolist(), "precision": precision.tolist()},
                      roc={"fpr": fpr.tolist(), "tpr": tpr.tolist()})
    return {
        "model": fitted.model_name, "rows": len(test.x),
        **probability_metrics(test.y, probabilities),
        "threshold": threshold, "accuracy": float(accuracy_score(test.y, labels)),
        "precision": float(precision_score(test.y, labels, zero_division=0)),
        "recall": float(recall_score(test.y, labels, zero_division=0)),
        "f1": float(f1_score(test.y, labels, zero_division=0)),
        "confusion_matrix": confusion_matrix(test.y, labels, labels=[0, 1]).tolist(),
    }


def resolve_csv(explicit, env_name: str, filename: str, data_dir=None) -> Path:
    """Explicit path > per-file env > data dir/env > repo/sibling discovery.

    An invalid explicit override fails; it never silently selects another file.
    Relative paths are interpreted relative to the process working directory.
    """
    override = explicit or os.environ.get(env_name)
    directory = data_dir or os.environ.get("HOME_CREDIT_DATA_DIR")
    if override:
        candidates = [Path(override)]
    elif directory:
        candidates = [Path(directory) / filename]
    else:
        repo = ROOT.parent
        candidates = [
            repo / "home-credit-default-risk" / filename,
            repo.parent / "home-credit-default-risk" / filename,
            ROOT / "data" / "home-credit-default-risk" / filename,
        ]
    candidates = [path.expanduser().resolve() for path in candidates]
    for path in candidates:
        if path.is_file():
            return path
    tried = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing {filename}. Tried:\n  {tried}\n"
        f"Pass a CSV path or set {env_name} / HOME_CREDIT_DATA_DIR. No simulated fallback."
    )


def load_home_credit(application_csv=None, description_csv=None, data_dir=None, *, full_audit=True):
    application_path = resolve_csv(application_csv, "HOME_CREDIT_APPLICATION_CSV", "application_train.csv", data_dir)
    # A supplied application path also locates its adjacent dictionary by default.
    dictionary_dir = data_dir
    if dictionary_dir is None and not os.environ.get("HOME_CREDIT_DATA_DIR"):
        dictionary_dir = application_path.parent
    description_path = resolve_csv(description_csv, "HOME_CREDIT_DESCRIPTION_CSV", "HomeCredit_columns_description.csv", dictionary_dir)
    if application_path.name.lower() != "application_train.csv":
        raise ValueError("This preparation step only accepts labeled application_train.csv")
    required = [ID_COLUMN, TARGET, *FEATURES, "AMT_INCOME_TOTAL"]
    try:
        frame = pd.read_csv(application_path, usecols=None if full_audit else required)
        # The official description file contains non-UTF8 bytes (e.g. accented words).
        descriptions = pd.read_csv(description_path, encoding="cp1252")
    except (ValueError, OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read Home Credit CSVs ({application_path}, {description_path}): {exc}") from exc
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"application_train.csv is missing required columns: {missing}")
    if frame.empty or frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("TARGET must be complete binary labels containing both 0 and 1")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].duplicated().any():
        raise ValueError("SK_ID_CURR must be nonmissing and unique; duplicate application IDs found")
    if not {"Table", "Row", "Description"}.issubset(descriptions.columns):
        raise ValueError("Description CSV must contain Table, Row and Description columns")
    dictionary = descriptions[descriptions["Table"].eq("application_{train|test}.csv")]
    absent = sorted(set(required) - set(dictionary["Row"]))
    if absent:
        raise ValueError(f"Description CSV lacks application definitions for: {absent}")
    validate_features(frame.loc[:, FEATURES])
    return frame, dictionary, application_path, description_path


def audit_dataset(frame, dictionary, application_path, description_path) -> dict:
    special = {}
    for name in frame.select_dtypes(include=["object", "string"]).columns:
        counts = {value: int(frame[name].eq(value).sum()) for value in SPECIAL_CATEGORIES}
        if any(counts.values()):
            special[name] = {key: count for key, count in counts.items() if count}
    definitions = dictionary.set_index("Row")["Description"].to_dict()
    missing = frame.isna().sum()
    return {
        "dataset": DATASET,
        "contract_version": CONTRACT_VERSION,
        "application_csv": str(application_path),
        "description_csv": str(description_path),
        "rows": len(frame), "columns": len(frame.columns),
        "target_meaning": TARGET_MEANING,
        "target_counts": {str(key): int(count) for key, count in frame[TARGET].value_counts().sort_index().items()},
        "target_rates": {str(key): float(count) for key, count in frame[TARGET].value_counts(normalize=True).sort_index().items()},
        "duplicate_ids": int(frame[ID_COLUMN].duplicated().sum()),
        "missing_ids": int(frame[ID_COLUMN].isna().sum()),
        "missing_cells": int(missing.sum()),
        "columns_with_missing": int(missing.gt(0).sum()),
        "missing_by_column": {name: int(value) for name, value in missing.items()},
        "dtypes": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "special_categories": special,
        "days_employed_365243": int(frame["DAYS_EMPLOYED"].eq(365243).sum()),
        "positive_relative_days": {
            name: int((frame[name] > 0).sum()) for name in frame if name.startswith("DAYS_")
        },
        "infinite_numeric_values": sum(int(np.isinf(frame[name]).sum()) for name in frame if pd.api.types.is_numeric_dtype(frame[name])),
        "nonpositive_amounts": {
            name: int((frame[name] <= 0).sum()) for name in ("AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY")
        },
        "numeric_features": NUMERIC, "categorical_features": CATEGORICAL,
        "dictionary": {name: definitions.get(name) for name in frame.columns},
        "status": "preparation_only_no_model_evaluation_or_export",
    }


def smoke_pipeline(frame: pd.DataFrame, sample_size: int = 256) -> dict:
    if sample_size < 10:
        raise ValueError("sample_size must be at least 10")
    # Stratified small sample; split BEFORE learning imputers/scaler/categories.
    if len(frame) > sample_size:
        sample, _ = train_test_split(frame, train_size=sample_size, stratify=frame[TARGET], random_state=42)
    else:
        sample = frame
    train, valid = train_test_split(sample, test_size=0.25, stratify=sample[TARGET], random_state=42)
    preprocessor = build_preprocessor()
    transformed_train = preprocessor.fit_transform(train.loc[:, FEATURES], train[TARGET])
    transformed_valid = preprocessor.transform(valid.loc[:, FEATURES])
    for values in (transformed_train, transformed_valid):
        if not np.isfinite(values.data if hasattr(values, "tocsr") else values).all():
            raise ValueError("Pipeline produced non-finite values")
    return {
        "sample_rows": len(sample), "fit_rows": len(train), "transform_rows": len(valid),
        "train_shape": list(transformed_train.shape), "validation_shape": list(transformed_valid.shape),
        "feature_names": preprocessor.get_feature_names_out().tolist(),
        "classifier_fitted": False, "model_exported": False,
    }


def train_version(frame, version: str, artifact_root=None, *, partitions=None, progress=None) -> dict:
    """Full run: lock CV choice/threshold, refit Training, evaluate Test ONCE.

    A new version directory is mandatory; existing versions are never overwritten.
    Training does not change the active serving pointer.
    """
    import joblib
    import sklearn
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", version):
        raise ValueError("Version must contain only letters, digits, dots, hyphens or underscores")
    destination = Path(artifact_root or get_settings().artifact_dir) / "versions" / version
    destination.mkdir(parents=True, exist_ok=False)
    training, test = partitions if partitions is not None else split_training_test(frame)
    cv = compare_on_training(training, progress=progress)
    selected = cv["selected_model"]
    best = cv["models"][selected]
    risk_review = {"status": "not_validated", "reason": "Business risk bands require separate outcome validation."}
    lock = {"cv": cv, "selected_model": selected, "classification_threshold": CLASSIFICATION_THRESHOLD,
            "threshold_policy": "fixed_0.5_before_test", "probability_calibration": CALIBRATION}
    (destination / "selection_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    message = f"Selection locked: {selected}; threshold=0.5. Refitting Training only."
    print(message, file=sys.stderr, flush=True)
    if progress:
        progress(message)
    fitted = refit_selected_on_training(training, selected, candidate=best["candidate"])
    binary = destination / "credit_pipeline.joblib"
    joblib.dump(fitted.pipeline, binary)
    if progress:
        progress("Evaluating the selected model on independent Test")
    (destination / "test_evaluation_started.txt").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
    curves = {}
    test_metrics = evaluate_on_test(fitted, test, curves=curves)
    (destination / "test_curves.json").write_text(json.dumps(curves, separators=(",", ":")), encoding="utf-8")
    print("Independent Test evaluated once.", file=sys.stderr, flush=True)
    sample = training.x.iloc[0].astype(object).where(training.x.iloc[0].notna(), None).to_dict()
    sample["amount_unit"] = "HOME_CREDIT_SOURCE"
    (destination / "sample_request.json").write_text(json.dumps(sample, indent=2), encoding="utf-8")
    metadata = {
        "version": version, "algorithm": selected, "training_dataset": DATASET,
        "contract_version": CONTRACT_VERSION, "features": FEATURES,
        "input_specification": FEATURE_SPEC, "amount_unit": "HOME_CREDIT_SOURCE",
        "target_meaning": TARGET_MEANING, "positive_class": 1,
        "n_rows": len(frame), "training_rows": len(training.y), "test_rows": len(test.y),
        "split": {"test_size": 0.2, "stratify": TARGET, "random_state": 42},
        "cv_results": cv, "selection_policy": SELECTION_POLICY,
        "classification_threshold": CLASSIFICATION_THRESHOLD, "threshold_policy": "fixed_0.5_before_test",
        "test_metrics": test_metrics, "test_evaluation_count": 1,
        "risk_bands": risk_review, "probability_calibrated": True, "probability_calibration": CALIBRATION,
        "parameters": best["parameters"],
        "refit": {"rows": len(training.y), "fit_seconds": fitted.fit_seconds,
                  "convergence_warnings": fitted.convergence_warnings,
                  "training_ids_sha256": _partition_digest(fitted.training_record_ids)},
        "test_ids_sha256": _partition_digest(test.record_ids),
        "curves_file": "test_curves.json",
        "credit_score_formula": "850 - 550 * payment_difficulty_probability (illustrative, not validated)",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "libraries": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    if selected == "DecisionTree":
        base = fitted.pipeline.calibrated_classifiers_[0].estimator
        metadata["feature_importance"] = dict(zip(
            base.named_steps["preprocess"].get_feature_names_out(),
            base.named_steps["model"].feature_importances_.tolist(),
        ))
    (destination / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "cv", "cv-smoke", "train"), default="prepare")
    parser.add_argument("--version", help="Required new version for --mode train; never activates automatically")
    parser.add_argument("--artifact-root", type=Path, default=Path(get_settings().artifact_dir))
    parser.add_argument("--application-csv")
    parser.add_argument("--description-csv")
    parser.add_argument("--data-dir")
    parser.add_argument("--sample-size", type=int, default=256, help="Preparation/CV-smoke sample size; real CV uses ALL Training")
    parser.add_argument("--report", type=Path, help="Optional preparation/CV/final-results JSON outside model artifact directories")
    parser.add_argument("--quiet", action="store_true", help="Suppress final JSON on stdout")
    args = parser.parse_args()
    try:
        frame, dictionary, application_path, description_path = load_home_credit(
            args.application_csv, args.description_csv, args.data_dir, full_audit=args.mode == "prepare",
        )
        training, held_out = split_training_test(frame)
        if args.mode == "train":
            if not args.version:
                raise ValueError("--mode train requires a new --version")
            report = train_version(frame, args.version, args.artifact_root, partitions=(training, held_out))
        elif args.mode == "cv":
            report = compare_on_training(training)
        elif args.mode == "cv-smoke":
            sample, _ = train_test_split(
                np.arange(len(training.y)), train_size=args.sample_size,
                stratify=training.y, random_state=42,
            )
            small = TrainingData(training.x.iloc[sample], training.y.iloc[sample], training.record_ids[sample])
            report = compare_on_training(small, profile="smoke")
            report["dataset"] = DATASET + " (Training subset smoke check)"
        else:
            report = audit_dataset(frame, dictionary, application_path, description_path)
            report["pipeline_smoke"] = smoke_pipeline(training.x.assign(**{TARGET: training.y}), args.sample_size)
        report["split"] = {
            **report.get("split", {}), "training_rows": len(training.x), "held_out_rows": len(held_out.y),
            "test_size": 0.2, "stratify": TARGET, "random_state": 42,
        }
        report["test_evaluated"] = args.mode == "train"
        if args.report:
            # Do not let a preparation report overwrite the active model or metadata.
            report_path = args.report.resolve()
            if "artifacts" in report_path.parts or report_path.suffix.lower() != ".json":
                raise ValueError("Write audit reports to a .json outside artifacts directories")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary = {key: value for key, value in report.items() if key not in {"dictionary", "missing_by_column"}}
        if not args.quiet:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Home Credit run failed: {exc}\n")


if __name__ == "__main__":
    main()
