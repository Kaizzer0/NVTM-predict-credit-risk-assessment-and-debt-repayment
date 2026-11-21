from __future__ import annotations

import hashlib
import json
from pathlib import Path
from threading import Lock

import joblib
import numpy as np

from app.config import get_settings
from app.home_credit_features import (
    CONTRACT_VERSION, DATASET, FEATURES, FEATURE_SPEC, TARGET_MEANING, frame_from_values,
)
from app.schemas import PredictRequest, PredictResponse

_LOCK = Lock()
_CACHE = None
# Used only while the isolated activation process verifies its candidate API.
_VERIFICATION_BUNDLE = None


class ModelUnavailable(RuntimeError):
    pass


def load_bundle(directory: Path):
    directory = Path(directory)
    info = json.loads((directory / "model_metadata.json").read_text(encoding="utf-8"))
    binary = directory / "credit_pipeline.joblib"
    if info.get("contract_version") != CONTRACT_VERSION or info.get("features") != FEATURES:
        raise ValueError("Model input contract does not match the Home Credit API")
    if info.get("input_specification") != FEATURE_SPEC or info.get("amount_unit") != "HOME_CREDIT_SOURCE":
        raise ValueError("Model input types/units/missing-value contract does not match the API")
    if info.get("training_dataset") != DATASET or info.get("positive_class") != 1:
        raise ValueError("Model source or positive class does not match Home Credit")
    if info.get("algorithm") not in {"DecisionTree", "LinearSVM", "ANN"} or not info.get("probability_calibrated"):
        raise ValueError("Train and activate a calibrated DecisionTree, LinearSVM or ANN artifact")
    if hashlib.sha256(binary.read_bytes()).hexdigest() != info.get("pipeline_sha256"):
        raise ValueError("Pipeline checksum mismatch")
    pipeline = joblib.load(binary)
    if list(pipeline.feature_names_in_) != FEATURES or list(pipeline.classes_) != [0, 1]:
        raise ValueError("Saved pipeline features/classes do not match metadata")
    return pipeline, info


def _active_bundle():
    global _CACHE
    if _VERIFICATION_BUNDLE is not None:
        return _VERIFICATION_BUNDLE
    root = Path(get_settings().artifact_dir).resolve()
    try:
        with _LOCK:
            raw = (root / "active_model.json").read_bytes()
            key = (root, raw)
            if _CACHE is None or _CACHE[0] != key:
                active = json.loads(raw)
                directory = (root / active["relative_dir"]).resolve()
                if not directory.is_relative_to(root / "versions"):
                    raise ValueError("Active model must be inside artifacts/versions")
                pipeline, info = load_bundle(directory)
                if info["version"] != active["version"]:
                    raise ValueError("Active model version mismatch")
                _CACHE = (key, (pipeline, info, directory))
            return _CACHE[1]
    except (OSError, ValueError, KeyError, ImportError) as exc:
        raise ModelUnavailable("No usable trained Home Credit model. Train and activate a new version.") from exc


def request_to_frame(payload: PredictRequest):
    return frame_from_values(payload.model_dump())


def predict(payload: PredictRequest) -> PredictResponse:
    pipeline, info, _ = _active_bundle()
    frame = request_to_frame(payload)
    probability = float(pipeline.predict_proba(frame)[0, 1])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability")
    cleaned = pipeline.calibrated_classifiers_[0].estimator.named_steps["preprocess"].named_steps["clean"].transform(frame)
    missing = cleaned.columns[cleaned.isna().iloc[0]].tolist()
    return PredictResponse(
        payment_difficulty_probability=probability,
        other_outcome_probability=1 - probability,
        predicted_payment_difficulty=probability >= info["classification_threshold"],
        classification_threshold=info["classification_threshold"],
        # Legacy risk-band defaults are deliberately not applied to Home Credit.
        risk_level="NOT_VALIDATED", risk_bands_validated=False,
        credit_score=round(max(300, min(850, 850 - 550 * probability)), 2),
        model_version=info["version"], algorithm=info["algorithm"],
        data_source=info["training_dataset"], outcome_definition=TARGET_MEANING["1"],
        imputed_features=missing, probability_calibrated=info["probability_calibrated"],
    )


def metadata() -> dict:
    return _active_bundle()[1]


def input_contract() -> dict:
    pipeline, _, _ = _active_bundle()
    specs = {name: dict(spec) for name, spec in FEATURE_SPEC.items()}
    encoder = pipeline.calibrated_classifiers_[0].estimator.named_steps["preprocess"].named_steps["columns"].named_transformers_["cat"].named_steps["encoder"]
    specs["OCCUPATION_TYPE"]["choices"] = [value for value in encoder.categories_[1].tolist() if value != "__MISSING__"]
    return {
        "contract_version": CONTRACT_VERSION, "features": specs,
        "amount_unit": "HOME_CREDIT_SOURCE", "json_schema": PredictRequest.model_json_schema(),
        "missing_policy": "Every key required. Explicit null only for nullable fields; fitted Pipeline imputes it.",
    }


def sample_request() -> dict:
    _, _, directory = _active_bundle()
    sample = json.loads((directory / "sample_request.json").read_text(encoding="utf-8"))
    return PredictRequest.model_validate(sample).model_dump()


def test_curves() -> dict:
    _, info, directory = _active_bundle()
    data = json.loads((directory / "test_curves.json").read_text(encoding="utf-8"))
    display = {}
    for name, curve in data.items():
        length = len(next(iter(curve.values())))
        indices = np.linspace(0, length - 1, min(length, 400), dtype=int)
        display[name] = {axis: [values[i] for i in indices] for axis, values in curve.items()}
    return {"version": info["version"], **display}
