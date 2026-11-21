"""Use a subset of the official Training partition for fast integration checks."""
import json
import sys

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold

from app.home_credit_features import FEATURES
from training import train


@pytest.fixture(scope="module")
def cv_run(real_sample):
    frame = real_sample.copy()
    training, test = train.split_training_test(frame)
    created = []
    original = train._new_pipeline

    original_fit = train._fit

    def track(name, *, profile, candidate):
        pipeline = original(name, profile=profile, candidate=candidate)
        created.append((name, pipeline))
        return pipeline

    def track_fit(pipeline, x, y):
        fit_indices[id(pipeline)] = y.index.to_numpy(copy=True)
        return original_fit(pipeline, x, y)

    fit_indices = {}

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "_new_pipeline", track)
        patch.setattr(train, "_fit", track_fit)
        report = train.compare_on_training(training, profile="smoke")
    return frame, training, test, report, [(name, fit_indices[id(pipe)], pipe) for name, pipe in created]


def test_single_reproducible_stratified_80_20_split(monkeypatch, real_sample):
    frame = real_sample
    original = train.train_test_split
    calls = []

    def track(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(train, "train_test_split", track)
    training, test = train.split_training_test(frame)
    assert len(calls) == 1
    assert calls[0]["test_size"] == 0.2 and calls[0]["random_state"] == 42
    assert calls[0]["stratify"].equals(frame["TARGET"])
    assert len(training.x) == 200 and len(test.x) == 50
    assert training.y.mean() == pytest.approx(frame.TARGET.mean(), abs=1 / len(training.y))
    assert test.y.mean() == pytest.approx(frame.TARGET.mean(), abs=1 / len(test.y))
    assert not set(training.record_ids) & set(test.record_ids)
    assert set(training.record_ids) | set(test.record_ids) == set(frame["SK_ID_CURR"])
    again, test_again = train.split_training_test(frame)
    np.testing.assert_array_equal(training.record_ids, again.record_ids)
    np.testing.assert_array_equal(test.record_ids, test_again.record_ids)
    assert list(training.x.columns) == FEATURES


def test_three_models_each_run_five_identical_folds(cv_run):
    _, training, test, report, created = cv_run
    assert len(created) == 15
    assert set(report["models"]) == set(train.MODEL_NAMES)
    assert report["selected_model"] == train.select_model(report["models"])
    assert report["selection_policy"] == train.SELECTION_POLICY
    assert report["test_evaluated"] is False
    expected = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(training.x, training.y))
    reference_fingerprints = None
    for name, result in report["models"].items():
        assert len(result["folds"]) == 5
        seen_validation_ids = []
        fingerprints = []
        for fold, (fit_pos, valid_pos) in zip(result["folds"], expected):
            fit_ids, valid_ids = training.record_ids[fit_pos], training.record_ids[valid_pos]
            assert not set(fit_ids) & set(valid_ids)
            assert not (set(fit_ids) | set(valid_ids)) & set(test.record_ids)
            assert fold["fit_ids_sha256"] == train._partition_digest(fit_ids)
            assert fold["validation_ids_sha256"] == train._partition_digest(valid_ids)
            fingerprints.append((fold["fit_ids_sha256"], fold["validation_ids_sha256"]))
            seen_validation_ids.extend(valid_ids)
            assert fold["fit_rows"] == 160 and fold["validation_rows"] == 40
            pipeline = next(p for n, indices, p in created if n == name and np.array_equal(indices, training.y.iloc[fit_pos].index))
            x_valid = training.x.iloc[valid_pos]
            if name == "LinearSVM":
                scores = pipeline.decision_function(x_valid)
                assert result["score_type"] == "decision_function"
            else:
                scores = pipeline.predict_proba(x_valid)[:, 1]
            from sklearn.metrics import average_precision_score
            assert fold["ap"] == pytest.approx(average_precision_score(training.y.iloc[valid_pos], scores))
        assert sorted(seen_validation_ids) == sorted(training.record_ids)
        if reference_fingerprints is None:
            reference_fingerprints = fingerprints
        assert fingerprints == reference_fingerprints
        for metric in ("ap", "roc_auc"):
            scores = [fold[metric] for fold in result["folds"]]
            assert result["summary"][f"{metric}_mean"] == pytest.approx(np.mean(scores))
            assert result["summary"][f"{metric}_std"] == pytest.approx(np.std(scores, ddof=0))
        assert "selection_score" not in result


def test_every_pipeline_and_preprocessor_is_new_and_fit_only_on_four_folds(cv_run):
    _, training, test, _, created = cv_run
    pipelines = [pipeline for _, _, pipeline in created]
    assert len({id(pipe) for pipe in pipelines}) == 15
    assert len({id(pipe.named_steps["preprocess"]) for pipe in pipelines}) == 15
    assert len({id(pipe.named_steps["model"]) for pipe in pipelines}) == 15
    for name, fit_indices, pipeline in created:
        assert not set(fit_indices) & set(test.x.index)
        assert len(fit_indices) == 160
        expected_fit = training.x.loc[fit_indices]
        pre = pipeline.named_steps["preprocess"]
        numeric = pre.named_steps["columns"].named_transformers_["num"]
        cleaned = pre.named_steps["clean"].transform(expected_fit)
        np.testing.assert_allclose(
            numeric.named_steps["imputer"].statistics_, cleaned[train.NUMERIC].median().to_numpy(),
        )
        if name == "DecisionTree":
            assert numeric.named_steps["scaler"] == "passthrough"
        else:
            assert numeric.named_steps["scaler"].n_samples_seen_ == 160
        encoder = pre.named_steps["columns"].named_transformers_["cat"].named_steps["encoder"]
        for field, categories in zip(train.CATEGORICAL, encoder.categories_):
            assert set(categories) == set(cleaned[field].fillna("__MISSING__"))


def test_test_partition_rejected_by_comparison_and_refit(cv_run):
    _, _, test, _, _ = cv_run
    with pytest.raises(TypeError, match="outside model selection"):
        train.compare_on_training(test)
    with pytest.raises(TypeError, match="outside model selection"):
        train.refit_selected_on_training(test, "DecisionTree")


def test_changing_only_test_cannot_change_cv(cv_run):
    _, training, test, original, _ = cv_run
    test = train.HeldOutTest(test.x.copy(), test.y.copy(), test.record_ids)
    # Test-only corruption/categories/extremes never reach any comparison call.
    test.x.loc[:, "AMT_CREDIT"] = 1e12
    test.x.loc[:, "OCCUPATION_TYPE"] = "TEST_ONLY_CATEGORY"
    test.y.loc[:] = 1 - test.y
    repeated = train.compare_on_training(training, profile="smoke")
    assert {n: m["summary"] for n, m in repeated["models"].items()} == {n: m["summary"] for n, m in original["models"].items()}


@pytest.mark.parametrize("name", train.MODEL_NAMES)
def test_refit_and_test_evaluation_are_separate_stages(monkeypatch, real_sample, name):
    training, test = train.split_training_test(real_sample)
    calls = []
    original_fit = train.Pipeline.fit

    def track_fit(self, x, y, **kwargs):
        calls.append(len(x))
        return original_fit(self, x, y, **kwargs)

    monkeypatch.setattr(train.Pipeline, "fit", track_fit)
    # A fixed test fixture choice, NOT an official model selection policy.
    fitted = train.refit_selected_on_training(training, name, profile="smoke")
    assert sorted(calls) == [133, 133, 134, 200]
    np.testing.assert_array_equal(fitted.training_record_ids, training.record_ids)
    probabilities = fitted.pipeline.predict_proba(test.x)
    assert np.all((0 <= probabilities) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    before = len(calls)
    metrics = train.evaluate_on_test(fitted, test)
    assert len(calls) == before  # evaluate only transforms/predicts; no fit.
    assert metrics["rows"] == 50
    assert set(metrics) == {"model", "rows", "ap", "roc_auc", "threshold", "precision", "recall", "f1", "confusion_matrix", "accuracy"}
    assert metrics["threshold"] == 0.5
    with pytest.raises(ValueError, match="overlap"):
        train.evaluate_on_test(fitted, train.HeldOutTest(training.x, training.y, training.record_ids))


@pytest.mark.parametrize("mode", ["cv", "cv-smoke"])
def test_cv_cli_never_evaluates_or_exports(monkeypatch, real_sample, mode):
    from pathlib import Path

    seen = []
    monkeypatch.setattr(train, "load_home_credit", lambda *a, **kw: (real_sample, None, Path("application_train.csv"), Path("dictionary.csv")))

    def compare(training, **kwargs):
        assert isinstance(training, train.TrainingData)
        seen.append(training.record_ids)
        return {"test_evaluated": False}

    def forbidden(*args, **kwargs):
        pytest.fail("CV must not refit, evaluate Test or export")

    monkeypatch.setattr(train, "compare_on_training", compare)
    for name in ("refit_selected_on_training", "evaluate_on_test", "train_version"):
        monkeypatch.setattr(train, name, forbidden)
    monkeypatch.setattr(sys, "argv", ["train.py", "--mode", mode, "--sample-size", "150"])
    train.main()
    _, held_out = train.split_training_test(real_sample)
    assert len(seen) == 1 and len(seen[0]) == (150 if mode == "cv-smoke" else 200)
    assert not set(seen[0]) & set(held_out.record_ids)
