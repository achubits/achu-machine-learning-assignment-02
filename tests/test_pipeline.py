"""Offline checks on synthetic data (the real dataset is not needed). Run: pytest -q"""

import numpy as np
import pandas as pd
import pytest

from ml_utils import NEGATIVE_LABEL, POSITIVE_LABEL, evaluate_frame, load_csv, map_target, prepare_dataset
from train_models import model_catalogue, tune

FEATURES = [f"feature_{i}" for i in range(14)]


def synthetic_frame(rows: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    malignant = rng.random(rows) < 0.4
    values = np.abs(rng.normal(size=(rows, len(FEATURES))) + malignant[:, None] * 1.5)
    frame = pd.DataFrame(values, columns=FEATURES)
    frame.insert(0, "id", np.arange(rows))
    frame.insert(1, "diagnosis", np.where(malignant, "M", "B"))
    return frame


def test_target_aliases():
    mapped = map_target(pd.Series(["M", "b", " Malignant ", "0", 1, np.nan]))
    expected = [POSITIVE_LABEL, NEGATIVE_LABEL, POSITIVE_LABEL, NEGATIVE_LABEL, POSITIVE_LABEL]
    assert list(mapped.dropna()) == expected
    with pytest.raises(ValueError):
        map_target(pd.Series(["X"]))


def test_load_csv_drops_blank_trailing_column(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text('"id","diagnosis","a",\n1,M,2.5\n2,B,1.5\n')
    assert list(load_csv(path).columns) == ["id", "diagnosis", "a"]


def test_minimum_feature_count_enforced():
    with pytest.raises(ValueError, match="At least 12"):
        prepare_dataset(synthetic_frame()[["id", "diagnosis", *FEATURES[:5]]])


def test_every_model_tunes_and_scores():
    frame = synthetic_frame()
    features, target, raw = prepare_dataset(frame)
    test_rows = raw.iloc[120:]
    for name, spec in model_catalogue(0).items():
        model, info = tune(spec, features.iloc[:120], target.iloc[:120].to_numpy(), folds=3, seed=0)
        result = evaluate_frame(model, test_rows, list(features.columns))
        for key in ("accuracy", "auc", "precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0, (name, key)
        assert -1.0 <= result["mcc"] <= 1.0
        assert sum(map(sum, result["confusion_matrix"])) == len(test_rows)
        assert {"best_params", "cv_mcc_mean", "cv_mcc_std"} <= set(info)


def test_lower_threshold_never_reduces_recall():
    frame = synthetic_frame()
    features, target, raw = prepare_dataset(frame)
    model, _ = tune(model_catalogue(0)["Logistic Regression"], features, target.to_numpy(), folds=3, seed=0)
    strict = evaluate_frame(model, raw, list(features.columns), threshold=0.8)
    loose = evaluate_frame(model, raw, list(features.columns), threshold=0.2)
    assert loose["recall"] >= strict["recall"]
    assert loose["auc"] == pytest.approx(strict["auc"])


def test_missing_columns_and_single_class_are_reported():
    frame = synthetic_frame()
    features, target, raw = prepare_dataset(frame)
    model, _ = tune(model_catalogue(0)["Naive Bayes"], features, target.to_numpy(), folds=3, seed=0)
    with pytest.raises(ValueError, match="Missing required input columns"):
        evaluate_frame(model, raw.drop(columns=["feature_3"]), list(features.columns))
    with pytest.raises(ValueError, match="both malignant"):
        evaluate_frame(model, raw[raw["diagnosis"] == "M"], list(features.columns))
