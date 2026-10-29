"""Data loading, target handling and evaluation shared by training and the app."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

TARGET_COLUMN = "diagnosis"
ID_COLUMN = "id"
POSITIVE_LABEL = "Malignant"
NEGATIVE_LABEL = "Benign"
CLASS_ORDER = [NEGATIVE_LABEL, POSITIVE_LABEL]
MIN_FEATURES = 12
METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}
_TARGET_ALIASES = {
    "M": POSITIVE_LABEL,
    "MALIGNANT": POSITIVE_LABEL,
    "1": POSITIVE_LABEL,
    "B": NEGATIVE_LABEL,
    "BENIGN": NEGATIVE_LABEL,
    "0": NEGATIVE_LABEL,
}


def load_csv(source: Any) -> pd.DataFrame:
    """Read a CSV and drop the empty trailing column that the Kaggle export contains."""
    frame = pd.read_csv(source)
    frame.columns = [str(column).strip() for column in frame.columns]
    blank = [c for c in frame.columns if c.startswith("Unnamed") and frame[c].isna().all()]
    return frame.drop(columns=blank)


def _normalise_label(value: Any) -> Any:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    if text.endswith(".0"):
        text = text[:-2]
    return _TARGET_ALIASES.get(text, text)


def map_target(values: pd.Series) -> pd.Series:
    """Map M/B (also Malignant/Benign or 1/0) to class labels; blank values stay NaN."""
    mapped = values.map(_normalise_label)
    unknown = mapped.notna() & ~mapped.isin(CLASS_ORDER)
    if unknown.any():
        examples = ", ".join(sorted({str(v) for v in values[unknown]})[:5])
        raise ValueError(
            f"Unrecognised '{TARGET_COLUMN}' values: {examples}. Expected M (malignant) or B (benign)."
        )
    return mapped


def feature_frame(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Select the model inputs in training order and coerce them to numbers."""
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing required input columns: " + ", ".join(missing))
    return frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce")


def prepare_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return numeric features, class labels, and the matching raw rows."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"The data must contain a '{TARGET_COLUMN}' column.")
    columns = [c for c in frame.columns if c not in (TARGET_COLUMN, ID_COLUMN)]
    if len(columns) < MIN_FEATURES:
        raise ValueError(f"At least {MIN_FEATURES} input features are required; found {len(columns)}.")

    target = map_target(frame[TARGET_COLUMN])
    keep = target.notna().to_numpy()
    raw = frame.loc[keep].reset_index(drop=True)
    features = feature_frame(raw, columns)
    empty = [c for c in columns if features[c].isna().all()]
    if empty:
        raise ValueError("Columns without numeric values: " + ", ".join(empty))
    return features, target.loc[keep].reset_index(drop=True), raw


def malignant_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    classes = list(estimator.classes_)
    if sorted(classes) != sorted(CLASS_ORDER):
        raise ValueError("The saved model was trained on different class labels.")
    return estimator.predict_proba(features)[:, classes.index(POSITIVE_LABEL)]


def score_metrics(actual: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Six assignment metrics plus confusion matrix, report and ROC curve."""
    predicted = np.where(probability >= threshold, POSITIVE_LABEL, NEGATIVE_LABEL)
    is_positive = (actual == POSITIVE_LABEL).astype(int)
    fpr, tpr, _ = roc_curve(is_positive, probability)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "auc": float(roc_auc_score(is_positive, probability)),
        "precision": float(precision_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
        "recall": float(recall_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
        "f1": float(f1_score(actual, predicted, pos_label=POSITIVE_LABEL, zero_division=0)),
        "mcc": float(matthews_corrcoef(actual, predicted)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=CLASS_ORDER).tolist(),
        "classification_report": classification_report(
            actual, predicted, labels=CLASS_ORDER, output_dict=True, zero_division=0
        ),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "predicted": predicted.tolist(),
        "probability": probability.tolist(),
        "threshold": float(threshold),
    }


def evaluate_frame(
    estimator: Any,
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score a saved pipeline on labelled raw rows. Never refits the model."""
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"The CSV must include the target column '{TARGET_COLUMN}' (M/B).")
    target = map_target(frame[TARGET_COLUMN])
    keep = target.notna().to_numpy()
    actual = target[keep].to_numpy()
    if len(set(actual)) < 2:
        raise ValueError("The test data must contain both malignant (M) and benign (B) rows.")

    features = feature_frame(frame.loc[keep], feature_columns)
    result = score_metrics(actual, malignant_probability(estimator, features), threshold)
    result["actual"] = actual.tolist()
    result["row_index"] = np.flatnonzero(keep).tolist()
    result["n_rows"] = int(keep.sum())
    result["class_counts"] = {label: int((actual == label).sum()) for label in CLASS_ORDER}
    return result
