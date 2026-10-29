"""Tune, train and evaluate five classifiers on the Breast Cancer Wisconsin (Diagnostic) data.

Usage:
    python train_models.py                   # reads breast_cancer.csv next to this file
    python train_models.py --data other.csv  # any CSV with the same columns
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from ml_utils import (
    CLASS_ORDER,
    ID_COLUMN,
    METRIC_LABELS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    evaluate_frame,
    load_csv,
    prepare_dataset,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
REPORT_DIR = ROOT / "reports"
TEST_DATA_PATH = ROOT / "test_data.csv"
README_PATH = ROOT / "README.md"
RESULTS_START = "<!-- MODEL_RESULTS_START -->"
RESULTS_END = "<!-- MODEL_RESULTS_END -->"
DEFAULT_SEED = 711

MODEL_INSIGHTS = {
    "Logistic Regression": (
        "On this split, Logistic Regression made one false negative and one false positive. Its "
        "linear boundary appears to capture most of the separation, while L2 regularisation helps "
        "manage the strong redundancy among radius, perimeter and area measurements."
    ),
    "Decision Tree": (
        "The tree is the easiest of these models to explain as a sequence of measurement-based "
        "splits. It made two false negatives and two false positives here, and its cross-validation "
        "spread was the widest, suggesting its result is more sensitive to the particular training split."
    ),
    "KNN": (
        "KNN made the same hard predictions as Logistic Regression on this test set: both missed one "
        "malignant tumour and raised one false alarm. Its AUC was slightly lower, and distance-based "
        "predictions depend on standardisation and on the many correlated size measurements."
    ),
    "Naive Bayes": (
        "Naive Bayes had the lowest MCC on this hold-out set, with two missed malignant tumours and "
        "three false alarms. Its AUC remained high, so its probability ranking was stronger than its "
        "0.5-threshold classifications. Correlated measurements challenge its conditional-independence "
        "assumption."
    ),
    "Support Vector Machine": (
        "The RBF SVM had the highest AUC on the test set and the strongest mean cross-validated MCC, "
        "but its test MCC was below Logistic Regression and KNN. This is a useful reminder that excellent "
        "ranking does not guarantee the best classifications at a fixed 0.5 threshold."
    ),
}


def model_catalogue(seed: int) -> dict[str, dict[str, Any]]:
    return {
        "Logistic Regression": {
            "file": "logistic_regression.joblib",
            "estimator": LogisticRegression(max_iter=5000),
            "grid": {"model__C": [0.01, 0.1, 1, 10, 100]},
        },
        "Decision Tree": {
            "file": "decision_tree.joblib",
            "estimator": DecisionTreeClassifier(random_state=seed),
            "grid": {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [3, 4, 5, 6, None],
                "model__min_samples_leaf": [1, 3, 5, 10],
            },
        },
        "KNN": {
            "file": "knn.joblib",
            "estimator": KNeighborsClassifier(),
            "grid": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15],
                "model__weights": ["uniform", "distance"],
            },
        },
        "Naive Bayes": {
            "file": "naive_bayes.joblib",
            "estimator": GaussianNB(),
            "grid": {"model__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]},
        },
        "Support Vector Machine": {
            "file": "svm.joblib",
            "estimator": SVC(kernel="rbf", probability=True, random_state=seed),
            "grid": {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.001, 0.01, 0.1]},
        },
    }


def build_pipeline(estimator: Any) -> Pipeline:
    # The imputer only matters for uploaded files with blanks; the source data has none.
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def tune(
    spec: dict[str, Any], features: pd.DataFrame, target: np.ndarray, folds: int, seed: int
) -> tuple[Pipeline, dict[str, Any]]:
    """Grid-search one model on training data only (scored by MCC) and refit the best."""
    search = GridSearchCV(
        build_pipeline(clone(spec["estimator"])),
        spec["grid"],
        scoring=make_scorer(matthews_corrcoef),
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    search.fit(features, target)
    best = search.best_index_
    return search.best_estimator_, {
        "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
        "cv_mcc_mean": float(search.cv_results_["mean_test_score"][best]),
        "cv_mcc_std": float(search.cv_results_["std_test_score"][best]),
    }


def fmt_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in params.items()) or "defaults"


def build_report(results: dict[str, dict], tuning: dict[str, dict], info: dict[str, Any]) -> str:
    names = list(results)
    mcc = {n: round(results[n]["mcc"], 4) for n in names}
    rank = {n: 1 + sum(other > mcc[n] for other in mcc.values()) for n in names}
    winner = max(
        names,
        key=lambda n: (round(results[n]["mcc"], 4), round(results[n]["auc"], 4), tuning[n]["cv_mcc_mean"]),
    )

    lines = [
        f"Hold-out test set (`test_data.csv`): **{info['test_rows']}** tumours "
        f"({info['test_malignant']} malignant, {info['test_benign']} benign), never used for training or "
        f"tuning. Models were trained on **{info['train_rows']}** tumours, with hyperparameters chosen by "
        f"{info['cv_folds']}-fold stratified cross-validation on the training set (scoring: MCC). "
        "Precision, recall and F1 treat **Malignant** as the positive class; the decision threshold is 0.5.",
        "",
        "### Comparison table (test data)",
        "",
        "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        lines.append(f"| {name} | " + " | ".join(f"{results[name][k]:.4f}" for k in METRIC_LABELS) + " |")

    lines += [
        "",
        "### Tuned hyperparameters",
        "",
        "| ML Model Name | Selected hyperparameters | Cross-validated MCC (mean ± sd) |",
        "|---|---|---:|",
    ]
    for name in names:
        t = tuning[name]
        lines.append(
            f"| {name} | {fmt_params(t['best_params'])} | {t['cv_mcc_mean']:.4f} ± {t['cv_mcc_std']:.4f} |"
        )

    lines += [
        "",
        "### Observations",
        "",
        "| ML Model Name | Observation about model performance |",
        "|---|---|",
    ]
    for name in names:
        r = results[name]
        (_, fp), (fn, tp) = r["confusion_matrix"]
        alarms = "false alarm" if fp == 1 else "false alarms"
        lines.append(
            f"| {name} | {MODEL_INSIGHTS[name]} On the test set: accuracy {r['accuracy']:.3f}, "
            f"AUC {r['auc']:.3f}, MCC {r['mcc']:.3f} (rank {rank[name]} of {len(names)} by MCC); "
            f"it missed {fn} of {fn + tp} malignant tumours and raised {fp} {alarms}. |"
        )

    best = results[winner]
    tied = [n for n in names if n != winner and mcc[n] == mcc[winner]]
    close = [n for n in names if n not in (winner, *tied) and mcc[winner] - mcc[n] < 0.01]
    summary = (
        f"**Overall winner: {winner}** - highest test MCC ({best['mcc']:.4f}) with accuracy "
        f"{best['accuracy']:.4f}, AUC {best['auc']:.4f} and F1 {best['f1']:.4f} "
        f"(cross-validated MCC {tuning[winner]['cv_mcc_mean']:.4f})."
    )
    if tied:
        summary += (
            f" {' and '.join(tied)} {'ties' if len(tied) == 1 else 'tie'} exactly on test MCC; "
            f"{winner} is placed first on the tie-break (higher AUC, then cross-validated MCC)."
        )
    if close:
        verb = "is" if len(close) == 1 else "are"
        summary += (
            f" {', '.join(close)} {verb} within 0.01 MCC on this {info['test_rows']}-row test set - "
            "a difference of about one tumour - so treat them as a near tie."
        )
    cv_best = max(names, key=lambda n: tuning[n]["cv_mcc_mean"])
    if cv_best != winner:
        summary += (
            f" Note that {cv_best} has the highest cross-validated MCC "
            f"({tuning[cv_best]['cv_mcc_mean']:.4f}, averaged over {info['cv_folds']} folds of the training data)"
        )
        auc_best = max(names, key=lambda n: results[n]["auc"])
        if auc_best == cv_best:
            summary += f" and the highest test AUC ({results[cv_best]['auc']:.4f})"
        summary += (
            f", ranking {rank[cv_best]} on test MCC. With only {info['test_rows']} test tumours one "
            "misclassification moves MCC by about 0.02, so it is an equally strong choice."
        )
    summary += (
        " MCC is used for the ranking because it uses all four cells of the confusion matrix and is not "
        "flattered by the 63/37 class imbalance."
    )
    lines += ["", summary]
    return "\n".join(lines)


def update_readme(block: str) -> None:
    if not README_PATH.exists():
        return
    content = README_PATH.read_text(encoding="utf-8")
    if RESULTS_START not in content or RESULTS_END not in content:
        raise SystemExit("README.md is missing the MODEL_RESULTS marker comments.")
    before, rest = content.split(RESULTS_START, maxsplit=1)
    _, after = rest.split(RESULTS_END, maxsplit=1)
    README_PATH.write_text(f"{before}{RESULTS_START}\n\n{block}\n\n{RESULTS_END}{after}", encoding="utf-8")


def to_json(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "breast_cancer.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()
    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test-size must be between 0.1 and 0.4.")
    if args.cv_folds < 3:
        parser.error("--cv-folds must be at least 3.")
    return args


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Dataset not found: {args.data}")

    features, target, raw = prepare_dataset(load_csv(args.data))
    if set(target) != set(CLASS_ORDER):
        raise SystemExit("The data must contain both malignant and benign rows.")
    feature_columns = list(features.columns)
    print(f"Loaded {len(target)} rows, {len(feature_columns)} features from {args.data.name}")

    train_idx, test_idx = train_test_split(
        np.arange(len(target)), test_size=args.test_size, stratify=target, random_state=args.seed
    )
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)

    # Written first and re-read so the reported metrics match exactly what the app computes.
    raw.iloc[test_idx].to_csv(TEST_DATA_PATH, index=False)
    test_frame = load_csv(TEST_DATA_PATH)
    train_features = features.iloc[train_idx]
    train_target = target.iloc[train_idx].to_numpy()

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    results: dict[str, dict] = {}
    tuning: dict[str, dict] = {}
    registry_models: dict[str, dict] = {}
    for name, spec in model_catalogue(args.seed).items():
        print(f"Tuning {name} ...", flush=True)
        model, info = tune(spec, train_features, train_target, args.cv_folds, args.seed)
        evaluation = evaluate_frame(model, test_frame, feature_columns)
        keep = (*METRIC_LABELS, "confusion_matrix", "classification_report", "roc_curve")
        results[name] = {key: evaluation[key] for key in keep}
        tuning[name] = info
        joblib.dump(model, MODEL_DIR / spec["file"], compress=3)
        registry_models[name] = {"file": spec["file"], **info}
        print(
            f"  best: {fmt_params(info['best_params'])} | CV MCC {info['cv_mcc_mean']:.4f} "
            f"| test MCC {evaluation['mcc']:.4f}, AUC {evaluation['auc']:.4f}"
        )

    test_target = target.iloc[test_idx]
    dataset_info = {
        "source_file": args.data.name,
        "total_rows": int(len(target)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "test_malignant": int((test_target == POSITIVE_LABEL).sum()),
        "test_benign": int((test_target == NEGATIVE_LABEL).sum()),
        "cv_folds": args.cv_folds,
        "random_seed": args.seed,
        "test_size": args.test_size,
    }
    registry = {
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "feature_columns": feature_columns,
        "labels": {"positive": POSITIVE_LABEL, "negative": NEGATIVE_LABEL},
        "decision_threshold": 0.5,
        "models": registry_models,
        "dataset": dataset_info,
        "library_versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (MODEL_DIR / "model_registry.json").write_text(json.dumps(registry, indent=2, default=to_json), encoding="utf-8")
    (REPORT_DIR / "model_metrics.json").write_text(
        json.dumps({"results": results, "tuning": tuning, "dataset": dataset_info}, indent=2, default=to_json),
        encoding="utf-8",
    )
    report = build_report(results, tuning, dataset_info)
    (REPORT_DIR / "model_comparison.md").write_text(report + "\n", encoding="utf-8")
    update_readme(report)

    print(f"\n{report}\n")
    print(f"scikit-learn {sklearn.__version__} | pandas {pd.__version__} | numpy {np.__version__}")
    print("Done: model/, reports/ and test_data.csv written; README.md results updated.")


if __name__ == "__main__":
    main()
