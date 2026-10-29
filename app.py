"""Tumour Insight: Streamlit dashboard that evaluates the saved breast cancer classifiers.

Models are loaded from model/ and used only for prediction; nothing is retrained here.
"""

from __future__ import annotations

import html
import io
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from matplotlib.figure import Figure

from ml_utils import (
    CLASS_ORDER,
    ID_COLUMN,
    METRIC_LABELS,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    TARGET_COLUMN,
    evaluate_frame,
    load_csv,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
REGISTRY_PATH = MODEL_DIR / "model_registry.json"
SAMPLE_PATH = ROOT / "test_data.csv"

ACCENT = "#C2185B"
BENIGN_COLOUR = "#2E7D6B"
INK = "#1F2933"
MODEL_COLOURS = {
    "Logistic Regression": "#C2185B",
    "Decision Tree": "#F57C00",
    "KNN": "#1976D2",
    "Naive Bayes": "#7B1FA2",
    "Support Vector Machine": "#2E7D6B",
}
METRIC_HINTS = {
    "accuracy": "share of all tumours classified correctly",
    "auc": "ranking quality, independent of threshold",
    "precision": "flagged malignant that truly are",
    "recall": "malignant tumours that were caught",
    "f1": "balance of precision and recall",
    "mcc": "overall correlation, from -1 to 1",
}

st.set_page_config(page_title="Tumour Insight | Breast Cancer Classifier", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; max-width: 1260px; }
    .hero {
        background: linear-gradient(120deg, #880E4F 0%, #C2185B 55%, #EC407A 100%);
        border-radius: 16px; padding: 1.5rem 2rem; margin-bottom: 1.2rem; color: #fff;
    }
    .hero h1 { color: #fff; font-size: 2.05rem; margin: 0; padding: 0; }
    .hero p { color: #FCE4EC; margin: .35rem 0 0 0; font-size: 1rem; }
    .tag {
        display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
        padding: .15rem .7rem; margin: .8rem .4rem 0 0; font-size: .78rem; color: #fff;
    }
    .kpi {
        background: #fff; border: 1px solid #F3D3E0; border-top: 4px solid #C2185B;
        border-radius: 12px; padding: .75rem 1rem; box-shadow: 0 1px 4px rgba(136,14,79,.07);
        height: 100%;
    }
    .kpi.soft { border-top-color: #F48FB1; }
    .kpi-label { font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: #6B7280; }
    .kpi-value { font-size: 1.65rem; font-weight: 700; color: #1F2933; line-height: 1.25; }
    .kpi-hint { font-size: .74rem; color: #9CA3AF; }
    .section { font-weight: 700; font-size: 1.06rem; color: #880E4F; margin: 1rem 0 .4rem 0; }
    .callout {
        background: #FDF2F6; border-left: 5px solid #C2185B; border-radius: 8px;
        padding: .8rem 1rem; margin: .6rem 0 1rem 0; color: #1F2933;
    }
    .step { background: #FDF2F6; border-radius: 12px; padding: 1rem 1.1rem; height: 100%; }
    .step b { color: #880E4F; }
    .footer { color: #9CA3AF; font-size: .78rem; text-align: center; margin-top: 2.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def card(label: str, value: str, hint: str = "", soft: bool = False) -> str:
    style = "kpi soft" if soft else "kpi"
    return (
        f'<div class="{style}"><div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value">{html.escape(value)}</div>'
        f'<div class="kpi-hint">{html.escape(hint)}</div></div>'
    )


def show_cards(items: list[tuple[str, str, str]], soft: bool = False) -> None:
    for column, (label, value, hint) in zip(st.columns(len(items)), items):
        column.markdown(card(label, value, hint, soft), unsafe_allow_html=True)


def section(title: str) -> None:
    st.markdown(f'<div class="section">{html.escape(title)}</div>', unsafe_allow_html=True)


def load_registry() -> dict:
    if not REGISTRY_PATH.is_file():
        st.error(
            "No saved models found. Run `python train_models.py` locally, commit the generated `model/` "
            "folder and `test_data.csv`, then redeploy."
        )
        st.stop()
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        st.error(f"`model/model_registry.json` could not be read: {error}")
        st.stop()


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def read_frame(csv_bytes: bytes) -> pd.DataFrame:
    return load_csv(io.BytesIO(csv_bytes))


@st.cache_data(show_spinner=False)
def run_evaluation(model_path: str, csv_bytes: bytes, columns: tuple[str, ...], threshold: float) -> dict:
    return evaluate_frame(load_model(model_path), read_frame(csv_bytes), columns, threshold)


def tidy_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def confusion_figure(matrix: list[list[int]], compact: bool = False) -> Figure:
    data = np.array(matrix)
    fig = Figure(figsize=(2.7, 2.4) if compact else (4.6, 3.9), dpi=120)
    ax = fig.subplots()
    ax.imshow(data, cmap="RdPu")
    short = ["B", "M"] if compact else CLASS_ORDER
    ax.set_xticks([0, 1], labels=[f"Pred. {s}" for s in short], fontsize=8 if compact else 9)
    ax.set_yticks([0, 1], labels=[f"Actual {s}" for s in short], fontsize=8 if compact else 9)
    names = [["True negative", "False positive"], ["False negative", "True positive"]]
    peak = data.max() or 1
    for i in range(2):
        for j in range(2):
            text = f"{data[i, j]}" if compact else f"{data[i, j]}\n{names[i][j]}"
            colour = "white" if data[i, j] > peak * 0.55 else INK
            ax.text(j, i, text, ha="center", va="center", color=colour, fontsize=11 if compact else 10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def roc_figure(curves: dict[str, tuple[list, list, float]], highlight: str | None = None) -> Figure:
    fig = Figure(figsize=(4.8, 3.9), dpi=120)
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#BDBDBD")
    for name, (fpr, tpr, auc) in curves.items():
        strong = highlight in (None, name)
        ax.plot(
            fpr, tpr, lw=2.4 if strong else 1.3, alpha=1 if strong else 0.5,
            color=MODEL_COLOURS.get(name, ACCENT), label=f"{name} (AUC {auc:.3f})",
        )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate", fontsize=9)
    ax.set_ylabel("True positive rate (recall)", fontsize=9)
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    tidy_axes(ax)
    fig.tight_layout()
    return fig


def probability_figure(probability: np.ndarray, actual: np.ndarray, threshold: float) -> Figure:
    fig = Figure(figsize=(9.6, 2.7), dpi=120)
    ax = fig.subplots()
    bins = np.linspace(0, 1, 21)
    for label, colour in ((NEGATIVE_LABEL, BENIGN_COLOUR), (POSITIVE_LABEL, ACCENT)):
        ax.hist(probability[actual == label], bins=bins, alpha=0.7, color=colour, label=f"Actual {label}")
    ax.axvline(threshold, color=INK, ls="--", lw=1.2)
    ax.annotate(f" threshold {threshold:.2f}", (threshold, 1), xycoords=("data", "axes fraction"),
                va="top", fontsize=8, color=INK)
    ax.set_xlabel("Predicted probability of malignancy", fontsize=9)
    ax.set_ylabel("Tumours", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    tidy_axes(ax)
    fig.tight_layout()
    return fig


def report_table(report: dict) -> pd.DataFrame:
    rows = {key: value for key, value in report.items() if isinstance(value, dict)}
    table = pd.DataFrame(rows).T[["precision", "recall", "f1-score", "support"]]
    table["support"] = table["support"].astype(int)
    return table.rename(columns=lambda c: c.replace("-", " ").title()).round(3)


# ---------------------------------------------------------------- header
registry = load_registry()
feature_columns = tuple(registry["feature_columns"])
models = {
    name: MODEL_DIR / info["file"]
    for name, info in registry["models"].items()
    if (MODEL_DIR / info["file"]).is_file()
}
if not models:
    st.error("The registry lists no model files that exist in `model/`.")
    st.stop()
dataset = registry.get("dataset", {})

st.markdown(
    f"""
    <div class="hero">
      <h1>Tumour Insight</h1>
      <p>Benign vs malignant classification of breast tumour biopsies (Wisconsin Diagnostic data).
      Upload labelled test data and see how each saved model performs - no retraining involved.</p>
      <span class="tag">{len(models)} trained models</span>
      <span class="tag">{len(feature_columns)} cell-nucleus features</span>
      <span class="tag">trained on {dataset.get('train_rows', '?')} tumours</span>
      <span class="tag">tuned with {dataset.get('cv_folds', '?')}-fold cross-validation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

trained_with = registry.get("library_versions", {}).get("scikit-learn")
if trained_with and trained_with != sklearn.__version__:
    st.warning(
        f"Models were trained with scikit-learn {trained_with}, but this app runs {sklearn.__version__}. "
        "Pin the same version in requirements.txt if predictions fail."
    )

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 1 · Test data")
    upload = st.file_uploader(
        "Upload a labelled test CSV", type="csv",
        help=f"Needs the `{TARGET_COLUMN}` column (M/B) and all {len(feature_columns)} feature columns.",
    )
    use_sample = False
    if SAMPLE_PATH.is_file():
        use_sample = st.toggle("Use bundled test_data.csv", value=True, disabled=upload is not None)
        st.download_button(
            "Download test_data.csv", SAMPLE_PATH.read_bytes(), file_name="test_data.csv",
            mime="text/csv",
        )

    st.markdown("### 2 · Model")
    model_name = st.selectbox("Classifier", list(models))
    details = registry["models"][model_name]
    params = ", ".join(f"{k}={v}" for k, v in details.get("best_params", {}).items()) or "defaults"
    st.caption(f"Tuned settings: `{params}`")
    if "cv_mcc_mean" in details:
        st.caption(f"Cross-validated MCC on training data: {details['cv_mcc_mean']:.3f} ± {details['cv_mcc_std']:.3f}")

    st.markdown("### 3 · Decision threshold")
    threshold = st.slider(
        "Flag as malignant when P(malignant) ≥", 0.05, 0.95, 0.50, 0.05,
        help="0.50 reproduces the README results. Lower values catch more malignant tumours "
        "but raise more false alarms. AUC does not depend on the threshold.",
    )
    st.divider()
    st.caption("Upload test data only - the hosted app has limited capacity.")

# ---------------------------------------------------------------- data source
if upload is not None:
    csv_bytes, source_name = upload.getvalue(), upload.name
elif use_sample:
    csv_bytes, source_name = SAMPLE_PATH.read_bytes(), "test_data.csv (bundled)"
else:
    steps = [
        ("Upload", "Add a labelled CSV in the sidebar, or switch on the bundled test file."),
        ("Choose a model", "Pick one of the five saved classifiers from the dropdown."),
        ("Read the results", "Metrics, confusion matrix, ROC curve and per-tumour predictions."),
    ]
    for column, (title, text) in zip(st.columns(3), steps):
        column.markdown(f'<div class="step"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)
    with st.expander("Required CSV columns"):
        st.code(", ".join([TARGET_COLUMN, *feature_columns]))
    st.stop()

try:
    frame = read_frame(csv_bytes)
except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError, ValueError) as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

try:
    result = run_evaluation(str(models[model_name]), csv_bytes, feature_columns, threshold)
except ValueError as error:
    st.error(str(error))
    st.stop()
except Exception as error:  # noqa: BLE001 - surface unexpected model errors in the UI
    st.error(f"Could not evaluate {model_name}: {error}")
    st.stop()

counts = result["class_counts"]
show_cards(
    [
        ("Data source", source_name if len(source_name) < 28 else source_name[:25] + "...", "evaluated file"),
        ("Tumours scored", f"{result['n_rows']:,}", "rows with a valid diagnosis"),
        ("Malignant", f"{counts[POSITIVE_LABEL]:,}", f"{counts[POSITIVE_LABEL] / result['n_rows']:.0%} of rows"),
        ("Benign", f"{counts[NEGATIVE_LABEL]:,}", f"{counts[NEGATIVE_LABEL] / result['n_rows']:.0%} of rows"),
    ],
    soft=True,
)
st.write("")

perf_tab, compare_tab, pred_tab, data_tab = st.tabs(
    ["Model performance", "Compare all models", "Predictions", "Data overview"]
)

# ---------------------------------------------------------------- selected model
with perf_tab:
    section(f"{model_name} - evaluation metrics")
    show_cards([(METRIC_LABELS[k], f"{result[k]:.4f}", METRIC_HINTS[k]) for k in METRIC_LABELS])
    if abs(threshold - 0.5) > 1e-9:
        st.info(f"Threshold is {threshold:.2f} (not 0.50), so all metrics except AUC differ from the README table.")

    (tn, fp), (fn, tp) = result["confusion_matrix"]
    st.markdown(
        f'<div class="callout">Caught <b>{tp}</b> of {tp + fn} malignant tumours, '
        f"missed <b>{fn}</b> (false negatives), and raised <b>{fp}</b> false alarm{'s' if fp != 1 else ''} "
        f"among {tn + fp} benign tumours.</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns(2, gap="large")
    with left:
        section("Confusion matrix")
        st.pyplot(confusion_figure(result["confusion_matrix"]))
    with right:
        section("ROC curve")
        st.pyplot(roc_figure({model_name: (result["roc_curve"]["fpr"], result["roc_curve"]["tpr"], result["auc"])}))

    section("How confident is the model?")
    st.pyplot(probability_figure(np.array(result["probability"]), np.array(result["actual"]), threshold))

    section("Classification report")
    st.dataframe(report_table(result["classification_report"]))

# ---------------------------------------------------------------- all models
with compare_tab:
    rows, curves, matrices = {}, {}, {}
    with st.spinner("Scoring all models..."):
        for name, path in models.items():
            r = run_evaluation(str(path), csv_bytes, feature_columns, threshold)
            rows[name] = {METRIC_LABELS[k]: r[k] for k in METRIC_LABELS}
            rows[name]["Missed malignant"] = r["confusion_matrix"][1][0]
            curves[name] = (r["roc_curve"]["fpr"], r["roc_curve"]["tpr"], r["auc"])
            matrices[name] = r["confusion_matrix"]

    table = pd.DataFrame(rows).T
    table["Missed malignant"] = table["Missed malignant"].astype(int)
    table.index.name = "ML Model Name"
    leader = table.sort_values(["MCC", "AUC"], ascending=False).index[0]
    st.markdown(
        f'<div class="callout">Best on this data: <b>{html.escape(leader)}</b> - '
        f"MCC {table.loc[leader, 'MCC']:.4f}, accuracy {table.loc[leader, 'Accuracy']:.4f}, "
        f"AUC {table.loc[leader, 'AUC']:.4f}. Highlighted cells mark the best value in each column.</div>",
        unsafe_allow_html=True,
    )
    metric_cols = list(METRIC_LABELS.values())
    styled = (
        table.style.format({c: "{:.4f}" for c in metric_cols})
        .highlight_max(subset=metric_cols, color="#F8BBD0")
        .highlight_min(subset=["Missed malignant"], color="#F8BBD0")
    )
    st.dataframe(styled)

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        section("Metrics side by side")
        long = table[metric_cols].reset_index().melt(id_vars="ML Model Name", var_name="Metric", value_name="Score")
        floor = max(0.0, np.floor((long["Score"].min() - 0.05) * 20) / 20)
        chart = (
            alt.Chart(long)
            .mark_bar(clip=True, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(
                x=alt.X("Metric:N", sort=metric_cols, title=None, axis=alt.Axis(labelAngle=0)),
                xOffset=alt.XOffset("ML Model Name:N"),
                y=alt.Y("Score:Q", scale=alt.Scale(domain=[floor, 1.0]), title="Score"),
                color=alt.Color(
                    "ML Model Name:N",
                    scale=alt.Scale(domain=list(MODEL_COLOURS), range=list(MODEL_COLOURS.values())),
                    legend=alt.Legend(orient="bottom", title=None, columns=3),
                ),
                tooltip=["ML Model Name", "Metric", alt.Tooltip("Score:Q", format=".4f")],
            )
            .properties(height=330)
        )
        st.altair_chart(chart, use_container_width=True)
    with right:
        section("ROC curves")
        st.pyplot(roc_figure(curves, highlight=model_name))

    section("Confusion matrices")
    for column, (name, matrix) in zip(st.columns(len(matrices)), matrices.items()):
        with column:
            st.caption(name)
            st.pyplot(confusion_figure(matrix, compact=True))

# ---------------------------------------------------------------- predictions
with pred_tab:
    section(f"Per-tumour predictions - {model_name}")
    index = result["row_index"]
    predictions = pd.DataFrame(
        {
            "Row": np.array(index) + 1,
            "Actual": result["actual"],
            "Predicted": result["predicted"],
            "P(malignant)": np.round(result["probability"], 4),
        }
    )
    if ID_COLUMN in frame.columns:
        predictions.insert(1, "Sample id", frame[ID_COLUMN].iloc[index].to_numpy())
    predictions["Correct"] = predictions["Actual"] == predictions["Predicted"]

    errors_only = st.toggle(f"Show only misclassified tumours ({int((~predictions['Correct']).sum())})")
    view = predictions[~predictions["Correct"]] if errors_only else predictions
    st.dataframe(
        view,
        hide_index=True,
        column_config={
            "P(malignant)": st.column_config.ProgressColumn(
                "P(malignant)", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "Correct": st.column_config.CheckboxColumn("Correct"),
        },
    )
    slug = model_name.lower().replace(" ", "_")
    st.download_button(
        "Download predictions (CSV)", view.to_csv(index=False).encode("utf-8"),
        file_name=f"predictions_{slug}.csv", mime="text/csv",
    )

# ---------------------------------------------------------------- data overview
with data_tab:
    present = [c for c in feature_columns if c in frame.columns]
    missing_cells = int(frame[present].isna().sum().sum())
    show_cards(
        [
            ("Rows", f"{len(frame):,}", "in the uploaded file"),
            ("Columns", f"{frame.shape[1]}", f"{len(present)} model features"),
            ("Missing values", f"{missing_cells}", "filled with training medians"),
            ("Extra columns", f"{frame.shape[1] - len(present) - 1}", "ignored (e.g. id)"),
        ],
        soft=True,
    )
    left, right = st.columns([1, 2], gap="large")
    with left:
        section("Class balance")
        balance = pd.DataFrame({"Diagnosis": CLASS_ORDER, "Tumours": [counts[c] for c in CLASS_ORDER]})
        st.altair_chart(
            alt.Chart(balance)
            .mark_arc(innerRadius=55)
            .encode(
                theta="Tumours:Q",
                color=alt.Color(
                    "Diagnosis:N",
                    scale=alt.Scale(domain=CLASS_ORDER, range=[BENIGN_COLOUR, ACCENT]),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["Diagnosis", "Tumours"],
            )
            .properties(height=260),
            use_container_width=True,
        )
    with right:
        section("Feature summary")
        st.dataframe(frame[present].describe().T[["mean", "std", "min", "max"]].round(4), height=300)
    section("First rows")
    st.dataframe(frame.head(25), hide_index=True)

st.markdown(
    '<div class="footer">Tumour Insight · BITS ML Assignment 2 · '
    "Academic demonstration only - not a diagnostic tool.</div>",
    unsafe_allow_html=True,
)
