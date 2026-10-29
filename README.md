# Tumour Insight - Breast Cancer Classification

Five machine-learning classifiers that label breast tumour biopsies as **malignant** or **benign**, compared on the same hold-out test set and served through an interactive Streamlit app. Built for BITS ML Assignment 2. This is an academic demonstration, not a diagnostic tool.

## a. Problem Statement

A fine-needle aspirate (FNA) of a breast mass is digitised and the cell nuclei in the image are measured. Given these 30 nucleus measurements, predict whether the tumour is **malignant (M)** or **benign (B)**.

This is a **binary classification** problem. Missing a malignant tumour (a false negative) is the costliest error, so alongside accuracy the project reports recall and MCC and shows how many malignant tumours each model misses.

## b. Dataset Description

| Item | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | UCI Machine Learning Repository - <https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic> (Kaggle mirror: <https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data>) |
| Instances | 569 tumours |
| Features | 30 numeric input features (plus `id`, which is not used) |
| Target | `diagnosis`: M = malignant (212, 37.3%), B = benign (357, 62.7%) |
| Missing values | None |
| Licence | CC BY 4.0 |

Citation: W. N. Street, W. H. Wolberg and O. L. Mangasarian (1993), *Nuclear feature extraction for breast tumor diagnosis*, IS&T/SPIE Symposium on Electronic Imaging.

**Features.** Ten properties are measured for every nucleus: radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. Each is summarised three ways across the image - `_mean`, `_se` (standard error) and `_worst` (mean of the three largest values) - giving 30 features.

**Exploratory findings**

- Clean data: no missing values, no duplicate rows or ids. The Kaggle file has one empty trailing column, which is dropped on load.
- The benign class is the majority (62.7%), so predicting every tumour as benign would give 62.7% accuracy. All five models are well above that baseline on the held-out test set.
- 13 tumours have concavity and concave points of exactly 0. These are genuinely smooth nuclei, not missing values, and all 13 are benign.
- The measurements contain a strong diagnostic signal. In a descriptive, single-feature check on the full dataset, `perimeter_worst` had the strongest separation (AUC 0.975), followed by `radius_worst`, `area_worst` (0.970) and `concave points_worst` (0.967). These are exploratory full-data values, not held-out model results. `texture_se`, `smoothness_se`, `symmetry_se` and `fractal_dimension_mean` were weak on their own (AUC 0.45-0.51).
- Heavy redundancy: 15 feature pairs correlate above 0.95 (radius, perimeter and area up to 0.998). Two principal components explain 63% of the variance and ten explain 95%.
- Scales differ by several orders of magnitude (0.001 to 4,254) and several features are strongly right-skewed (`area_se` skewness 5.4), so features are standardised before modelling.

**Preparation**

- `diagnosis` is mapped to Malignant / Benign; `id` is excluded from the features.
- One stratified 80/20 split (seed 711) gives 455 training and 114 test tumours. The test rows are saved unchanged as `test_data.csv`.
- Every model is a pipeline of median imputation (a safety net for uploaded files - the source data has no gaps), standard scaling and the classifier, fitted on the training rows only.
- Hyperparameters are tuned per model with 5-fold stratified cross-validation on the training set, scored by MCC. The test set is used once, for the final comparison.

## c. GitHub Repository Link

TODO: paste the repository URL here.

## d. Models Used

| Model | Implementation | Hyperparameters searched |
|---|---|---|
| Logistic Regression | `LogisticRegression` (L2) | C |
| Decision Tree | `DecisionTreeClassifier` | criterion, max_depth, min_samples_leaf |
| KNN | `KNeighborsClassifier` | n_neighbors, weights |
| Naive Bayes | `GaussianNB` (Gaussian, suited to continuous measurements) | var_smoothing |
| Support Vector Machine | `SVC`, RBF kernel, Platt-scaled probabilities | C, gamma |

AUC is computed from each model's predicted probability of malignancy; the other metrics use a 0.5 threshold.

<!-- MODEL_RESULTS_START -->

Hold-out test set (`test_data.csv`): **114** tumours (42 malignant, 72 benign), never used for training or tuning. Models were trained on **455** tumours, with hyperparameters chosen by 5-fold stratified cross-validation on the training set (scoring: MCC). Precision, recall and F1 treat **Malignant** as the positive class; the decision threshold is 0.5.

### Comparison table (test data)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9964 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Decision Tree | 0.9649 | 0.9714 | 0.9524 | 0.9524 | 0.9524 | 0.9246 |
| KNN | 0.9825 | 0.9959 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |
| Naive Bayes | 0.9561 | 0.9897 | 0.9302 | 0.9524 | 0.9412 | 0.9064 |
| Support Vector Machine | 0.9737 | 0.9987 | 0.9535 | 0.9762 | 0.9647 | 0.9439 |

### Tuned hyperparameters

| ML Model Name | Selected hyperparameters | Cross-validated MCC (mean ± sd) |
|---|---|---:|
| Logistic Regression | C=1 | 0.9405 ± 0.0111 |
| Decision Tree | criterion=entropy, max_depth=5, min_samples_leaf=5 | 0.8828 ± 0.0595 |
| KNN | n_neighbors=7, weights=uniform | 0.9301 ± 0.0330 |
| Naive Bayes | var_smoothing=1e-09 | 0.8494 ± 0.0509 |
| Support Vector Machine | C=10, gamma=0.01 | 0.9539 ± 0.0251 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performed best on the held-out test set by MCC, at 0.9623. It classified 41 of 42 malignant tumours correctly and made one false alarm among the benign tumours. The close results for a regularised linear model fit with the strong signal and substantial overlap among size-related measurements. |
| Decision Tree | The Decision Tree was the easiest model to explain as a series of feature-based splits, but it had the lowest cross-validation MCC (0.8828) and the widest fold-to-fold variation (±0.0595). On the test set it missed two malignant tumours and raised two false alarms. Its result may be more sensitive to the particular training sample than the results of the linear models. |
| KNN | KNN matched Logistic Regression on test accuracy, precision, recall, F1 and MCC. Its AUC was slightly lower (0.9959 versus 0.9964), so Logistic Regression takes first place under the stated tie-break. KNN's strong result depends on standardising measurements before calculating distances. |
| Naive Bayes | Naive Bayes had the lowest test MCC (0.9064), with two missed malignant tumours and three false alarms. Its AUC was still 0.9897, which means its ranking of cases was strong even though its predictions at the fixed 0.5 threshold were less accurate. The many correlated measurements challenge its feature-independence assumption. |
| Support Vector Machine | The SVM achieved the highest test AUC (0.9987) and the highest mean cross-validation MCC (0.9539), but its test MCC was 0.9439. It missed one malignant tumour and raised two false alarms. The difference between its excellent ranking and its lower fixed-threshold score shows why AUC and threshold-based metrics tell different parts of the story. |

**Overall winner for this hold-out set: Logistic Regression.** It shares the best MCC with KNN (0.9623), so this is a tie in MCC; Logistic Regression is listed first because its AUC is marginally higher (0.9964 versus 0.9959). The SVM led cross-validation and AUC, so it is also a strong contender. The test set contains only 114 tumours, including 42 malignant cases, and a single changed prediction can shift a metric noticeably. These results describe this reproducible split; they do not establish that one model will always outperform the others.

MCC is the primary ranking metric because it incorporates true and false positives and negatives. AUC evaluates how well the model ranks cases across thresholds; the other reported metrics use the specified 0.5 threshold. In a real screening context, missed malignant tumours matter greatly, but changing the threshold should be evaluated on training/validation data and with clinical oversight. This coursework app is not a diagnostic tool.

<!-- MODEL_RESULTS_END -->

## Streamlit App

Live app: TODO - paste the Streamlit Community Cloud URL here.

| Assignment requirement | Where it is in the app |
|---|---|
| Dataset upload (CSV) | Sidebar uploader, plus a switch to use the bundled `test_data.csv` and a download button for it |
| Model selection dropdown | Sidebar "Classifier" dropdown |
| Evaluation metrics | Six metric cards on the *Model performance* tab |
| Confusion matrix / classification report | Both on the *Model performance* tab |
| Results of the different models | *Compare all models* tab: table, bar chart, ROC curves and all five confusion matrices |

Also included: a decision-threshold slider (to trade missed cancers against false alarms), a predicted-probability chart, a per-tumour predictions table with CSV download, and a data overview tab. The app only loads the saved pipelines from `model/` - it never retrains on uploaded data. Uploaded files must contain `diagnosis` (M/B) and all 30 feature columns; extra columns such as `id` are ignored.

## Repository Structure

```text
app.py                 Streamlit app (loads saved models, never retrains)
train_models.py        Tuning, training, evaluation; writes all artifacts
ml_utils.py            Data loading, target mapping and metrics shared by both
requirements.txt       Pinned dependencies
breast_cancer.csv      Source dataset (569 rows)
test_data.csv          Hold-out test rows used for every reported metric (generated)
model/                 Five saved pipelines (.joblib) + model_registry.json (generated)
reports/               model_metrics.json, model_comparison.md (generated)
tests/                 Offline pytest checks on synthetic data
.streamlit/config.toml App theme
```

## How to Run

Versions are pinned because a saved model must be loaded with the same scikit-learn version that trained it.

**BITS Virtual Lab or local machine (Python 3.10-3.12)**

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

Training finishes in about a minute. Commit the generated `model/`, `reports/`, `test_data.csv` and the updated `README.md`.

## Deploy on Streamlit Community Cloud

1. Push the repository, including `model/` and `test_data.csv`, to GitHub.
2. Go to <https://streamlit.io/cloud>, sign in with GitHub and click **New app**.
3. Select the repository and branch (`main`), set the main file to `app.py`, and under *Advanced settings* choose Python 3.12.
4. Deploy, then open the app and check every model on `test_data.csv`.
