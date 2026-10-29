# Implementation Tasks

- [x] Analyse `breast_cancer.csv` (quality, class balance, separability, correlation) and choose the models.
- [x] `ml_utils.py`: CSV loading, target mapping, metric calculation shared by training and app.
- [x] `train_models.py`: stratified split, per-model grid-search CV, saved pipelines, `test_data.csv`, reports, README results.
- [x] `app.py`: themed dashboard with upload, model dropdown, metrics, confusion matrix, report, model comparison, predictions.
- [x] `requirements.txt` pinned, `.streamlit/config.toml` theme, offline tests, README and spec.
- [ ] User: in Colab or BITS Lab run `pytest -q` and `python train_models.py`; review the generated observations and add your own wording.
- [ ] User: commit in several steps, deploy on Streamlit Cloud, fill the two README links, take the BITS Lab screenshot, build the PDF.

## Note

The local environment has no scikit-learn, so this code has been syntax-checked but not executed here. The model comparison in the analysis was done with numpy re-implementations; final numbers come from the Colab/BITS Lab run.
