# Project Definition - Quickstart

This quickstart explains how to prepare the environment and reproduce core experiments and inference from the repository.

1) Create a Python environment and install dependencies:

```
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

2) Inspect processed data and vectorizers (already available in repo):

- `data/processed/` contains prepared train/valid/test CSVs and labels.
- `data/vectorizers/` contains the fitted TF-IDF vectorizer and sparse matrices.

3) Run notebooks (recommended order):

- `notebooks/01_data_acquisition.ipynb`
- `notebooks/02_eda.ipynb`
- `notebooks/03_data_preprocessing.ipynb`
- `notebooks/04_feature_engineering.ipynb`
- modeling notebooks `05_*.ipynb` → `14_comparsion.ipynb`

4) Run the demo/app:

```
streamlit run app.py
```

Notes:
- Preprocessed artifacts and trained model joblib files are stored under `data/processed`, `data/vectorizers`, and `data/models` to speed up reproduction.

