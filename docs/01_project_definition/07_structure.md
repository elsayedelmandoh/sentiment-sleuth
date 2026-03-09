# Project Definition - Project Structure (Where is the code?)

```text
sentiment-analysis-of-amazon-reviews-using-machine-learning/
├── data/
│   ├── models/          # Saved model files (.joblib)
│   ├── predictions/     # Model prediction outputs (CSV)
│   ├── processed/       # Cleaned & feature-engineered datasets
│   │   ├── processed_train.csv
│   │   ├── processed_valid.csv
│   │   ├── processed_test.csv
│   │   └── feat_eng_train.csv
│   ├── raw/             # Original immutable dataset
│   │   ├── train.csv
│   │   └── test.csv
│   ├── samples/         # Small sample files for quick testing
│   └── vectorizers/     # Saved vectorizers and sparse matrices (TF-IDF)
│       ├── tfidf_vectorizer.joblib
│       ├── X_train_tfidf.npz
│       └── X_test_tfidf.npz
|
├── docs/
│   ├── 00_research/
│   │   ├── datasets.md
│   │   ├── references.md
│   │   └── related_projects.md
│   └── 01_project_definition/
│       ├── 00_quickstart.md
│       ├── 01_problem.md
│       ├── 02_goal.md
│       ├── 03_solution.md
│       ├── 04_stack.md
│       ├── 05_architecture.md
│       ├── 06_workflow.md
│       ├── 07_structure.md    
│       └── 08_report.md
|
├── notebooks/          
│   ├── 00_quickstartt.ipynb
│   ├── 01_data_acquisition.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_logistic_regression.ipynb
│   ├── 06_naive_bayes.ipynb
│   ├── 07_support_vector_machine.ipynb
│   ├── 08_k_nearest_neighbors.ipynb
│   ├── 09_decision_trees.ipynb
│   ├── 10_random_forest.ipynb
│   ├── 11_stochastic_gradient_descent.ipynb
│   ├── 12_xgboost.ipynb
│   ├── 13_lightgbm.ipynb
│   └── 14_comparsion.ipynb
|
├── src/                 # Production-style source code and helpers
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py   # configuration values and constants
│   └── utils/
│       ├── __init__.py
|       └── helpers.py     # Helper functions used by notebooks and app
|
├── .env                 # Environment variables
├── .gitignore           # List of files to ignore by git
├── .env.example         # Example of environment variables
├── .gitattributes
├── .gitignore
├── app.py               # App/runner for model inference or demo
├── README.md            # Project overview and instructions to run
└── requirements.txt     # List of dependencies (pandas, scikit-learn, etc.)
```
