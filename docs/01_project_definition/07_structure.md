# Project Definition - Project Structure (Where is the code?)

```text
sentiment-analysis-of-amazon-reviews-using-machine-learning/
├── data/
│   ├── models/          # Saved model files (.joblib)
│   ├── processed/       # Cleaned & feature-engineered datasets
│   │   ├── balanced_sample_train.csv
│   │   ├── feat_eng_train.csv
│   │   ├── processed_test.csv
│   │   ├── processed_train.csv
│   │   ├── processed_valid.csv
│   │   ├── y_test.csv
│   │   ├── y_train.csv
│   │   └── y_valid.csv
│   ├── raw/             # Original immutable dataset
│   │   ├── readme.txt
│   │   ├── train.csv
│   │   └── test.csv
│   ├── samples/         # Small sample files for quick testing
│   |   ├── sample_test.csv
│   |   ├── sample_train.csv
│   |   └── sample_valid.csv
│   └── vectorizers/     # Saved vectorizers and sparse matrices (TF-IDF)
│       ├── tfidf_vectorizer.joblib
│       ├── X_test_tfidf.npz
│       ├── X_train_tfidf.npz
│       └── X_valid_tfidf.npz
|
├── docs/
│   ├── 00_research/
│   │   ├── datasets.md
│   │   ├── references.md
│   │   └── related_projects.md
│   ├── 01_project_definition/
│   |   ├── 00_quickstart.md
│   |   ├── 01_problem.md
│   |   ├── 02_goal.md
│   |   ├── 03_solution.md
│   |   ├── 04_stack.md
│   |   ├── 05_architecture.md
│   |   ├── 06_workflow.md
│   |   └── 07_structure.md    
│   ├── 02_results/     
│   ├── 03_report/          
│   └── 04_presentation/     # Presentation slides and speaker notes
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
│       ├── helpers.py      # Helper functions used by notebooks and app
│       ├── hf_loader.py    # loader for GUI
|       └── hf_uploader.py  # uploader for Hugging Face  
|
├── .env                 # Environment variables
├── .env.example         # Example of environment variables
├── .gitattributes
├── .gitignore           # List of files to ignore by git
├── app.py               # App/runner for model inference or demo
├── LICENSE              # License file
├── README.md            # Project overview and instructions to run
└── requirements.txt     # List of dependencies (pandas, scikit-learn, etc.)
```
