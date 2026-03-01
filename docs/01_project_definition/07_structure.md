# Project Definition - Project Structure (Where is the code?)

```text
sentiment-analysis-of-amazon-reviews-using-machine-learning/
├── data/
│   ├── raw/             # Original, immutable Kaggle dataset
│   ├── processed/       # Cleaned data ready for modeling
│   ├── predictions/     # Model predictions on test set
│   └── models/          # Saved model files
|
├── docs/
│   └── 00_research/     # 
|       ├── datasets.md
|       ├── references.md
|       └── related_projects.md
|
│   └── 01_project_definition/          # 
|       ├── 00_quickstart.md
|       ├── 01_problem.md
|       ├── 08_report.md
|       └── proposal_sentiment_analysis.pdf
|
├── notebooks/           
│   ├── 00_quickstart.ipynb     # 
│   ├── 01_logistic_regression.ipynb     # 
│   ├── 02_naive_bayes.ipynb     # 
│   ├── 03_support_vector_machines.ipynb     # 
│   ├── 04_k_nearest_neighbors.ipynb     # 
│   ├── 05_decision_trees.ipynb     # 
│   ├── 06_random_forest.ipynb     # 
│   ├── 07_stochastic_gradient_descent.ipynb     # 
│   └── 08_comparsion.ipynb # 
|
├── src/                 # Production-style source code
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings     # Configuration files
│   ├── utils/
│   |   ├── __init__.py
│   |   └── helpers.py    
│   └── __init__.py
|
├── .env             # 
├── .env.example             # 
├── .gitattributes             # 
├── .gitignore             # 
├── appy.py             # 
├── README.md            # Project overview and instructions to run
└── requirements.txt     # List of dependencies (pandas, scikit-learn, etc.)