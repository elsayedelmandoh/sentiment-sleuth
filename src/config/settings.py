from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "Sentiment Sleuth")
PROJECT_DESCRIPTION = os.getenv(
	"PROJECT_DESCRIPTION", "ML-Powered Amazon Review Sentiment Analysis"
)

# Hugging Face / asset settings
HF_ASSETS_REPO = os.getenv("HF_ASSETS_REPO", "")
HF_ASSETS_REPO_TYPE = os.getenv("HF_ASSETS_REPO_TYPE", "")


# Default asset paths (kept here so settings is authoritative)
ASSET_PATHS = [
	'data/vectorizers/tfidf_vectorizer.joblib',
	'data/models/05_logistic_regression_classifier.joblib',
	'data/models/06_naive_bayes_classifier.joblib',
	'data/models/07_ft_svm_classifier.joblib',
	'data/models/07_linear_svm_classifier.joblib',
	'data/models/08_knn_classifier.joblib',
	'data/models/09_decision_tree_classifier.joblib',
	'data/models/10_random_forest_classifier.joblib',
	'data/models/11_stochastic_gradient_descent_classifier.joblib',
	'data/models/12_xgboost_classifier.joblib',
	'data/models/13_lightgbm_classifier.joblib',
]

# Cache directory for downloaded assets
ASSET_CACHE_DIR = Path(os.getenv("ASSET_CACHE_DIR", "data/remote_cache"))
