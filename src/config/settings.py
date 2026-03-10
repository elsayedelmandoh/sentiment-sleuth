from dotenv import load_dotenv
import os

# Load environment variables from project .env (if present)
load_dotenv()

PROJECT_NAME = os.getenv("PROJECT_NAME", "Sentiment Sleuth")
PROJECT_DESCRIPTION = os.getenv(
	"PROJECT_DESCRIPTION", "ML-Powered Amazon Review Sentiment Analysis"
)

# Hugging Face / asset settings
HF_ASSETS_REPO = os.getenv("HF_ASSETS_REPO", "")
HF_ASSETS_REPO_TYPE = os.getenv("HF_ASSETS_REPO_TYPE", "")

# Optional comma-separated override for asset filenames (paths relative to repo)
# Example: data/models/modelA.joblib,data/vectorizers/tfidf_vectorizer.joblib
HF_ASSET_FILES = os.getenv("HF_ASSET_FILES", "")

from pathlib import Path

# Default asset paths (kept here so settings is authoritative)
DEFAULT_ASSET_PATHS = [
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

if HF_ASSET_FILES:
	ASSET_PATHS = [p.strip() for p in HF_ASSET_FILES.split(",") if p.strip()]
else:
	ASSET_PATHS = DEFAULT_ASSET_PATHS

# Cache directory for downloaded assets
ASSET_CACHE_DIR = Path(os.getenv("ASSET_CACHE_DIR", "data/remote_cache"))
