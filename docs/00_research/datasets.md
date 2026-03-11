# Research - Datasets

We have selected the Amazon Reviews dataset hosted on Kaggle, which serves as a refined subset of the larger SNAP Amazon dataset.[1]

Citation: Character-level Convolutional Networks for Text Classification.[2]

Dataset: Amazon reviews.[3]

Dataset link (Kaggle): https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/

Description: The dataset consists of approximately 1.8M training samples and 200K testing samples. It features three key columns:

- Polarity: The target label (1=negative and 2=positive).
- Title: The review summary.
- Text: The full review body.

Scale: Given the constraints of the Anaconda/Jupyter environment and the 4-week timeline, we will utilize a stratified subset of this data. This ensures we maintain a balanced distribution of classes while keeping training times feasible for iterative experimentation.

Current project files (workspace snapshot):

- data/raw/: `train.csv`, `test.csv`, `readme.txt`
- data/processed/: `processed_train.csv`, `processed_valid.csv`, `processed_test.csv`, `feat_eng_train.csv`, `balanced_sample_train.csv`, `y_train.csv`, `y_valid.csv`, `y_test.csv`
- data/vectorizers/: `tfidf_vectorizer.joblib`, `X_train_tfidf.npz`, `X_valid_tfidf.npz`, `X_test_tfidf.npz`
- data/models/: several pre-trained model artifacts (see docs/01_project_definition/07_structure.md for full tree)

Notes:
- The workspace stores preprocessed train/valid/test splits under `data/processed` to allow reproducible training and evaluation without re-running heavy preprocessing steps.
- TF-IDF artifacts are persisted under `data/vectorizers` and are used to transform text into sparse matrices for model training and inference.