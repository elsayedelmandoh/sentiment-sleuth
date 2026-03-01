# Project Definition - Project Architecture (JUST How project works End to End?) 

The data pipeline is divided into 5 layers.

1. **Ingestion Layer:** Load Kaggle Amazon Review dataset (`.csv`).
2. **Cleaning Layer:** Apply regex and NLTK filtering to the raw text columns.
3. **Transformation Layer:** Fit a TF-IDF/BoW vectorizer on the training data and transform both train and test sets.
4. **Training Layer:** Pass the sparse matrices into the Scikit-learn estimators.
5. **Evaluation Layer:** Generate predictions on the test set, compute metrics (Accuracy, F1, Precision, Recall, Time), and plot the Confusion Matrix.