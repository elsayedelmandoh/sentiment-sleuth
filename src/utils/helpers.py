from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import html
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import unicodedata

import contractions
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# data acquisition notebook
def save(base_path, df: pd.DataFrame = None, vectorizer=None, model=None,
		 df_name: str = "dataset.csv", vectorizer_name: str = "vectorizer.joblib",
		 model_name: str = "model.joblib", verbose: bool = True):
	"""
	Save a dataframe (CSV), a vectorizer (joblib) and/or a model (joblib) to disk.
	Any of df/vectorizer/model can be None and will be skipped.
	"""
	base = Path(base_path)
	base.mkdir(parents=True, exist_ok=True)

	saved = {}
	# CSV
	if df is not None:
		path = base / df_name
		df.to_csv(path, index=False)
		saved['csv'] = path
		if verbose:
			print(f"Saved dataframe to {path}")

	# joblib (vectorizer and model)
	if vectorizer is not None:
		_dump = lambda obj, p: joblib.dump(obj, p)
		path = base / vectorizer_name
		_dump(vectorizer, path)
		saved['vectorizer'] = path
		if verbose:
			print(f"Saved vectorizer to {path}")

	if model is not None:
		_dump = lambda obj, p: joblib.dump(obj, p)
		path = base / model_name
		_dump(model, path)
		saved['model'] = path
		if verbose:
			print(f"Saved model to {path}")

	return saved


# eda notebook
def apply_balance(df: pd.DataFrame, target_col: str = "target", random_state: int = 42) -> pd.DataFrame:
	"""Return a balanced dataframe by undersampling majority classes to the minority count.

	If the dataframe is already balanced (all classes equal), it's returned unchanged.
	Args:
		df (pd.DataFrame): The input dataframe to balance.
		target_col (str, optional): The name of the target column. Defaults to "target".
		random_state (int, optional): Random state for reproducibility. Defaults to 42.
	Returns:
		pd.DataFrame: A balanced dataframe.
	"""
	counts = df[target_col].value_counts()
	if counts.nunique() == 1:
		return df.reset_index(drop=True)

	target_n = counts.min()
	parts = [
		grp.sample(n=target_n, replace=False, random_state=random_state)
		for _, grp in df.groupby(target_col)
	]
	balanced = pd.concat(parts, axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
	return balanced


def plot_top_ngrams(corpus, n=1, top_k=20, stop_words='english', max_features=20000, figsize=(10,6), title=None):
    """
    Compute and plot the top n-grams from a text corpus.

    Parameters
    ----------
    corpus : iterable-like
        Iterable of text documents (e.g., pandas Series).
    n : int, optional
        The n in n-grams (uses ngram_range=(n,n)). Default is 1 (unigrams).
    top_k : int, optional
        Number of top n-grams to show. Default is 20.
    stop_words : str or list, optional
        Stop words parameter forwarded to CountVectorizer. Default 'english'.
    max_features : int, optional
        Max features for the vectorizer. Default 20000.
    figsize : tuple, optional
        Figure size for the plot.
    title : str, optional
        Custom title for the plot. If None, a default title is used.

    Returns
    -------
    list of (term, count)
        The top n-grams and their counts (sorted descending).
    """

    vec = CountVectorizer(ngram_range=(n, n), stop_words=stop_words, max_features=max_features)
    X = vec.fit_transform(corpus)
    sums = np.array(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())

    if terms.size == 0:
        print("No terms found for the given corpus/parameters.")
        return []

    top_idx = sums.argsort()[::-1][:top_k]
    top_terms = terms[top_idx]
    top_counts = sums[top_idx]

    # Plot horizontal bar chart with largest on top
    plt.figure(figsize=figsize)
    plt.barh(top_terms[::-1], top_counts[::-1], color='steelblue')
    plt.xlabel("Count")
    plt.tight_layout()
    if title is None:
        title = f"Top {min(top_k, len(top_terms))} {n}-grams"
    plt.title(title)
    plt.show()

    return list(zip(top_terms, top_counts))



# preprocessing notebook
def clean_text(s):
	"""
	Professional NLP preprocessing for Sentiment Analysis.
	Accepts a str, pandas.Series or pandas.DataFrame (with 'review_content').
	Returns cleaned str or pandas.Series of cleaned strs.
	"""
	# Initialize NLTK resources and lemmatizer once
	if not hasattr(clean_text, "_nltk_initialized"):
		nltk.download('stopwords', quiet=True)
		nltk.download('wordnet', quiet=True)
		clean_text._lemmatizer = WordNetLemmatizer()
		clean_text._stopwords = set(stopwords.words('english'))
		clean_text._nltk_initialized = True

	lemmatizer = clean_text._lemmatizer
	all_stopwords = clean_text._stopwords

	# DataFrame: apply on 'review_content' column
	if isinstance(s, pd.DataFrame):
		if 'review_content' not in s.columns:
			raise ValueError("DataFrame must contain 'review_content' column")
		return s['review_content'].apply(clean_text)

	# Series: apply element-wise
	if isinstance(s, pd.Series):
		return s.apply(clean_text)

	# Non-string inputs -> return empty string
	if not isinstance(s, str):
		return ''

	# PRO TIP: keep negations and some modifiers
	sentiment_exceptions = {'not', 'no', 'nor', 'against', 'but', 'however', 'very', 'too'}
	custom_stopwords = all_stopwords - sentiment_exceptions

	# 1. Decode HTML & Unicode
	s = html.unescape(s)
	s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii', errors='ignore')

	# 2. Lowercase
	s = s.lower()

	# 3. Emojis to text
	s = emoji.demojize(s, delimiters=(" ", " "))

	# 4. Expand contractions
	s = contractions.fix(s)

	# 5. Remove URLs and HTML tags
	s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
	s = re.sub(r'<[^>]+>', ' ', s)

	# 6. Limit repeated characters (e.g., "loooove" -> "loove")
	s = re.sub(r'(.)\1{2,}', r'\1\1', s)

	# 7. Keep only letters, digits, whitespace and underscores (emoji text)
	s = re.sub(r'[^a-z0-9\s_]', ' ', s)

	# 8. Tokenize, remove stopwords, lemmatize
	words = s.split()
	cleaned_words = [
		lemmatizer.lemmatize(word)
		for word in words
		if word not in custom_stopwords and len(word) > 1
	]

	# 9. Rejoin and collapse extra whitespace
	s = ' '.join(cleaned_words)
	s = re.sub(r'\s+', ' ', s).strip()

	return s


# feature engineering notebook
def top_n_grams(corpus, n=None, ngram_range=(1,1), top_k=20):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=20000)
    X = vec.fit_transform(corpus)
    sums = np.array(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = sums.argsort()[::-1][:top_k]
    return list(zip(terms[top_idx], sums[top_idx]))



