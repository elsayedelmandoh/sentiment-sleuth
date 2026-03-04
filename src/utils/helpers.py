from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import html
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import joblib
import unicodedata

from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
from sklearn.utils import check_random_state
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
def top_n_grams(corpus, ngram_range=(1,1), top_k=20, stop_words='english', max_features=20000):
	vec = CountVectorizer(ngram_range=ngram_range, stop_words=stop_words, max_features=max_features)
	X = vec.fit_transform(corpus)
	sums = np.array(X.sum(axis=0)).ravel()
	terms = np.array(vec.get_feature_names_out())
	if terms.size == 0:
		return []
	top_idx = sums.argsort()[::-1][:top_k]
	return list(zip(terms[top_idx], sums[top_idx]))


def show_top_ngrams_by_class(df, target_col='review_target', text_col='review_cleaned',
							 ngram_ranges=((1,1),(2,2)), top_k=(15,12),
							 stop_words='english', max_features=20000,
							 plot=True, figsize=(8,5)):
	"""
	For each class in df[target_col], print and (optionally) plot top n-grams per ngram_range.
	Returns a nested dict: {class: {ngram_range: [(term, count), ...]}}
	"""
	if target_col not in df.columns:
		raise KeyError(f"Target column '{target_col}' not found")

	# Normalize ngram_ranges: allow a single range like (1,1) to be passed and wrap it
	if isinstance(ngram_ranges, tuple) and len(ngram_ranges) == 2 and all(isinstance(x, int) for x in ngram_ranges):
		ngram_ranges = (ngram_ranges,)
	elif isinstance(ngram_ranges, list):
		ngram_ranges = tuple(ngram_ranges)

	classes = (df[target_col].cat.categories
			   if hasattr(df[target_col], 'cat') else np.unique(df[target_col].astype(str)))

	results = {}
	for cls in classes:
		cls_mask = df[target_col] == cls if cls in df[target_col].values else df[target_col].astype(str) == str(cls)
		subset = df.loc[cls_mask, text_col].fillna("").astype(str)
		results.setdefault(cls, {})
		for i, rg in enumerate(ngram_ranges):
			k = top_k[i] if (isinstance(top_k, (list,tuple)) and i < len(top_k)) else (top_k if isinstance(top_k, int) else 20)
			top = top_n_grams(subset, ngram_range=rg, top_k=k, stop_words=stop_words, max_features=max_features)
			results[cls][rg] = top

			# Print
			nname = ("unigrams" if rg==(1,1) else "bigrams" if rg==(2,2) else f"{rg[0]}-{rg[1]}grams")
			print(f'--- Top {nname} for class {cls} ---')
			print(top)
			print()

			# Plot
			if plot and top:
				terms, counts = zip(*top)
				plt.figure(figsize=figsize)
				plt.barh(terms[::-1], counts[::-1], color='steelblue')
				plt.title(f"Top {len(terms)} {nname} for class {cls}")
				plt.xlabel("Count")
				plt.tight_layout()
				plt.show()

	return results


def add_basic_meta_features(df: pd.DataFrame, text_col: str = 'review_content') -> pd.DataFrame:
	"""
	Add basic meta-features to `df` based on the text column `text_col`.
	Features added: exclamation_count, question_count, uppercase_count, uppercase_ratio,
	word_count, avg_word_length, punctuation_count.

	The function is tolerant if the column is missing (raises KeyError).
	"""
	if text_col not in df.columns:
		raise KeyError(f"Text column '{text_col}' not found in dataframe")

	s = df[text_col].fillna("").astype(str)
	df = df.copy()
	df['exclamation_count'] = s.str.count('!')
	df['question_count'] = s.str.count('\?')
	df['punctuation_count'] = s.str.count(r"[^\w\s]")
	df['word_count'] = s.str.split().apply(lambda ws: len(ws) if isinstance(ws, list) else 0)
	df['avg_word_length'] = s.str.split().apply(lambda ws: np.mean([len(w) for w in ws]) if isinstance(ws, list) and len(ws) else 0)
	# Uppercase counts and ratio (use string length to avoid division by zero)
	df['uppercase_count'] = s.apply(lambda x: sum(1 for c in x if c.isupper()))
	lengths = s.str.len().replace(0, 1)
	df['uppercase_ratio'] = df['uppercase_count'] / lengths
	return df


def plot_dimensionality_reduction(X, labels, method='PCA', sample=1000, random_state: int = 42, figsize=(8,6)):
	"""
	Reduce `X` to 2D and plot colored by `labels`.

	- If `X` is sparse, uses TruncatedSVD for initial reduction.
	- `method` can be 'PCA' or 'TSNE'. For 'TSNE', X is first reduced to 50 components
	  (when high-dimensional) using TruncatedSVD for speed.
	- `sample` controls maximum number of points to plot (random sampling).

	Returns the (n_samples,2) embedding array.
	"""
	# Handle sparse matrices
	is_sparse = hasattr(X, 'tocsr') or hasattr(X, 'tocsc')

	n_samples = X.shape[0]
	rng = check_random_state(random_state)
	if sample is not None and n_samples > sample:
		idx = rng.choice(n_samples, size=sample, replace=False)
		if is_sparse:
			X_sample = X[idx]
		else:
			X_sample = X[idx, :]
		y_sample = np.asarray(labels)[idx]
	else:
		X_sample = X
		y_sample = np.asarray(labels)

	# Produce 2D embedding
	if method.upper() == 'PCA':
		if is_sparse:
			svd = TruncatedSVD(n_components=2, random_state=random_state)
			emb = svd.fit_transform(X_sample)
		else:
			pca = PCA(n_components=2, random_state=random_state)
			emb = pca.fit_transform(X_sample)
	elif method.upper() == 'TSNE':
		# For TSNE, pre-reduce if needed
		if is_sparse:
			pre_n = min(50, X.shape[1])
			pre = TruncatedSVD(n_components=pre_n, random_state=random_state)
			X_pre = pre.fit_transform(X_sample)
		else:
			X_pre = X_sample
		tsne = TSNE(n_components=2, random_state=random_state)
		emb = tsne.fit_transform(X_pre)
	else:
		raise ValueError("Unsupported method. Choose 'PCA' or 'TSNE'.")

	# Plot
	plt.figure(figsize=figsize)
	unique_labels, label_idx = np.unique(y_sample, return_inverse=True)
	cmap = plt.get_cmap('tab10')
	for i, ul in enumerate(unique_labels):
		mask = label_idx == i
		plt.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.8, label=str(ul), color=cmap(i % 10))
	plt.legend(title='label', bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.tight_layout()
	plt.xlabel('dim1')
	plt.ylabel('dim2')
	plt.title(f'{method} projection')
	plt.show()

	return emb


# logistic regression notebook
