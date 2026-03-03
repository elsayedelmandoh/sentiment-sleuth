from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


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

# preprocessing notebook
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = html.unescape(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\\.\S+', ' ', s)
    s = re.sub(r'<.*?>', ' ', s)
    s = re.sub(r'[^a-z0-9\\s]', ' ', s)
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

# feature engineering notebook
def top_n_grams(corpus, n=None, ngram_range=(1,1), top_k=20):
    vec = CountVectorizer(ngram_range=ngram_range, stop_words='english', max_features=20000)
    X = vec.fit_transform(corpus)
    sums = np.array(X.sum(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = sums.argsort()[::-1][:top_k]
    return list(zip(terms[top_idx], sums[top_idx]))

