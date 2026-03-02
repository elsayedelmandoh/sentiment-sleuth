from pathlib import Path
import pandas as pd


def _project_root() -> Path:
	return Path(__file__).resolve().parents[2]


def _check_and_balance(df: pd.DataFrame, target_col: str = "target", random_state: int = 42) -> pd.DataFrame:
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
