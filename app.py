import streamlit as st
from src.utils import helpers
from src.config import settings

st.set_page_config(page_title=settings.PROJECT_NAME, page_icon="🕵️", layout="centered")

@st.cache_resource
def load_cached_assets():
	# Best-effort: ensure NLTK resources (silent failures allowed)
	try:
		helpers.ensure_nltk_resources()
	except Exception:
		pass

	# Return the full assets tuple from helpers.load_assets()
	assets = helpers.load_assets()
	return assets


def _safe_predict(model, X):
	"""Try to predict and return (pred, prob, error_str).

	On failure, attempt a dense fallback for sparse `X` and surface the exception message.
	"""
	if model is None:
		return None, None, None

	def _try_predict(input_X):
		pred = model.predict(input_X)[0]
		prob = None
		# First try predict_proba when available
		if hasattr(model, "predict_proba"):
			try:
				probs = model.predict_proba(input_X)[0]
				if hasattr(model, "classes_"):
					try:
						idx = int((model.classes_ == pred).nonzero()[0][0])
						prob = float(probs[idx])
					except Exception:
						prob = float(probs.max())
				else:
					prob = float(probs.max())
			except Exception:
				prob = None
		# If no predict_proba, try decision_function fallback for an approximate confidence
		elif hasattr(model, "decision_function"):
			try:
				score = model.decision_function(input_X)
				# decision_function can return (n_samples,) or (n_samples, n_classes)
				if hasattr(score, '__len__') and getattr(score, 'ndim', 0) == 1:
					score_val = float(score[0])
					# convert distance to a pseudo-probability via a sigmoid
					prob_pos = 1.0 / (1.0 + __import__('math').exp(-score_val))
					# If classes_ available, align probability to predicted class
					if hasattr(model, 'classes_') and len(model.classes_) >= 2:
						# assume classes_[1] corresponds to the positive side of decision_function
						if pred == model.classes_[1]:
							prob = float(prob_pos)
						else:
							prob = float(1.0 - prob_pos)
					else:
						prob = float(max(min(prob_pos, 1.0), 0.0))
				else:
					# multi-dimensional decision function — skip
					prob = None
			except Exception:
				prob = None
		return pred, prob

	try:
		return (*_try_predict(X), None)
	except Exception as e1:
		# If X is sparse, try dense fallback
		try:
			if hasattr(X, "toarray"):
				X_dense = X.toarray()
				try:
					return (*_try_predict(X_dense), None)
				except Exception as e2:
					return None, None, f"predict failed: {e2}"
		except Exception:
			pass
		return None, None, f"predict failed: {e1}"


def map_label(pred, model=None):
	"""Map a raw model prediction to a human label.

	Supports both common encodings used in this repo:
	- {0,1} where 0 -> Negative, 1 -> Positive
	- {1,2} where 1 -> Negative, 2 -> Positive

	If `model` is provided and has `classes_`, we use that to disambiguate.
	"""
	if pred is None:
		return "Unknown"

	# If model provides classes_, prefer that mapping
	try:
		if model is not None and hasattr(model, 'classes_'):
			classes = tuple(model.classes_)
			if set(classes) == {0, 1}:
				p = int(pred)
				return "Negative" if p == 0 else "Positive"
			if set(classes) == {1, 2}:
				p = int(pred)
				return "Negative" if p == 1 else "Positive"
	except Exception:
		pass

	# Fallback heuristics
	try:
		p = int(pred)
		if p == 0:
			return "Negative"
		if p == 1:
			return "Positive"
		if p == 2:
			return "Positive"
	except Exception:
		pass

	if isinstance(pred, str):
		l = pred.lower()
		if "neg" in l:
			return "Negative"
		if "pos" in l:
			return "Positive"

	return str(pred)


def main():
	st.title(settings.PROJECT_NAME)
	if settings.PROJECT_DESCRIPTION:
		st.caption(settings.PROJECT_DESCRIPTION)

	assets = load_cached_assets()
	# unpack expected assets (helpers.load_assets returns many entries)
	vectorizer = assets[0] if len(assets) > 0 else None
	lr_model = assets[1] if len(assets) > 1 else None
	nb_model = assets[2] if len(assets) > 2 else None
	ft_svm_model = assets[3] if len(assets) > 3 else None
	linear_svm_model = assets[4] if len(assets) > 4 else None
	knn_model = assets[5] if len(assets) > 5 else None
	decision_tree_model = assets[6] if len(assets) > 6 else None
	random_forest_model = assets[7] if len(assets) > 7 else None
	sgd_model = assets[8] if len(assets) > 8 else None
	xgboost_model = assets[9] if len(assets) > 9 else None
	lightgbm_model = assets[10] if len(assets) > 10 else None

	st.markdown("---")
	st.subheader("Analyze a custom Amazon review")

	review = st.text_area("Paste a review here", height=200)
	_, btn_col, _ = st.columns([1, 2, 1])
	analyze = btn_col.button("Analyze")

	if analyze:
		if not review or not str(review).strip():
			st.warning("Please enter a review to analyze.")
			st.stop()

		try:
			cleaned = helpers.clean_text(str(review))
		except Exception as e:
			st.error(f"Error during text cleaning: {e}")
			st.stop()

		if not cleaned:
			st.warning("Input text became empty after cleaning. Try a different review.")
			st.stop()

		if vectorizer is None:
			st.error("TF-IDF vectorizer not available. Ensure `data/vectorizers/tfidf_vectorizer.joblib` exists.")
			st.stop()

		try:
			X = vectorizer.transform([cleaned])
		except Exception as e:
			st.error(f"Error during vectorization: {e}")
			st.stop()

		# collect models and display names in the desired order
		model_list = [
			("Logistic Regression", lr_model),
			("Naive Bayes", nb_model),
			("FT SVM", ft_svm_model),
			("Linear SVM", linear_svm_model),
			("KNN", knn_model),
			("Decision Tree", decision_tree_model),
			("Random Forest", random_forest_model),
			("SGD", sgd_model),
			("XGBoost", xgboost_model),
			("LightGBM", lightgbm_model),
		]

		cols = st.columns(3)
		for i, (name, model) in enumerate(model_list):
			col = cols[i % 3]
			with col:
				st.subheader(name)
				raw, prob, err = _safe_predict(model, X)
				label = map_label(raw, model)
				if label == "Positive":
					st.success(label)
				elif label == "Negative":
					st.error(label)
				elif label == "Unknown":
					if err:
						st.write("Model error:")
						st.caption(err)
					else:
						st.write("Model unavailable or prediction failed.")
				else:
					st.info(label)
				if prob is not None:
					st.caption(f"Confidence: {prob:.2%}")
				elif err:
					# show short error hint for debugging
					st.caption(err)

		st.markdown("---")
		st.subheader("Details")
		st.write("**Original**:")
		st.write(review)
		st.write("**Cleaned (used for inference)**:")
		st.write(cleaned)


if __name__ == "__main__":
	main()
