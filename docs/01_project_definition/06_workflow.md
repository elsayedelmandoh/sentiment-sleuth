# Project Definition - Operational Workflow (JUST workflow based on stack and architecture files )

# ML Engineering Workflow

We will follow an iterative, agile approach to this project:

1. **Exploratory Data Analysis (EDA):** Understand class distribution (Are there more positive reviews than negative?) and text length statistics.

2. **Data Split:** Hold out 20% of the data strictly for final testing to prevent data leakage.

3. **The "Dummy" Baseline:** First, train a Naive Bayes model. This sets our minimum performance threshold. If a complex Random Forest can't beat Naive Bayes, we don't use it.

4. **Experimentation:** Loop through our list of models (Logistic Regression, SVM, etc.). Log the F1-score and training time for each.

5. **Hyperparameter Tuning:** Take the top 2 performing models and use `GridSearchCV` to optimize their parameters.

6. **Final Evaluation:** Run the optimized models on the held-out test set and generate final visual reports.