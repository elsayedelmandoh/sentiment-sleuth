# Presentation Script: Sentiment Analysis of Amazon Reviews
# Total Time: ~10 minutes (50-60 seconds per slide)
# Delivery Tips: Speak clearly, pause for emphasis, engage with visuals

---

## Slide 1: Title Slide (30 seconds)
"Greeting, salutations, to all those present, our distinguished viewers.  I'm Mohamed Hasn with the rest of the members of Group 12. Today, We'll present our project on Sentiment Analysis of Amazon Reviews using traditional Machine Learning algorithms. Our team consists of A. Mohamed Hasan, B. Elsayed Elmandouh, and C. Mahmoud Abdel Aleem. The repository is available at this link on github and a live demo at this link on hugging face, that we'll explore as we move through, today."

---

## Slide 2: Problem Statement (1 minute)
"Our project addresses a critical need in e-commerce: understanding customer sentiment from millions of Amazon product reviews. 

What we're doing: Classifying sentiment as positive or negative using traditional ML algorithms instead of resource-heavy deep learning.

Why it matters: This enables scalable customer insights, automated content moderation, and better recommendation systems. And for our main challenge, Assessing whether lightweight ML can compete with LLMs for efficiency.

Scope: We work with English reviews from a public dataset, focusing on text review only that contains titles and content (its body). It's binary classification: 1 for negative, 2 for positive.

[Point to word cloud and charts] As you can see from our data exploration, reviews contain rich vocabulary with clear sentiment indicators."

---

## Slide 3: Project Goal & Success Criteria (1 minute)
"Our primary goal was to predict review sentiment with high accuracy using lightweight, traditional ML algorithms, establishing a baseline that proves these methods can compete with deep learning approaches.

Success metrics: We targeted F1 score of at least 0.82 and accuracy of 0.85. We exceeded both - achieving 0.8627 F1 with LightGBM and 0.8903 accuracy with SGD.

Constraints included limited compute resources, so we focused on reproducible notebooks and clear model documentation.

[Show gauge and bars] The gauge shows we're well into the 'excellent' zone, and our progress bars confirm we met and exceeded our targets."

---

## Slide 4: Data & Preprocessing (1 minute)
"We used the Amazon Reviews dataset from Kaggle with nearly 2 million samples, taking a stratified balanced subset to ensure equal positive and negative classes.

Key preprocessing steps: Cleaning HTML, URLs, and emojis to plain text; normalizing with lowercasing and contraction expansion; tokenizing while filtering stopwords with lemmatization.

Features: TF-IDF vectors with 1-3 n-grams and up to 100k features, plus review length and rating embeddings.

[Point to charts] Our class balance is perfect at 50/50, and the word cloud reveals the vocabulary our models work with."

---

## Slide 5: Solution Overview — Workflow (1 minute)
"Our end-to-end pipeline follows standard ML practice: Ingest data, clean it, engineer features, train models, and evaluate results.

The workflow includes exploratory data analysis, data splitting, feature engineering, establishing a baseline, experimentation, hyperparameter tuning, and final evaluation.

[Show time allocation chart] As the chart indicates, we spent most time on EDA and preprocessing (30%), followed by evaluation and comparison (25%), which was crucial for understanding our data thoroughly."

---

## Slide 6: Models Built & Tuning Strategy (1 minute)
"We implemented 9 different models across categories:

Linear models: Logistic Regression and Stochastic Gradient Descent with regularization tuning.

SVM and Naive Bayes: LinearSVC and Multinomial NB with comprehensive parameter search.

Tree-based and ensembles: Decision Trees, Random Forest, XGBoost, and LightGBM.

Non-parametric: K-Nearest Neighbors.

All models used TF-IDF features with 3-5 fold cross-validation for hyperparameter tuning.

[Show pie chart] The distribution shows our focus on linear models (33%) and ensembles (22%), covering a wide range of approaches."

---

## Slide 7: Related Project to Surpass (1 minute)
"To put our work in context, we compared against a related academic project by M. Dedeepya et al. that used advanced deep learning models like BERT, GPT, and BART.

Their work achieved up to 90% accuracy but at massive computational cost with 78% F1-score that's impactfuly low.

Our project seeks to address this efficiency gap, showing that well-tuned traditional algorithms can achieve satisfactory performance for binary classification without the overhead of deep learning.

[Show comparative chart] As you can see, BERT leads in all metrics, but our traditional ML approaches offer a very compelling alternative for resource-constrained environments."

---

## Slide 8: Results & Comparison (1 minute)
"Our results show strong performance across models. The best contenders are XGBoost for highest F1 (0.8545) but with longest training time, and Logistic Regression/SVM offering competitive accuracy with simpler deployment.

Key highlights: Best F1 of 0.8627 with LightGBM, best accuracy 0.8903 with SGD, fastest training with Naive Bayes at 17 seconds.

[Show charts] The stacked bars show combined performance, while individual charts highlight accuracy and training time trade-offs."

---

## Slide 9: Accuracy vs Training Time (1 minute)
"This scatter plot clearly illustrates the efficiency frontier - the sweet spot between performance and computational cost.

[Point to points] Naive Bayes offers the best value: fast training and high accuracy. SGD, SVM, and Logistic Regression cluster in the upper-middle with good balance. XGBoost requires significant time for moderate gains.

This visualization makes our key finding evident: simple models can deliver excellent results without excessive compute requirements."

---

## Slide 10: Analysis, Challenges & Learnings (1 minute)
"Key observations: Models struggled most with borderline reviews containing ambiguous or mixed sentiment language.

Challenges included label noise from mapping 1-5 star ratings to binary sentiment, and domain-specific vocabulary in long-tail product categories.

Mitigations we tried: threshold tuning, class weighting, resampling, and specialized tokenization.

[Show feature importance and confusion matrix] The feature importance reveals words like 'not', 'great', 'love' drive decisions. The confusion matrix shows strong diagonal performance with limited misclassifications.

Learning: Simple models with strong TF-IDF features perform surprisingly well; ensembles provide marginal gains at high computational cost."

---

## Slide 11: Repository Structure & Reproducibility (1 minute)
"Our project emphasizes reproducibility with clear organization:

Notebooks 05-14 contain all experiments and visualizations.

Src/ holds utilities and configurations.

Data/ stores raw, processed data, and saved models in joblib format.

Docs/ includes research, definitions, and references.

[Show folder structure] Everything runs in a conda environment with requirements.txt. Notebooks execute in sequence, and the comparison notebook provides the full analysis.

This ensures anyone can reproduce our results."

---

## Conclusion & Q&A (30 seconds)
"In conclusion, we successfully demonstrated that traditional ML algorithms can achieve high performance - F1 0.8627, accuracy 0.8903 - for sentiment analysis, offering an efficient alternative to deep learning approaches.

Future work could include active learning, transformer distillation, and multi-lingual analysis.

---

# Total Estimated Time: 10 minutes