# Project Definition - Report

                    [Sentiment Analysis of Amazon Reviews Using Machine Learning]
                                [CSAI 801] | [Winter 2026] | Group [12]

| Name | ID | Email |
|---|---:|---|
| Elsayed Elmandoua | 20596379 | 25XRVL@queensu.ca |
| Mohamed Hasan | 20595450 | 25QBD3@queensu.ca |
| Mahmoud Abdel Aleem | 20596359 | 25lfmb@queensu.ca |

## 1. Abstract
This project looked into the problem of analyzing a lot of Amazon product reviews at once, since it is no longer possible to do so by hand because of how quickly and how many reviews are posted online. The research evaluated the effectiveness of conventional machine learning techniques relative to deep learning for binary sentiment classification [7]. We used the Amazon Reviews dataset from Kaggle as a test case. It is part of the larger SNAP Amazon corpus. During text preprocessing, we got rid of punctuation, filtered out stop words, tokenized the text, and created TF-IDF vectors based on unigram, bigram, and trigram features from the titles and bodies of the reviews. We examined several classical classifiers, including Naive Bayes, Logistic Regression, Support Vector Machine, K-Nearest Neighbors, Decision Tree, Random Forest, Stochastic Gradient Descent, XGBoost, and LightGBM.

We used stratified sampling to make sure the data was still fair while we were doing the experiments. We did this when we made the working subset and when we split the data into sets for training, validation, and testing. We checked the model's accuracy, precision, recall, and F1-score to see how well it worked. The results showed that optimized classical machine learning models kept their effectiveness and competitiveness. This supports the project's claim that well-tuned ML pipelines can be good alternatives to deep learning methods for sentiment analysis that use more resources.

## 2. Introduction
Sentiment analysis is an important part of how modern e-commerce works. When practitioners can correctly categorize what people say in online reviews, they can keep an eye on how happy customers are, find flaws in products, and help people make decisions. As digital marketplaces grow, it has become harder and more important for businesses to tell the difference between
positive and negative sentiment. Keyword-based or rule-based methods don't work anymore because reviews often use slang, sarcasm, spelling mistakes, and mixed opinions that are hard to capture by hand [1]. Researchers have thus transitioned to machine learning techniques that directly assimilate statistical patterns in text, including term frequency and contextual co-occurrence data extracted from review content [2].

Initial research showed that statistical features at the text level can be used to classify sentiment without having to create rules by hand. This was done by using machine learning methods on review corpora and document representations [1]. Later studies found that more complex text encodings, like character-level and n-gram-based representations, can tell the difference between sentiment classes very well [2]. More recent research is using transformer models and other deep architectures more and more to get the best results on review  lassification tasks. But these systems usually need a lot more computing power, longer training
times, and more complicated deployment pipelines. This project poses a pragmatic inquiry: can a finely calibrated classical machine learning pipeline, utilizing unigram, bigram, and trigram TF-IDF features alongside stratified sampling, maintain competitive performance against deep learning methodologies for binary Amazon review sentiment classification?

## 3. Methodology
### 3.1 Dataset and Preprocessing
Researchers made the Amazon Reviews dataset and hosted it on Kaggle. They used data extraction tools to get text review features from Amazon product review pages. There are two classes in the dataset: Negative (the customer is unhappy with the product) and Positive (the customer is happy with the product). The complete dataset has about 1.8 million training samples and 200,000 testing samples. Because of limits on computing power, a stratified subset of the dataset was used while keeping the class distribution balanced to make sure that model training and evaluation were fair.

There were a number of steps that had to be done in order to preprocess. To stop cross-split leakage between training and testing data, duplicate rows were removed first. Next, we got rid of rows with missing or invalid values, leaving us with a clean dataset that could be used for modeling. Then, the text data was cleaned by changing all the text to lowercase, taking out punctuation, and getting rid of stop words. Tokenization was used to break sentences down into their parts. Then, feature extraction was done using TF-IDF vectorization with unigram, bigram, and trigram representations. This way, the models could learn not only single words that show sentiment, but also short phrases and patterns in the local context. We also looked at other factors, like the length of the review and the frequency of words. The last feature  epresentation was made up of TF-IDF weighted text features taken from the title and body of the review.

### 3.2 Splitting, Scaling, and Dimensionality Reduction
Using stratified random sampling, the cleaned dataset was split into three parts: training (70%), validation (15%), and test (15%).

The original class distribution was kept. To avoid data leakage, all preprocessing transformers, like the TF-IDF vectorizer, were only fitted on the training data and then used on the validation and test sets. TF-IDF's feature vectors were normalized to make sure that all features were scaled the same way. We didn't use PCA or any other method to reduce the number of dimensions in this project because sparse TF-IDF n-gram features already work well with classical text classifiers. The goal of the project was to see how well standard machine learning can still compete with deep models using a simple pipeline.

### 3.3 Class Balancing
The Amazon Reviews dataset is very massive, but if you don't keep the same class proportions when sampling, the experimental subsets can become biased. To make sure that the model training was fair and the evaluation was correct, a stratified sampling strategy was used to make the working subset and split up the data. This method made sure that the training, validation, and test sets all had the same number of positive and negative reviews. This gave the data balance without making up fake observations.

The experimental workflow kept stratification going, which meant that each partition had the same ratio of positive to negative sentiments as the sampled dataset. Because the subset was made in a balanced and controlled way, there was no need for a synthetic oversampling method. This choice kept the evaluation closer to how the chosen machine learning models really work and avoided the risk of adding fake text patterns to the feature space.

### 3.4 Classifiers and Hyperparameter Optimization
The Scikit-learn library was used to evaluate six machine learning classifiers. These models include Logistic Regression as a linear baseline model, Naive Bayes based on probabilistic text classification principles, Random Forest as an ensemble learning method built on multiple decision trees, Support Vector Machine (SVM) which identifies an optimal separating hyperplane in a high-dimensional text space, K-Nearest Neighbors (KNN) which classifies samples based on proximity to neighboring data points, and Decision Tree which performs hierarchical rule-based classification. To get a baseline performance, each model was first trained with the default settings. We then used Grid Search and 5-fold stratified cross-validation to find the best hyperparameters. We used accuracy and F1-score as the main metrics for evaluation. Cross-validation and test-set performance were used to choose the best model. Feature analysis concentrated on significant terms and n-gram patterns that influence sentiment prediction to enhance comprehension of model behavior.

## 4. Experiments
All six model variants followed the same evaluation protocol, which is in line with best practices for supervised learning. This included training on the stratified training set, monitoring model selection on the validation set, and final evaluation on the held-out test set, which was only available once per model to avoid optimistic bias from repeated access. The main measure was the F1-score, which gives a balanced view of how accurate and how well the two sentiment classes remember things. Accuracy was also reported to go along with the F1-score and show how well the classification worked overall. To check for overfitting risk, the generalization gap train F1-score minus test F1-score was calculated for each variant.

Baseline experiments validated the anticipated model hierarchy. Logistic Regression only got an F1 score of 0.86 with a 4% train-test gap, which shows that a simple linear decision boundary doesn't always work for all sentiment patterns. Support Vector Machine showed a clear difference in accuracy F1 (0.90 vs. 0.87), which means it has a strong ability to rank but can only slightly adjust the threshold at the default decision boundary, which is a known problem in noisy text classification problems [3].

Both Random Forest and Gradient Boosting achieved a baseline F1-score of around 0.89, demonstrating that ensemble
tree-based models are effective for this task. The results of hyperparameter optimization were different for different types of classifiers. Random Forest went from 0.88 to 0.91, which is a +0.03 gain and the biggest absolute improvement of all the models.

This shows that the default tree-depth settings were too strict. K-Nearest Neighbors went down a little, which means that the search grid didn't completely cover the best hyperparameter area. The experimental comparison validated the project hypothesis that classical machine learning models, when integrated with stratified sampling and TF-IDF n-gram features, can maintain competitive performance relative to more resource-intensive deep learning methodologies for Amazon review sentiment
analysis.

## 5. Results
Table 1 presents the test-set performance of all nine model variants ranked by test accuracy. The optimized Stochastic Gradient Descent (SGD) model achieved the best overall performance, obtaining a test accuracy of 0.8903 and an F1-score of 0.8902. This result slightly outperformed the next-best configurations, Logistic Regression (0.8898 accuracy) and Support Vector Machine (0.8891 accuracy). Although the margin between these top models is relatively small (approximately 0.05%–0.12%), the findings are important because they show that lightweight classical machine learning can still remain competitive, especially when combined with stratified sampling and unigram, bigram, and trigram TF-IDF features. Table 2 presents the detailed per-class precision, recall, F1-score, and support values for the champion model.


### Table 1 – Model Performance Comparison

| Model | Training Time (s) | Test Accuracy | F1-Score | Hyperparameter Tuning | Notes |
|---|---:|---:|---:|---|---|
| Logistic Regression | 588.67 | 0.8898 | 0.8901 | RandomizedSearchCV (5-fold, 50 iters) | Linear model — C, solver |
| Naive Bayes | 16.78 | 0.8683 | 0.8737 | RandomizedSearchCV (5-fold, 60 iters) | Probabilistic — alpha |
| Support Vector Machine | 85.40 | 0.8891 | 0.8888 | GridSearchCV (3-fold) | LinearSVC — C, loss |
| K-Nearest Neighbors | 414.54 | 0.7716 | 0.7645 | RandomizedSearchCV (3-fold, 20 iters) | Brute-force — k, metric |
| Decision Trees | 21.12 | 0.7398 | 0.7315 | GridSearchCV (3-fold) | Depth-controlled — max_depth |
| Random Forest | 115.07 | 0.8533 | 0.8531 | RandomizedSearchCV (3-fold, 50 iters) | Ensemble — n_estimators |
| Stochastic Gradient Descent (SGD) | 96.59 | 0.8903 | 0.8902 | GridSearchCV (3-fold) | Fast linear — loss, penalty |
| XGBoost | 1713.93 | 0.8525 | 0.8545 | RandomizedSearchCV (3-fold, 20 iters) | Gradient boosting — n_estimators |
| LightGBM | 736.70 | 0.8612 | 0.8627 | RandomizedSearchCV (3-fold, 20 iters) | Leaf-wise — num_leaves |

### Table 2 – Per-Class Performance (Champion Model: SGD)

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Negative | 0.89 | 0.89 | 0.89 | 9966 |
| Positive | 0.89 | 0.89 | 0.89 | 10034 |


A wide range of reviews have mixed opinions and unclear language, which makes it harder to find the negative sentiment class.

The Decision Tree model had a lower overall F1 score of 0.7315, while the K-Nearest Neighbors model had a higher score of 0.7645. This shows that even weaker models had trouble with harder negative examples [1]. The best Stochastic Gradient Descent (SGD) model raised the negative-class F1-score to about 0.89. This means that 89% of the true negative reviews in the test set were correctly identified. This result shows that the behavior of the minority class got better without using synthetic oversampling, just by using careful stratified sampling, balanced partitioning, and stronger feature modeling. The generalization analysis showed that the training F1-score was about 0.89 and the test F1-score was 0.89. This means that there wasn't much of a generalization gap, which is fine for high-dimensional text classification tasks. The results from the validation set (F1-score ≈ 0.89, accuracy ≈ 0.89) were very close to those from the test set. This shows that checking the model's performance several times during selection did not affect the evaluation in any way.

The optimized Support Vector Machine (SVM) had a small difference between accuracy and F1 (0.8891 vs. 0.8888), which is what you would expect when there is a threshold sensitivity problem with classification tasks that use noisy text data [3]. After the fact, feature analysis of the champion model showed that the strongest signals came from very positive and very negative terms in the corpus, especially unigrams and short multiword phrases that were full of sentiment and were shown through bigram and trigram structures. These features clearly show polarity, emotional intensity in customer opinions, and differences in direction between positive and negative reviews. This level of interpretability lets experts compare model predictions with actual patterns of customer feedback, which makes people more likely to trust automated sentiment analysis systems.

## 6. Conclusion
This project illustrated how TF-IDF textual features, when put through a strict preprocessing pipeline and combined with stratified sampling, are enough to separate Amazon customer reviews into two different sentiment groups. The optimized Stochastic Gradient Descent (SGD) model had an F1-score of 0.8902 and an accuracy of 89.03%. This was better than all eight other model configurations that were tested in this study. The most important thing is that the main claim of the project is supported by the experimental results: classical machine learning still works well and can compete with deeper, more resource-intensive models for real-world sentiment analysis tasks.

The three most important technical contributions are:

(1) Stratified sampling kept the class balance across the working subset and all experimental partitions. This was very important for fair learning and reliable evaluation.

(2) The TF-IDF feature transformation worked well for text data with a lot of dimensions. It cut down on noise and made it faster for the model to learn.

(3) systematic Grid Search hyperparameter optimization led to measurable performance improvements across several classifier families, particularly for the SGD and Random Forest models.
Interpretability analysis showed that the model used domain-specific sentiment indicators taken from the review corpus, such as high-impact unigrams, bigrams, and trigrams that were related to positive and negative customer opinion trends. These clear text signals help professionals connect model predictions to patterns they can see in review data.

We need to be aware of a number of limitations. The Amazon Reviews dataset is just a small part of a much bigger dataset, so it might not show all the different ways people use language in real reviews. TF-IDF representations also don't look at how words are related to each other in context, which could make it harder for the model to get a deeper understanding of what they mean.

Finally, the computer's resources limited the size of the hyperparameter search grid that could be used during optimization.

Future research should investigate real-time sentiment classification pipelines capable of analyzing customer reviews as they are generated. The model would be even better at generalizing if it were tested on bigger and more varied review datasets. Also, a direct comparison with modern deep learning architectures like LSTM networks and transformer-based models like BERT would make the main point of this project stronger by showing exactly where classical machine learning is still useful and where deep models start to show real improvements. Finally, machine learning techniques that protect privacy could make it possible to use on a large scale while keeping customer data safe.

## 7. References
[1] Pang, B., & Lee, L. (2008). Opinion mining and sentiment analysis. Foundations and Trends in Information Retrieval, 2 (1–2), pp. 1–135.
https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html?utm_.com

[2] Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level convolutional networks for text classification. Advances in Neural Information Processing Systems, 28, pp. 649–657.https://doi.org/10.48550/arXiv.1509.01626

[3] Liu, B. (2012). Sentiment analysis and opinion mining. Synthesis Lectures on Human Language Technologies, 5(1), pp.
     1–167.https://doi.org/10.2200/S00416ED1V01Y201204HLT016

[4] SNAP. (n.d.). Amazon review data (2018): Stanford Network Analysis Project. Stanford University.
     https://snap.stanford.edu/data/web-Amazon.html

[5] Jain, K. (n.d.). Amazon reviews dataset. Kaggle.
https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/

[6] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, pp. 2825–2830.
     https://doi.org/10.48550/arXiv.1201.0490

[7] Dedeepya, M., Kokatnoor, S. A., & Kumar, S. Sentiment Analysis for Online Shopping Reviews Using Machine Learning.
     https://doi.org/10.1007/978-981-97-7571-2_35



