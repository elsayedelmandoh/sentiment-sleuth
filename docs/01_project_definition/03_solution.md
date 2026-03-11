# Project Definition - Solution

**1. Data Preprocessing:**   
Clean the unstructured text to reduce noise. This includes text preprocessing (tokenization, stop-word removal

**2. Feature Engineering (Vectorization):**   
Vectorization using TF-IDF or Bag-of-Words) to convert unstructured text into a format suitable for standard algorithms.


**3. Modeling:**   
Models: We will train and compare the following classifiers:

*1. Naive Bayes:* As our primary baseline due to its speed and effectiveness in text data.   
*2. Logistic Regression:* For its interpretability in binary classification.    
*3. Support Vector Machines (SVM):* To test performance in high-dimensional feature spaces.   
*4. K-Nearest Neighbors (KNN):*
*5. Decision Trees:*   
*6. Random Forest:* To evaluate ensemble methods against linear models.   
*7. Stochastic Gradient Descent (SGD):* For efficient handling of large-scaled data.
*8. XgBoost:* To assess the performance of gradient boosting methods.
*9. LightGBM:* To evaluate the performance of gradient boosting methods.


**4. Evaluation Plan:**   
To rigorously assess performance, we will look beyond simple accuracy. We will use a Confusion Matrix to calculate and analyze:

*- Precision and Recall:* To understand the trade-off between false positives and false negatives.   
*- F1-Score:* To provide a single metric for model balance, which is critical if our subset retains any class imbalance.    
*- Training Time:* We will also log the time required to train each model to quantitatively support our argument regarding computational efficiency.

