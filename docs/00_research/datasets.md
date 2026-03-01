# Research - Datasets

We have selected the Amazon Reviews dataset hosted on Kaggle, which serves as a refined subset of the larger SNAP Amazon dataset.[1]

Citation: Character-level Convolutional Networks for Text Classification.[2]

Dataset: Amazon reviews.[3]

Description: The dataset consists of approximately 1.8M training samples and 200K testing samples. It features three key columns:

- Polarity: The target label (1=negative and 2=positive).
- Title: The review summary.
- Text: The full review body.

Scale: Given the constraints of the Anaconda/Jupyter environment and the 4-week timeline, we will utilize a stratified subset of this data. This ensures we maintain a balanced distribution of classes while keeping training times feasible for iterative experimentation.