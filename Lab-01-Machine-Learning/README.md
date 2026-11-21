# Lab 01 — Machine Learning

## Laboratory Session

Lab 01 — Sentiment Analysis Using Traditional Machine Learning

## Objective

Practice basic NLP tasks: text preprocessing, Bag-of-Words representation, and training a
traditional machine learning classifier (Logistic Regression / Linear SVM) for sentiment
analysis, per [Lab sheet 1](resources/Lab%20sheet%201.pdf).

## Tasks and Implementation

Full implementation: [notebook/lab1_sentiment_analysis.ipynb](notebook/lab1_sentiment_analysis.ipynb)

1. Downloaded and loaded the [UCI Sentiment Labelled Sentences dataset](https://archive.ics.uci.edu/dataset/331/sentiment%2Blabelled%2Bsentences) (Amazon, IMDb, Yelp reviews).
2. Preprocessed text: lowercasing, punctuation removal, whitespace normalisation, with a
   `clean_text` column kept alongside the original for comparison.
3. Explored the data: class balance, example positive/negative reviews, most frequent words per
   class.
4. Split the dataset (80/20, stratified).
5. Built a Bag-of-Words representation with `CountVectorizer`.
6. Trained two models — Logistic Regression and Linear SVM — on the BoW features.
7. Evaluated both models with accuracy, precision, recall and F1.
8. Tested the best model on five new, unseen sentences.

Conceptual background for this lab: [notes.md](notes.md).

## Results and Evidence

- Vocabulary size: 4,516 words; train/test split: 2,198 / 550 sentences.
- **Logistic Regression:** 82.2% accuracy.
- **Linear SVM:** 81.8% accuracy.
- Full classification reports (precision/recall/F1 per class), a confusion matrix, and the
  model's predictions on 5 new unseen sentences are all in the notebook's executed output.

*(Screenshots to be added here.)*

## Observations

- A simple BoW + Logistic Regression pipeline reached 82.2% accuracy on real review text with
  almost no tuning — a good demonstration of why BoW was the default approach for a long time
  before embeddings took over.
- BoW's "no semantics" limitation showed up directly in the word-frequency lists: "good" appears
  as a top word for *both* positive and negative reviews, which a frequency-only model can't
  resolve without surrounding context (e.g. "not good" vs. "really good").
- Logistic Regression and Linear SVM performed almost identically (82.2% vs. 81.8%), which makes
  sense since both are just finding a linear decision boundary over the same BoW features.

Full reflection: [reflection.md](reflection.md).

## Learning Outcomes

- Built a complete, working text-classification pipeline: preprocessing → vectorization →
  training → evaluation → inference on new input.
- Learned to use `CountVectorizer`, `LogisticRegression`, `LinearSVC`, and scikit-learn's
  evaluation utilities (`classification_report`, `confusion_matrix`).
- Understood concretely (not just conceptually) why Bag-of-Words loses word order and semantic
  meaning, and why that eventually motivates moving to embeddings and attention-based models in
  later labs.
