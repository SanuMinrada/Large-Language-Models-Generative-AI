# Lab 01: Sentiment Analysis Using Traditional Machine Learning

## Laboratory Session
Lab 01, Sentiment Analysis Using Traditional Machine Learning (PUSL 3205)

## Objective
Practise basic NLP tasks: text preprocessing, Bag-of-Words feature extraction,
and training and evaluating a traditional ML classifier for sentiment analysis.

## Tasks and Implementation
- Dataset: UCI Sentiment Labelled Sentences (Amazon, IMDB and Yelp; 1,000 sentences each, binary labels)
- Preprocessing: lowercase, remove punctuation/special characters, remove extra whitespace, `clean_text` column
- Exploration: examples, word frequencies, positive vs negative comparison
- Split: 80/20, stratified, random_state=42
- Features: Bag-of-Words (`CountVectorizer`)
- Model: Logistic Regression and Linear SVM, compared side by side (plus a top-coefficients comparison as an extension)
- Evaluation: accuracy, precision, recall, F1
- Notebook: [notebook/lab1_sentiment_analysis.ipynb](notebook/lab1_sentiment_analysis.ipynb)

## Results and Evidence
- Vocabulary size: 4,516 words; train/test split: 2,198 / 550 sentences.
- Logistic Regression: 82.2% accuracy. Linear SVM: 81.8% accuracy.
- Extension: a side-by-side comparison table (accuracy, precision, recall, F1 for both models) and
  each model's top positive/negative coefficient words.
- Full classification reports, confusion matrix, and predictions on 5 new unseen sentences are in
  the notebook's executed output.
- Screenshots: [screenshots/](screenshots/) (01–14, covering the full pipeline plus the comparison extension).

## Observations
- A simple BoW + Logistic Regression pipeline reached 82.2% accuracy with almost no tuning — a
  good demonstration of why BoW was the default approach before embeddings took over.
- BoW's lack of semantics showed up directly: "good" appears as a top word for *both* positive and
  negative reviews, since a frequency-only model can't use context like "not good" vs "really good".
- Logistic Regression and Linear SVM scored almost identically (82.2% vs 81.8%), which makes sense
  since both just find a linear boundary over the same BoW features.
- Looking at each model's top coefficients, Logistic Regression's were far more intuitive
  ("great", "excellent", "poor", "bad") than Linear SVM's, which picked up some less obvious words —
  so similar accuracy doesn't necessarily mean similarly sensible decision-making underneath.


## Learning Outcomes
- Built a complete text-classification pipeline: preprocessing → vectorization → training →
  evaluation → inference on new input.
- Used `CountVectorizer`, `LogisticRegression`, `LinearSVC`, and scikit-learn's evaluation tools
  (`classification_report`, `confusion_matrix`, `precision_recall_fscore_support`).
- Understood concretely why Bag-of-Words loses word order and meaning, motivating the move to
  embeddings and attention-based models in later labs.
- Learned that comparing models on accuracy alone can hide real differences — inspecting a model's
  learned coefficients can reveal whether it's picking up sensible signal or not.
