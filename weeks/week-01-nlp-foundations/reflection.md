# Week 1 Reflection — NLP Foundations

## What I learned

Working through the [notes](notes.md) and then actually building the [Lab 1 sentiment classifier](lab/lab1_sentiment_analysis.ipynb) made the preprocessing pipeline concrete in a way the slides alone didn't: lowercasing, punctuation stripping, and the clean/raw text comparison all made obvious sense once I saw them applied to real Amazon/IMDb/Yelp reviews. The Bag-of-Words representation stopped being an abstract diagram and became a 4,516-column sparse matrix I could actually inspect. I was a little surprised that a simple BoW + Logistic Regression pipeline reached 82.2% accuracy on real review text with almost no tuning — it's a good demonstration of why BoW was the default approach for so long before embeddings took over.

## What was challenging

Understanding BoW's limitations *in the abstract* (no semantics, no word order) was easy to nod along to in the slides, but it only really clicked when I looked at the top-word lists from the lab: "good" shows up as a top word for *both* positive and negative reviews (rank 2 positive, rank 10 negative), which is exactly the kind of ambiguity a frequency-only model can't resolve on its own — it needs the surrounding context ("not good" vs "really good"), which BoW throws away. Also found it slightly non-intuitive at first that Logistic Regression and Linear SVM performed almost identically (82.2% vs 81.8%) — had to remind myself that both are just finding a linear decision boundary over the same BoW features, so similar performance is expected rather than a coincidence.

## How it connects to earlier weeks

This is the first week of the module, so there isn't an earlier week to connect to yet — but the lecture slides explicitly set up where this is heading: the "What comes next?" slide after the BoW limitations (N-grams → TF-IDF → embeddings) is clearly foreshadowing Week 2's attention mechanism, which is the real fix for "no word order" and "no semantics."

## Questions I still have

- How much would switching from BoW to TF-IDF actually move the 82% accuracy on this dataset — is the gap large or marginal for a dataset this small (3,000 sentences)?
- The lecture mentioned sub-word tokenization avoids out-of-vocabulary problems — how big a difference would that make on a dataset like this, where the vocabulary is already small and mostly everyday English?
- Why did Linear SVM very slightly underperform Logistic Regression here specifically — is that just this dataset/split, or a more general pattern for short-text sentiment tasks?

## What I'd try next

- Re-run Lab 1 with a `TfidfVectorizer` instead of `CountVectorizer` and compare metrics directly.
- Try a pretrained embedding-based baseline (e.g. averaged word vectors) to get a concrete before/after number for "BoW vs. embeddings," rather than just the conceptual comparison from the slides.
- Look more closely at the misclassified test examples (not just the aggregate accuracy) to see what kinds of sentences the BoW model actually gets wrong.
