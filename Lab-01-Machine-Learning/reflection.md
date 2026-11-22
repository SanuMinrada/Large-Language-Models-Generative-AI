# Week 1 Reflection — NLP Foundations

## What I learned

Doing Lab 1 helped me actually understand text preprocessing instead of just reading about it. Cleaning the text (lowercasing, removing punctuation) and turning it into a Bag-of-Words matrix made a lot more sense once I saw it done on real Amazon, IMDb, and Yelp reviews. I was surprised that such a simple model (BoW + Logistic Regression) got 82.2% accuracy without much tuning at all.

## What was challenging

It was hard to really get why BoW has limitations until I saw it for myself. The word "good" showed up as a top word in both positive and negative reviews, which shows the model can't tell the difference between "good" and "not good" since it ignores word order. I also found it a bit confusing that Logistic Regression and Linear SVM gave almost the same accuracy (82.2% vs 81.8%), but I realised that's because they're both just drawing a straight line to separate the two classes using the same features.

## Model comparison

At the end of the lab, I compared Logistic Regression and Linear SVM side by side in a table instead of just looking at them separately. Logistic Regression came out slightly better overall (82.2% accuracy vs 81.8% for SVM), though the difference is small enough that it's not a huge deal.

What stood out more to me was looking at the actual words each model relied on most. Logistic Regression's top words made a lot of sense — things like "great", "excellent", and "delicious" for positive, and "not", "poor", "bad" for negative. Linear SVM's top words were a lot less obvious, with some pretty random words showing up near the top. So even though both models got a similar accuracy score, Logistic Regression seems to be learning something closer to what a person would actually expect, which made me trust it more.


## What I'd try next

- Try `TfidfVectorizer` instead of `CountVectorizer` and compare the results.
- Try a pretrained embedding-based model to get a real comparison instead of just a conceptual one.
- Look at the actual sentences the model got wrong, not just the overall accuracy number.
