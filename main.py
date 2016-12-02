import os

import pandas as pd
import numpy as np

from itertools import permutations
from itertools import combinations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV

from sklearn.pipeline import Pipeline

#text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('lrcv', LogisticRegressionCV())])


def generate_word(word):
    """
    This method generates a random sequence of characters, a word, of the same length
    as the original word provided, excluding characters which it contains
    """
    import random, string
    select_from = ''.join(set(string.ascii_letters) - set(word))
    return ''.join(random.choice(select_from) for _ in range(len(word))).lower()

# A word that we want to train our model on
word = 'elucidate'
# word = 'donghuang'

# Create features from all possible permutations and also create target label
X = [''.join(p) for p in list(permutations(word, len(word)))]
y = [word for e in range(len(X))]

# Add random words generated in a way to exclude characters used in the original word
# also label them as 'unknown'. The noise will account for X% of the permutations
for i in range(round(len(X)*.50)):
    X.append(generate_word(word))
    y.append('unknown')

# Get bag-of-words
vec = CountVectorizer(analyzer='char')
X_train_count = vec.fit_transform(X)

# Transform the bag into TF-IDF
tfidf = TfidfTransformer(use_idf=True).fit(X_train_count)
X_train_tfidf = tfidf.fit_transform(X_train_count),

# Train the model
clf = LogisticRegressionCV().fit(X_train_tfidf, y)

# Predict new
docs_new = ['alucidyte', 'elucidite', 'elusive', 'lucrative', 'account']
X_new_counts = vec.transform(docs_new)
X_new_tfidf = tfidf.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

# Print predicted outcome out
print(predicted)