import os

import pandas as pd
import numpy as np

from itertools import permutations
from itertools import combinations

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegressionCV

from sklearn.pipeline import Pipeline

import pickle


def generate_word(word):
    """
    This method generates a random sequence of characters, a word, of the same length
    as the original word provided, excluding characters which it contains
    """
    import random, string
    select_from = ''.join(set(string.ascii_letters) - set(word))
    return ''.join(random.choice(select_from) for _ in range(len(word))).lower()

model_save = False
model_retrieve = True

model_filename = 'clf_model.sav'
vec_filename = 'vec_model.sav'
tfidf_filename = 'tfidf_model.sav'

# A word that we want to train our model on
word = 'elucidate'

# Either train or retrieve model
if model_retrieve:
    vec = pickle.load(open(vec_filename, 'rb'))
    tfidf = pickle.load(open(tfidf_filename, 'rb'))
    clf = pickle.load(open(model_filename, 'rb'))

else:
    # Create features from all possible permutations and also create target label
    X = [''.join(p) for p in list(permutations(word, len(word)))]
    y = [word for e in range(len(X))]

    # Add random words generated in a way to exclude characters used in the original word
    # also label them as 'unknown'. The noise will account for X% of the permutations
    for i in range(round(len(X)*.50)):
        X.append(generate_word(word))
        y.append('unknown')

    print(sorted(X))

    # Get bag-of-words
    vec = CountVectorizer(analyzer='char')
    X_train_count = vec.fit_transform(X)

    # Transform the bag into TF-IDF
    tfidf = TfidfTransformer(use_idf=True).fit(X_train_count)
    X_train_tfidf = tfidf.fit_transform(X_train_count)

    # Train the model or retrieve
    clf = LogisticRegressionCV().fit(X_train_tfidf, y)


# save the model to disk, if necessary
if not model_retrieve:
    pickle.dump(clf, open(model_filename, 'wb'))
    pickle.dump(vec, open(vec_filename, 'wb'))
    pickle.dump(tfidf, open(tfidf_filename, 'wb'))

# Predict new
docs_new = ['alucidyte', 'elucidite', 'elusive', 'elusivade', 'lucrative', 'account']
X_new_counts = vec.transform(docs_new)

X_new_tfidf = tfidf.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)

# Print predicted outcome out
print(predicted)