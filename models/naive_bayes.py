import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features.ngram_feature_set import NGramFeatureSet
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation

# in order to run me, call me from the root of the project like so:
# python -m models.naive_bayes

featureSet = NGramFeatureSet()
X = featureSet.smartLoad("data_cache/unigrams.npa", binary = True, lowercase = True, ngram_range = (1, 1))
X1 = featureSet.smartLoad("data_cache/bigrams.npa", binary = True, lowercase = True, ngram_range = (2, 2))
X2 = featureSet.smartLoad("data_cache/multigrams.npa", binary = True, lowercase = True, ngram_range = (1, 2))

pizza_data = pd.read_json('data/train.json')
y = pizza_data['requester_received_pizza']

nb = MultinomialNB()

nb.fit(X, y)
scores = cross_validation.cross_val_score(nb, X, y, cv = 10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

nb.fit(X1, y)
scores = cross_validation.cross_val_score(nb, X1, y, cv = 10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

nb.fit(X2, y)
scores = cross_validation.cross_val_score(nb, X2, y, cv = 10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
