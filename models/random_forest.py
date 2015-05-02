import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from operator import itemgetter
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
import matplotlib.pyplot as plt
from time import time

from features.tfidf_feature_set import TfidfFeatureSet
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
from sklearn import random_projection

# in order to run me, call me from the root of the project like so:
# python -m models.random_forest

featureSet = TfidfFeatureSet()
'''
for i in range(1, 25, 2):
    print("Using min_df of %d" % i)
    X = featureSet.smartLoad("data_cache/tfidf.npa", min_df=i)
    print(X.shape)

    pizza_data = pd.read_json('data/train.json')
    y = pizza_data['requester_received_pizza']

    rf = RandomForestClassifier()

    oldTime = time()
    rf.fit(X, y)
    print("Training Time: %d" % int(1000.0 * (time() - oldTime)))

    oldTime = time()
    scores = cross_validation.cross_val_score(rf, X, y, cv = 10)
    print("CV Time: %d" % int(1000.0 * (time() - oldTime)))

    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Confusion Matrix")
    print(confusion_matrix(y, rf.predict(X)))
'''

X = featureSet.smartLoad("data_cache/tfidf.npa", min_df=10, ngram_range = (1,1))
transformer = random_projection.GaussianRandomProjection(eps=0.2)
print "Shape before random projection: " + str(X.shape)
X = transformer.fit_transform(X) 
print X
print "Shape after random projection: " + str(X.shape)
pizza_data = pd.read_json('data/train.json')
y = np.array(pizza_data['requester_received_pizza'])

X_positive = X[y]
X_negative = X[np.logical_not(y)]
y_positive = y[y]
y_negative = y[np.logical_not(y)]

(X_positive_train, X_positive_test, y_positive_train, y_positive_test) = \
    cross_validation.train_test_split(X_positive, y_positive, train_size=750, random_state=42)

(X_negative_train, X_negative_test, y_negative_train, y_negative_test) = \
    cross_validation.train_test_split(X_negative, y_negative, train_size=750, random_state=42)

X_train = np.concatenate((X_positive_train, X_negative_train))
X_test = np.concatenate((X_positive_test, X_negative_test))
y_train = np.concatenate((y_positive_train, y_negative_train))
y_test = np.concatenate((y_positive_test, y_negative_test))

np.random.seed(42)

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# build a classifier
clf = RandomForestClassifier(n_estimators=50)

# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, 5, 7, 9, 11, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(1, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search, n_jobs=2)

start = time()
random_search.fit(X_train, y_train)
print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % ((time() - start), n_iter_search))
report(random_search.grid_scores_)
print("Final Score: %.2f" % random_search.score(X_test, y_test))
print("Confusion Matrix")
print(confusion_matrix(y_test, random_search.predict(X_test)))
