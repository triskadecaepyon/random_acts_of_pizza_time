import numpy as np
import pandas as pd
import scipy as sp

from sklearn import random_projection
from sklearn.preprocessing import scale

from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.qda import QDA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import TruncatedSVD

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, matthews_corrcoef, make_scorer
from sklearn.preprocessing import StandardScaler

from features.optional_transformer import OptionalTransformer

from operator import itemgetter
from time import time
import multiprocessing
from sklearn import base

NOT_LEET = -1337.0

# in order to run me, call me from the root of the project like so:
# python -m models.{file_name_without_extension}

# Utility function to report best scores
def reportNBest(grid_scores, classifier, X_train, y_train, X_test, y_test, n):
    topScores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n]
    usefulModelResults = [] # A useful model is one that does better than random => matthew_coeff > 0
    for i, score in enumerate(topScores):
        if score.mean_validation_score <= 0.0:
            break # No need to look at anymore models

        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(score.mean_validation_score, np.std(score.cv_validation_scores)))

        parameters = score.parameters
        print("Parameters: {0}".format(parameters))
        
        classifier = base.clone(classifier)
        classifier.set_params(**parameters)

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        usefulModelResults.append(y_pred)
        print("Test score (matthews_corrcoef): %.3f" % classifier.score(X_test, y_test))

        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report")
        target_names = ['no_pizza', 'pizza']
        print(classification_report(y_test, y_pred, target_names=target_names))

    if not usefulModelResults:
        print("No useful models found!")

    return usefulModelResults

def groupAndRenameParams(*baseParamsList, **stepParametersMap):
    params = {}
    for baseParams in baseParamsList:
        params.update(baseParams)

    for stepName in stepParametersMap:
        parameterValuesMap = stepParametersMap[stepName]
        for parameterName in parameterValuesMap:
            parameterValues = parameterValuesMap[parameterName]
            params['%s__%s' % (stepName, parameterName)] = parameterValues

    return params

#np.random.seed(1337)

### LOAD DATA ###
pizza_data = pd.read_json('data/train.json')
X = pizza_data['request_title'] + ' ' + pizza_data['request_text_edit_aware']
y = np.array(pizza_data['requester_received_pizza'])

X_positive = X[y]
y_positive = y[y]
X_negative = X[np.logical_not(y)]
y_negative = y[np.logical_not(y)]


(X_negative, X_negative_remainder, y_negative, y_negative_remainder) = \
    train_test_split(X_negative, y_negative, train_size=len(X_positive))

X = np.concatenate((X_positive, X_negative))
y = np.concatenate((y_positive, y_negative))

(X_train, X_test, y_train, y_test) = \
    train_test_split(X, y, train_size=0.75)

print("Training set size: %d" % X_train.shape[0])
print("Test set size: %d" % X_test.shape[0])

### END LOAD DATA ###


### DEFINE MODELS AND PARAMETER GRIDS ###
transformerParams = groupAndRenameParams(
    tfidf = {
        'stop_words': ['english'],
        #'max_df': sp.stats.uniform(0.2, 0.5),
        'min_df': [5],
        'lowercase': [True],
        'sublinear_tf': [True, False],
        'analyzer': ['word'],
        'ngram_range': [(1, 1), (1, 2)]
    },
    dimred__kbest = {
        'score_func': [f_classif, chi2],
        'k': [50],
    },
    dimred__tsvd = {
        'n_components': [150],
    },
)

positivePrior = y.sum() / float(len(y))
priors = [positivePrior, 1.0 - positivePrior]

models = {
    #GaussianNB(): {
    #},
    NuSVC(): {
        "nu": sp.stats.uniform(0.1, 1.0),
        "degree": sp.stats.randint(3, 11),
        "kernel": ['poly', 'rbf', 'sigmoid']
    },
    #LogisticRegression(): {
    #    "dual": [True, False],
    #    "penalty": ['l1', 'l2'],
    #    "solver": ['newton-cg', 'lbfgs', 'liblinear'],
    #    "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    #},
    RandomForestClassifier(): {
        "n_estimators": sp.stats.randint(10, 50),
        "max_depth": sp.stats.randint(1, 15),
        "max_features": sp.stats.randint(1, 15),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    },
    AdaBoostClassifier(): {
        "base_estimator": NuSVC(),
        "algorithm": ['SAMME.R'],
        "n_estimators": [200]
    },
    AdaBoostClassifier(): {
        "base_estimator": DecisionTreeClassifier(),
        "algorithm": ['SAMME'],
        "n_estimators": [200]
    },
    KNeighborsClassifier(): {

    }
}


### END DEFINE MODELS AND PARAMETER GRIDS ###


### TRAIN MODELS ###
usefulModelResults = []
n_iter_search = 8
for model in models:
    print("Performing GridSearch on %s" % model.__class__.__name__)

    parameters = models[model]
    steps = [
        ('tfidf', TfidfVectorizer()),
        ('dimred', FeatureUnion([('kbest', SelectKBest()), ('tsvd', TruncatedSVD())])),
        ('model', model)
    ]
    clf = Pipeline(steps)

    params = groupAndRenameParams(transformerParams, model = parameters)

    searchGrid = RandomizedSearchCV(
        clf, 
        param_distributions = params, 
        n_iter = n_iter_search, 
        n_jobs = multiprocessing.cpu_count(), # multiprocessing FTW!!!
        error_score = NOT_LEET, # Provides robustness against classifiers that fail because of bad parameters
        scoring = make_scorer(matthews_corrcoef),
        cv = 5,
        refit = False, # Don't try to refit the best model on all the data. We'll do that on all the models at the end.
        verbose = True
    )

    startTime = time()
    searchGrid.fit(X_train, y_train)
    totalTime = time() - startTime
    ### END TRAIN MODELS ###

    ### OUTPUT RESULTS ###
    grid_scores = searchGrid.grid_scores_
    print("GridSearch took %.2f seconds for %d candidates parameter settings." % (totalTime, len(grid_scores)))
    print("Printing N best models")
    usefulModelResults.extend(reportNBest(grid_scores, clf, X_train, y_train, X_test, y_test, n = 3))
    print("Finished with %s" % model.__class__.__name__)
    print("")
    print("")
    print("")
    print("")
    # picklize me, Captain!


finalResults = sp.stats.mode(usefulModelResults)

print("Put it all together!")
print("Confusion Matrix")
print(confusion_matrix(y_test, finalResults[0][0]))

print("Classification Report")
target_names = ['no_pizza', 'pizza']
print(classification_report(y_test, finalResults[0][0], target_names=target_names))

    ### END OUTPUT RESULTS ###
