import numpy as np
import pandas as pd
import scipy as sp

from sklearn.grid_search import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn import base

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import make_scorer

from operator import itemgetter
from time import time
import multiprocessing

NOT_LEET = -1337.0

# Utility function to report best scores
def reportNBest(grid_scores, classifier, X_train, y_train, X_test, y_test, scoring_function, n_best):
    topScores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_best]
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

        if scoring_function:
            print("Test score: %.3f" % scoring_function(y_test, y_pred))
        print("Accuracy: %.3f" % classifier.score(X_test, y_test))
        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred))

        print("Classification Report")
        target_names = ['no_pizza', 'pizza']
        print(classification_report(y_test, y_pred, target_names=target_names))

    if not usefulModelResults:
        print("No useful models found!")

    return usefulModelResults

# helper function for separateModelStepsAndParameters
def flattenParametersWithNamespace(parameters, namespace):
    flattenedParameters = {}
    for (parameterName, parameterValue) in parameters.iteritems():
        namespacedParameterName = '%s__%s' % (namespace, parameterName)
        if type(parameterValue) == dict:
            flattenedParameters.update(flattenParametersWithNamespace(parameterValue, namespacedParameterName))
        else:
            flattenedParameters[namespacedParameterName] = parameterValue

    return flattenedParameters

# convenience function for creating model steps and parameters
def separateModelStepsAndParameters(modelStepsAndParameters):
    modelSteps = []
    modelParameters = {}

    for modelStepAndParameters in modelStepsAndParameters:
        name = modelStepAndParameters['name']
        model = modelStepAndParameters['model']
        parameters = modelStepAndParameters['params']

        modelSteps.append((name, model))
        modelParameters.update(flattenParametersWithNamespace(parameters, name))

    return (modelSteps, modelParameters)


# modelSteps: a list of 2-tuples of the form ('name', transformer_or_model)
# modelParameters: a dictionary of parameters, where the keys correspond
#   to the 'name' part of the 2-tuples in modelSteps 
# modelData: a 4-tuple of the form (X_train, X_test, y_train, y_test)
def runModel(
    modelSteps,
    modelParameters,
    modelData,
    n_search_iters = 10, 
    scoring_function = None,
    cv = 5,
    n_best = 3,
    verbose = False
):
    classifierName = modelSteps[-1].__class__.__name__
    classifier = Pipeline(modelSteps)

    print("Running randomized parameter search on %s" % classifierName)

    parameterSearcher = RandomizedSearchCV(
        classifier, 
        param_distributions = modelParameters, 
        n_iter = n_search_iters, 
        n_jobs = multiprocessing.cpu_count(), # multiprocessing FTW!!!
        error_score = NOT_LEET, # Provides robustness against classifiers that fail because of bad parameters
        scoring = make_scorer(scoring_function),
        cv = cv,
        refit = False, # Don't try to refit the best model on all the data. We'll do that on all the models at the end.
        verbose = verbose
    )

    (X_train, X_test, y_train, y_test) = modelData

    startTime = time()
    parameterSearcher.fit(X_train, y_train)
    totalTime = time() - startTime
    ### END TRAIN MODELS ###

    ### OUTPUT RESULTS ###
    gridScores = parameterSearcher.grid_scores_
    print("Randomized search took %.2f seconds for %d candidates parameter settings." % (totalTime, len(gridScores)))
    print("Printing %d best models" % n_best)
    reportNBest(gridScores, classifier, X_train, y_train, X_test, y_test, scoring_function, n_best)
    print("Finished with %s" % classifierName)
    print("")
    print("")
    print("")
    print("")
    # picklize me, Captain!
    ### END OUTPUT RESULTS ###
