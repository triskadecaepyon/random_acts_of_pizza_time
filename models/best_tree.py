import numpy as np
import pandas as pd
import scipy as sp

from features.raop_field_extractor import RAOPFieldExtractor
from features.length_transformer import LengthTransformer
from features.raop_numerical_field_extractor import RAOPNumericalFieldExtractor
from models import model_runner

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split, StratifiedKFold, StratifiedShuffleSplit

from sklearn.metrics import matthews_corrcoef

from sklearn.pipeline import Pipeline

np.random.seed(1337)

removeRecordsWithZeroLengthRequests = False
useEqualNumberOfPositiveAndNegativeResults = True
useStratifiedSplit = False

### LOAD DATA ###
pizza_data = pd.read_json('data/train.json')
X = pizza_data
y = np.array(pizza_data['requester_received_pizza'])

if removeRecordsWithZeroLengthRequests:
    length = np.array(LengthTransformer().transform(X['request_text_edit_aware']))
    X = X[length > 0]
    y = y[(length > 0).reshape((1, -1))[0]]

if useEqualNumberOfPositiveAndNegativeResults:
    X_positive = X[y]
    y_positive = y[y]
    X_negative = X[np.logical_not(y)]
    y_negative = y[np.logical_not(y)]

    (X_negative, X_negative_remainder, y_negative, y_negative_remainder) = \
        train_test_split(X_negative, y_negative, train_size=len(X_positive))

    X = pd.concat([X_positive, X_negative])
    y = np.concatenate([y_positive, y_negative])
#
if useStratifiedSplit:
    sss = StratifiedShuffleSplit(y, n_iter=1, train_size=0.75)
    for trainIndex, testIndex in sss:
        modelData = (X.iloc[trainIndex], X.iloc[testIndex], y[trainIndex], y[testIndex])
else:
    #(X_train, X_test, y_train, y_test)
    modelData = train_test_split(X, y, train_size=0.75)



print("Training set size: %d" % modelData[0].shape[0])
print("Test set size: %d" % modelData[1].shape[0])
### END LOAD DATA ###


### DEFINE MODEL STEPS AND PARAMETERS ###
modelStepsAndParameters = [
    {
        'name': 'feature_combiner',
        'model': FeatureUnion(
            [
                ('title_grams', Pipeline(
                    [
                        ('extract', RAOPFieldExtractor('request_title')),
                        ('count', CountVectorizer()),
                        ('dimred', TruncatedSVD())
                    ]
                )),
                ('best_words', Pipeline(
                    [
                        ('extract', RAOPFieldExtractor('request_text_edit_aware')),
                        ('tfidf', TfidfVectorizer()),
                        ('kbest', SelectKBest())
                    ]
                )),
                ('text_length', Pipeline(
                    [
                        ('extract', RAOPFieldExtractor('request_text_edit_aware')),
                        ('length', LengthTransformer())
                    ]
                )),
                ('numerical_fields', Pipeline(
                    [
                        ('extract', RAOPNumericalFieldExtractor())
                    ]
                ))
                # Add more features here!
            ]
        ),
        'params': {
            'title_grams': {
                'count': {
                    'analyzer': ['char'],
                    'ngram_range': [(2, 2)]
                },
                'dimred': {
                    'n_components': [50]
                }
            },
            'best_words': {
                'tfidf': {
                    'stop_words': ['english'],
                    'max_df': sp.stats.uniform(0.2, 0.5),
                    'min_df': [5],
                    'lowercase': [True],
                    'sublinear_tf': [True, False],
                    'analyzer': ['word'],
                    'ngram_range': [(1, 1), (1, 2)]
                },
                'kbest': {
                    'score_func': [f_classif, chi2],
                    'k': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200]
                }
            }
        },
    },
    {
        'name': 'decision_tree',
        'model': RandomForestClassifier(n_estimators=25),
        'params': {
            "max_depth": sp.stats.randint(1, 15),
            "max_features": sp.stats.randint(1, 15),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }
    }
]
(modelSteps, modelParameters) = model_runner.separateModelStepsAndParameters(modelStepsAndParameters)

### END DEFINE MODEL STEPS AND PARAMETERS ###


### RUN MODEL ###

# Optional parameters for runModel()
#   n_search_iters = 10, 
#   scorer = None,
#   n_folds = 5,
#   n_best = 3,
#   verbose = False
bestModel = model_runner.runModel(
    modelSteps, 
    modelParameters, 
    modelData,
    scoring_function = matthews_corrcoef,
    n_search_iters = 5,
    n_best = 10,
    cv = 5, #StratifiedKFold(modelData[2], n_folds=5),
    verbose = True
) 

# Run the best model on the test set and lets see what we get on Kaggle!
pizza_data_test = pd.read_json('data/test.json')
bestModel.fit(X, y)
prediction = bestModel.predict(pizza_data_test)
pizza_data_test['requester_received_pizza'] = prediction.astype(int)
pizza_data_test.to_csv('tree_final_results.csv', columns=['request_id', 'requester_received_pizza'], index=False)



### END RUN MODEL ###


