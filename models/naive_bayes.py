import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

def extractFeatures(document):
    documentWords = set(word_tokenize(document))
    features = {}
    for word in allWords:
        features['contains (%s)' % word] = (word in documentWords)
    return features

def extractFeaturesAndLabel(dataItem):
    return (extractFeatures(dataItem['request_text']), dataItem['requester_received_pizza'])

# necessary for the word tokenizer
nltk.download('punkt')

# Import the datasets via the read_json method from pandas
pizza_data = pd.read_json('../data/train.json')

# the test set doesn't have the 'request_text' columns or the 'requester_received_pizza', so I'm not using it
#pizza_data_test = pd.read_json('../data/test.json')[:20]

allWords = set()
for document in pizza_data['request_text']:
    for word in word_tokenize(document):
        allWords.add(word)

# Tweak these values as desired. Be careful though, this model uses a lot of memory!
trainingSize = 500
testSize = int(trainingSize * 0.1)

pizza_data_train = pizza_data[0:trainingSize].apply(extractFeaturesAndLabel, axis=1)
pizza_data_test = pizza_data[trainingSize:trainingSize + testSize].apply(extractFeaturesAndLabel, axis=1)

classifier = NaiveBayesClassifier.train(pizza_data_train)

print classifier.show_most_informative_features()
print "Model Accuracy: %f" % nltk.classify.accuracy(classifier, pizza_data_test)
