import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_set import FeatureSet

# some possible CountVectorizer parameters
# binary: try both True and false
# lowercase: may want to try a False version
# max_df: need to play with this parameter
# min_df: need to play with this parameter
# stop_words: something more to consider
# ngram_range: (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)

class TfidfFeatureSet(FeatureSet):

    def __init__(self):
        FeatureSet.__init__(self, verbose=True, useBinaryFileMode=True)

    def _extract(self, outputFile, **kwargs):
        # Import the datasets via the read_json method from pandas
        pizza_data = pd.read_json('data/train.json')
        vectorizer = TfidfVectorizer(**kwargs)

        # turns out that some requests are empty because the requester included all the information in the title
        titlesAndRequests = pizza_data['request_title'] + " " + pizza_data['request_text'] 
        tfidf = vectorizer.fit_transform(titlesAndRequests).toarray()
        
        np.save(outputFile, tfidf)

        return tfidf

    def _load(self, outputFile):
        return np.load(outputFile)

if __name__ == '__main__':
    featureSet = TfidfFeatureSet()
    featureSet.smartLoad()

