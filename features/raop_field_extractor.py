import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class RAOPFieldExtractor(BaseEstimator):

    def __init__(self, raopFields):
        self.raopFields = raopFields

    def fit(self, X, y = None, **fit_params):
        return self

    def transform(self, X):
        return X[self.raopFields]
