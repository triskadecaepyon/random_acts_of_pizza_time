from sklearn.feature_extraction.text import CountVectorizer

class LengthTransformer():

    def get_params(self, deep = True):
        return {}

    def set_params(self, **params):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # simply count the number of words without worrying about stop words of min document frequency
        return CountVectorizer().fit_transform(X).sum(axis = 1)
