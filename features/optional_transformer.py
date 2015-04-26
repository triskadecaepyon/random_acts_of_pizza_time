from sklearn.base import BaseEstimator, TransformerMixin

class OptionalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, **kwargs):
        OptionalTransformer.set_params(self, **kwargs)

    def get_params(self, deep = True):
        params = self.transformer.get_params(deep)
        params['exclude'] = self.exclude
        params['transformer'] = self.transformer

        return params

    def set_params(self, **params):
        if 'exclude' in params:
            self.exclude = params['exclude']
            del params['exclude']
        else:
            self.exclude = False

        if 'transformer' in params:
            self.transformer = params['transformer']
            del params['transformer']

        BaseEstimator.set_params(self.transformer, **params)

    def fit(self, X, y=None):
        if not self.exclude:
            self.transformer.fit(X, y)

        return self

    def transform(self, X):
        if not self.exclude:
            X = self.transformer.transform(X)

        return X

