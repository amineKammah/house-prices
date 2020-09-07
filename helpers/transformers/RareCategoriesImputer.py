from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np


class RareCategoriesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.03):
        self.threshold = threshold

        self._frequent_categories = dict()

    def fit(self, X, y=None):
        self._frequent_categories = dict()

        for feature in X.columns:
            frequency = X.groupby([feature])[feature].count() / np.float(len(X))
            frequent_cat = [x for x in frequency.loc[frequency > self.threshold].index.values]

            self._frequent_categories[feature] = frequent_cat

        return self

    def transform(self, X, y=None):
        X = X.copy()
        for feature in X.columns:
            X.loc[:, feature] = np.where(X[feature].isin(self._frequent_categories[feature]), X[feature], 'Rare')

        return X