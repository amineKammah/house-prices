from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

import numpy as np

import pandas as pd


class Imputer(BaseEstimator, TransformerMixin):
    def __init__(self, missing_values=np.nan, strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value
        self.verbose = verbose
        self.copy = copy
        self.add_indicator = add_indicator

        self._imputers = {}

    def fit(self, X, y=None):
        self._imputers = {}
        for feature in X.columns:
            self._imputers[feature] = SimpleImputer(
                missing_values=self.missing_values, strategy=self.strategy, fill_value=self.fill_value, verbose=self.verbose, copy=self.copy, add_indicator=self.add_indicator
            ).fit(X.loc[:, [feature]])

        return self

    def transform(self, X, y=None):
        for feature in X.columns:
            transformed = self._imputers[feature].transform(X.loc[:, [feature]])
            if transformed.shape[1] == 2:
                X.loc[:, feature + '_is_na'] = 0
                X.loc[:, [feature, feature + '_is_na']] = pd.DataFrame(transformed.copy(), columns=[feature, feature + '_is_na'], index=X.index)
            elif transformed.shape[1] == 1:
                X.loc[:, feature] = self._imputers[feature].transform(X[[feature]]).flatten()
            else:
                raise ValueError

        return X