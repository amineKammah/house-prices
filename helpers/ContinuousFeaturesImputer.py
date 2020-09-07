from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

import numpy as np


class ContinuousFeaturesImputer(BaseEstimator, TransformerMixin):
    def __init__(self, create_is_0=False, impute_zeros=False, drop_features=False, keep_outliers=False,
                 sensitivity=2.8):
        self.create_is_0 = create_is_0
        self.impute_zeros = impute_zeros
        self.drop_features = drop_features
        self.keep_outliers = keep_outliers
        self.sensitivity = sensitivity

        self._high_percentage_0 = []
        self._power_transformer = None
        self._means = dict()
        self._z_scores = dict()

    def fit(self, X, y):
        X = X.copy()

        self._high_percentage_0 = X.columns[((X == 0).sum() / X.shape[0]) > 0.3]

        if self.create_is_0:
            X[self._high_percentage_0 + '_is_0'] = X[self._high_percentage_0] == 0

        if self.impute_zeros:
            self._means = {}
            for feature in self._high_percentage_0:
                mean = X[feature][X[feature] != 0].mean()
                X.loc[:, feature] = X.loc[:, feature].replace(0, mean)
                self._means[feature] = mean

        if self.drop_features:
            X = X.drop(self._high_percentage_0, axis=1)

        self._power_transformer = PowerTransformer().fit(X)

        if not self.keep_outliers:
            self._z_scores = {}
            for feature in X.columns:
                mean, std = X[feature].mean(), X[feature].std()
                z_score = (X[feature] - mean) / std
                self._z_scores[feature] = z_score

        return self

    def transform(self, X, y):
        X = X.copy()

        if self.create_is_0:
            X[self._high_percentage_0 + '_is_0'] = X[self._high_percentage_0] == 0

        if self.impute_zeros:
            for feature in self._high_percentage_0:
                X.loc[:, feature] = X[feature].replace(0, self._means[feature])

        if self.drop_features:
            X = X.drop(self._high_percentage_0, axis=1)

        X = self._power_transformer.transform(X)

        if not self.keep_outliers:
            for feature in X.columns:
                X.loc[:, feature] = np.clip(X[feature], - self.sensitivity * self._z_scores[feature],
                                            self.sensitivity * self._z_scores[feature])

        return X
