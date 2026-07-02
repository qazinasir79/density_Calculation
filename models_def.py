import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin


class PhysicsInspiredModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = LinearRegression()

    def _engineer_features(self, X):
        T, P = X[:, 0], X[:, 1]
        return np.column_stack([
            T, P, P / T, P**2, P**2 / T, np.sqrt(P), 1.0 / T, P * T, T**2, P**3
        ])

    def fit(self, X, y):
        X_feat = self._engineer_features(X)
        self.model.fit(X_feat, y)
        return self

    def predict(self, X):
        X_feat = self._engineer_features(X)
        return self.model.predict(X_feat)
