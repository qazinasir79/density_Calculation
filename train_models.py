import numpy as np
import pandas as pd
import joblib
import os
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge, BayesianRidge, HuberRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from models_def import PhysicsInspiredModel

fluid = sys.argv[1] if len(sys.argv) > 1 else 'ethane'
csv_path = f'data/{fluid}_data.csv'
model_dir = f'models/{fluid}'
os.makedirs(model_dir, exist_ok=True)

df = pd.read_csv(csv_path)
X = df[['T_K', 'P_MPa']].values
y = df['Density_kgm3'].values
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models_def = [
    ('Gradient Boosting', GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        min_samples_leaf=2, random_state=42
    )),
    ('Random Forest', RandomForestRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=2, random_state=42
    )),
    ('Extra Trees', ExtraTreesRegressor(
        n_estimators=500, max_depth=10, min_samples_leaf=2, random_state=42
    )),
    ('HistGradient Boosting', HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.03, max_depth=4,
        min_samples_leaf=2, random_state=42
    )),
    ('AdaBoost', AdaBoostRegressor(
        n_estimators=300, learning_rate=0.05, random_state=42
    )),
    ('Huber Regressor', HuberRegressor(epsilon=1.35, max_iter=1000)),
    ('Polynomial Regression', Pipeline([
        ('poly', PolynomialFeatures(degree=4, include_bias=False)),
        ('ridge', Ridge(alpha=0.1))
    ])),
    ('Physics Inspired', PhysicsInspiredModel()),
    ('Bayesian Ridge', Pipeline([
        ('scaler', StandardScaler()),
        ('br', BayesianRidge())
    ])),
    ('KNN', Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=3, weights='distance'))
    ])),
    ('Gaussian Process', Pipeline([
        ('scaler', StandardScaler()),
        ('gp', GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=[100, 50]) + WhiteKernel(noise_level=1),
            alpha=1e-6, normalize_y=True, random_state=42,
            n_restarts_optimizer=5
        ))
    ])),
    ('SVR', Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
    ])),
]

# CatBoost omitted — no Python 3.14 wheel on Streamlit Cloud

nn_model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32), activation='relu',
    solver='adam', max_iter=10000, random_state=42,
    early_stopping=True, validation_fraction=0.1,
    alpha=0.001, batch_size=8
)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
nn_model.fit(X_scaled, y_scaled)
joblib.dump((nn_model, scaler_X, scaler_y), f'{model_dir}/Neural Network.pkl')
y_pred_nn = scaler_y.inverse_transform(nn_model.predict(X_scaled).reshape(-1, 1)).ravel()

results = []
for name, model in models_def:
    try:
        model.fit(X, y)
        joblib.dump(model, f'{model_dir}/{name}.pkl')
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
    except Exception as e:
        r2, mae, rmse, mape = 0, 0, 0, 0
        print(f"  {name}: FAILED - {e}")
    results.append({
        'Model': name, 'R2_train': r2, 'MAE_train': mae,
        'RMSE_train': rmse, 'MAPE_train': mape
    })

results.append({
    'Model': 'Neural Network',
    'R2_train': r2_score(y, y_pred_nn),
    'MAE_train': mean_absolute_error(y, y_pred_nn),
    'RMSE_train': np.sqrt(mean_squared_error(y, y_pred_nn)),
    'MAPE_train': np.mean(np.abs((y - y_pred_nn) / y)) * 100
})

results_df = pd.DataFrame(results).sort_values('R2_train', ascending=False)
results_df = results_df[results_df['R2_train'] > 0].reset_index(drop=True)
results_df.to_csv(f'{model_dir}/model_performance.csv', index=False)

print(f"=== {fluid.upper()} FULL DATASET PERFORMANCE ===")
print(results_df.to_string(index=False))

print(f"\n=== {fluid.upper()} 5-FOLD CROSS-VALIDATION R² ===")
for name, model in models_def:
    try:
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        print(f"  {name:30s}: {scores.mean():.4f} ± {scores.std():.4f}")
    except Exception as e:
        print(f"  {name:30s}: FAILED - {e}")

from sklearn.base import BaseEstimator, RegressorMixin
class NNWrapper(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu',
                                  solver='adam', max_iter=5000, random_state=42,
                                  early_stopping=True, validation_fraction=0.1,
                                  alpha=0.001, batch_size=8)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
    def fit(self, X, y):
        self.scaler_X.fit(X)
        self.scaler_y.fit(y.reshape(-1, 1))
        X_s = self.scaler_X.transform(X)
        y_s = self.scaler_y.transform(y.reshape(-1, 1)).ravel()
        self.model.fit(X_s, y_s)
        return self
    def predict(self, X):
        X_s = self.scaler_X.transform(X)
        y_s = self.model.predict(X_s)
        return self.scaler_y.inverse_transform(y_s.reshape(-1, 1)).ravel()

scores = cross_val_score(NNWrapper(), X, y, cv=kf, scoring='r2')
print(f"  {'Neural Network':30s}: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"\nModels saved to {model_dir}/")
