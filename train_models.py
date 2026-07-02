import numpy as np
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor,
    AdaBoostRegressor, HistGradientBoostingRegressor, StackingRegressor
)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, HuberRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin

from models_def import PhysicsInspiredModel

df = pd.read_csv('data/ethane_data.csv')
X = df[['T_K', 'P_MPa']].values
y = df['Density_kgm3'].values

os.makedirs('models', exist_ok=True)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

models_def = [
    ('Gradient Boosting', GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.03, max_depth=4,
        min_samples_leaf=2, random_state=42
    )),
]

has_xgb = False
has_lgb = False
has_cat = False
try:
    import xgboost as xgb
    models_def.append(('XGBoost', Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42
        ))
    ])))
    has_xgb = True
except:
    print("XGBoost not available")

try:
    import lightgbm as lgb
    models_def.append(('LightGBM', Pipeline([
        ('scaler', StandardScaler()),
        ('lgb', lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=4,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            verbose=-1
        ))
    ])))
    has_lgb = True
except:
    print("LightGBM not available")

try:
    import catboost as cb
    models_def.append(('CatBoost', cb.CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=4,
        verbose=0, random_seed=42, early_stopping_rounds=50,
        min_data_in_leaf=2
    )))
    has_cat = True
except:
    print("CatBoost not available")

if has_xgb:
    models_def.append(('Stacking Ensemble', StackingRegressor(
        estimators=[
            ('gb', GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)),
            ('xgb', __import__('xgboost').XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42)),
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=3
    )))

models_def += [
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
    ('Polynomial Regression', Pipeline([
        ('poly', PolynomialFeatures(degree=4, include_bias=False)),
        ('ridge', Ridge(alpha=0.1))
    ])),
    ('Physics Inspired', PhysicsInspiredModel()),
    ('SVR', Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1))
    ])),
    ('Gaussian Process', Pipeline([
        ('scaler', StandardScaler()),
        ('gp', GaussianProcessRegressor(
            kernel=ConstantKernel(1.0) * RBF(length_scale=[100, 50]) + WhiteKernel(noise_level=1),
            alpha=1e-6, normalize_y=True, random_state=42,
            n_restarts_optimizer=5
        ))
    ])),
    ('Bayesian Ridge', Pipeline([
        ('scaler', StandardScaler()),
        ('br', BayesianRidge())
    ])),
    ('AdaBoost', AdaBoostRegressor(
        n_estimators=300, learning_rate=0.05, random_state=42
    )),
    ('Huber Regressor', HuberRegressor(epsilon=1.35, max_iter=1000)),
    ('KNN', Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=3, weights='distance'))
    ])),
]

# Neural Network (handled separately due to scaling)
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
joblib.dump((nn_model, scaler_X, scaler_y), 'models/Neural Network.pkl')
y_pred_nn = scaler_y.inverse_transform(nn_model.predict(X_scaled).reshape(-1, 1)).ravel()

# Train and evaluate
results = []
for name, model in models_def:
    try:
        model.fit(X, y)
        joblib.dump(model, f'models/{name}.pkl')
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
results_df.to_csv('models/model_performance.csv', index=False)
print("=== FULL DATASET PERFORMANCE ===")
print(results_df.to_string(index=False))

print("\n=== 5-FOLD CROSS-VALIDATION R² ===")
for name, model in models_def:
    try:
        scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        print(f"  {name:30s}: {scores.mean():.4f} ± {scores.std():.4f}")
    except Exception as e:
        print(f"  {name:30s}: FAILED - {e}")

# NN cross-val
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

print(f"\nModels saved to models/ directory")
