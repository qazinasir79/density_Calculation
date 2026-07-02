import numpy as np
import pandas as pd
import joblib
import os
import warnings
import json
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    explained_variance_score, max_error, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, KFold
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from models_def import PhysicsInspiredModel

df = pd.read_csv('data/ethane_data.csv')
X = df[['T_K', 'P_MPa']].values
y_true = df['Density_kgm3'].values

MODEL_FILES = sorted([
    f.replace('.pkl', '') for f in os.listdir('models')
    if f.endswith('.pkl') and f != 'model_performance.csv'
])

def load_model(name):
    if name == 'Neural Network':
        model, sx, sy = joblib.load(f'models/{name}.pkl')
        return model, sx, sy, 'nn'
    return joblib.load(f'models/{name}.pkl'), None, None, 'sklearn'

def predict(name, Xin):
    obj = load_model(name)
    if obj[3] == 'nn':
        return obj[2].inverse_transform(obj[0].predict(obj[1].transform(Xin)).reshape(-1, 1)).ravel()
    return obj[0].predict(Xin)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name in MODEL_FILES:
    y_pred = predict(name, X)
    residuals = y_pred - y_true
    rel_err = residuals / y_true

    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - 2 - 1)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs(rel_err)) * 100
    max_err = max_error(y_true, y_pred)
    bias = np.mean(residuals)
    ev = explained_variance_score(y_true, y_pred)

    n = len(y_true); p = 2
    sse = np.sum(residuals**2)
    aic = n * np.log(sse/n) + 2 * p
    bic = n * np.log(sse/n) + p * np.log(n)

    cv_scores = []
    try:
        obj = load_model(name)
        if obj[3] == 'nn':
            from sklearn.base import BaseEstimator, RegressorMixin
            class W(BaseEstimator, RegressorMixin):
                def fit(self, X, y): self.m, self.sx, self.sy = load_model(name)[:3]; return self
                def predict(self, X): return self.sy.inverse_transform(self.m.predict(self.sx.transform(X)).reshape(-1,1)).ravel()
            cv_scores = cross_val_score(W(), X, y_true, cv=kf, scoring='r2')
        else:
            cv_scores = cross_val_score(obj[0], X, y_true, cv=kf, scoring='r2')
    except:
        cv_scores = [np.nan] * 5

    r_p, _ = pearsonr(y_true, y_pred)
    r_s, _ = spearmanr(y_true, y_pred)

    sw_stat, sw_p = stats.shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)
    jb_stat, jb_p = stats.jarque_bera(residuals)
    bp_stat, bp_p = 0, 1.0
    try:
        from sklearn.linear_model import LinearRegression
        X_bp = np.column_stack([X, X**2])
        bp_mod = LinearRegression().fit(X_bp, residuals**2)
        bp_stat = bp_mod.score(X_bp, residuals**2) * len(residuals)
        bp_p = 1 - stats.chi2.cdf(bp_stat, X_bp.shape[1] - 1)
    except:
        pass

    rmspe = np.sqrt(np.mean((rel_err)**2)) * 100
    mad = np.median(np.abs(residuals - np.median(residuals)))

    q25, q75 = np.percentile(residuals, [25, 75])
    iqr = q75 - q25

    within_1sigma = np.mean(np.abs(residuals - bias) < np.std(residuals)) * 100
    within_2sigma = np.mean(np.abs(residuals - bias) < 2 * np.std(residuals)) * 100

    results.append({
        'Model': name,
        'R²': round(r2, 6),
        'Adj. R²': round(adj_r2, 6),
        'R (Pearson)': round(r_p, 6),
        'ρ (Spearman)': round(r_s, 6),
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE (%)': round(mape, 4),
        'RMSPE (%)': round(rmspe, 4),
        'Max Error': round(max_err, 4),
        'Bias': round(bias, 4),
        'MAD': round(mad, 4),
        'IQR of Residuals': round(iqr, 4),
        'Explained Var': round(ev, 6),
        'AIC': round(aic, 2),
        'BIC': round(bic, 2),
        'CV R² mean': round(np.nanmean(cv_scores), 4),
        'CV R² std': round(np.nanstd(cv_scores), 4),
        'Shapiro-Wilk W': round(sw_stat, 4),
        'Shapiro-Wilk p': round(sw_p, 6),
        'Jarque-Bera p': round(jb_p, 6),
        'BP p-value': round(bp_p, 6),
        'Within 1σ (%)': round(within_1sigma, 2),
        'Within 2σ (%)': round(within_2sigma, 2),
    })

res_df = pd.DataFrame(results).sort_values('R²', ascending=False).reset_index(drop=True)
res_df.to_csv('models/full_analysis.csv', index=False)
print("Full analysis saved to models/full_analysis.csv")
