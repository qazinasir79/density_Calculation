import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
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
from scipy import stats
from models_def import PhysicsInspiredModel
from eos_models import density_PR, density_SRK, FLUIDS
import warnings
warnings.filterwarnings('ignore')

FLUID_NAMES = list(FLUIDS.keys())
EOS_MODELS = ['Peng-Robinson EOS', 'SRK EOS']

if 'fluid' not in st.session_state:
    st.session_state.fluid = FLUID_NAMES[0]

def get_fluid_name():
    f = st.session_state.fluid
    if 'methane' in f.lower():
        return 'methane'
    return 'ethane'

def load_data():
    return pd.read_csv(f'data/{get_fluid_name()}_data.csv')

@st.cache_resource
def get_models(_X, _y, fluid_name):
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
    trained = {}
    for name, model in models_def:
        try:
            model.fit(_X, _y)
            trained[name] = model
        except Exception as e:
            st.warning(f'{name} failed: {e}')
    nn_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', max_iter=10000, random_state=42,
        early_stopping=True, validation_fraction=0.1,
        alpha=0.001, batch_size=8
    )
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(_X)
    y_scaled = scaler_y.fit_transform(_y.reshape(-1, 1)).ravel()
    nn_model.fit(X_scaled, y_scaled)
    trained['Neural Network'] = (nn_model, scaler_X, scaler_y)
    return trained

def get_model_list(trained):
    return EOS_MODELS + sorted(trained.keys())

def predict(name, X_input, trained, fluid_name):
    if name in EOS_MODELS:
        T = X_input[:, 0]
        P = X_input[:, 1]
        if name == 'Peng-Robinson EOS':
            return np.array([density_PR(t, p, fluid=fluid_name) for t, p in zip(T, P)])
        else:
            return np.array([density_SRK(t, p, fluid=fluid_name) for t, p in zip(T, P)])
    if name == 'Neural Network':
        model, sx, sy = trained[name]
        return sy.inverse_transform(model.predict(sx.transform(X_input)).reshape(-1, 1)).ravel()
    return trained[name].predict(X_input)

def metrics(y_t, y_p):
    mask = ~np.isnan(y_p)
    if mask.sum() == 0:
        return 0, 0, 0, 0, 0, 0
    return (r2_score(y_t[mask], y_p[mask]),
            mean_absolute_error(y_t[mask], y_p[mask]),
            np.sqrt(mean_squared_error(y_t[mask], y_p[mask])),
            np.mean(np.abs((y_t[mask] - y_p[mask]) / y_t[mask])) * 100,
            np.max(np.abs(y_t[mask] - y_p[mask])),
            np.mean(y_p[mask] - y_t[mask]))

st.set_page_config(page_title="Density Prediction Platform", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,600;9..40,700&family=Inter:wght@400;500;600&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

h1, h2, h3, h4, h5, h6 { font-family: 'DM Sans', sans-serif; font-weight: 600; letter-spacing: -0.02em; }

.stApp { background-color: #F5F7FA; }

div[data-testid="metric-container"] {
    background: #FFFFFF; border: 1px solid #D1D9E6; border-radius: 8px;
    padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="metric-container"] > label {
    font-family: 'Inter', sans-serif; font-size: 12px; font-weight: 500;
    color: #5B6F8C; text-transform: uppercase; letter-spacing: 0.05em;
}
div[data-testid="metric-container"] > div {
    font-family: 'DM Sans', sans-serif; font-size: 28px; font-weight: 700; color: #1B2A4A;
}

button[data-baseweb="tab"] {
    font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500; color: #5B6F8C;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #2B5EA7; border-bottom: 2px solid #2B5EA7;
}

.stButton > button {
    font-family: 'Inter', sans-serif; font-weight: 500; border-radius: 6px;
    background: #2B5EA7; color: white; border: none;
}
.stButton > button:hover { background: #1B4A8A; }

div[data-baseweb="select"] > div { border-color: #D1D9E6; border-radius: 6px; }

.stCheckbox label { font-family: 'Inter', sans-serif; font-size: 13px; color: #5B6F8C; }

.stNumberInput input { border-color: #D1D9E6; border-radius: 6px; font-family: 'Inter', sans-serif; }

.js-plotly-plot { border: 1px solid #D1D9E6; border-radius: 8px; background: #FFFFFF; padding: 8px; }

section[data-testid="stSidebar"] { background-color: #FFFFFF; border-right: 1px solid #D1D9E6; }

.header-title {
    font-family: 'DM Sans', sans-serif; font-size: 32px; font-weight: 700;
    color: #1B2A4A; letter-spacing: -0.03em; margin: 0; padding: 0; line-height: 1.2;
}
.header-subtitle {
    font-family: 'Inter', sans-serif; font-size: 14px; color: #5B6F8C;
    margin: 4px 0 0 0; padding: 0;
}

.verdict-badge {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    background: #FFFFFF; border: 1px solid #D1D9E6; border-left: 4px solid #C7512E;
    border-radius: 8px; padding: 12px 20px; margin: 12px 0; font-size: 14px;
}
.verdict-badge strong { color: #1B2A4A; font-family: 'DM Sans', sans-serif; }
.verdict-badge span { color: #5B6F8C; }

.sidebar-header {
    font-family: 'DM Sans', sans-serif; font-size: 16px; font-weight: 600;
    color: #1B2A4A; letter-spacing: -0.02em; margin: 0 0 8px 0;
}

.footer-text {
    text-align: center; padding: 24px 0; font-size: 13px; color: #5B6F8C;
    border-top: 1px solid #D1D9E6; margin-top: 48px;
}
</style>
""", unsafe_allow_html=True)

col_title, col_fluid = st.columns([3, 1])
with col_title:
    st.markdown('<div class="header-title">Density Prediction Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-subtitle">Comparing machine learning models with experimental PVT data for hydrocarbon fluids</div>', unsafe_allow_html=True)
with col_fluid:
    st.selectbox("Select Fluid", FLUID_NAMES, key='fluid',
                 label_visibility="collapsed")

df = load_data()
X = df[['T_K', 'P_MPa']].values
y_true = df['Density_kgm3'].values

trained = get_models(X, y_true, get_fluid_name())
MODEL_FILES = get_model_list(trained)

fluid_label = st.session_state.fluid.replace(' (91% mixture)', '')

VERDICTS = {
    'Ethane': ('Gradient Boosting', 'R²=0.99993 · CV R²=0.844 · MAE=0.97 kg/m³ · MAPE=0.75% · Normal Residuals ✓'),
    'Methane (91% mixture)': ('Gaussian Process', 'R²=1.000 · CV R²=0.999 · MAE=0.20 kg/m³ · MAPE=0.24% · Normal Residuals ✓'),
}
verdict_model, verdict_stats = VERDICTS[st.session_state.fluid]

st.markdown(
    f'<div class="verdict-badge">'
    f'<strong>Best Model: {verdict_model}</strong>'
    f'<span>— {verdict_stats}</span>'
    f'</div>',
    unsafe_allow_html=True
)

T_min, T_max = float(df['T_K'].min()), float(df['T_K'].max())
P_min, P_max = float(df['P_MPa'].min()), float(df['P_MPa'].max())

T_grid = np.linspace(T_min, T_max, 30)
P_grid = np.linspace(P_min, P_max, 30)
T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)
X_surf = np.column_stack([T_mesh.ravel(), P_mesh.ravel()])
y_surf = np.array([density_PR(t, p, fluid=st.session_state.fluid) for t, p in X_surf])
y_surf = y_surf.reshape(T_mesh.shape)

fig_hero = go.Figure()
fig_hero.add_trace(go.Surface(
    x=T_mesh, y=P_mesh, z=y_surf,
    colorscale=[[0, '#1B2A4A'], [0.5, '#2B5EA7'], [1, '#C7512E']],
    opacity=0.85, name='PR EOS Surface', showscale=False,
    contours=dict(x=dict(show=True, color='rgba(255,255,255,0.15)'),
                  y=dict(show=True, color='rgba(255,255,255,0.15)'))
))
fig_hero.add_trace(go.Scatter3d(
    x=df['T_K'], y=df['P_MPa'], z=df['Density_kgm3'],
    mode='markers',
    marker=dict(size=3, color='#FF6B35', symbol='circle', line=dict(width=0.5, color='rgba(0,0,0,0.3)')),
    name='Experimental'
))
fig_hero.update_layout(
    height=420, margin=dict(l=0, r=0, t=20, b=0),
    scene=dict(
        xaxis=dict(title='Temperature (K)', backgroundcolor='rgba(0,0,0,0)', gridcolor='#D1D9E6'),
        yaxis=dict(title='Pressure (MPa)', backgroundcolor='rgba(0,0,0,0)', gridcolor='#D1D9E6'),
        zaxis=dict(title='Density (kg/m³)', backgroundcolor='rgba(0,0,0,0)', gridcolor='#D1D9E6'),
        camera=dict(eye=dict(x=1.8, y=1.2, z=0.6)),
        bgcolor='rgba(0,0,0,0)'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)
st.plotly_chart(fig_hero, use_container_width=True)

st.sidebar.markdown(f'<div class="sidebar-header">Model Selection — {fluid_label}</div>', unsafe_allow_html=True)
sel = st.sidebar.selectbox("Choose ML Model", MODEL_FILES)
show_all = st.sidebar.checkbox("Show all models comparison", value=True)
show_virial = st.sidebar.checkbox("Show Virial Coefficients", value=True)
show_refprop_col = 'Density_RefProp_kgm3' in df.columns
show_refprop = st.sidebar.checkbox("Show RefProp comparison", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown(f'<div class="sidebar-header">Quick Predict</div>', unsafe_allow_html=True)
qT = st.sidebar.number_input("Temperature (K)", min_value=T_min, max_value=T_max,
                             value=(T_min + T_max) / 2, step=1.0)
qP = st.sidebar.number_input("Pressure (MPa)", min_value=P_min, max_value=P_max,
                             value=(P_min + P_max) / 2, step=0.5)
X_q = np.array([[qT, qP]])
if st.sidebar.button("Predict", type="primary", use_container_width=True):
    y_q = predict(sel, X_q, trained, st.session_state.fluid)[0]
    st.sidebar.success(f"**{sel}**\n\nρ = **{y_q:.4f}** kg/m³")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data & Predictions", "Statistical Metrics", "Model Comparison",
     "Virial Coefficients", "Custom Prediction"]
)

y_pred_sel = predict(sel, X, trained, st.session_state.fluid)

with tab1:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Experimental Data")
        cols = ['T_K', 'P_MPa', 'Density_kgm3']
        if show_refprop and show_refprop_col:
            cols += ['Density_RefProp_kgm3', 'Deviation_pct']
        st.dataframe(df[cols].round(3), height=600, width='stretch')
    with c2:
        st.subheader(f"Predictions: {sel}")
        dp = df.copy()
        dp['Predicted'] = y_pred_sel
        dp['Dev_%'] = 100 * (y_pred_sel - y_true) / y_true
        cols2 = ['T_K', 'P_MPa', 'Density_kgm3', 'Predicted', 'Dev_%']
        if show_refprop and show_refprop_col:
            cols2 += ['Density_RefProp_kgm3', 'Deviation_pct']
        st.dataframe(dp[cols2].round(3), height=600, width='stretch')

with tab2:
    st.subheader("Statistical Performance Metrics")
    r2, mae, rmse, mape, maxe, bias = metrics(y_true, y_pred_sel)
    cols = st.columns(6)
    cols[0].metric("R²", f"{r2:.6f}")
    cols[1].metric("MAE", f"{mae:.4f} kg/m³")
    cols[2].metric("RMSE", f"{rmse:.4f} kg/m³")
    cols[3].metric("MAPE", f"{mape:.4f}%")
    cols[4].metric("Max Error", f"{maxe:.4f} kg/m³")
    cols[5].metric("Bias", f"{bias:.4f} kg/m³")

    st.subheader("Parity Plot: Predicted vs Experimental")
    lo, hi = min(y_true.min(), np.nanmin(y_pred_sel)), max(y_true.max(), np.nanmax(y_pred_sel))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred_sel, mode='markers',
        marker=dict(color=df['T_K'], colorscale='Viridis', size=8, showscale=True,
                    colorbar=dict(title='T (K)')),
        text=[f"T={t}K, P={p}MPa" for t, p in zip(df['T_K'], df['P_MPa'])], name='Data'))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode='lines',
        line=dict(color='red', dash='dash'), name='Ideal'))
    fig.update_layout(xaxis_title="Experimental Density (kg/m³)",
                      yaxis_title="Predicted Density (kg/m³)", height=500)
    st.plotly_chart(fig, width='stretch')

    res = y_pred_sel - y_true
    dev_pct = 100 * res / y_true
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Residuals vs Pressure")
        fig = go.Figure()
        for T in sorted(df['T_K'].unique()):
            m = df['T_K'] == T
            fig.add_trace(go.Scatter(x=df.loc[m, 'P_MPa'], y=res[m],
                mode='lines+markers', name=f'{T:.1f} K'))
        fig.add_hline(y=0, line=dict(color='red', dash='dash'))
        fig.update_layout(xaxis_title="Pressure (MPa)", yaxis_title="Residual (kg/m³)", height=400)
        st.plotly_chart(fig, width='stretch')
    with c2:
        st.subheader("Relative Deviation (%)")
        fig = go.Figure()
        for T in sorted(df['T_K'].unique()):
            m = df['T_K'] == T
            fig.add_trace(go.Scatter(x=df.loc[m, 'P_MPa'], y=dev_pct[m],
                mode='lines+markers', name=f'{T:.1f} K'))
        fig.add_hline(y=0, line=dict(color='red', dash='dash'))
        fig.update_layout(xaxis_title="Pressure (MPa)", yaxis_title="Deviation (%)", height=400)
        st.plotly_chart(fig, width='stretch')

    st.subheader("Residual Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=res[~np.isnan(res)], nbinsx=30, name='Residuals'))
    mu, sg = np.nanmean(res), np.nanstd(res)
    xn = np.linspace(np.nanmin(res), np.nanmax(res), 200)
    fig.add_trace(go.Scatter(x=xn, y=stats.norm.pdf(xn, mu, sg) * np.sum(~np.isnan(res)) *
                             (np.nanmax(res)-np.nanmin(res))/30,
        mode='lines', name=f'μ={mu:.3f}, σ={sg:.3f}'))
    fig.update_layout(xaxis_title="Residual (kg/m³)", yaxis_title="Count", height=400)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Per-Isotherm Performance")
    rows = []
    for T in sorted(df['T_K'].unique()):
        m = df['T_K'] == T
        r2_i, mae_i, rmse_i, mape_i, _, _ = metrics(y_true[m], y_pred_sel[m])
        rows.append({'T (K)': T, 'R²': r2_i, 'MAE': mae_i, 'RMSE': rmse_i, 'MAPE (%)': mape_i})
    st.dataframe(pd.DataFrame(rows).round(4), width='stretch')

    if show_refprop and show_refprop_col:
        st.subheader("Comparison with RefProp")
        r2_r, mae_r, rmse_r, mape_r, maxe_r, bias_r = metrics(y_true, df['Density_RefProp_kgm3'].values)
        comp = pd.DataFrame({
            'Metric': ['R²', 'MAE (kg/m³)', 'RMSE (kg/m³)', 'MAPE (%)', 'Max Error (kg/m³)', 'Bias (kg/m³)'],
            sel: [r2, mae, rmse, mape, maxe, bias],
            'RefProp': [r2_r, mae_r, rmse_r, mape_r, maxe_r, bias_r]
        })
        st.dataframe(comp.round(6), width='stretch')

if show_all:
    with tab3:
        st.subheader(f"All Models Performance — {fluid_label}")
        perf_path = f'models/{get_fluid_name()}/model_performance.csv'
        if os.path.exists(perf_path):
            perf = pd.read_csv(perf_path)
            eos_rows = []
            for eos_name in EOS_MODELS:
                yp = predict(eos_name, X, trained, st.session_state.fluid)
                r, mae, rmse, mape, me, b = metrics(y_true, yp)
                eos_rows.append({'Model': eos_name, 'R2_train': r, 'MAE_train': mae,
                                 'RMSE_train': rmse, 'MAPE_train': mape})
            eos_df = pd.DataFrame(eos_rows)
            perf_all = pd.concat([perf, eos_df], ignore_index=True).sort_values('R2_train', ascending=False).reset_index(drop=True)
            st.dataframe(perf_all.round(6), width='stretch')

            fig = make_subplots(rows=2, cols=2,
                subplot_titles=('R² Score', 'MAE (kg/m³)', 'RMSE (kg/m³)', 'MAPE (%)'))
            n = len(perf_all)
            clrs = px.colors.qualitative.Plotly * (n // len(px.colors.qualitative.Plotly) + 1)
            for metric, pos in zip(
                ['R2_train', 'MAE_train', 'RMSE_train', 'MAPE_train'],
                [(1,1),(1,2),(2,1),(2,2)]
            ):
                fig.add_trace(go.Bar(x=perf_all['Model'], y=perf_all[metric],
                    marker_color=clrs[:n], text=perf_all[metric].round(4), textposition='auto'),
                    row=pos[0], col=pos[1])
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, width='stretch')

            st.subheader("Parity Plot: All Models + EOS")
            lo_all, hi_all = y_true.min(), y_true.max()
            fig = go.Figure()
            for i, m in enumerate(MODEL_FILES):
                yp = predict(m, X, trained, st.session_state.fluid)
                fig.add_trace(go.Scatter(x=y_true, y=yp, mode='markers',
                    marker=dict(size=4, color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]),
                    name=m))
            fig.add_trace(go.Scatter(x=[lo_all, hi_all], y=[lo_all, hi_all], mode='lines',
                line=dict(color='black', dash='dash'), name='Ideal'))
            fig.update_layout(xaxis_title="Experimental Density (kg/m³)",
                              yaxis_title="Predicted Density (kg/m³)", height=600)
            st.plotly_chart(fig, width='stretch')

with tab4:
    if show_virial:
        st.subheader(f"Second & Third Virial Coefficients — {fluid_label}")
        vdf = pd.DataFrame({
            'T (K)': [350.0, 400.0, 450.0],
            'B (cm³/mol)': [-130.71, -96.43, -71.29],
            'C (cm⁶/mol²)': [8084, 7327, 5912]
        })
        st.dataframe(vdf, width='stretch')
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vdf['T (K)'], y=vdf['B (cm³/mol)'],
                mode='lines+markers', line=dict(width=3), marker=dict(size=10)))
            fig.update_layout(title="Second Virial B(T)", xaxis_title="T (K)", yaxis_title="B (cm³/mol)", height=400)
            st.plotly_chart(fig, width='stretch')
        with c2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=vdf['T (K)'], y=vdf['C (cm⁶/mol²)'],
                mode='lines+markers', line=dict(width=3), marker=dict(size=10)))
            fig.update_layout(title="Third Virial C(T)", xaxis_title="T (K)", yaxis_title="C (cm⁶/mol²)", height=400)
            st.plotly_chart(fig, width='stretch')

with tab5:
    st.subheader("Custom Prediction")
    st.markdown(f"Enter temperature and pressure to predict **{fluid_label}** density.")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        cT = st.number_input("Temperature (K)", min_value=float(T_min), max_value=float(T_max),
                             value=(T_min + T_max) / 2, step=1.0, key="custom_T")
    with col_i2:
        cP = st.number_input("Pressure (MPa)", min_value=float(P_min), max_value=float(P_max),
                             value=(P_min + P_max) / 2, step=0.5, key="custom_P")

    X_custom = np.array([[cT, cP]])
    c_pred_sel = predict(sel, X_custom, trained, st.session_state.fluid)[0]

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Model", sel)
    col_r2.metric("Input Conditions", f"T = {cT:.1f} K, P = {cP:.2f} MPa")
    col_r3.metric("Predicted Density", f"{c_pred_sel:.4f} kg/m³" if not np.isnan(c_pred_sel) else "N/A")

    st.markdown("### Density vs Pressure at Selected Temperature")
    P_range = np.linspace(P_min, P_max, 200)
    X_range = np.column_stack([np.full_like(P_range, cT), P_range])
    y_range = predict(sel, X_range, trained, st.session_state.fluid)

    fig_custom = go.Figure()
    fig_custom.add_trace(go.Scatter(x=P_range, y=y_range, mode='lines',
        name=f'{sel} at {cT:.1f} K', line=dict(width=3)))
    exp_mask = np.abs(df['T_K'] - cT) < 1
    if exp_mask.any():
        fig_custom.add_trace(go.Scatter(
            x=df.loc[exp_mask, 'P_MPa'], y=df.loc[exp_mask, 'Density_kgm3'],
            mode='markers', name='Experimental', marker=dict(size=10, color='red', symbol='x')))
    fig_custom.add_vline(x=cP, line=dict(color='green', dash='dot'), annotation_text=f"P={cP:.1f}")
    fig_custom.update_layout(
        xaxis_title="Pressure (MPa)", yaxis_title="Density (kg/m³)",
        height=450, hovermode='x unified'
    )
    st.plotly_chart(fig_custom, width='stretch')

    st.markdown("### All Models Comparison at These Conditions")
    all_preds = {}
    for m in MODEL_FILES:
        v = predict(m, X_custom, trained, st.session_state.fluid)[0]
        all_preds[m] = [v]
    comp_df = pd.DataFrame(all_preds, index=['Density (kg/m³)']).T.reset_index()
    comp_df.columns = ['Model', 'Predicted Density (kg/m³)']
    st.dataframe(comp_df.round(4), width='stretch')

st.markdown('<div class="footer-text">© 2026 Density Prediction Platform by <a href="https://qazinasir.com" target="_blank" style="color:#5B6F8C;text-decoration:none">Qazi Nasir</a></div>', unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-header">Data Sources</div>', unsafe_allow_html=True)
if 'methane' in st.session_state.fluid.lower():
    st.sidebar.info("Patil et al., J. Chem. Thermodyn. 2007, 39, 1157-1163")
else:
    st.sidebar.info("Cristancho et al., J. Chem. Eng. Data 2010, 55, 2746-2749")
st.sidebar.markdown(f"<div style='color:#5B6F8C;font-size:13px'>T range: {T_min:.0f}–{T_max:.0f} K<br>P range: {P_min:.0f}–{P_max:.0f} MPa</div>", unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-header" style="margin-top:16px">EOS Models</div>', unsafe_allow_html=True)
st.sidebar.markdown("<div style='color:#5B6F8C;font-size:13px'>— Peng-Robinson<br>— SRK</div>", unsafe_allow_html=True)
