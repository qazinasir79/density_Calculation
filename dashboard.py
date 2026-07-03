import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from models_def import PhysicsInspiredModel
from eos_models import density_PR, density_SRK, FLUIDS
import warnings
warnings.filterwarnings('ignore')

FLUID_NAMES = list(FLUIDS.keys())
EOS_MODELS = ['Peng-Robinson EOS', 'SRK EOS']

if 'fluid' not in st.session_state:
    st.session_state.fluid = FLUID_NAMES[0]

def get_fluid_dir():
    fname = st.session_state.fluid.lower().split('(')[0].strip().replace(' ', '_')
    if 'methane' in st.session_state.fluid.lower():
        return 'models/methane'
    return 'models/ethane'

def load_data():
    fname = st.session_state.fluid.lower().split('(')[0].strip().replace(' ', '_')
    if 'methane' in st.session_state.fluid.lower():
        return pd.read_csv('data/methane_data.csv')
    return pd.read_csv('data/ethane_data.csv')

def get_model_files():
    d = get_fluid_dir()
    if not os.path.exists(d):
        return []
    return sorted([f.replace('.pkl', '') for f in os.listdir(d) if f.endswith('.pkl')])

@st.cache_resource
def load_model(name, fluid_dir):
    if name in EOS_MODELS:
        return (None, None, None, 'eos')
    path = f'{fluid_dir}/{name}.pkl'
    if not os.path.exists(path):
        return (None, None, None, 'missing')
    if name == 'Neural Network':
        model, scaler_X, scaler_y = joblib.load(path)
        return (model, scaler_X, scaler_y, 'nn')
    return (joblib.load(path), None, None, 'sklearn')

@st.cache_data
def predict(name, X_input, fluid_dir, fluid_name):
    obj = load_model(name, fluid_dir)
    if obj[3] == 'eos':
        T = X_input[:, 0]
        P = X_input[:, 1]
        if name == 'Peng-Robinson EOS':
            return np.array([density_PR(t, p, fluid=fluid_name) for t, p in zip(T, P)])
        else:
            return np.array([density_SRK(t, p, fluid=fluid_name) for t, p in zip(T, P)])
    if obj[3] == 'missing':
        return np.full(len(X_input), np.nan)
    if obj[3] == 'nn':
        model, sx, sy, _ = obj
        return sy.inverse_transform(model.predict(sx.transform(X_input)).reshape(-1, 1)).ravel()
    return obj[0].predict(X_input)

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

st.set_page_config(page_title="Density Prediction ML Dashboard", layout="wide")

col_title, col_fluid = st.columns([3, 1])
with col_title:
    st.title("Density Prediction: ML Models vs Experimental Data")
with col_fluid:
    st.selectbox("Select Fluid", FLUID_NAMES, key='fluid',
                 label_visibility="collapsed")

df = load_data()
model_dir = get_fluid_dir()
MODEL_FILES = EOS_MODELS + get_model_files()
X = df[['T_K', 'P_MPa']].values
y_true = df['Density_kgm3'].values

fluid_label = st.session_state.fluid.replace(' (91% mixture)', '')

VERDICTS = {
    'Ethane': ('CatBoost', 'R²=0.999 · CV R²=0.888 · MAE=4.05 kg/m³ · MAPE=2.95% · Normal Residuals ✓'),
    'Methane (91% mixture)': ('Gaussian Process', 'R²=1.000 · CV R²=0.999 · MAE=0.20 kg/m³ · MAPE=0.24% · Normal Residuals ✓'),
}
verdict_model, verdict_stats = VERDICTS[st.session_state.fluid]

st.markdown(f"""
<div style="background:linear-gradient(135deg,#667eea,#764ba2);padding:18px 24px;border-radius:12px;margin-bottom:20px">
  <span style="color:white;font-size:20px;font-weight:700">🏆 Final Verdict: {verdict_model} is the Best Model for {fluid_label}</span>
  <span style="color:#ddd;font-size:14px;margin-left:16px">
    {verdict_stats}
  </span>
</div>
""", unsafe_allow_html=True)
st.sidebar.header(f"Model Selection — {fluid_label}")
sel = st.sidebar.selectbox("Choose ML Model", MODEL_FILES)
show_all = st.sidebar.checkbox("Show all models comparison", value=True)
show_virial = st.sidebar.checkbox("Show Virial Coefficients", value=True)
show_refprop_col = 'Density_RefProp_kgm3' in df.columns
show_refprop = st.sidebar.checkbox("Show RefProp comparison", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Quick Predict")
T_min, T_max = float(df['T_K'].min()), float(df['T_K'].max())
P_min, P_max = float(df['P_MPa'].min()), float(df['P_MPa'].max())
qT = st.sidebar.number_input("Temperature (K)", min_value=T_min, max_value=T_max,
                             value=(T_min + T_max) / 2, step=1.0)
qP = st.sidebar.number_input("Pressure (MPa)", min_value=P_min, max_value=P_max,
                             value=(P_min + P_max) / 2, step=0.5)
X_q = np.array([[qT, qP]])
if st.sidebar.button("Predict", type="primary", use_container_width=True):
    y_q = predict(sel, X_q, model_dir, st.session_state.fluid)[0]
    st.sidebar.success(f"**{sel}**\n\nρ = **{y_q:.4f}** kg/m³")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Data & Predictions", "Statistical Metrics", "Model Comparison",
     "Virial Coefficients", "Custom Prediction"]
)

y_pred_sel = predict(sel, X, model_dir, st.session_state.fluid)

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
        perf_path = f'{model_dir}/model_performance.csv'
        if os.path.exists(perf_path):
            perf = pd.read_csv(perf_path)
            eos_rows = []
            for eos_name in EOS_MODELS:
                yp = predict(eos_name, X, model_dir, st.session_state.fluid)
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
                yp = predict(m, X, model_dir, st.session_state.fluid)
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
    c_pred_sel = predict(sel, X_custom, model_dir, st.session_state.fluid)[0]

    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("Model", sel)
    col_r2.metric("Input Conditions", f"T = {cT:.1f} K, P = {cP:.2f} MPa")
    col_r3.metric("Predicted Density", f"{c_pred_sel:.4f} kg/m³" if not np.isnan(c_pred_sel) else "N/A")

    st.markdown("### Density vs Pressure at Selected Temperature")
    P_range = np.linspace(P_min, P_max, 200)
    X_range = np.column_stack([np.full_like(P_range, cT), P_range])
    y_range = predict(sel, X_range, model_dir, st.session_state.fluid)

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
        v = predict(m, X_custom, model_dir, st.session_state.fluid)[0]
        all_preds[m] = [v]
    comp_df = pd.DataFrame(all_preds, index=['Density (kg/m³)']).T.reset_index()
    comp_df.columns = ['Model', 'Predicted Density (kg/m³)']
    st.dataframe(comp_df.round(4), width='stretch')

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Sources")
if 'methane' in st.session_state.fluid.lower():
    st.sidebar.info("Patil et al., J. Chem. Thermodyn. 2007, 39, 1157-1163")
else:
    st.sidebar.info("Cristancho et al., J. Chem. Eng. Data 2010, 55, 2746-2749")
st.sidebar.markdown(f"### T range: {T_min:.0f}-{T_max:.0f} K, P range: {P_min:.0f}-{P_max:.0f} MPa")
st.sidebar.markdown("### EOS Models (thermodynamic)")
st.sidebar.markdown("- Peng-Robinson EOS")
st.sidebar.markdown("- SRK EOS")
