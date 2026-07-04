# Density Prediction Platform

ML-based density prediction dashboard for ethane, methane (91% mixture), and propane.

## Features
- **Fluid Selector**: Switch between Ethane, Methane (91% mixture), and Propane
- **14 ML Models**: Gradient Boosting, Gaussian Process, Random Forest, etc.
- **Equation of State**: Peng-Robinson and SRK with fluid-specific properties
- **3D PVT Surface**: Interactive visualisation of the density surface as a function of T and P
- **Custom Prediction**: Interactive density prediction at any T/P
- **Statistical Analysis**: R², MAE, RMSE, MAPE, residual diagnostics, etc.

## Data Sources
- **Ethane**: Cristancho et al., J. Chem. Eng. Data 2010, 55, 2746-2749
- **Methane (91% mixture)**: Patil et al., J. Chem. Thermodyn. 2007, 39, 1157-1163
- **Propane**: Glos et al., J. Chem. Thermodyn. 2004, 36, 1037-1059

## Run Locally
```bash
pip install -r requirements.txt
python3 train_models.py ethane
python3 train_models.py methane
python3 train_models.py propane
python3 analysis.py ethane
python3 analysis.py methane
python3 analysis.py propane
streamlit run dashboard.py
```

## Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repo, set main file to `dashboard.py`
