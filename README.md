# Density Calculation

ML-based density prediction dashboard for ethane and methane (91% natural gas mixture).

## Features
- **Fluid Selector**: Switch between Ethane and Methane (91% mixture)
- **14 ML Models**: CatBoost, Gradient Boosting, Gaussian Process, Random Forest, etc.
- **Equation of State**: Peng-Robinson and SRK with fluid-specific properties
- **Custom Prediction**: Interactive density prediction at any T/P
- **Statistical Analysis**: R², MAE, RMSE, MAPE, residual diagnostics, etc.
- **Cross-Validation**: 5-fold CV for honest generalization estimates

## Data Sources
- **Ethane**: Cristancho et al., J. Chem. Eng. Data 2010, 55, 2746-2749
- **Methane (91% mixture)**: Patil et al., J. Chem. Thermodyn. 2007, 39, 1157-1163

## Run Locally
```bash
pip install -r requirements.txt
python3 train_models.py ethane
python3 train_models.py methane
python3 analysis.py ethane
python3 analysis.py methane
streamlit run dashboard.py
```

## Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to https://streamlit.io/cloud
3. Connect repo, set main file to `dashboard.py`
