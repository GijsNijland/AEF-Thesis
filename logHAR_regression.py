import numpy as np
import pandas as pd
import statsmodels.api as sm

# Import time-series
from Data_handling import build_series
data, mid_5m, logret_5m, rv_daily = build_series()

# Construct regressor space dataframe for log‑HAR
log_rv = np.log(rv_daily).dropna()

log_rv_lag1  = log_rv.shift(1)
log_rv_week  = log_rv.shift(1).rolling(5).mean()
log_rv_month = log_rv.shift(1).rolling(22).mean()

loghar_df = pd.DataFrame({
    "logrv": log_rv,
    "logrv_lag1": log_rv_lag1,
    "logrv_week": log_rv_week,
    "logrv_month": log_rv_month
}).dropna()

#Divide data in to training and test set (80/20)
split_idx  = int(0.8 * len(loghar_df))
split_date = loghar_df.index[split_idx]

loghar_train = loghar_df[loghar_df.index < split_date]
loghar_test  = loghar_df[loghar_df.index >= split_date]

#Function for calculating mean squared error 
def mse(pred, actual):
    return ((pred - actual) ** 2).mean()

#Function for including constant
def _add_const_and_align(X_like: pd.DataFrame, ref_cols: pd.Index) -> pd.DataFrame:
    Xc = sm.add_constant(X_like, has_constant="add")
    Xc = Xc.reindex(columns=ref_cols, fill_value=0.0)
    if "const" in Xc.columns:
        Xc["const"] = 1.0
    return Xc

#Function for computing rolling forecasts
def rolling_forecast_ols(train_df, test_df, feature_cols, y_col="logrv"):
    forecasts_log, actual_log = [], []
    base_train = train_df.copy()

    for i, t in enumerate(test_df.index):
        in_sample = pd.concat([base_train, test_df.iloc[:i]]) if i > 0 else base_train

        X = sm.add_constant(in_sample[feature_cols], has_constant='add')
        y = in_sample[y_col]
        model = sm.OLS(y, X).fit()

        # Align design matrix
        X_t = _add_const_and_align(test_df.loc[[t], feature_cols], X.columns)
        pred_log = model.predict(X_t).iloc[0]

        forecasts_log.append(pred_log)
        actual_log.append(test_df.loc[t, y_col])

    return (
        pd.Series(forecasts_log, index=test_df.index),
        pd.Series(actual_log, index=test_df.index)
    )
    

#Model forecasts and MSE

#log-HAR
print("\n log-HAR")
loghar_fc_log, loghar_actual_log = rolling_forecast_ols(
    loghar_train, loghar_test,
    feature_cols=["logrv_lag1", "logrv_week", "logrv_month"],
    y_col="logrv"
)

# Jensen correction as in paper
resid_var = (loghar_actual_log - loghar_fc_log).var()

loghar_fc = np.exp(loghar_fc_log + 0.5 * resid_var)
loghar_actual = np.exp(loghar_actual_log)

print("MSE:", mse(loghar_fc, loghar_actual))