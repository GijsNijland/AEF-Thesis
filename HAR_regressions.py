import numpy as np
import pandas as pd
import statsmodels.api as sm


# Import time-series
from Data_handling import build_series
data, mid_5m, logret_5m, rv_daily = build_series()

#Construct regressor space dataframe for HAR
rv = rv_daily.copy()

rv_lag1  = rv.shift(1)
rv_week  = rv.shift(1).rolling(5).mean()
rv_month = rv.shift(1).rolling(22).mean()

har_df = pd.DataFrame({
    "rv": rv,
    "rv_lag1": rv_lag1,
    "rv_week": rv_week,
    "rv_month": rv_month
}).dropna()

#Construct regressor space dataframe for lev-HAR
daily_ret = logret_5m.resample("1D").sum().reindex(rv.index)
r_neg = daily_ret.clip(upper=0)

lev_df = pd.DataFrame({
    "rv": rv,
    "rv_lag1": rv_lag1,
    "rv_week": rv_week,
    "rv_month": rv_month,
    "rneg_lag1": r_neg.shift(1),
    "rneg_week": r_neg.shift(1).rolling(5).mean(),
    "rneg_month": r_neg.shift(1).rolling(22).mean()
}).dropna()

#Construct regressor space dataframe for SHAR
rv_up   = logret_5m.clip(lower=0).pow(2).resample("1D").sum().reindex(rv.index)
rv_down = logret_5m.clip(upper=0).pow(2).resample("1D").sum().reindex(rv.index)

shar_df = pd.DataFrame({
    "rv": rv,
    "rv_up_lag1": rv_up.shift(1),
    "rv_down_lag1": rv_down.shift(1),
    "rv_week": rv_week,
    "rv_month": rv_month
}).dropna()

#Construct regressor space dataframe forHARQ
n_per_day = int(logret_5m.groupby(logret_5m.index.date).size().median())
rq = (n_per_day / 3.0) * logret_5m.pow(4).resample("1D").sum()
rq = rq.reindex(rv.index)

harq_df = pd.DataFrame({
    "rv": rv,
    "rv_lag1": rv_lag1,
    "rv_week": rv_week,
    "rv_month": rv_month,
    "sqrt_rq_lag1": rq.shift(1).pow(0.5)
}).dropna()
harq_df["interaction"] = harq_df["sqrt_rq_lag1"] * harq_df["rv_lag1"]

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
split_idx = int(0.8 * len(har_df))
split_date = har_df.index[split_idx]

har_train  = har_df[har_df.index <  split_date]
har_test   = har_df[har_df.index >= split_date]

lev_train  = lev_df[lev_df.index <  split_date]
lev_test   = lev_df[lev_df.index >= split_date]

shar_train = shar_df[shar_df.index <  split_date]
shar_test  = shar_df[shar_df.index >= split_date]

harq_train = harq_df[harq_df.index <  split_date]
harq_test  = harq_df[harq_df.index >= split_date]

split_idx  = int(0.8 * len(loghar_df))
split_date = loghar_df.index[split_idx]

loghar_train = loghar_df[loghar_df.index < split_date]
loghar_test  = loghar_df[loghar_df.index >= split_date]


#Function for calculating mean squared error 
def mse(pred, actual):
    return ((pred - actual) ** 2).mean()

#Function for including constant
def _add_const_and_align(X_like: pd.DataFrame, ref_cols: pd.Index) -> pd.DataFrame:
    """
    Force-add a constant and align prediction exog to training exog columns.
    Missing non-const columns are filled with 0.0; const set to 1.0.
    """
    Xc = sm.add_constant(X_like, has_constant='add')
    Xc = Xc.reindex(columns=ref_cols, fill_value=0.0)
    if 'const' in Xc.columns:
        Xc['const'] = 1.0
    return Xc

#Function for computing rolling forecasts
def rolling_forecast_ols(train_df, test_df, feature_cols, y_col="rv"):
    
    forecasts, actuals = [], []
    base_train = train_df.copy()

    for i, t in enumerate(test_df.index):
        in_sample = pd.concat([base_train, test_df.iloc[:i]]) if i > 0 else base_train

        X = sm.add_constant(in_sample[feature_cols], has_constant='add')
        y = in_sample[y_col]
        model = sm.OLS(y, X).fit()

        # 1-row design matrix for time t, aligned to training columns
        X_t = _add_const_and_align(test_df.loc[[t], feature_cols], X.columns)
        pred = model.predict(X_t).iloc[0]

        forecasts.append(pred)
        actuals.append(test_df.loc[t, y_col])

    return pd.Series(forecasts, index=test_df.index), pd.Series(actuals, index=test_df.index)


#Model forecasts and MSE

#HAR model
print("\n HAR")
har_fc, har_actual = rolling_forecast_ols(
    har_train, har_test,
    feature_cols=["rv_lag1", "rv_week", "rv_month"],
    y_col="rv"
)
print("MSE:", mse(har_fc, har_actual))


# LevHAR model
print("\n LevHAR")
lev_fc, lev_actual = rolling_forecast_ols(
    lev_train, lev_test,
    feature_cols=["rv_lag1","rv_week","rv_month","rneg_lag1","rneg_week","rneg_month"],
    y_col="rv"
)
print("MSE:", mse(lev_fc, lev_actual))

# SHAR model
print("\n SHAR")
shar_fc, shar_actual = rolling_forecast_ols(
    shar_train, shar_test,
    feature_cols=["rv_up_lag1","rv_down_lag1","rv_week","rv_month"],
    y_col="rv"
)
print("MSE:", mse(shar_fc, shar_actual))

# HARQ model
print("\n HARQ")
harq_fc, harq_actual = rolling_forecast_ols(
    harq_train, harq_test,
    feature_cols=["rv_lag1","interaction","rv_week","rv_month"],
    y_col="rv"
)
print("MSE:", mse(harq_fc, harq_actual))

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