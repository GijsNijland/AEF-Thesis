import numpy as np
import pandas as pd
import statsmodels.api as sm

from HAR_regressions import rolling_forecast_ols, mse

from Data_handling import build_series
data, mid_5m, logret_5m, rv_daily = build_series()

rv = rv_daily.copy().sort_index()
idx = rv.index

# Function to load exogenous variables
def load_XVAR_excel(
    filename: str,
    date_col: str = "observation_date"
) -> pd.Series:
    
    df = pd.read_excel(filename, engine="openpyxl", decimal=",")
    if date_col not in df.columns:
        raise ValueError(f"{filename}: expected a '{date_col}' column")
    
    value_cols = [c for c in df.columns if c != date_col]
    if not value_cols:
        raise ValueError(f"{filename}: no value column besides '{date_col}'")
    val_col = value_cols[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    s = pd.to_numeric(df[val_col], errors="coerce")
    ser = pd.Series(s.values, index=df[date_col], name=val_col).dropna().sort_index()
    ser = ser[~ser.index.duplicated(keep="last")]
    return ser


def align_to_idx(s: pd.Series, target_idx: pd.DatetimeIndex) -> pd.Series:
    """
    Align a daily series to the RV trading-day index and forward fill gaps.
    """
    return s.reindex(target_idx).ffill()


#Import exogenous variables
EPU     = load_XVAR_excel("EPUIndexUS.xlsx")
FEDFUN  = load_XVAR_excel("FedfundsrateUS.xlsx")
INFL    = load_XVAR_excel("InflationUS.xlsx")
NASDAQ  = load_XVAR_excel("NasdaqcompositeUS.xlsx")
SP500   = load_XVAR_excel("SP500US.xlsx")
T10Y    = load_XVAR_excel("TenyearTrateUS.xlsx")
VIX     = load_XVAR_excel("VIXUS.xlsx")

# Align to RV index
EPU    = align_to_idx(EPU, idx)
FEDFUN = align_to_idx(FEDFUN, idx)
INFL   = align_to_idx(INFL, idx)
NASDAQ = align_to_idx(NASDAQ, idx)
SP500  = align_to_idx(SP500, idx)
T10Y   = align_to_idx(T10Y, idx)
VIX    = align_to_idx(VIX, idx)

#Compute returns for indices (SP500 and NASDAQ)
SP500_ret  = np.log(SP500 / SP500.shift(1))
NASDAQ_ret = np.log(NASDAQ / NASDAQ.shift(1))

#Endogenous regressors
rv_lag1  = rv.shift(1)
rv_week  = rv.shift(1).rolling(5).mean()
rv_month = rv.shift(1).rolling(22).mean()

# Construct regressor space for HAR-X
harx_df = pd.DataFrame({
    "rv": rv,
    "rv_lag1":  rv_lag1,
    "rv_week":  rv_week,
    "rv_month": rv_month,

    # Exogenous regressors
    "VIX_lag1":     VIX.shift(1),
    "T10Y_lag1":    T10Y.shift(1),
    "SP500_lag1":   SP500_ret.shift(1),
    "NASDAQ_lag1":  NASDAQ_ret.shift(1),
    "INFL_lag1":    INFL.shift(1),
    "FEDFUN_lag1":  FEDFUN.shift(1),
    "EPU_lag1":     EPU.shift(1)
}).dropna()

# Construct regressor space for L-HAR-X
log_rv        = np.log(rv)
log_rv_lag1   = np.log(rv_lag1)
log_rv_week   = np.log(rv_week)
log_rv_month  = np.log(rv_month)

lharx_df = pd.DataFrame({
    "logrv":        log_rv,
    "logrv_lag1":   log_rv_lag1,
    "logrv_week":   log_rv_week,
    "logrv_month":  log_rv_month,

    # Exogenous regressors NOT logged:
    "VIX_lag1":     VIX.shift(1),
    "T10Y_lag1":    T10Y.shift(1),
    "SP500_lag1":   SP500_ret.shift(1),
    "NASDAQ_lag1":  NASDAQ_ret.shift(1),
    "INFL_lag1":    INFL.shift(1),
    "FEDFUN_lag1":  FEDFUN.shift(1),
    "EPU_lag1":     EPU.shift(1)
}).dropna()

#Divide data in to training and test set (80/20)
split_idx = int(0.8 * len(harx_df))
split_date = harx_df.index[split_idx]

harx_train = harx_df[harx_df.index < split_date]
harx_test  = harx_df[harx_df.index >= split_date]

split_idx = int(0.8 * len(lharx_df))
split_date = lharx_df.index[split_idx]

lharx_train = lharx_df[lharx_df.index < split_date]
lharx_test  = lharx_df[lharx_df.index >= split_date]

feature_cols_harx = [
    "rv_lag1", "rv_week", "rv_month",
    "VIX_lag1", "T10Y_lag1", "SP500_lag1",
    "NASDAQ_lag1", "INFL_lag1", "FEDFUN_lag1", "EPU_lag1"
]

#HAR-X model forecasts and MSE
if __name__ == "__main__":
    print("\n HAR-X forecasts MSE")
    harx_fc, harx_actual = rolling_forecast_ols(
        harx_train, harx_test,
        feature_cols=[
    "logrv_lag1", "logrv_week", "logrv_month",
    "VIX_lag1", "T10Y_lag1", "SP500_lag1",
    "NASDAQ_lag1", "INFL_lag1", "FEDFUN_lag1", "EPU_lag1"
        ],
        y_col="rv"
    )
    print(mse(harx_fc, harx_actual))

# L-HAR-X forecasts and MSE
if __name__ == "__main__":
    print("\n L-HAR-X forecasts MSE")
    lharx_fc_log, lharx_actual_log = rolling_forecast_ols(
        lharx_train, lharx_test,
        feature_cols=[
    "logrv_lag1", "logrv_week", "logrv_month",
    "VIX_lag1", "T10Y_lag1", "SP500_lag1",
    "NASDAQ_lag1", "INFL_lag1", "FEDFUN_lag1", "EPU_lag1"
    ],
        y_col="logrv"
    )
    # Jensen correction 
    resid_var = (lharx_actual_log - lharx_fc_log).var()

    lharx_fc     = np.exp(lharx_fc_log + 0.5 * resid_var)
    lharx_actual = np.exp(lharx_actual_log)

    print(mse(lharx_fc, lharx_actual))