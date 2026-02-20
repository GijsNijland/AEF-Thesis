import numpy as np
import pandas as pd
import statsmodels.api as sm

# --- Reuse what you already have in HAR_regressions.py ---
# rolling_forecast_ols uses _add_const_and_align inside HAR_regressions' namespace,
# so importing the function is sufficient.
from HAR_regressions import rolling_forecast_ols, mse

# --- EURUSD realized variance input (same source you use everywhere) ---
from Data_handling import build_series
data, mid_5m, logret_5m, rv_daily = build_series()

# Keep trading-day index sorted and define index
rv = rv_daily.copy().sort_index()
idx = rv.index


# ---------------------------
# Helpers to load exogenous Excel files
# ---------------------------
def load_one_series_excel(
    filename: str,
    date_col: str = "observation_date"
) -> pd.Series:
    """
    Load a one-series Excel with columns:
      - observation_date
      - <variable>
    Values use European decimal commas.
    Returns a pd.Series with a DateTimeIndex.
    """
    df = pd.read_excel(filename, engine="openpyxl", decimal=",")
    if date_col not in df.columns:
        raise ValueError(f"{filename}: expected a '{date_col}' column")
    # The value column is the first non-date column
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
    Align a daily series to the RV trading-day index and forward-fill small gaps.
    """
    return s.reindex(target_idx).ffill()


# ---------------------------
# Load your exogenous variables (all daily; some missing dates)
# Filenames as provided:
#   EPUIndexUS.xlsx, FedfundsrateUS.xlsx, InflationUS.xlsx,
#   NasdaqcompositeUS.xlsx, SP500US.xlsx, TenyearTrateUS.xlsx, VIXUS.xlsx
# ---------------------------
EPU     = load_one_series_excel("EPUIndexUS.xlsx")
FEDFUN  = load_one_series_excel("FedfundsrateUS.xlsx")
INFL    = load_one_series_excel("InflationUS.xlsx")
NASDAQ  = load_one_series_excel("NasdaqcompositeUS.xlsx")
SP500   = load_one_series_excel("SP500US.xlsx")
T10Y    = load_one_series_excel("TenyearTrateUS.xlsx")
VIX     = load_one_series_excel("VIXUS.xlsx")

# Align to RV index and forward-fill
EPU    = align_to_idx(EPU, idx)
FEDFUN = align_to_idx(FEDFUN, idx)
INFL   = align_to_idx(INFL, idx)
NASDAQ = align_to_idx(NASDAQ, idx)
SP500  = align_to_idx(SP500, idx)
T10Y   = align_to_idx(T10Y, idx)
VIX    = align_to_idx(VIX, idx)

# Equity indices as returns (levels also fine; returns are common in HAR-X)
SP500_ret  = np.log(SP500 / SP500.shift(1))
NASDAQ_ret = np.log(NASDAQ / NASDAQ.shift(1))


# ---------------------------
# Construct HAR backbone (same logic as in HAR_regressions.py)
# ---------------------------
rv_lag1  = rv.shift(1)
rv_week  = rv.shift(1).rolling(5).mean()
rv_month = rv.shift(1).rolling(22).mean()

# ---------------------------
# Construct HAR-X dataframe (lag X's by 1 day for RV_{t+1} forecast)
# ---------------------------
harx_df = pd.DataFrame({
    "rv": rv,

    # HAR backbone
    "rv_lag1":  rv_lag1,
    "rv_week":  rv_week,
    "rv_month": rv_month,

    # Exogenous regressors (lagged by 1 day)
    "VIX_lag1":     VIX.shift(1),
    "T10Y_lag1":    T10Y.shift(1),
    "SP500_lag1":   SP500_ret.shift(1),
    "NASDAQ_lag1":  NASDAQ_ret.shift(1),
    "INFL_lag1":    INFL.shift(1),
    "FEDFUN_lag1":  FEDFUN.shift(1),
    "EPU_lag1":     EPU.shift(1)
}).dropna()


# ---------------------------
# 80/20 split, identical methodology
# ---------------------------
split_idx = int(0.8 * len(harx_df))
split_date = harx_df.index[split_idx]

harx_train = harx_df[harx_df.index < split_date]
harx_test  = harx_df[harx_df.index >= split_date]

# Feature list (order explicit and stable)
feature_cols_harx = [
    "rv_lag1", "rv_week", "rv_month",
    "VIX_lag1", "T10Y_lag1", "SP500_lag1",
    "NASDAQ_lag1", "INFL_lag1", "FEDFUN_lag1", "EPU_lag1"
]


# ---------------------------
# Run rolling OLS HAR-X forecast and print MSE
# ---------------------------
if __name__ == "__main__":
    print("\n HAR-X forecasts MSE")
    harx_fc, harx_actual = rolling_forecast_ols(
        harx_train, harx_test,
        feature_cols=feature_cols_harx,
        y_col="rv"
    )
    print(mse(harx_fc, harx_actual))