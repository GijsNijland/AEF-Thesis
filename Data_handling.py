import pandas as pd
import numpy as np

#All data
files = [
    "EURUSD_November2025.xlsx",
    "EURUSD_December2025.xlsx",
    "EURUSD_January2026.xlsx",
]

# Merge files
dfs = []
for f in files:
    df = pd.read_excel(f, engine="openpyxl", header=None)
    
    #Assign column names
    df.columns = ["symbol", "timestamp", "bid", "ask"]

    #Timestamp
    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        format="%Y%m%d %H:%M:%S.%f",
        errors="raise"
    )
    
    df = df[["symbol", "datetime", "bid", "ask"]]
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data = data.sort_values("datetime").set_index("datetime")

#Rescale bid-ask prices to spot units
rescale_to_spot = True
rescale_factor = 1e5  
if rescale_to_spot:
    data["bid"] = data["bid"] / rescale_factor
    data["ask"] = data["ask"] / rescale_factor

#Calculate midprice
data["mid"] = (data["bid"] + data["ask"]) / 2.0

#Calculate 5-minute midprice (last price in each  bin)
mid_5m = data["mid"].resample("5min").last().dropna()

#Calculate 5-minute log-returns
logret_5m = np.log(mid_5m / mid_5m.shift(1)).dropna()

#Calculate daily realized variance
#(RV = sum of squared intraday 5-minute log-returns)
rv_daily = logret_5m.pow(2).resample("1D").sum().dropna()

print(rv_daily.head())

