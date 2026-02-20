import pandas as pd
import numpy as np

#Function to construct the following time-series: 5-Minute mid-price (mid_5m),
#5-minute log-return(logret_5m), Daily realized variance (rv_daily)
def build_series(
    files=[
        "EURUSD_November2025.xlsx",
        "EURUSD_December2025.xlsx",
        "EURUSD_January2026.xlsx",
    ],
    rescale_to_spot=True,
    rescale_factor=1e5
):
    # Merge files
    dfs=[]
    for f in files:
        df=pd.read_excel(f, engine="openpyxl", header=None)
        
        # Assign column names
        df.columns=["symbol", "timestamp", "bid", "ask"]

        # Timestamp
        df["datetime"]=pd.to_datetime(
            df["timestamp"],
            format="%Y%m%d %H:%M:%S.%f",
            errors="raise"
        )
        
        df=df[["symbol", "datetime", "bid", "ask"]]
        dfs.append(df)

    data=pd.concat(dfs, ignore_index=True)
    data=data.sort_values("datetime").set_index("datetime")

    # Rescale bid-ask prices to spot units
    if rescale_to_spot:
        data["bid"]=data["bid"]/rescale_factor
        data["ask"]=data["ask"]/rescale_factor

    # Calculate midprice
    data["mid"]=(data["bid"]+data["ask"])/2.0

    # Calculate 5-minute midprice (last price in each bin)
    mid_5m=data["mid"].resample("5min").last().dropna()

    # Calculate 5-minute log-returns
    logret_5m=np.log(mid_5m / mid_5m.shift(1)).dropna()

    # Calculate daily realized variance (RV = sum of squared intraday returns)
    rv_daily=logret_5m.pow(2).resample("1D").sum().dropna()
    rv_daily=rv_daily.replace(0, np.nan).dropna()
    rv_daily=rv_daily[rv_daily > 0]
    rv_daily=rv_daily.sort_index()

    return data, mid_5m, logret_5m, rv_daily


if __name__=="__main__":
    _, _, _, rv_daily = build_series()
    rv_daily.to_csv("rv_daily.csv", index=True)
   

