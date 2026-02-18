import pandas as pd


df = pd.read_excel("EURUSD_November2025.xlsx", header=None)
df.columns = ["symbol", "timestamp", "bid", "ask"]
print(df.head())
print("ii")
