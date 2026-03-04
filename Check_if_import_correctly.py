from pathlib import Path
import pandas as pd

file_path = Path(r"C:\Users\gnijland\Documents\Code_Thesis_AEF\AEF-Thesis\Stockprice_data\obcdpimqcwp9eaz2.csv")  # <-- adjust this

# If your file isn’t comma-separated, try sep=";" or sep="\t"
df = pd.read_csv(file_path, encoding="utf-8-sig")  # change to encoding="latin-1" if needed

print("\n=== Head (first 5 rows) ===")
print(df.head())

print("\n=== Dimensions (rows, columns) ===")
print(df.shape)