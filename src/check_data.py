import pandas as pd

df = pd.read_excel("/Users/sarunas/Documents/LT Elektros kainų prognozuotojas/LT_elektros_kain-_prognozuotojas/data/raw/Electricity price Litgrid/Combined electricity prices Litgrid.xlsx")

print(df.head())
print(df.info())

print("\nColumns:")
print(df.columns)

print("\nDatetime range:")
print(df.iloc[:,0].min(), "→", df.iloc[:,0].max())

print("\nDuplicates:")
print(df.duplicated().sum())