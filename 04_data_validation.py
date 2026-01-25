import pandas as pd

df = pd.read_csv("master_dataset.csv")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nDuplicate user_id count:")
print(df["user_id"].duplicated().sum())

print("\nData types:")
print(df.dtypes)
