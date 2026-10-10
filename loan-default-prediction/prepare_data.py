"""Convert Kaggle 'Give Me Some Credit' (cs-training.csv) into data/loan_data.csv.

Usage (from the project root):
    python prepare_data.py

Download cs-training.csv from https://www.kaggle.com/competitions/GiveMeSomeCredit/data
(free Kaggle login, accept the competition rules) and put it in the project root.
"""
import pandas as pd

df = pd.read_csv("cs-training.csv")
df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])   # unnamed index column
df = df.rename(columns={"SeriousDlqin2yrs": "default"})
df = df[df["age"] >= 18]                                                   # drop impossible ages
feats = [c for c in df.columns if c != "default"]
df[feats] = df[feats].clip(upper=df[feats].quantile(0.995), axis=1)        # tame extreme outliers
df.to_csv("data/loan_data.csv", index=False)
print(df.shape, f"default rate = {df['default'].mean():.4f}")
