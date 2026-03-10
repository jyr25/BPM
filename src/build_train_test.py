import pandas as pd
import numpy as np

INPUT_PATH = "data/processed/phase2_dataset.csv"
TRAIN_OUT = "data/processed/train.csv"
TEST_OUT = "data/processed/test.csv"

df = pd.read_csv(INPUT_PATH)

# 1. Robust Date Sorting
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]) # Safety check
df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

# 2. Smarter Null Handling
# Instead of 0 for everything, we use the median for stats 
# so 'missing' players don't look like superstars or total busts.
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 3. Time-Based Split (Preventing mid-series splits)
# Finding the exact date that marks the 80% cutoff
split_date = df.iloc[int(len(df) * 0.8)]["date"]

train = df[df["date"] < split_date].copy()
test = df[df["date"] >= split_date].copy()

# 4. Final Save
train.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
test.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")

print(f"📊 Dataset Split Complete:")
print(f"📅 Split Date: {split_date.date()}")
print(f"✅ Train rows: {len(train)} (Games before {split_date.date()})")
print(f"🚀 Test rows: {len(test)} (Games from {split_date.date()} onwards)")