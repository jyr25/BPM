import pandas as pd

INPUT_PATH = "data/processed/phase2_dataset.csv"
TRAIN_OUT = "data/processed/train.csv"
TEST_OUT = "data/processed/test.csv"

df = pd.read_csv(INPUT_PATH)

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

df = df.fillna(0)

split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

train.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")
test.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")

print("train rows:", len(train))
print("test rows:", len(test))